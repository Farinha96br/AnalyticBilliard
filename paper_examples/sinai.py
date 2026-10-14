import os
os.environ.setdefault("AB_LOG", "INFO")  # AB_LOG=DEBUG for details
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colormaps
from matplotlib.patches import Circle
from analyticBilliards import (Elipse, ElipseArc, LineSegment, Portal, Scene,
                               compileScene, updateScene)
from analyticBilliards.binding import EV_PORTAL
from analyticBilliards.utils import plotScene

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

# Periodic Sinai billiard (a Lorentz gas): a unit cell [-1/2, 1/2]^2 with a disc
# of radius R at the centre and a quarter disc at each corner. Four one-way
# portals glue opposite edges, so a particle leaving through one edge comes in
# through the other: the cell tiles the plane. The discs stay apart only for
# R < L / (2 sqrt 2); past that every particle is shut in a pocket.
#
# This lattice keeps open corridors along the diagonals for every R, an infinite
# horizon, so the MSD grows like t log t and D(t) = MSD / (4t) drifts slowly
# with t. That is the physics, not the numerics.
#
# Reference: G. Cristadoro, T. Gilbert, M. Lenci and D. P. Sanders, "Machta-
# Zwanzig regime of anomalous diffusion in infinite-horizon billiards", Phys.
# Rev. E 90, 050102(R) (2014), arXiv:1408.0349. Their table is square cells of
# side ell with corner discs of radius rho and corridors delta = ell - 2 rho
# along the cell axes. Ours is that table turned by 45 degrees, ell = L / sqrt 2,
# rho = R, and it has only those corridors for R > L / 4 (below that the axis
# corridors of our cell open as well, which their model does not include).

L = 1.0
H = L / 2
RADII = np.r_[np.linspace(0.05, 0.33, 12), 0.34, 0.345]
TF = 1000.0
N, BATCH = 6000, 2000
T_GRID = np.linspace(0.0, TF, 1001)
OUT = Path(__file__).parent


def edge(x0, y0, x1, y1):
    return LineSegment(x0, y0, x1, y1)


def sinaiScene(R):
    # each corner keeps only the quarter of its disc that lies inside the cell
    corners = [(H, H, np.pi, 1.5 * np.pi), (-H, H, 1.5 * np.pi, 2 * np.pi),
               (-H, -H, 0.0, 0.5 * np.pi), (H, -H, 0.5 * np.pi, np.pi)]
    right, left = edge(H, -H, H, H), edge(-H, -H, -H, H)
    top, bottom = edge(-H, H, H, H), edge(-H, -H, H, -H)
    return Scene(
        g=0.0,
        solidObjects=[Elipse(0.0, 0.0, R, R, 0.0)]
        + [ElipseArc(cx, cy, R, R, 0.0, p0, p1) for cx, cy, p0, p1 in corners],
        # same orientation on both sides and no flips: a pure translation by L
        portalObjects=[Portal(right, left), Portal(left, right),
                       Portal(top, bottom), Portal(bottom, top)])


def initialConditions(R, n, rng):
    """uniform in the free part of the cell, unit speed in a random direction"""
    centres = np.array([(0, 0), (H, H), (-H, H), (-H, -H), (H, -H)])
    x, y = np.empty(0), np.empty(0)
    while x.size < n:
        px, py = rng.uniform(-H, H, (2, 2 * n))
        free = np.all(np.hypot(px[:, None] - centres[:, 0],
                               py[:, None] - centres[:, 1]) > R, axis=1)
        x, y = np.r_[x, px[free]], np.r_[y, py[free]]
    angle = rng.uniform(0.0, 2 * np.pi, n)
    return x[:n], y[:n], np.cos(angle), np.sin(angle)


def runUntil(billiard, x0, y0, vx0, vy0, tf, iterations):
    """the event table, with enough events that every particle reaches tf"""
    while True:
        rec = billiard.recordWithIterations(x0, y0, vx0, vy0, iterations,
                                            save=["t", "x", "y", "evType"])
        rows = np.arange(rec.counts.size)
        assert (rec.counts == iterations + 1).all(), "a particle stopped early"
        if (rec.t[rows, rec.counts - 1] >= tf).all():
            return rec, iterations
        iterations *= 2


def realPositions(rec):
    """undo the portals. A portal row holds where the particle came OUT, so the
    edge it is on says which way it really went: out of the left edge means in
    through the right, one cell further along +x"""
    portal = rec.evType == EV_PORTAL

    def side(z):  # +1 out of the low edge, -1 out of the high one, else 0
        return (np.isclose(z, -H, atol=1e-9).astype(int)
                - np.isclose(z, H, atol=1e-9).astype(int))

    shiftX, shiftY = L * portal * side(rec.x), L * portal * side(rec.y)
    return rec.x + np.cumsum(shiftX, axis=1), rec.y + np.cumsum(shiftY, axis=1)


def onGrid(t, X, Y):
    """flights are straight at constant speed (g = 0): interpolating between
    events is exact"""
    Xg = np.array([np.interp(T_GRID, t[i], X[i]) for i in range(t.shape[0])])
    Yg = np.array([np.interp(T_GRID, t[i], Y[i]) for i in range(t.shape[0])])
    return Xg, Yg


def reference(R, t):
    """-> (D_MZ, D_log, D(t)) from Cristadoro et al., Eqs. (1), (2) and (8):

        MSD / (4t) = D_MZ + D_log [log t + O(1)],  D_MZ = ell^2 / (4 tau_R),
        D_log = delta ell / (4 tau_R),             tau_R = pi A / (4 p0 delta)

    tau_R is the mean time to leave a cell through one of its 4 gaps, A the free
    area of their cell and p0 = 1 the speed. The O(1) is not known, so D(t) drops
    it; the expansion is first order in delta / ell, a narrow-corridor result.
    NaN outside 0.25 < R < L / (2 sqrt 2), where it does not apply"""
    ell = L / np.sqrt(2)
    delta = ell - 2 * R
    if not (L / 4 < R < ell / 2):
        return np.nan, np.nan, np.nan
    tauR = np.pi * (ell ** 2 - np.pi * R ** 2) / (4 * delta)
    DMZ, Dlog = ell ** 2 / (4 * tauR), delta * ell / (4 * tauR)
    return DMZ, Dlog, DMZ + Dlog * np.log(t)


rng = np.random.default_rng(1)
billiard = compileScene(sinaiScene(RADII[0]), backend="openmp")

msd, D = [], []
iterations = int(4 * TF)
for R in RADII:
    updateScene(sinaiScene(R), billiard)
    total = np.zeros_like(T_GRID)
    x0, y0, vx0, vy0 = initialConditions(R, N, rng)
    for b in range(0, N, BATCH):
        part = slice(b, b + BATCH)
        rec, iterations = runUntil(billiard, x0[part], y0[part], vx0[part],
                                   vy0[part], TF, iterations)
        Xg, Yg = onGrid(rec.t, *realPositions(rec))
        total += ((Xg - Xg[:, :1]) ** 2 + (Yg - Yg[:, :1]) ** 2).sum(axis=0)
    msd.append(total / N)
    D.append(msd[-1][-1] / (4 * TF))  # D(tf), the quantity Eq. (8) predicts

ref = np.array([reference(R, TF) for R in RADII])
ell = L / np.sqrt(2)
print(f"D(t) = MSD / (4t) at t = {TF:g} vs Cristadoro et al., PRE 90, 050102 (2014)")
print("     R  delta/ell     D_MZ    D_log   D_ref(t)     D(t)  D/D_ref")
for R, d, (DMZ, Dlog, Dref) in zip(RADII, D, ref):
    if np.isnan(Dref):
        print(f"{R:6.3f}  {'-':>9}  {'-':>7}  {'-':>7}  {'-':>9}  {d:7.4f}  {'-':>7}")
    else:
        print(f"{R:6.3f}  {(ell - 2 * R) / ell:9.3f}  {DMZ:7.4f}  {Dlog:7.4f}  "
              f"{Dref:9.4f}  {d:7.4f}  {d / Dref:7.2f}")
np.savetxt(OUT / "sinai_diffusion.csv", np.c_[RADII, D, ref], delimiter=",",
           header="R,D,D_MZ,D_log,D_ref", comments="", fmt="%.6g")

# --- figure -------------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.4),
                                    gridspec_kw={"width_ratios": [1, 1.1, 1.1]})

# (a) one particle's real path over the tiled lattice, unit cell highlighted
R_SHOW = 0.2
scene = sinaiScene(R_SHOW)
updateScene(scene, billiard)
rec = billiard.recordWithIterations([0.0], [0.3], [np.cos(0.4)], [np.sin(0.4)],
                                    iterations=60, save=["t", "x", "y", "evType"])
X, Y = realPositions(rec)
for i in range(-2, 3):
    for j in range(-2, 3):
        for cx, cy in ((i, j), (i + H, j + H)):
            axA.add_patch(Circle((cx, cy), R_SHOW, fc="0.85", ec="0.5", lw=0.3))
plotScene(scene, axA)
axA.get_legend().remove()
axA.plot(X[0], Y[0], color=colormaps["rainbow"](0.0), lw=0.4)
axA.plot(X[0, 0], Y[0, 0], "^", color=colormaps["rainbow"](0.0), ms=3)
axA.set_xlim(-1.5, 1.5)
axA.set_ylim(-1.5, 1.5)
axA.set_title(rf"(a) $R = {R_SHOW}$")

# (b) the MSD, one curve per radius
colors = colormaps["rainbow"](np.linspace(0.0, 1.0, RADII.size))
for R, m, c in zip(RADII, msd, colors):
    axB.plot(T_GRID, m, color=c, lw=0.6)
axB.set_xlabel("$t$")
axB.set_ylabel(r"$\langle |\mathbf{r}(t)-\mathbf{r}(0)|^2 \rangle$")
axB.set_title("(b) mean squared displacement")
sm = plt.cm.ScalarMappable(cmap="rainbow",
                           norm=plt.Normalize(RADII[0], RADII[-1]))
fig.colorbar(sm, ax=axB, label="$R$", pad=0.02)

# (c) the diffusion coefficient
axC.plot(RADII, D, "o-", color="k", ms=3, lw=0.6, label="this work")
Rs = np.linspace(0.251, 0.3535, 200)
refCurve = np.array([reference(R, TF) for R in Rs])
axC.plot(Rs, refCurve[:, 2], "--", color="C3", lw=0.8, label="Ref., Eq. (8)")
axC.plot(Rs, refCurve[:, 0], ":", color="C3", lw=0.8, label=r"$D_{MZ}$")
axC.legend(fontsize=6, frameon=False)
axC.set_xlabel("$R$")
axC.set_yscale("log")
axC.set_ylim(bottom=1e-2)
axC.set_ylabel(rf"$D(t) = \langle r^2 \rangle / 4t$, $t = {TF:g}$")
axC.set_title("(c) diffusion coefficient")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"sinai.{ext}", bbox_inches="tight", pad_inches=0.02)
