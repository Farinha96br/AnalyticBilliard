import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frequentFunctions import SIDE, birkoffToSeries, compileScene, makeTable

PORTAL_WIDTH = 0.1
RESOLUTION = 512
MAX_TIME = 10000.0
STARTING_E = 0.01

MAX_EVENTS = 10000
N_OFFSETS = 5
N_ENERGIES = 10
CHUNK = 1024
G = 0.5
L = SIDE
BACKEND = "openmp"

here = pathlib.Path(__file__).parent

theta = np.linspace(0.0, 4.0 * L, RESOLUTION, endpoint=False)
p = np.linspace(-1.0, 1.0, RESOLUTION)
thetaGrid, pGrid = np.meshgrid(theta, p)

lefts = np.linspace(0.5 * L, 0.5 * PORTAL_WIDTH, N_OFFSETS)
energies = np.geomspace(STARTING_E, 1.1 * G * L, N_ENERGIES)

cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#e8e8e8")

for dLeft in lefts:
    dRight = L - dLeft
    delta = dRight - dLeft
    table = makeTable(g=G, dLeft=dLeft, dRight=dRight,
                      h=PORTAL_WIDTH, L=L)
    scene = compileScene(table, backend=BACKEND)

    for E0 in energies:
        x, y, vx, vy, allowed = birkoffToSeries(thetaGrid, pGrid, E0, G)
        x, y, vx, vy = x[allowed], y[allowed], vx[allowed], vy[allowed]
        nLive = x.size

        maxE, meanE = np.empty(nLive), np.empty(nLive)
        ranOut = np.empty(nLive, dtype=bool)

        for lo in range(0, nLive, CHUNK):
            block = slice(lo, lo + CHUNK)
            record = scene.recordWithIterations(x[block], y[block], vx[block],
                                                vy[block],
                                                iterations=MAX_EVENTS,
                                                save=["t", "Energy"])

            live = ((np.arange(record.t.shape[1]) < record.counts[:, None])
                    & (record.t <= MAX_TIME))
            maxE[block] = np.where(live, record.energy, -np.inf).max(axis=1)
            meanE[block] = (record.energy * live).sum(axis=1) / live.sum(axis=1)

            lastT = record.t[np.arange(record.counts.size), record.counts - 1]
            ranOut[block] = lastT < MAX_TIME

        print(f"delta = {delta:.3f}, E0 = {E0:.4f}: {nLive} conditions, "
              f"E in [{maxE.min():.4f}, {maxE.max():.4f}], "
              f"mean of meanE {meanE.mean():.4f}, "
              f"{100.0 * ranOut.mean():.1f}% out of events")

        figure, panels = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                      facecolor="white")

        for axes, values, name in zip(panels, (maxE - E0, meanE - E0),
                                      ("max", "mean")):
            picture = np.full(thetaGrid.shape, np.nan)
            picture[allowed] = values

            image = axes.imshow(picture, origin="lower", cmap=cmap,
                                aspect="auto", interpolation="nearest",
                                extent=(0.0, 4.0 * L, -1.0, 1.0))
            figure.colorbar(image, ax=axes)

            for corner in (L, 2.0 * L, 3.0 * L):
                axes.axvline(corner, color="white", lw=0.8, ls="--", alpha=0.6)

            axes.axvspan(4.0 * L - dLeft - 0.5 * PORTAL_WIDTH,
                         4.0 * L - dLeft + 0.5 * PORTAL_WIDTH,
                         color="#e8730a", alpha=0.55, lw=0.0,
                         label="left portal")
            axes.axvspan(L + dRight - 0.5 * PORTAL_WIDTH,
                         L + dRight + 0.5 * PORTAL_WIDTH,
                         color="#1f77b4", alpha=0.55, lw=0.0,
                         label="right portal")

            axes.set_title(f"{name} E $-$ E0", fontsize=10, loc="left")
            axes.set_xticks(np.arange(5) * L)
            axes.set_yticks([-1.0, 0.0, 1.0])
            axes.set_xlabel(r"$\theta$  (arclength)")
            axes.set_ylabel(r"$p = \cos\alpha$")

        panels[0].legend(loc="lower left", fontsize=8, framealpha=0.85)

        figure.suptitle(
            f"energy gained from each initial condition,   E0 = {E0:.4f},   "
            f"$\\Delta$ = {delta:.3f}   "
            f"(dLeft = {dLeft:.3f}, dRight = {dRight:.3f})\n"
            f"h = {PORTAL_WIDTH}, g = {G}, L = {L}, "
            f"{RESOLUTION}x{RESOLUTION} grid, {MAX_EVENTS} events or "
            f"t = {MAX_TIME}", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.94))

        name = here / f"energyMap_E{E0:.4f}_d{delta:.3f}.png"
        figure.savefig(name, dpi=150, facecolor="white")
        plt.close(figure)
        print(f"  wrote {name.name}")
