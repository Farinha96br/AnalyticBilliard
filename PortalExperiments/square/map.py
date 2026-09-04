# the phase space of the square billiard in gravity, in the coordinates of
# "Circular, elliptic and oval billiards in a gravitational field": a point per
# collision, at (theta, p).
#
# theta is ARCLENGTH here rather than an angle -- a square has no natural angle
# to a boundary point -- measured counterclockwise from the bottom-left corner,
# so it runs ground, right wall, top, left wall and ends at 4L. p is the
# outgoing velocity's tangential component once it is normalised, cos of the
# angle the paper calls alpha, which is what makes the pair canonical.
#
# this starts at h = 0, where the portals close up and the table is a plain box.
# That case is the control: a rectangle in gravity SEPARATES -- the horizontal
# motion is free bouncing between two walls, the vertical is a ball on a floor
# under a ceiling, and neither one feeds the other -- so it is integrable and the
# map has to come out as curves rather than a chaotic sea. Anything that is not
# regular once h > 0 is the portals' doing and not the box's

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the table, the initial conditions and the way a run is read back all live
# beside this script, and that module is what puts the repo root on the path
from frequentFunctions import (SIDE, compileScene, heightAt, makeTable,
                               realRows, seriesToBirkoff, startWithFixedEnergy)

here = pathlib.Path(__file__).parent    # figures land beside the script

G = 0.5
L = SIDE
H = 0.0             # the portals are closed: a plain box, and the control case
NPARTICLES = 60
NBOUNCES = 5000

# the ceiling sits at y = L, so g*L is the energy that just reaches it -- 0.5
# here. The set below straddles that: the first two clear it, the last three are
# shut under it and never see the top wall at all, and E = 0.5 is exactly
# marginal. A particle arrives at the ceiling there with no speed left, so it
# never actually strikes it: that panel's top band comes out empty without being
# shaded, which is right -- the wall is reachable, just not hittable
energies = [5.0, 1.0, 0.5, 0.25, 0.1, 0.05]

# the ground is at y = 0 and the solver measures the potential from y = 0 too,
# so E here is E in the paper's convention with nothing to reconcile -- unlike
# the circle, where the paper counts from the bottom of the boundary and the
# energies have to be shifted by g*R
table = makeTable(g=G, dLeft=0.0, dRight=0.0, h=H, L=L)
billiard = compileScene(table, backend="gpu_openmp")
print(f"{len(billiard.types)} objects -> {billiard.so_path.name}, "
      f"cached: {billiard.cached}")

figure, axesGrid = plt.subplots(2, 3, figsize=(15.0, 8.0), facecolor="white")

for axes, E in zip(axesGrid.ravel(), energies):
    x0, y0, vx0, vy0 = startWithFixedEnergy(E, NPARTICLES, G, L, seed=0)

    # bounces only. Row 0 is the initial state, not a collision, and its
    # velocity is whatever the particle was launched with rather than something
    # a wall just turned -- the angle to the wall would be meaningless there
    run = billiard.recordWithIterations(x0, y0, vx0, vy0, iterations=NBOUNCES,
                                        save=["x", "y", "vx", "vy"],
                                        eventType=["bounce"])

    x, y, vx, vy = realRows(run)
    theta, p = seriesToBirkoff(x, y, vx, vy)

    # what the energy shuts out: at arclength theta the wall is at height
    # heightAt(theta), and a particle needs E >= g*y to be there at all. It
    # depends on theta alone, so it is a band rather than a region
    grid = np.linspace(0.0, 4.0 * L, 2000)
    axes.fill_between(grid, -1.0, 1.0, where=E < G * heightAt(grid, L),
                      color="#e8e8e8", zorder=0)

    axes.plot(theta, p, ",", color="black", alpha=0.5, zorder=1)

    # the corners, where one wall's tangent gives way to the next
    for corner in (L, 2.0 * L, 3.0 * L):
        axes.axvline(corner, color="tab:blue", lw=0.9, ls="--", alpha=0.8,
                     zorder=2)

    axes.set_xlim(0.0, 4.0 * L)
    axes.set_ylim(-1.0, 1.0)
    axes.set_xticks(np.arange(5) * L)
    axes.set_yticks([-1.0, 0.0, 1.0])
    axes.set_xlabel(r"$\theta$  (arclength)")
    axes.set_ylabel(r"$p = \cos\alpha$")
    axes.set_title(f"E = {E}", loc="left", fontsize=11)

figure.suptitle(
    f"square billiard in gravity, h = {H} (the portals closed), g = {G}, L = {L}"
    f"   ({NPARTICLES} particles x {NBOUNCES} collisions)\n"
    f"dashed: the corners, ground | right | top | left   "
    f"grey: the wall there is higher than E can reach", fontsize=11)
figure.tight_layout()

name = here / f"map_h{H:.2f}.png"
figure.savefig(name, dpi=150, facecolor="white")
plt.close(figure)
print(f"wrote {name.name}")
