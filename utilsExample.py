"""The plotting helpers in hpcBilliards.utils, on one scene.

A unit square under gravity, two portals joining the side walls, and a basin in
the top-right corner -- drawn, but never asked for, so it changes nothing here.

Writes utilsExample.png: the table left, the Birkhoff section right.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # straight to a file, no window
import matplotlib.pyplot as plt

from hpcBilliards import (Elipse, LineSegment, Portal, Scene, compileScene)
from hpcBilliards.utils import plotScene, plotTrajectory, toBirkoff

# geometry. the table is the unit square; the mouths are its only gaps
G = 0.5
H = 0.1                      # mouth length, the same on both walls
DLEFT, DRIGHT = 0.2, 0.7     # mouth centres, left wall and right wall
BASIN_CORNER = (1.0, 1.0)
BASIN_RADIUS = 0.2

# the run. same energy E for all, so the start height fixes the start speed
E = 1.0
NPARTICLES = 60
NBOUNCES = 600

# what to draw: NSHOWN trajectories over INTERVAL, RESOLUTION points per arc
NSHOWN = 4
INTERVAL = (0, 25)
RESOLUTION = 60

# 1. the scene
halfH = 0.5 * H
leftMouth = LineSegment(0.0, DLEFT - halfH, 0.0, DLEFT + halfH)
rightMouth = LineSegment(1.0, DRIGHT - halfH, 1.0, DRIGHT + halfH)

scene = Scene(
    g=G,
    deadTime=1e-10,

    # anticlockwise from the origin, each side wall in two pieces
    solidObjects=[
        LineSegment(0.0, 0.0, 1.0, 0.0),
        LineSegment(1.0, 0.0, 1.0, DRIGHT - halfH),
        LineSegment(1.0, DRIGHT + halfH, 1.0, 1.0),
        LineSegment(1.0, 1.0, 0.0, 1.0),
        LineSegment(0.0, 1.0, 0.0, DLEFT + halfH),
        LineSegment(0.0, DLEFT - halfH, 0.0, 0.0),
    ],

    # one per direction. the same mouths the walls were cut around
    portalObjects=[
        Portal(entry=leftMouth, exit=rightMouth),
        Portal(entry=rightMouth, exit=leftMouth),
    ],

    basinObjects={
        1: [Elipse(*BASIN_CORNER, BASIN_RADIUS, BASIN_RADIUS, 0.0)],
    },
)

billiard = compileScene(scene, backend="openmp")
print(f"{len(billiard.types)} objects -> {billiard.so_path.name}, "
      f"cached: {billiard.cached}")

# 2. initial conditions: uniform in the square, random direction, shell speed
rng = np.random.default_rng(0)
x0 = rng.uniform(0.0, 1.0, NPARTICLES)
y0 = rng.uniform(0.0, 1.0, NPARTICLES)
eta = rng.uniform(0.0, 2.0 * np.pi, NPARTICLES)
speed = np.sqrt(2.0 * (E - G * y0))

run = billiard.recordWithIterations(x0, y0, speed * np.cos(eta),
                                    speed * np.sin(eta), iterations=NBOUNCES,
                                    save=["x", "y", "vx", "vy"],
                                    eventType=["bounce"])

# 3. the Birkhoff section. keep the filled rows; drop column 0, the start
written = np.arange(run.x.shape[1])[None, :] < run.counts[:, None]
written[:, 0] = False

theta, p = toBirkoff(run.x[written], run.y[written],
                     run.vx[written], run.vy[written])
print(f"{written.sum()} collisions, theta in [{theta.min():.3f}, "
      f"{theta.max():.3f}], p in [{p.min():.3f}, {p.max():.3f}]")

# 4. the figure
figure, (table, birkoff) = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                        facecolor="white")

# the table
plotScene(scene, table)

# then the trajectories
shown = np.flatnonzero(run.counts >= INTERVAL[1])[:NSHOWN]
plotTrajectory(table, run.x, run.y, run.vx, run.vy, RESOLUTION, G, shown,
               INTERVAL, lw=0.9, alpha=0.85, zorder=2)
table.set_title(f"the table, events {INTERVAL[0]} to {INTERVAL[1]} of "
                f"{len(shown)} particles", fontsize=11, loc="left")

# the section
birkoff.plot(theta, p, ",", color="black", alpha=0.5)
for corner in (1.0, 2.0, 3.0):  # theta is 0 to 4, one unit per wall
    birkoff.axvline(corner, color="tab:blue", lw=0.9, ls="--", alpha=0.8)
birkoff.set_xlabel("theta, arc length around the boundary")
birkoff.set_ylabel("p, velocity along the wall")
birkoff.set_title(f"the section, every collision of {NPARTICLES} particles",
                  fontsize=11, loc="left")

figure.suptitle(f"g = {G}, E = {E}, portals of length {H} at {DLEFT} and "
                f"{DRIGHT}, a circular basin of radius {BASIN_RADIUS} at "
                f"{BASIN_CORNER}", fontsize=11)
figure.tight_layout()
figure.savefig("utilsExample.png", dpi=150, facecolor="white")
print("wrote utilsExample.png")
