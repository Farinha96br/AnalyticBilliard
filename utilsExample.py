"""The plotting helpers in hpcBilliards.utils, on one scene.

A unit square under gravity, with a pair of portals that teleport a particle
from one side wall to the other, and a circular basin in the top-right corner
that would end a run -- here we never ask for basins, so it is drawn but has no
effect on the trajectories.

Writes utilsExample.png: the table on the left, the Birkhoff section on the
right.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, the figure goes straight to a file
import matplotlib.pyplot as plt

from hpcBilliards import compileScene
from hpcBilliards.utils import plotScene, plotTrajectory, toBirkoff

# geometry. the table is the unit square, so a portal mouth is given by its
# height on the wall and the two mouths are the only gaps in it
G = 0.5
H = 0.1                      # mouth length, the same on both walls
DLEFT, DRIGHT = 0.2, 0.7     # mouth centres, left wall and right wall
BASIN_CORNER = (1.0, 1.0)
BASIN_RADIUS = 0.2

# the run. every particle starts with the same total energy E, so where it
# starts fixes how fast it starts
E = 1.0
NPARTICLES = 60
NBOUNCES = 600

# what to draw. only NSHOWN trajectories go on the table, each over the events
# in INTERVAL, with RESOLUTION points per parabola
NSHOWN = 4
INTERVAL = (0, 25)
RESOLUTION = 60

# 1. the scene
halfH = 0.5 * H
leftMouth = [0.0, DLEFT - halfH, 0.0, DLEFT + halfH]
rightMouth = [1.0, DRIGHT - halfH, 1.0, DRIGHT + halfH]

scene = {
    "g": G,
    "deadTime": 1e-10,

    # the square, walked anticlockwise from the origin. the two side walls come
    # in two pieces each, leaving the mouths open
    "solidObjects": [
        {"type": "lineSegment", "params": [0.0, 0.0, 1.0, 0.0]},
        {"type": "lineSegment", "params": [1.0, 0.0, 1.0, DRIGHT - halfH]},
        {"type": "lineSegment", "params": [1.0, DRIGHT + halfH, 1.0, 1.0]},
        {"type": "lineSegment", "params": [1.0, 1.0, 0.0, 1.0]},
        {"type": "lineSegment", "params": [0.0, 1.0, 0.0, DLEFT + halfH]},
        {"type": "lineSegment", "params": [0.0, DLEFT - halfH, 0.0, 0.0]},
    ],

    # one portal per direction: entering either mouth puts the particle at the
    # other, keeping both components of its velocity
    "portalObjects": [
        {"entry": "lineSegment", "entryParams": leftMouth,
         "exit": "lineSegment", "exitParams": rightMouth,
         "normalFlip": 1.0, "tangentFlip": 1.0},
        {"entry": "lineSegment", "entryParams": rightMouth,
         "exit": "lineSegment", "exitParams": leftMouth,
         "normalFlip": 1.0, "tangentFlip": 1.0},
    ],

    "basinObjects": {
        1: [{"type": "elipse",
             "params": [*BASIN_CORNER, BASIN_RADIUS, BASIN_RADIUS, 0.0]}],
    },
}

billiard = compileScene(scene, backend="openmp")
print(f"{len(billiard.types)} objects -> {billiard.so_path.name}, "
      f"cached: {billiard.cached}")

# 2. initial conditions: uniform over the square, in a uniformly random
# direction, at the speed that puts every particle on the same energy shell
rng = np.random.default_rng(0)
x0 = rng.uniform(0.0, 1.0, NPARTICLES)
y0 = rng.uniform(0.0, 1.0, NPARTICLES)
eta = rng.uniform(0.0, 2.0 * np.pi, NPARTICLES)
speed = np.sqrt(2.0 * (E - G * y0))

run = billiard.recordWithIterations(x0, y0, speed * np.cos(eta),
                                    speed * np.sin(eta), iterations=NBOUNCES,
                                    save=["x", "y", "vx", "vy"],
                                    eventType=["bounce"])

# 3. the Birkhoff section. rows are padded out to NBOUNCES, so keep only the
# ones a particle actually filled, and drop column 0 -- that is where it
# started, not somewhere it hit
written = np.arange(run.x.shape[1])[None, :] < run.counts[:, None]
written[:, 0] = False

theta, p = toBirkoff(run.x[written], run.y[written],
                     run.vx[written], run.vy[written])
print(f"{written.sum()} collisions, theta in [{theta.min():.3f}, "
      f"{theta.max():.3f}], p in [{p.min():.3f}, {p.max():.3f}]")

# 4. the figure
figure, (table, birkoff) = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                        facecolor="white")

# plot the table
plotScene(scene, table)

# Plot the trajectories
shown = np.flatnonzero(run.counts >= INTERVAL[1])[:NSHOWN]
plotTrajectory(table, run.x, run.y, run.vx, run.vy, RESOLUTION, G, shown,
               INTERVAL, lw=0.9, alpha=0.85, zorder=2)
table.set_title(f"the table, events {INTERVAL[0]} to {INTERVAL[1]} of "
                f"{len(shown)} particles", fontsize=11, loc="left")


# plotting the mapped points on the Birkhoff section. the horizontal axis is the arc length
birkoff.plot(theta, p, ",", color="black", alpha=0.5)
for corner in (1.0, 2.0, 3.0):  # theta runs 0 to 4, one unit per wall
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
