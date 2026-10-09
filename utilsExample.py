import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpcBilliards import compileScene
from hpcBilliards.utils import plotScene, plotTrajectory, toBirkoff

G = 0.5
H = 0.1
DLEFT, DRIGHT = 0.2, 0.7
BASIN_CORNER = (1.0, 1.0)
BASIN_RADIUS = 0.2
E = 1.0
NPARTICLES = 60
NBOUNCES = 600
NSHOWN = 4
INTERVAL = (0, 25)
RESOLUTION = 60

left = [0.0, DLEFT - 0.5 * H, 0.0, DLEFT + 0.5 * H]
right = [1.0, DRIGHT - 0.5 * H, 1.0, DRIGHT + 0.5 * H]

scene = {
    "g": G,
    "deadTime": 1e-10,
    "solidObjects": [
        {"type": "lineSegment", "params": [0.0, 0.0, 1.0, 0.0]},
        {"type": "lineSegment", "params": [1.0, 0.0, 1.0, DRIGHT - 0.5 * H]},
        {"type": "lineSegment", "params": [1.0, DRIGHT + 0.5 * H, 1.0, 1.0]},
        {"type": "lineSegment", "params": [1.0, 1.0, 0.0, 1.0]},
        {"type": "lineSegment", "params": [0.0, 1.0, 0.0, DLEFT + 0.5 * H]},
        {"type": "lineSegment", "params": [0.0, DLEFT - 0.5 * H, 0.0, 0.0]},
    ],
    "portalObjects": [
        {"entry": "lineSegment", "entryParams": left,
         "exit": "lineSegment", "exitParams": right,
         "normalFlip": 1.0, "tangentFlip": 1.0},
        {"entry": "lineSegment", "entryParams": right,
         "exit": "lineSegment", "exitParams": left,
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

rng = np.random.default_rng(0)
x0 = rng.uniform(0.0, 1.0, NPARTICLES)
y0 = rng.uniform(0.0, 1.0, NPARTICLES)
speed = np.sqrt(2.0 * (E - G * y0))
eta = rng.uniform(0.0, 2.0 * np.pi, NPARTICLES)

run = billiard.recordWithIterations(x0, y0, speed * np.cos(eta),
                                    speed * np.sin(eta), iterations=NBOUNCES,
                                    save=["x", "y", "vx", "vy"],
                                    eventType=["bounce"])

written = np.arange(run.x.shape[1])[None, :] < run.counts[:, None]
written[:, 0] = False
theta, p = toBirkoff(run.x[written], run.y[written],
                     run.vx[written], run.vy[written])

print(f"{written.sum()} collisions, theta in [{theta.min():.3f}, "
      f"{theta.max():.3f}], p in [{p.min():.3f}, {p.max():.3f}]")

figure, (table, birkoff) = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                        facecolor="white")

plotScene(scene, table)

shown = np.flatnonzero(run.counts >= INTERVAL[1])[:NSHOWN]
plotTrajectory(table, run.x, run.y, run.vx, run.vy, RESOLUTION, G, shown,
               INTERVAL, lw=0.9, alpha=0.85, zorder=2)
table.set_title(f"the table, with events {INTERVAL[0]} to {INTERVAL[1]} of "
                f"{len(shown)} particles", fontsize=11, loc="left")

birkoff.plot(theta, p, ",", color="black", alpha=0.5)
for corner in (1.0, 2.0, 3.0):
    birkoff.axvline(corner, color="tab:blue", lw=0.9, ls="--", alpha=0.8)


figure.suptitle(f"g = {G}, E = {E}, portals of length {H} at {DLEFT} and "
                f"{DRIGHT}, a circular basin of radius {BASIN_RADIUS} at "
                f"{BASIN_CORNER}", fontsize=11)
figure.tight_layout()
figure.savefig("utilsExample.png", dpi=150, facecolor="white")
print("wrote utilsExample.png")
