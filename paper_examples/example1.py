import os
os.environ.setdefault("AB_LOG", "DEBUG")  # AB_LOG=DEBUG for details
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from analyticBilliards import Ellipse, Scene, compileScene
from analyticBilliards.utils import plotScene, plotTrajectory

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

# unit circle: centre (0, 0), radii a = b = 1, no tilt
scene = Scene(g=9.8, solidObjects=[Ellipse(0.0, 0.0, 1.0, 1.0, 0.0)])
billiard = compileScene(scene, backend="openmp")

n = 3
x0, y0 = np.zeros(n), np.full(n, 0.3)
angle = np.linspace(0.3, 1.2, n)
vx0, vy0 = np.cos(angle), np.sin(angle)

# one row per collision (row 0 = initial state); position and velocity are
# all plotTrajectory needs to redraw the parabolas between collisions
run = billiard.recordWithIterations(x0, y0, vx0, vy0, iterations=200,
                                    save=["x", "y", "vx", "vy"])

fig, ax = plt.subplots(figsize=(4, 4))
plotScene(scene, ax)
plotTrajectory(ax, run.x, run.y, run.vx, run.vy, resolution=50, g=scene.g,
               particles=range(n), interval=(0, 200))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
for ext in ("png", "pdf"):
    plt.savefig(Path(__file__).with_name(f"circle.{ext}"), bbox_inches="tight",
                pad_inches=0.02)
