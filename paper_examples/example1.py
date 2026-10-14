import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from analyticBilliards import compileScene
from analyticBilliards.commomTables import circle
from analyticBilliards.utils import plotScene, plotTrajectory

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

scene = circle(R=1.0, g=1.0)
billiard = compileScene(scene, backend="openmp")

n = 3
x0, y0 = np.zeros(n), np.full(n, -0.5)
angle = np.linspace(0.3, 1.2, n)
vx0, vy0 = np.cos(angle), np.sin(angle)

run = billiard.recordWithIterations(x0, y0, vx0, vy0, iterations=50)

fig, ax = plt.subplots(figsize=(4, 4))
plotScene(scene, ax)
plotTrajectory(ax, run.x, run.y, run.vx, run.vy, resolution=50, g=scene.g,
               particles=range(n), interval=(0, 50))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
for ext in ("png", "pdf"):
    plt.savefig(Path(__file__).with_name(f"circle.{ext}"), bbox_inches="tight",
                pad_inches=0.02)
