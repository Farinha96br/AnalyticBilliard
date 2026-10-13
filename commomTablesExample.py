import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpcBilliards import commomTables as T
from hpcBilliards.utils import plotScene

TABLES = {
    "circle(R=1)": T.circle(1.0),
    "oval(a=1, b=0.5, tilt=0.3)": T.oval(1.0, 0.5, 0.3),
    "stadium(r=1, a=1)": T.stadium(1.0, 1.0),
    "lemon(B=0.5, R=1)": T.lemon(0.5, 1.0),
    "mushroom(R=1, w=0.5, h=1, offset=0.2)": T.mushroom(1.0, 0.5, 1.0, 0.2),
    "poliLine(R=1, N=5, theta0=0.2)": T.poliLine(1.0, 5, 0.2),
    "poliLineSeg(R=1, N=6, theta0=0.1)": T.poliLineSeg(1.0, 6, 0.1),
    "flower(R=1, N=5, theta0=0, X=0.9)": T.flower(1.0, 5, 0.0, 0.9),
    "flower(R=1, N=4, theta0=0.3, X=-0.5)": T.flower(1.0, 4, 0.3, -0.5),
    "PolyCircle(N=3, R=1, r=0.15, d=0.5)": T.PolyCircle(3, 1.0, 0.15, 0.5),
}

figure, axes = plt.subplots(2, 5, figsize=(20.0, 8.0), facecolor="white")
for ax, (name, scene) in zip(axes.ravel(), TABLES.items()):
    if name.startswith("poliLine("):
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    plotScene(scene, ax)
    ax.set_title(name, fontsize=10, loc="left")

figure.tight_layout()
figure.savefig("commomTablesExample.png", dpi=150, facecolor="white")
