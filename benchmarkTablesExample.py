import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyticBilliards import commonTables as T
from analyticBilliards.utils import plotScene

CM = 1 / 2.54
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.size": 7})

# the stacks are drawn short: at their benchmark length, L = 1e8, nothing but a
# line would show
TABLES = [
    (r"(a) \texttt{rectangle(a, b)}", T.rectangle(1.0, 1.0)),
    (r"(b) \texttt{nestedEllipses(4)}", T.nestedEllipses(4)),
    (r"(c) \texttt{nestedEllipseArcs(4)}", T.nestedEllipseArcs(4)),
    (r"(d) \texttt{portalBox(6)}", T.portalBox(6)),
    (r"(e) \texttt{stackedLineSegs(4, L=3)}", T.stackedLineSegs(4, L=3.0)),
    (r"(f) \texttt{stackedLines(4)}", T.stackedLines(4)),
    (r"(g) \texttt{lorentzGas(w=0.3)}", T.lorentzGas(0.3)),
]
# the infinite lines are drawn across whatever the axes show, so they get the
# limits of their finite twin
LIMITS = {r"(f) \texttt{stackedLines(4)}": ((-3.3, 3.3), (-3.3, 0.3))}

figure, axes = plt.subplots(2, 4, figsize=(17 * CM, 8.4 * CM))
for ax, (title, scene) in zip(axes.flat, TABLES):
    if title in LIMITS:
        ax.set_xlim(*LIMITS[title][0])
        ax.set_ylim(*LIMITS[title][1])
    plotScene(scene, ax)
    if ax.get_legend():  # the portal legend would cover a panel this small
        ax.get_legend().remove()
    for line in ax.lines:
        line.set_linewidth(0.9)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.axis("off")
    ax.set_title(title, fontsize=6, loc="left", pad=2)
for ax in axes.flat[len(TABLES):]:
    ax.remove()

figure.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.01,
                       wspace=0.04, hspace=0.12)
figure.savefig("benchmarkTablesExample.pdf")
figure.savefig("benchmarkTablesExample.png", dpi=300)
