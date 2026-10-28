import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from analyticBilliards import commonTables as T
from analyticBilliards.utils import plotScene

CM = 1 / 2.54
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.size": 7})

DIM = "#2a78d6"
DIM_STYLE = {"color": DIM, "lw": 0.6, "zorder": 6}
ARROW = {"shrinkA": 0, "shrinkB": 0, "mutation_scale": 5, **DIM_STYLE}
TEXT_STYLE = {"color": DIM, "fontsize": 6, "ha": "center", "va": "center",
              "zorder": 7, "bbox": {"fc": "white", "ec": "none", "pad": 0.3}}


def dim(ax, p, q, text, at=0.5):
    ax.annotate("", q, p, arrowprops={"arrowstyle": "<->", **ARROW})
    ax.text(*(np.array(p) + at * (np.array(q) - np.array(p))), text,
            **TEXT_STYLE)


def radius(ax, c, q, text, at=0.55):
    ax.plot(*c, "+", ms=3, **DIM_STYLE)
    ax.annotate("", q, c, arrowprops={"arrowstyle": "->", **ARROW})
    ax.text(*(np.array(c) + at * (np.array(q) - np.array(c))), text,
            **TEXT_STYLE)


def guide(ax, p, q):
    ax.plot(*zip(p, q), ls="--", **{**DIM_STYLE, "lw": 0.4})


def angle(ax, c, r, a0, a1, text, at):
    guide(ax, c, (c[0] + 1.3 * r * np.cos(a0), c[1] + 1.3 * r * np.sin(a0)))
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=np.degrees(a0),
                     theta2=np.degrees(a1), **DIM_STYLE))
    ax.text(c[0] + at[0], c[1] + at[1], text, **TEXT_STYLE)


def polar(r, a):
    return r * np.cos(a), r * np.sin(a)


def specCircle(ax):
    radius(ax, (0, 0), polar(1.0, np.pi / 4), r"$R$")


def specOval(ax):
    t = 0.3
    radius(ax, (0, 0), polar(1.0, t), r"$a$", at=0.6)
    radius(ax, (0, 0), polar(0.5, t + np.pi / 2), r"$b$")
    angle(ax, (0, 0), 0.3, 0.0, t, r"$\theta_0$", (0.45, -0.13))


def specStadium(ax):
    guide(ax, (-1, -1), (-1, -1.35))
    guide(ax, (1, -1), (1, -1.35))
    dim(ax, (-1, -1.3), (1, -1.3), r"$2a$")
    radius(ax, (1, 0), (1 + np.cos(np.pi / 4), np.sin(np.pi / 4)), r"$r$")
    ax.plot(-1, 0, "+", ms=3, **DIM_STYLE)


def specLemon(ax):
    dim(ax, (-0.5, 0), (0.5, 0), r"$2B$")
    radius(ax, (-0.5, 0), (0, np.sqrt(0.75)), r"$R$")
    ax.plot(0.5, 0, "+", ms=3, **DIM_STYLE)


def specMushroom(ax):
    s0, s1 = 0.2 - 0.25, 0.2 + 0.25
    radius(ax, (0, 0), polar(1.0, 2 * np.pi / 3), r"$R$")
    guide(ax, (s0, -1), (s0, -1.3))
    guide(ax, (s1, -1), (s1, -1.3))
    dim(ax, (s0, -1.25), (s1, -1.25), r"$w$")
    guide(ax, (s1, -1), (s1 + 0.4, -1))
    dim(ax, (s1 + 0.3, 0), (s1 + 0.3, -1), r"$h$")
    guide(ax, (0, 0), (0, -0.75))
    guide(ax, (0.2, 0), (0.2, -0.75))
    dim(ax, (0, -0.6), (0.2, -0.6), r"$\mathrm{offset}$",
        at=-1.7)


TABLES = [
    (r"(a) \texttt{circle(R)}", T.circle(1.0), specCircle),
    (r"(b) \texttt{oval(a, b, theta0)}", T.oval(1.0, 0.5, 0.3), specOval),
    (r"(c) \texttt{stadium(r, a)}", T.stadium(1.0, 1.0), specStadium),
    (r"(d) \texttt{lemon(B, R)}", T.lemon(0.5, 1.0), specLemon),
    (r"(e) \texttt{mushroom(R, w, h, offset)}", T.mushroom(1.0, 0.5, 1.0, 0.2),
     specMushroom),
]

figure, axes = plt.subplots(1, 5, figsize=(17 * CM, 3.7 * CM))
for ax, (title, scene, spec) in zip(axes, TABLES):
    plotScene(scene, ax)
    for line in ax.lines:
        line.set_linewidth(0.9)
    spec(ax)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.axis("off")
    ax.set_title(title, fontsize=6, loc="left", pad=2)

figure.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01,
                       wspace=0.04)
figure.savefig("commonTablesExample.pdf")
figure.savefig("commonTablesExample.png", dpi=300)
