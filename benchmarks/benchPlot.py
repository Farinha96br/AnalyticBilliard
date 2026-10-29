"""What the figures share: one colour and one marker per object type, in every
figure, and the per-host results folder they read from.

Colour follows the object type, never its position in a legend: a scene is
drawn in the colour of the object it is made of, so the Ellipse scene and the
'ellipse' compile series match. The colours are the package's own portal
palette, PORTAL_COLORS in analyticBilliards.utils, and every series carries its
own marker too, so no line is told apart by colour alone.
"""

import argparse
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import benchCommon as bc  # noqa: E402
from analyticBilliards.utils import PORTAL_COLORS  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TEX = shutil.which("latex") is not None

OBJECTS = ["line", "lineSegment", "ellipse", "ellipseArc", "portal", "rand"]
COLOR = dict(zip(OBJECTS, PORTAL_COLORS))
MARKER = dict(zip(OBJECTS, ["o", "s", "^", "v", "D", "P"]))
SCENE_OBJECT = {"Line": "line", "LineSeg": "lineSegment", "Ellipse": "ellipse",
                "ElipseArc": "ellipseArc", "Portal": "portal"}
BACKENDS = ["linear", "openmp", "gpu_openmp"]
BACKEND_NAME = {"linear": "linear", "openmp": "OpenMP (CPU)",
                "gpu_openmp": "OpenMP offload (GPU)"}
# for figures coloured by backend instead: Chell's two portals and P-body's red
BACKEND_COLOR = dict(zip(BACKENDS, [PORTAL_COLORS[0], PORTAL_COLORS[1],
                                    PORTAL_COLORS[5]]))
BACKEND_MARKER = dict(zip(BACKENDS, ["o", "s", "D"]))
INK, MUTED, GRID = "black", "#52514e", "#e4e3df"

plt.rcParams.update({
    "text.usetex": TEX, "font.family": "serif", "font.size": 8,
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
})


def style(objType):
    """kwargs for one series: its colour, its marker, a surface ring"""
    return {"color": COLOR[objType], "marker": MARKER[objType],
            "markeredgecolor": "white", "markeredgewidth": 0.5}


def backendStyle(backend):
    return {"color": BACKEND_COLOR[backend], "marker": BACKEND_MARKER[backend],
            "markeredgecolor": "white", "markeredgewidth": 0.5}


def legendTopLeft(ax, handles, labels):
    """one row above `ax`, flush with its left edge. On the axes, not the
    figure: bbox_inches="tight" crops when saving, which shifts anything placed
    in figure coordinates"""
    ax.legend(handles, labels, loc="lower left", ncol=len(labels),
              bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, borderpad=0.0,
              handlelength=1.6, columnspacing=1.2)


def panelLabel(ax, text, where="upper left"):
    """the panel's name, inside it"""
    x, ha = (0.04, "left") if "left" in where else (0.96, "right")
    y, va = (0.95, "top") if "upper" in where else (0.05, "bottom")
    ax.text(x, y, tex(text), transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=INK)


def options(doc, extra=None):
    """`extra(p)` adds a script's own arguments"""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--results", type=pathlib.Path,
                   default=HERE / "results" / bc.host(),
                   help="a results/<host> folder written by runBenchmarks.py")
    p.add_argument("--name", help="output file name, without extension "
                   "(default: the script's usual one)")
    if extra:
        extra(p)
    return p.parse_args()


def machine(results):
    f = results / "machine.json"
    return json.loads(f.read_text()) if f.exists() else {}


def tex(s):
    """plain text that survives usetex"""
    return s.replace("_", r"\_") if TEX else s


def save(fig, results, name):
    for ext in ("pdf", "png"):
        fig.savefig(results / f"{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.02, dpi=200)
    print(f"wrote {results / name}.pdf/.png")
