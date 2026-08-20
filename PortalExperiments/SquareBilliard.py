# the square analogue of CircularEnergy.py: a box of side L under gravity, with
# one portal cut into each of the two vertical walls
#
# the two portals are the same length h, and each one's position is given as the
# distance from the ground to where the portal STARTS -- so dLeft and dRight are
# the heights of the lower end, and the portal runs from d up to d + h
#
# this file only builds the table and draws it. no billiard is run

import pathlib

import numpy as np
import matplotlib.pyplot as plt

SIDE = 1.0          # the box is [0, L] x [0, L], so the ground is y = 0
here = pathlib.Path(__file__).parent    # figures land beside the script


def makeTable(g, dLeft, dRight, h, L=SIDE, normalFlip=+1.0, tangentFlip=+1.0):
    """the box with a gap in each vertical wall, the two joined.
    """
    # d is where the portal starts, not where it is centred, so the gap runs
    # from d up to d + h
    l0, l1 = dLeft, dLeft + h                         # the gap in the left wall
    r0, r1 = dRight, dRight + h                       # the gap in the right wall

    # a piece of zero length is a degenerate lineSegment, which the solver
    # refuses to build, and dropping it instead would change the object count and
    # so force a recompile. Keeping every portal strictly off the corners keeps
    # one compiled library good for a whole sweep
    # for name, d, lo, hi in (("dLeft", dLeft, l0, l1), ("dRight", dRight, r0, r1)):
    #     if not (0.0 < lo and hi < L):
    #         raise ValueError(
    #             f"{name} = {d} puts a portal of length {h} at [{lo}, {hi}], "
    #             f"which does not fit strictly inside a wall of length {L}: "
    #             f"keep 0 < {name} < {L - h}")

    left = [0.0, l0, 0.0, l1]           # bottom end first: u grows upwards
    right = [L, r0, L, r1]              # likewise, so the two agree on u

    return {
        "g": g,             # gravity, pulling towards -y
        "deadTime": 1e-10,  # roots below this are the collision just resolved

        # counterclockwise from the bottom-left corner, with the two vertical
        # walls split around the portal each one carries
        "solidObjects": [
            {"type": "lineSegment", "params": [0.0, 0.0, L, 0.0]},
            {"type": "lineSegment", "params": [L, 0.0, L, r0]},
            {"type": "lineSegment", "params": [L, r1, L, L]},
            {"type": "lineSegment", "params": [L, L, 0.0, L]},
            {"type": "lineSegment", "params": [0.0, L, 0.0, l1]},
            {"type": "lineSegment", "params": [0.0, l0, 0.0, 0.0]},
        ],

        # both segments are written bottom end first, so u -- the distance along
        # the entry that the map carries over -- is measured from each portal's
        # own start. With the default flips the pair is then a rigid shift: a
        # particle leaving the left wall a height u above dLeft re-enters at the
        # right wall the same u above dRight, with its velocity untouched
        "portalObjects": [
            {"entry": "lineSegment", "entryParams": left,
             "exit":  "lineSegment", "exitParams":  right,
             "normalFlip": normalFlip, "tangentFlip": tangentFlip},
            {"entry": "lineSegment", "entryParams": right,
             "exit":  "lineSegment", "exitParams":  left,
             "normalFlip": normalFlip, "tangentFlip": tangentFlip},
        ],
    }



def plotTable(table, L=SIDE, axes=None):

    if axes is None:
        _, axes = plt.subplots(figsize=(5.5, 5.5))

    # every solid object the solver was given, not a rectangle drawn from L, so
    # the picture cannot drift out of step with the table: the gaps the walls
    # were split around are exactly the ones that show here
    for wall in table["solidObjects"]:
        x0, y0, x1, y1 = wall["params"]
        axes.plot([x0, x1], [y0, y1], color="#101010", lw=1.5, zorder=3,
                  solid_capstyle="butt")

    colours = ["#e8730a", "#1f77b4"]
    offset = 0.07 * L       # how far outside the wall a dimension line is drawn

    for k, portal in enumerate(table["portalObjects"]):
        x0, y0, x1, y1 = portal["entryParams"]
        ex0, ey0, ex1, ey1 = portal["exitParams"]
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        colour = colours[k % len(colours)]

        axes.plot([x0, x1], [y0, y1], color=colour, lw=3, zorder=5,
                  solid_capstyle="butt", label=f"portal {k + 1}")


        nx, ny = -(y1 - y0), x1 - x0
        scale = offset / np.hypot(nx, ny)
        if nx * (mx - 0.5 * L) + ny * (my - 0.5 * L) < 0.0:
            scale = -scale
        nx, ny = nx * scale, ny * scale

        cx, cy = x0, 0.0
        axes.annotate("", xy=(x0 + nx, y0 + ny), xytext=(cx + nx, cy + ny),
                      zorder=4,
                      arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=0.9,
                                      shrinkA=0, shrinkB=0))
        axes.annotate(f"{y0:.2f}", xy=(0.5 * (cx + x0) + nx, 0.5 * (cy + y0) + ny),
                      color=colour, fontsize=9, zorder=4,
                      ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        axes.plot(cx, cy, "o", ms=4, color="#101010", zorder=6)

    axes.set_aspect("equal")
    axes.set_xlim(-0.25 * L, 1.25 * L)
    axes.set_ylim(-0.15 * L, 1.15 * L)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend(loc="upper center", fontsize=9)
    return axes


g = 0.5
h = 0.05
L = SIDE




has = np.linspace(0.00, 0.9 - h, 5)      

E = [0.5]
ITERATIONS = 1e4    # events per particle, not a duration
NPARTICLES = 5

for ha in has:
    for Es in E:
        print(f"initial energy: {Es}")
        print(f"left portal height: {ha}")

        hb = L - h
        table = makeTable(g=g, dLeft=ha, dRight=hb, h=h, L=L)

        fig, axes = plt.subplots(figsize=(5.5, 5.5))
        plotTable(table, L=L, axes=axes)
        axes.set_title(f"h = {h}, dLeft = {ha}, dRight = {L-ha}", fontsize=11)
        fig.tight_layout()

        filename = here / f"square_table_l{ha:.2f}_r{hb:.2f}.png"
        fig.savefig(str(filename), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {filename.name}")




