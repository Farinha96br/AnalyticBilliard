#!/usr/bin/env python3
"""Visualize the bouncing table produced by Plane.cpp.

    ./plane > collisions.dat && python3 plot_plane.py collisions.dat

The scene is read from the '# line a b c' header rows the program emits.
The table only stores collisions, so the arcs between them are rebuilt from
the same closed form the simulation solves -- sampling here is for drawing only.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#f4c72e", "#00420d"]  # categorical slots 1-5
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"
# portals are not walls, so they do not get the wall colour. teal specifically:
# plasma runs blue -> purple -> magenta -> orange and never passes near it, so a
# portal cannot be mistaken for a stretch of the path lying on top of it
PORTAL = "#0d9488"

# one path crosses itself over and over, and nothing about a uniform stroke says
# which pass came first. colouring by time does: every arc carries its own place
# in the run, and the colorbar turns that back into a collision number.
# plasma stops at 0.85 because its last stretch is a pale yellow that vanishes
# against SURFACE -- the late arcs are exactly the ones that must stay legible
TIME_CMAP = LinearSegmentedColormap.from_list(
    "plasma_dark", plt.cm.plasma(np.linspace(0.0, 0.85, 256)))


def load(path):
    """header -> scene objects, table -> {particle id: rows of (t, x, y, vx, vy)}"""
    scene, skip = [], 0
    with open(path) as f:
        for line in f:
            skip += 1
            tok = line.split()
            if tok[0] == "#":
                scene.append((tok[1], [float(v) for v in tok[2:]]))
            else:
                break  # the 'ID t x y ...' header line: table starts next
    raw = np.loadtxt(path, skiprows=skip)
    return scene, {int(i): raw[raw[:, 0] == i][:, 1:] for i in np.unique(raw[:, 0])}


def arc(row, t_next, grav, n=80):
    """the exact parabola leaving one collision, up to the next one"""
    t0, x, y, vx, vy = row
    t = np.linspace(0.0, t_next - t0, n)
    return x + vx * t, y + vy * t - 0.5 * grav * t * t


def draw(ax, rows, grav, color, label, mark_start=True, mark_end=True,
         arrow_every=1, cmap=None):
    """one particle: its flight path, then the collisions on top

    with a cmap the path is coloured by how far through the run each arc is,
    which is the only thing that makes a single self-crossing path readable.
    without one every arc takes `color`, so several particles stay told apart --
    a colour that meant time and particle at once would mean neither.
    """
    # how far through the run each row is: 0 at the start, 1 at the final state.
    # arc i leaves row i, so the two share an index and take the same colour
    age = np.linspace(0.0, 1.0, len(rows))
    hue = (lambda k: cmap(age[k])) if cmap else (lambda k: color)

    for i in range(len(rows) - 1):
        px, py = arc(rows[i], rows[i + 1][0], grav)
        ax.plot(px, py, color=hue(i), lw=1.7, solid_capstyle="round",
                label=label if i == 0 else None, zorder=2, alpha=0.9)

        # which way along the arc the particle went. the head is sized in points,
        # so it stays legible on a short hop and does not swamp a long one.
        # a teleport pair spans no time at all, leaving every sample on the same
        # spot with no direction to draw -- skip those
        if arrow_every and i % arrow_every == 0:
            m = len(px) // 2
            if px[m] != px[m + 1] or py[m] != py[m + 1]:
                ax.annotate("", xy=(px[m + 1], py[m + 1]), xytext=(px[m], py[m]),
                            zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=hue(i), lw=1.0,
                                            shrinkA=0, shrinkB=0))

    ax.scatter(rows[1:, 1], rows[1:, 2], s=16,
               c=(cmap(age[1:]) if cmap else color),
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    # the ends of whatever slice was handed in. in the top panel that is the
    # initial condition and wherever the window was cut; in the bottom, wherever
    # it was cut and the final state. between them they say which way to read the
    # path, which the collision dots alone cannot
    if mark_start:
        ax.plot(rows[0, 1], rows[0, 2], "*", ms=14, color=hue(0),
                mec=INK, mew=0.8, zorder=4)
    if mark_end:
        ax.plot(rows[-1, 1], rows[-1, 2], "s", ms=7, color=hue(len(rows) - 1),
                mec=INK, mew=0.8, zorder=4)


def _sense(ax, ox, oy, px, py, flip, length=1.6):
    """an arrow at (ox, oy) showing which way u grows, reversed when flip < 0"""
    dx, dy = px - ox, py - oy
    n = np.hypot(dx, dy)
    if n == 0.0:
        return
    dx, dy = flip * length * dx / n, flip * length * dy / n
    ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=PORTAL, lw=1.4,
                                shrinkA=0, shrinkB=0))


def draw_scene(ax, scene):
    """the objects, clipped to the view the particles established"""
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    for kind, p in scene:
        if kind == "line":
            a, b, c = p
            if b == 0.0:  # vertical: x is fixed, y spans the view
                ax.plot([-c / a] * 2, [ylo, yhi], color=MUTED, lw=2, zorder=1)
            else:
                ax.plot([xlo, xhi], [-(a * xlo + c) / b, -(a * xhi + c) / b],
                        color=MUTED, lw=2, zorder=1)
        elif kind == "segment":  # its endpoints are the whole shape: no clipping
            x0, y0, x1, y1 = p
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=2, zorder=1)
        elif kind == "elipse":  # width/height are the full axes, angle in degrees
            cx, cy, a, b, th = p
            ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(th),
                                 fill=False, ec=MUTED, lw=2, zorder=1))
        elif kind == "elipsearc":  # theta1/theta2 are degrees relative to angle
            cx, cy, a, b, th, phi0, phi1 = p
            ax.add_patch(Arc((cx, cy), 2 * a, 2 * b, angle=np.degrees(th),
                             theta1=np.degrees(phi0), theta2=np.degrees(phi1),
                             ec=MUTED, lw=2, zorder=1))
        elif kind == "portal":  # entry frame, then a dotted link to where it leads
            a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y, _fn, _ft = p
            ax.plot([a0x, a1x], [a0y, a1y], color=PORTAL, lw=1,
                    solid_capstyle="butt", zorder=5)
            ax.plot([(a0x + a1x) / 2, (b0x + b1x) / 2],
                    [(a0y + a1y) / 2, (b0y + b1y) / 2],
                    color=PORTAL, lw=0.8, ls=":", alpha=0.6, zorder=1)
            # which way u runs on each side, and so which end maps to which. the
            # two frames carry that on their own: the flips move the velocity and
            # never the position, so neither belongs in these arrows. arrows that
            # oppose mean the partner's endpoints were written the other way round
            _sense(ax, a0x, a0y, a1x, a1y, 1.0)
            _sense(ax, b0x, b0y, b1x, b1y, 1.0)
        elif kind == "circle":
            ax.add_patch(plt.Circle((p[0], p[1]), p[2], fill=False,
                                    ec=MUTED, lw=2, zorder=1))
    ax.set_xlim(xlo, xhi)  # the scene must not stretch the view
    ax.set_ylim(ylo, yhi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", help="output of ./plane: '# line ...' scene, then ID t x y vx vy")
    p.add_argument("--ids", type=int, nargs="+", help="particles to draw (max 5)")
    p.add_argument("--bounces", type=int, default=100, help="bounces to draw per particle")
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--arrows", type=int, default=1, metavar="N",
                   help="direction arrow on every Nth arc; 0 for none")
    p.add_argument("--out", default="plane.png")
    args = p.parse_args()

    scene, tracks = load(args.table)
    ids = (args.ids if args.ids else sorted(tracks))[:len(SERIES)]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), facecolor=SURFACE)

    # top: the first N collisions from the initial condition;
    # bottom: the last N recorded, where any long-run drift would show.
    # both are starred at their first row, so a path can be picked up in either
    panels = [
        (axes[0], slice(None, args.bounces + 1),
         f"First {args.bounces} collisions   * the initial condition, "
         f"\u25a0 where the window ends"),
        (axes[1], slice(-(args.bounces + 1), None),
         f"Last {args.bounces} collisions   * where the window begins, "
         f"\u25a0 the final state"),
    ]
    # with a single path there is no particle to tell apart, so the colour is
    # free to carry time instead -- which is the one thing a still picture of a
    # path that crosses itself cannot otherwise say
    solo = len(ids) == 1

    for ax, rows_of, title in panels:
        ax.set_facecolor(SURFACE)

        for slot, i in enumerate(ids):
            rows = tracks[i][rows_of]
            draw(ax, rows, args.g, SERIES[slot], None, mark_start=True,
                 mark_end=True, arrow_every=args.arrows,
                 cmap=TIME_CMAP if solo else None)

            # the two panels are different windows on the same run, so each
            # colorbar is labelled with the row numbers it actually covers
            if solo:
                first = np.arange(len(tracks[i]))[rows_of]
                bar = fig.colorbar(plt.cm.ScalarMappable(
                    Normalize(first[0], first[-1]), TIME_CMAP),
                    ax=ax, pad=0.015, fraction=0.03, aspect=12)
                # row, not collision: a portal crossing writes two of them, the
                # arrival and the departure, against the one instant
                bar.set_label("row #", color=MUTED)
                bar.ax.tick_params(colors=MUTED, length=0)
                bar.outline.set_visible(False)

        draw_scene(ax, scene)
        ax.set_aspect("equal")  # circles must look round, boxes square

        ax.set_title(title, color=INK, loc="left", fontsize=12)
        ax.set_ylabel("y", color=MUTED)
        ax.tick_params(colors=MUTED, length=0)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel("x", color=MUTED)

    # one legend for both panels, below everything: the same particle keeps the
    # same color in each. a lone particle is on the colorbar instead, and naming
    # it "particle 0" under a bar that already says what the colour means would
    # only invite reading the bar as a second particle
    if not solo:
        handles = [Line2D([], [], color=SERIES[s], lw=1.5) for s in range(len(ids))]
        fig.legend(handles, [f"particle {i}" for i in ids], frameon=False,
                   labelcolor=INK, ncol=len(ids), loc="lower center")

    fig.tight_layout(rect=(0, 0.0 if solo else 0.05, 1, 1))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
