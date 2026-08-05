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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#f4c72e", "#00420d"]  # categorical slots 1-5
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"


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


def draw(ax, rows, grav, color, label, mark_start=True):
    """one particle: its flight path, then the collisions on top"""
    for i in range(len(rows) - 1):
        px, py = arc(rows[i], rows[i + 1][0], grav)
        ax.plot(px, py, color=color, lw=1.0, solid_capstyle="round",
                label=label if i == 0 else None, zorder=2, alpha=0.3)
    ax.plot(rows[1:, 1], rows[1:, 2], "o", ms=4, color=color,
            mec=SURFACE, mew=0.8, zorder=3)
    if mark_start:  # only meaningful when rows begin at the initial condition
        ax.plot(rows[0, 1], rows[0, 2], "*", ms=13, color=color,
                mec=SURFACE, mew=1, zorder=4)  # where it started


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
    p.add_argument("--out", default="plane.png")
    args = p.parse_args()

    scene, tracks = load(args.table)
    ids = (args.ids if args.ids else sorted(tracks))[:len(SERIES)]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), facecolor=SURFACE)

    # top: the first N collisions from the initial condition (starred);
    # bottom: the last N recorded, where any long-run drift would show
    panels = [
        (axes[0], slice(None, args.bounces + 1),
         f"First {args.bounces} collisions, * marks the start", True),
        (axes[1], slice(-(args.bounces + 1), None),
         f"Last {args.bounces} collisions", False),
    ]
    for ax, rows_of, title, is_first in panels:
        ax.set_facecolor(SURFACE)

        for slot, i in enumerate(ids):
            draw(ax, tracks[i][rows_of], args.g, SERIES[slot],
                 None, mark_start=is_first)

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

    # one legend for both panels, below everything: the same particle keeps
    # the same color in each. Proxy lines, because the arcs themselves are
    # alpha 0.1 and a legend swatch that faint would be unreadable
    handles = [Line2D([], [], color=SERIES[slot], lw=1.5) for slot in range(len(ids))]
    fig.legend(handles, [f"particle {i}" for i in ids], frameon=False,
               labelcolor=INK, ncol=len(ids), loc="lower center")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
