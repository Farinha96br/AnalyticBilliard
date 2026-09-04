# the square analogue of CircularEnergy.py: a box of side L under gravity, with
# one portal cut into each of the two vertical walls
#
# the two portals are the same length h, and each one's position is given as the
# distance from the ground to the portal's CENTRE -- so dLeft and dRight are the
# heights of the midpoint, and the portal runs from d - h/2 up to d + h/2
#
# this file runs a fan of particles from the middle of the box and draws the two
# ends of each run over the table it was run on

import pathlib

import numpy as np
import matplotlib.pyplot as plt

# the table, the picture of it and the initial conditions all live beside this
# script, and that module is what puts the repo root on the path
from frequentFunctions import SIDE, arcs, compileScene, fan, makeTable, plotTable

here = pathlib.Path(__file__).parent    # figures land beside the script

g = 0.5
h = 0.05
L = SIDE

DMIN = 0.5 * h      # the lowest a portal can be centred and still fit on its wall
DMAX = L - 0.5 * h  # and the highest

# the two portals move in opposite directions: hb = L - ha, so as the left one
# climbs off the ground the right one comes down from the ceiling to meet it. The
# step a crossing costs is hb - ha = L - 2*ha, which runs from +(L - h) to
# -(L - h) across the sweep and passes through zero exactly in the middle -- an
# odd number of placements, so that level one is in the list rather than skipped
has = np.linspace(DMIN, DMAX, 5)

E = [0.5]
ITERATIONS = 1e4    # events per particle, not a duration
NPARTICLES = 5
ARCS = 20           # how many flights to draw at each end of the run
LAUNCH = (0.5 * L, 0.5 * L)     # every particle starts here, in the middle

# neither orange nor blue: those two belong to the portals, and a path must not
# be mistakable for the thing it is passing through
SERIES = ["#7a3fa0", "#1baf7a", "#c1272d", "#00420d", "#8a6d3b"]

for ha in has:
    hb = L - ha             # one goes up, the other comes down to meet it

    for Es in E:
        print(f"initial energy: {Es}")
        print(f"portal heights: left {ha:.4f}, right {hb:.4f}")

        table = makeTable(g=g, dLeft=ha, dRight=hb, h=h, L=L)
        scene = compileScene(table, backend="openmp")

        # one row per event rather than per tick of a clock: between two events
        # the flight is a parabola in closed form, so the rows are all the
        # trajectory there is to have
        x, y, vx, vy = fan(Es, g, NPARTICLES, *LAUNCH)
        record = scene.recordWithIterations(x, y, vx, vy, iterations=ITERATIONS,
                                            save=["t", "x", "y", "vx", "vy"])

        # the two ends of the run, side by side: what the particle does straight
        # away on the left, what it has settled into ten thousand events later on
        # the right. The same table under both, so the only difference between
        # the panels is the trajectory
        fig, panels = plt.subplots(1, 2, figsize=(11.0, 5.75))
        for axes in panels:
            plotTable(table, L=L, axes=axes)
            axes.get_legend().remove()      # one legend for the figure, below

        for i in range(NPARTICLES):
            rows = record.trim(i)       # rows past counts[i] were never written
            path = arcs(rows, g)
            colour = SERIES[i % len(SERIES)]

            for axes, window in zip(panels, (path[:ARCS], path[-ARCS:])):
                for px, py in window:
                    axes.plot(px, py, color=colour, lw=0.9, alpha=0.85,
                              zorder=2, solid_capstyle="round")

            # each panel gets the end of the run it actually shows
            panels[0].plot(rows["x"][0], rows["y"][0], "*", ms=11, color=colour,
                           mec="#101010", mew=0.6, zorder=6)
            panels[1].plot(rows["x"][-1], rows["y"][-1], "s", ms=5, color=colour,
                           mec="#101010", mew=0.6, zorder=6)

        panels[0].set_title(f"first {ARCS} arcs   * the initial condition",
                            fontsize=10)
        panels[1].set_title(f"last {ARCS} arcs   ■ the final state",
                            fontsize=10)

        handles, labels = panels[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                   frameon=False)
        fig.suptitle(f"E = {Es}, h = {h}, dLeft = {ha:.2f}, dRight = {hb:.2f}",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        filename = here / f"square_table_l{ha:.2f}_r{hb:.2f}.png"
        fig.savefig(str(filename), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {len(scene.types)} objects, events per particle "
              f"{record.counts.min()} to {record.counts.max()}, "
              f"wrote {filename.name}")
