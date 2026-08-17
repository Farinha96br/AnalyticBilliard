#!/usr/bin/env python3
"""Drive the billiard from Python, then plot what came back.

    python3 usageExample2.py

Read it top to bottom. It builds one scene, compiles it once, and then runs it
four different ways WITHOUT compiling again -- which is the whole point of the
library: object types are fixed at compile time, everything else is not.
"""

import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpcBilliards import compileScene, updateScene


# ---------------------------------------------------------------------------
# Colours. Nothing clever, just names so the plotting code reads in English.
# ---------------------------------------------------------------------------

CREAM = "#fcfcfb"   # the page
INK = "#0b0b0b"     # titles, marker edges
GREY = "#52514e"    # walls, axis labels
GRIDLINE = "#dcdbd5"
TEAL = "#0d9488"    # portals, which are not walls and should not look like them


def colourForTime(fraction):
    """Dark blue at the start of a run, orange at the end.

    Stops at 0.85 because plasma finishes in a pale yellow that disappears
    against the cream background, and the late arcs are the ones you most want
    to still be able to see.
    """
    return plt.cm.plasma(0.85 * fraction)


# ---------------------------------------------------------------------------
# The scene. A box with two portals facing each other across it.
#
# One portal dict is ONE ONE-WAY portal: a particle entering `entryParams`
# comes out at `exitParams`. To make a two-way pair you write two dicts, which
# is what happens below -- the second is the first one backwards.
# ---------------------------------------------------------------------------

boxScene = {
    "g": 1.0,
    "deadTime": 1e-9,

    "solidObjects": [
        {"type": "line", "params": [0.0, 1.0, 0.0]},     # floor    y = 0
        {"type": "line", "params": [0.0, 1.0, -10.0]},   # ceiling  y = 10
        {"type": "line", "params": [1.0, 0.0, 20.0]},    # wall     x = -20
        {"type": "line", "params": [1.0, 0.0, -20.0]},   # wall     x = 20
    ],

    "portalObjects": [
        # left portal -> right portal
        {"entry": "lineSegment", "entryParams": [-8.0, 2.0, -8.0, 8.0],
         "exit":  "lineSegment", "exitParams":  [8.0, 2.0, 8.0, 8.0],
         "normalFlip": +1.0, "tangentFlip": +1.0},

        # right portal -> left portal
        {"entry": "lineSegment", "entryParams": [8.0, 2.0, 8.0, 8.0],
         "exit":  "lineSegment", "exitParams":  [-8.0, 2.0, -8.0, 8.0],
         "normalFlip": +1.0, "tangentFlip": +1.0},
    ],
}

# One particle, thrown up and to the right from the bottom left corner.
# These are lists because a real run would have thousands of entries.
startX = [-16.0]
startY = [1.0]
startVX = [5.75]
startVY = [4.25]


# ---------------------------------------------------------------------------
# Compile. This is the only step that runs a compiler.
# ---------------------------------------------------------------------------

billiard = compileScene(boxScene, backend="openmp")

print(f"compiled {len(billiard.types)} objects into {billiard.so_path.name}")
if billiard.cached:
    print("  (it was already built, so this was instant)")


# ---------------------------------------------------------------------------
# Run 1: stop after a fixed number of events.
#
# You get one row per event, plus one for the initial state, so 2000 iterations
# means exactly 2001 rows.
# ---------------------------------------------------------------------------

run = billiard.recordWithIterations(startX, startY, startVX, startVY,
                                    iterations=2000)

rowsUsed = run.counts[0]
eventTypes = run.evType[0, :rowsUsed]
portalCrossings = int((eventTypes == 2).sum())

print()
print(f"iterations=2000 gave {rowsUsed} rows")
print(f"  {portalCrossings} of them are portal crossings (evType 2)")
print(f"  the rest are bounces (evType 1), and row 0 is the start (evType 0)")


# ---------------------------------------------------------------------------
# Run 2: the same trajectory as a TIME SERIES rather than an event table.
#
# recordWithTime samples a uniform grid, t = 0, dt, 2*dt, ... dt is not a step
# size and nothing is integrated: between collisions the state is an exact
# parabola, and this evaluates it wherever you ask. A smaller dt buys more
# samples of the very same flight and costs no accuracy, because there is none
# to lose.
#
# floor(tf / dt) + 1 rows, the same grid for every particle, and the count is
# known before the run starts -- so there is nothing to say about how much room
# to give it, and nothing to check afterwards.
# ---------------------------------------------------------------------------

timedRun = billiard.recordWithTime(startX, startY, startVX, startVY,
                                   tf=120.0, dt=0.05)

print()
print(f"tf=120, dt=0.05 gave {timedRun.counts[0]} rows, "
      f"t from {timedRun.t[0, 0]:.2f} to {timedRun.t[0, -1]:.2f}")
print(f"  mean height over the run = {timedRun.y[0].mean():.4f}")
print(f"  that is a time average, which only evenly spaced samples can give:")
print(f"  in the event table above, rows crowd where the bounces are")


# ---------------------------------------------------------------------------
# Run 3: ask for one column only, and only every tenth event.
#
# save= decides what gets allocated, not just what you get handed back. Asking
# for Energy alone allocates one array instead of six, which matters when the
# run is long, and stride= shrinks the other axis the same way: 20000 events
# recorded 2001 rows deep. Every event in between is still simulated -- stride
# subsamples the record, never the physics. eventType= then throws away every
# row that is not a bounce.
# ---------------------------------------------------------------------------

energyRun = billiard.recordWithIterations(startX, startY, startVX, startVY,
                                          iterations=20000,
                                          stride=10,
                                          save=["Energy"],
                                          eventType=["bounce"])

energy = energyRun.energy[0, :energyRun.counts[0]]

print()
print(f"energy at {energy.size} bounces (kinetic + potential, per unit mass)")
print(f"  starts at {energy[0]:.6f}")
print(f"  drifts by {energy[-1] - energy[0]:+.2e} over the whole run")


# ---------------------------------------------------------------------------
# Run 4: four scenes, one compile.
#
# Each variant below changes only PARAMETERS -- the flip signs, or gravity. The
# object count and the object types never move, so updateScene pushes the new
# numbers straight into the library that is already loaded. No compiler runs.
#
# Adding or removing an object, or changing one's type, WOULD need a rebuild,
# and updateScene raises rather than silently doing the wrong thing.
# ---------------------------------------------------------------------------

# (a) exactly the scene from the top of the file
plainScene = copy.deepcopy(boxScene)

# (b) reverse the velocity ALONG the portal surface: goes in climbing, comes
#     out descending, at the same height
flipTangentScene = copy.deepcopy(boxScene)
for portal in flipTangentScene["portalObjects"]:
    portal["tangentFlip"] = -1.0

# (c) reverse the velocity THROUGH the surface: leaves by the face it arrived
#     at, so it turns back instead of carrying on
flipNormalScene = copy.deepcopy(boxScene)
for portal in flipNormalScene["portalObjects"]:
    portal["normalFlip"] = -1.0

# (d) same portals, weaker gravity: flatter arcs, and the particle spends far
#     longer in the air between bounces
lowGravityScene = copy.deepcopy(boxScene)
lowGravityScene["g"] = 0.5

panels = [
    ("normalFlip +1, tangentFlip +1  (straight through)", plainScene),
    ("tangentFlip -1  (in climbing, out descending)", flipTangentScene),
    ("normalFlip -1  (turns back)", flipNormalScene),
    ("g = 0.5  (same portals, weaker gravity)", lowGravityScene),
]


def drawPanel(axes, record, scene, title, arcsToDraw=40):
    """Draw one scene and the beginning of one trajectory on it."""

    rows = min(int(record.counts[0]), arcsToDraw)
    t = record.t[0, :rows]
    x = record.x[0, :rows]
    y = record.y[0, :rows]
    vx = record.vx[0, :rows]
    vy = record.vy[0, :rows]
    evType = record.evType[0, :rows]
    gravity = scene["g"]

    # The walls first, so the trajectory sits on top of them.
    for solid in scene["solidObjects"]:
        a, b, c = solid["params"]
        if b == 0.0:                       # vertical: x is fixed
            axes.plot([-c / a, -c / a], [-1, 11], color=GREY, lw=2, zorder=1)
        else:                              # everything else: solve for y
            leftY = -(a * -21 + c) / b
            rightY = -(a * 21 + c) / b
            axes.plot([-21, 21], [leftY, rightY], color=GREY, lw=2, zorder=1)

    for portal in scene["portalObjects"]:
        x0, y0, x1, y1 = portal["entryParams"]
        axes.plot([x0, x1], [y0, y1], color=TEAL, lw=3, zorder=5)

    # Now the flight path. Every row is the state a flight STARTS from, and
    # that flight ends at the next row's time -- so the arc between row i and
    # row i+1 is just the parabola, sampled for drawing.
    #
    # this is by hand because the panel wants each arc coloured separately.
    # recordWithTime does the same sampling in the kernel when you want the
    # whole path as one array instead.
    for i in range(rows - 1):
        flightTime = t[i + 1] - t[i]
        step = np.linspace(0.0, flightTime, 60)
        arcX = x[i] + vx[i] * step
        arcY = y[i] + vy[i] * step - 0.5 * gravity * step * step

        howFarThroughTheRun = i / max(rows - 2, 1)
        axes.plot(arcX, arcY, color=colourForTime(howFarThroughTheRun),
                  lw=1.5, alpha=0.9, zorder=2)

    # Where a portal put the particle down. These always land ON a portal, so
    # they are drawn hollow -- a filled teal dot on a teal bar is invisible.
    isPortalExit = evType == 2
    axes.scatter(x[~isPortalExit], y[~isPortalExit], s=10, color=GREY, zorder=3)
    axes.scatter(x[isPortalExit], y[isPortalExit], s=44,
                 facecolors=CREAM, edgecolors=INK, linewidths=1.2, zorder=4,
                 label="portal exit")

    # And where it all started.
    axes.plot(x[0], y[0], "*", ms=15, color=colourForTime(0.0),
              mec=INK, mew=0.8, zorder=6)

    axes.set_xlim(-21, 21)
    axes.set_ylim(-1, 11)
    axes.set_aspect("equal")
    axes.set_facecolor(CREAM)
    axes.set_title(title, color=INK, loc="left", fontsize=11)
    axes.grid(color=GRIDLINE, lw=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=GREY, length=0)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)


figure, axesGrid = plt.subplots(2, 2, figsize=(13, 7), facecolor=CREAM)

libraryTimestampBefore = billiard.so_path.stat().st_mtime

for axes, (title, scene) in zip(axesGrid.ravel(), panels):
    updateScene(scene, billiard)          # <- no compiler runs on this line
    record = billiard.recordWithIterations(startX, startY, startVX, startVY,
                                           iterations=400)
    drawPanel(axes, record, scene, title)

libraryTimestampAfter = billiard.so_path.stat().st_mtime
assert libraryTimestampBefore == libraryTimestampAfter, "the library was rebuilt!"

print()
print("all four panels came from the same .so, which was never rebuilt")

figure.suptitle("one compile, four scenes: parameters change without a rebuild",
                color=INK, fontsize=13)
handles, labels = axesGrid[0, 0].get_legend_handles_labels()
figure.legend(handles, labels, frameon=False, labelcolor=INK, loc="lower center")
figure.tight_layout(rect=(0, 0.04, 1, 0.97))
figure.savefig("usageExample2.png", dpi=150, facecolor=CREAM)
print("wrote usageExample2.png")


# ---------------------------------------------------------------------------
# One more picture: the energy from run 3, to show how little it wanders.
# ---------------------------------------------------------------------------

energyFigure, energyAxes = plt.subplots(figsize=(9, 3.2), facecolor=CREAM)

energyAxes.plot(np.arange(energy.size), energy - energy[0],
                color=colourForTime(0.35), lw=1)

energyAxes.set_facecolor(CREAM)
energyAxes.set_xlabel("bounce", color=GREY)
energyAxes.set_ylabel("E - E(0)", color=GREY)
energyAxes.set_title(f"energy drift over {energy.size} bounces, "
                     f"total spread {np.ptp(energy):.1e}",
                     color=INK, loc="left", fontsize=11)
energyAxes.grid(color=GRIDLINE, lw=0.6)
energyAxes.set_axisbelow(True)
energyAxes.tick_params(colors=GREY, length=0)
for side in ("top", "right", "left", "bottom"):
    energyAxes.spines[side].set_visible(False)

energyFigure.tight_layout()
energyFigure.savefig("usageExample2_energy.png", dpi=150, facecolor=CREAM)
print("wrote usageExample2_energy.png")
