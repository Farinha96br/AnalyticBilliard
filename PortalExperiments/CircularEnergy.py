# this experiment uses a circular billiard with g = 0.5
# the cirlce has radius 1
# the portals have central positions at phi1, and phi2
# the portals have size h

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb

# this script sits in a subfolder, so python puts PortalExperiments/ on the path
# and not the repo root, which is where hpcBilliards lives. Adding it here is
# what lets the script run from any directory, with no PYTHONPATH to remember
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from hpcBilliards import compileScene

R0 = 1.0            # the outer circle's radius
R1 = 0.5            # the inner circle: a solid obstacle, so the table is an annulus
here = pathlib.Path(__file__).parent    # figures land beside the script


def makeTable(g, phi1, phi2, h, normalFlip=-1.0, tangentFlip=+1.0):

    delta = np.arcsin(0.5 * h / R0)     # half the angle the portalSegment subtends

    def portalSegment(phi):
        a, b = phi - delta, phi + delta
        return [R0 * np.cos(a), R0 * np.sin(a),
                R0 * np.cos(b), R0 * np.sin(b)]

    first, second = portalSegment(phi1), portalSegment(phi2)

    return {
        "g": g,             # gravity, pulling towards -y
        "deadTime": 1e-10,  # roots below this are the collision just resolved

        "solidObjects": [
            {"type": "elipse", "params": [0.0, 0.0, R0, R0, 0.0]},
            {"type": "elipse", "params": [0.0, 0.0, R1, R1, 0.0]},
        ],

        "portalObjects": [
            {"entry": "lineSegment", "entryParams": first,
             "exit":  "lineSegment", "exitParams":  second,
             "normalFlip": normalFlip, "tangentFlip": tangentFlip},
            {"entry": "lineSegment", "entryParams": second,
             "exit":  "lineSegment", "exitParams":  first,
             "normalFlip": normalFlip, "tangentFlip": tangentFlip},
        ],
    }


def plotTable(table, axes=None):

    if axes is None:
        _, axes = plt.subplots(figsize=(5.5, 5.5))

    # every solid object the solver was given, not just the outer wall, so the
    # picture cannot drift out of step with the table
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    for wall in table["solidObjects"]:
        cx, cy, rx, ry, tilt = wall["params"]
        ex, ey = rx * np.cos(t), ry * np.sin(t)
        axes.plot(cx + ex * np.cos(tilt) - ey * np.sin(tilt),
                  cy + ex * np.sin(tilt) + ey * np.cos(tilt),
                  color="#101010", lw=0.5, zorder=3)

    colours = ["#e8730a", "#1f77b4"]
    for k, portal in enumerate(table["portalObjects"]):
        x0, y0, x1, y1 = portal["entryParams"]
        colour = colours[k % len(colours)]

        # the cap this portalSegment cuts off: the minor arc between its two ends
        a0, a1 = np.arctan2(y0, x0), np.arctan2(y1, x1)
        while a1 - a0 > np.pi:
            a1 -= 2.0 * np.pi
        while a1 - a0 < -np.pi:
            a1 += 2.0 * np.pi


    axes.set_aspect("equal")
    axes.set_xlim(-1.15 * R0, 1.15 * R0)
    axes.set_ylim(-1.15 * R0, 1.15 * R0)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    return axes


g = 0.5
h = 0.05
RESOLUTION = 256
iterations = 1e4    # events per particle, not a duration
angles = np.linspace(0.0, np.pi/2.0, 5, endpoint=True)
initialE = [0.5,1.3,2.0]
for E in initialE:
    for angle in angles:
        print(f"initial energy: {E}")
        print(f"angle: {angle}")
        phi1 = angle
        phi2 = angle + np.pi
        print(f"phi1: {angle}, phi2: {phi2}")
        table = makeTable(g=g, phi1=phi1, phi2=phi2, h=h)

        # initial conditions on the boundary, as in escapeBasins.py: the grid is
        # (theta, alpha), the place on the wall and the angle to the tangent
        theta = np.linspace(0.0, 2.0 * np.pi, RESOLUTION)
        alpha = np.linspace(0.0, np.pi, RESOLUTION)
        thetaGrid, alphaGrid = np.meshgrid(theta, alpha)

        # a particle on the wall at angle theta needs E >= U(theta) to exist
        potential = g * (R0 * np.sin(thetaGrid) + 1.0)

        # and a start ON a portal is not a start in the billiard: that arc is
        # the cap the chord crops away
        delta = np.arcsin(0.5 * h / R0)
        gap1 = np.abs(np.arctan2(np.sin(thetaGrid - phi1), np.cos(thetaGrid - phi1)))
        gap2 = np.abs(np.arctan2(np.sin(thetaGrid - phi2), np.cos(thetaGrid - phi2)))

        # launched a hair inside the wall rather than exactly on it
        startR = R0 - 1e-5
        startX = startR * np.cos(thetaGrid)
        startY = startR * np.sin(thetaGrid)

        # the inner circle is solid, so its interior is a separate region that
        # nothing in the annulus can reach: a particle started in there would be
        # a different billiard altogether
        insideInner = np.hypot(startX, startY) <= R1

        usable = ((E >= potential) & (gap1 > delta) & (gap2 > delta)
                  & ~insideInner)

        # leaving the wall at angle alpha to the tangent, which for a circle
        # points along theta + pi/2
        speed = np.sqrt(2.0 * (E - potential[usable]))
        eta = thetaGrid[usable] + 0.5 * np.pi + alphaGrid[usable]
        x = startX[usable]
        y = startY[usable]
        vx = speed * np.cos(eta)
        vy = speed * np.sin(eta)

        # one row per event rather than per tick of a clock. Energy is a
        # staircase -- conserved along every flight and across every bounce, and
        # moving only when a portal drops the particle at a different height --
        # so an event is exactly where something can happen to it
        scene = compileScene(table, backend="openmp")
        record = scene.recordWithIterations(x, y, vx, vy, iterations=iterations,
                                            save=["Energy"])

        # the average over the events a particle actually reached. Rows past
        # counts[i] need no masking: the buffer starts at zero and those rows
        # were never written, so they add nothing to the sum
        #
        # the solver measures potential from y = 0, the initial E above from
        # y = -R0 like escapeBasins does, so the two differ by g*R0. Shift the
        # record, not the input, and the map is on the same scale as the E that
        # produced it
        meanEnergy = record.energy.sum(axis=1) / record.counts + g * R0

        picture = np.full(thetaGrid.shape, np.nan)
        picture[usable] = meanEnergy

        # the table on the left, what it does to the energy on the right
        fig, (axesTable, axes) = plt.subplots(1, 2, figsize=(11.0, 4.5))
        plotTable(table, axes=axesTable)

        # why a cell was thrown out, painted underneath: grey for the energy,
        # and each portal's own colour for the arc it crops. In that order, so
        # a cell excluded for both reasons shows the portal -- those bands are
        # only 2*delta wide and would be lost under the grey
        background = np.ones(thetaGrid.shape + (3,))
        background[E < potential] = to_rgb("#e8e8e8")
        background[gap1 <= delta] = to_rgb("#f6c9a3")   # phi1, the orange one
        background[gap2 <= delta] = to_rgb("#aecde8")   # phi2, the blue one
        axes.imshow(background, origin="lower", aspect="auto",
                    extent=(0.0, 2.0 * np.pi, 0.0, np.pi),
                    interpolation="nearest")

        colours = plt.get_cmap("viridis").copy()
        colours.set_bad("none")         # excluded cells: let the background show
        # log colours: the averages run over orders of magnitude once a portal
        # pair pumps, and a linear scale shows one bright cell on a flat floor.
        # The range comes from the positive cells only, since LogNorm has
        # nothing to say about zero
        positive = picture[np.isfinite(picture) & (picture > 0.0)]
        image = axes.imshow(picture, origin="lower", aspect="auto", cmap=colours,
                            norm=LogNorm(vmin=positive.min(), vmax=positive.max()),
                            extent=(0.0, 2.0 * np.pi, 0.0, np.pi),
                            interpolation="nearest")
        fig.colorbar(image, ax=axes, label="average energy")
        axes.set_title("grey: E below the potential\n"
                       "orange / blue: starting on that portal", fontsize=9)
        fig.suptitle(f"E = {E}, h = {h}, "
                     rf"$\varphi_1$ = {np.rad2deg(phi1):.0f}$^\circ$, "
                     rf"$\varphi_2$ = {np.rad2deg(phi2):.0f}$^\circ$",
                     fontsize=11)
        axes.set_xticks(np.arange(5) * np.pi / 2)
        axes.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        axes.set_yticks(np.arange(3) * np.pi / 2)
        axes.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
        axes.set_xlabel(r"$\theta$")
        axes.set_ylabel(r"$\alpha$")
        fig.tight_layout()
        filename = here / f"circular_avg_energy_E{E:.1f}_{angle:.2f}.png"
        fig.savefig(str(filename), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  usable {usable.sum():6d} of {usable.size}, "
              f"average energy {np.nanmin(picture):.3f} to "
              f"{np.nanmax(picture):.3f}, wrote {filename.name}")





    



