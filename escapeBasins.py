"""Escape basins of an open gravitational circular billiard, in (theta, alpha).

Following Haerter et al., Chaos, Solitons and Fractals 210 (2026) 118700. A
point particle bounces elastically inside a unit circle under uniform gravity,
and two holes of angular half-width h are cut in the boundary at theta = 0 and
theta = pi. Which hole a trajectory eventually finds is a fractal function of
where on the boundary it started and at what angle.

Initial conditions live on the boundary, not in the interior: a particle is
launched from the wall at angular position theta, at an angle alpha to the
tangent there, with whatever speed the total energy E leaves it. So the picture
is drawn in (theta, alpha) -- the billiard's own phase space -- rather than in
x and y.

The holes are basinObjects: arcs of the same circle that end a run instead of
reflecting it. Everything else is a solid arc, so the boundary is the four
pieces the two holes cut it into.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from hpcBilliards import compileScene, getBasins, updateScene

G = 0.5             # the paper's gravitational acceleration
RB = 1.0            # boundary radius
HOLE = 0.04         # angular HALF-width of each hole; the paper's 2h is the full width
ENERGIES = (0.5, 0.7, 1.3, 2.0)   # Fig. 2: two hyperbolic, two mixed phase space
RESOLUTION = 256    # per axis of the (theta, alpha) grid
COLLISIONS = 1e4   # give up after this many bounces (the paper affords 10^6)

# potential energy is measured from the bottom of the billiard, y = -1, so
# U(theta) = g*(sin(theta) + 1) and a particle needs E >= U to exist at all
POTENTIAL = lambda theta: G * (RB * np.sin(theta) + 1.0)


def openCircle(h):
    """the unit circle with two holes: two solid arcs, two basin arcs.

    An arc runs counterclockwise from phi0 to phi1. The holes sit at theta = 0
    and theta = pi, so the solid pieces are what is left between them.
    """
    def arc(phi0, phi1):
        return {"type": "elipseArc",
                "params": [0.0, 0.0, RB, RB, 0.0, phi0, phi1]}

    return {
        "g": G,
        "deadTime": 1e-9,

        "solidObjects": [
            arc(h, np.pi - h),                    # upper wall
            arc(np.pi + h, 2.0 * np.pi - h),      # lower wall
            ],

        # basin 1 is the hole at theta = 0, basin 2 the one at theta = pi.
        # label 0 is reserved: it is what a particle that never escaped reports
        "basinObjects": {
            1: [arc(-h, h)],
            2: [arc(np.pi - h, np.pi + h)],
            },
    }


def launch(theta, alpha, energy):
    """(theta, alpha) on the boundary -> the cartesian state the solver wants.

    The particle sits on the wall at angle theta and leaves it at angle alpha to
    the tangent, so alpha in (0, pi) always points into the billiard. For a
    circle the tangent direction is simply theta + pi/2, which is why no arctan
    appears here.

    -> (x, y, vx, vy, allowed). `allowed` is False where the energy is below the
    potential at that height, and no real speed exists.
    """
    allowed = energy >= POTENTIAL(theta)
    speed = np.sqrt(np.maximum(2.0 * (energy - POTENTIAL(theta)), 0.0))

    eta = theta + 0.5 * np.pi + alpha        # direction of travel
    return (RB * np.cos(theta), RB * np.sin(theta),
            speed * np.cos(eta), speed * np.sin(eta), allowed)


scene = compileScene(openCircle(HOLE), backend="openmp")
print(f"{len(scene.types)} objects -> {scene.so_path.name}, cached: {scene.cached}")
print(f"hole half-width h = {HOLE}, g = {G}, grid {RESOLUTION}x{RESOLUTION}")

theta = np.linspace(0.0, 2.0 * np.pi, RESOLUTION)
alpha = np.linspace(0.0, np.pi, RESOLUTION)
thetaGrid, alphaGrid = np.meshgrid(theta, alpha)

# -1 forbidden by energy, 0 never escaped, 1 and 2 the two holes
LABELS = ListedColormap(["#e8e8e8", "#ffffff", "#e8730a", "#101010"])
NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], 4)

figure, axesGrid = plt.subplots(2, 2, figsize=(11.5, 9.0))

for axes, energy in zip(axesGrid.ravel(), ENERGIES):
    x, y, vx, vy, allowed = launch(thetaGrid, alphaGrid, energy)

    # only the energetically possible initial conditions are simulated at all.
    # the paper's criterion is a collision count, not a time, which is what
    # iterations= says: no trajectory is stored either way, just the ending
    esc = getBasins(scene, x[allowed], y[allowed], vx[allowed], vy[allowed],
                    iterations=COLLISIONS, save=["t", "id"])

    picture = np.full(thetaGrid.shape, -1, dtype=np.int32)
    picture[allowed] = esc.id

    tally = esc.tally()
    escaped = esc.id != 0
    meanEscape = esc.t[escaped].mean() if escaped.any() else float("nan")
    print(f"  E = {energy:.1f}: {allowed.sum():6d} allowed, "
          f"hole1 {tally.get(1, 0):6d}, hole2 {tally.get(2, 0):6d}, "
          f"trapped {tally.get(0, 0):6d}, mean escape time {meanEscape:7.2f}")

    axes.imshow(picture, origin="lower", aspect="auto", cmap=LABELS, norm=NORM,
                extent=(0.0, 2.0 * np.pi, 0.0, np.pi), interpolation="nearest")
    axes.set_title(f"E = {energy}", fontsize=11)
    axes.set_xticks(np.arange(5) * np.pi / 2)
    axes.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axes.set_yticks(np.arange(3) * np.pi / 2)
    axes.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
    axes.set_xlabel(r"$\theta$")
    axes.set_ylabel(r"$\alpha$")

figure.suptitle(f"escape basins of the open gravitational circular billiard "
                f"(h = {HOLE}, g = {G})\n"
                f"orange: escaped at "r"$\theta=0$""   black: escaped at "
                r"$\theta=\pi$""   white: trapped   grey: E below the potential",
                fontsize=12)
figure.tight_layout(rect=(0, 0, 1, 0.94))
figure.savefig("escapeBasins.png", dpi=150)
print("wrote escapeBasins.png")

