"""Scenes, initial conditions and the reference portal map the tests compare to.

Plain functions rather than fixtures: SCENE builders are needed at collection
time by parametrize, and the initial conditions are cheap enough that sharing
them would only couple the tests together.

Everything here is seeded. A billiard is chaotic, so an unseeded run is a test
that reports a different number every time it fails.
"""

import math

import numpy as np

from hpcBilliards import Elipse, Line, LineSegment, Portal, Scene, updateScene

# an exact claim -- a reflected speed, a teleported height -- is asserted at the
# width of double rounding, not at a physics tolerance
TOL = 1e-12

# a geometric one is not: a point is snapped onto its surface and deadTime is
# 1e-9, so "on the wall" and "inside the box" are true to about that
GEOM_EPS = 1e-9

SIDE = 1.0      # the unit square every box test uses
RADIUS = 1.0    # the round billiard
ENERGY = 2.0    # total energy per unit mass: over g*SIDE, so every start moves


# --------------------------------------------------------------------- scenes
#
# the structure -- the slot types in order -- is what picks the .so, and dlopen
# hands back the same library for the same path. two scenes of the SAME shape
# therefore share one set of C globals, `g` among them, so a test must push its
# own numbers before it runs rather than trusting the ones a fixture arrived
# with. push() below is that call, and every run in this suite goes through it.


def squareScene(g):
    """the unit box, as four segments.

    Segments and not infinite lines on purpose: a particle that slips past a
    corner of this box really does leave, and meets nothing ever again, which
    is the leak test_leakage.py is there to catch. Four infinite lines could
    not leak if the solver were badly broken.
    """
    return Scene(g=g, solidObjects=[
        LineSegment(0.0, 0.0, SIDE, 0.0),      # floor
        LineSegment(SIDE, 0.0, SIDE, SIDE),    # right
        LineSegment(SIDE, SIDE, 0.0, SIDE),    # ceiling
        LineSegment(0.0, SIDE, 0.0, 0.0),      # left
    ])


def circleScene(g, R=RADIUS):
    """a circular billiard. There is no circle class: it is an equal-radii Elipse,
    which means the hit time is the Ferrari quartic, the sharpest numerics here."""
    return Scene(g=g, solidObjects=[Elipse(0.0, 0.0, R, R, 0.0)])


def portalScene(normalFlip=1.0, tangentFlip=1.0, exitSeg=None, g=0.0):
    """an x-periodic channel: walls at y = 0 and y = 1, and a portal taking the
    face at x = 1 to the face at x = 0.

    The exit is a destination and not a wall, so there is nothing at x = 0 to
    bounce off -- a particle put through lands there and flies on.
    """
    return Scene(g=g,
                 solidObjects=[Line(0.0, 1.0, 0.0),        # y = 0
                               Line(0.0, 1.0, -SIDE)],     # y = 1
                 portalObjects=[Portal(PORTAL_ENTRY,
                                       exitSeg or PORTAL_EXIT,
                                       normalFlip, tangentFlip)])


# named, because the reference transform has to be handed the very same pair
PORTAL_ENTRY = LineSegment(SIDE, 0.0, SIDE, SIDE)   # x = 1, pointing +y
PORTAL_EXIT = LineSegment(0.0, 0.0, 0.0, SIDE)      # x = 0, pointing +y
PORTAL_EXIT_REVERSED = LineSegment(0.0, SIDE, 0.0, 0.0)   # the same face, end for end
PORTAL_EXIT_LONG = LineSegment(0.0, 0.0, 0.0, 2.0 * SIDE)  # twice as long


def basinScene(g):
    """the box with its ceiling absorbing instead of reflecting.

    Three walls and one basin, so a particle under gravity climbs out of the top
    and the run ends: getBasins has both a 1 to report and, for the ones still
    going when the limit runs out, a 0.
    """
    return Scene(g=g,
                 solidObjects=[LineSegment(0.0, 0.0, SIDE, 0.0),
                               LineSegment(SIDE, 0.0, SIDE, SIDE),
                               LineSegment(0.0, SIDE, 0.0, 0.0)],
                 basinObjects={1: [LineSegment(0.0, SIDE, SIDE, SIDE)]})


def freeFlightScene(g=0.0):
    """one wall, far enough away that a short run never reaches it.

    stepScene answers MAGIC_NO_COLLISION when nothing is ever hit, and the grid
    recorder then coasts the whole window off the initial state in one go -- so
    this isolates trajectory() with no collision anywhere near it.
    """
    return Scene(g=g, solidObjects=[Line(1.0, 0.0, -1000.0)])   # x = 1000


def floorScene(g):
    """a single floor at y = 0, for a vertical throw. Same structure as
    freeFlightScene, hence the same .so -- which is exactly why push() exists."""
    return Scene(g=g, solidObjects=[Line(0.0, 1.0, 0.0)])


def push(compiled, scene):
    """put this scene's numbers into the shared library, then hand it back.

    updateScene never compiles; it compares the structure and pushes. Calling it
    at the top of a test is what keeps a g from one test out of the next one.
    """
    updateScene(scene, compiled)
    return compiled


# -------------------------------------------------------- initial conditions
#
# all at one fixed energy, by inverting E = v^2 / 2 + g*y for the speed. That
# puts every particle on the same energy shell, so "energy is conserved" has a
# single number to be conserved and a drift shows up as a spread.


def _launch(x0, y0, g, E, rng):
    speed = np.sqrt(2.0 * (E - g * y0))
    eta = rng.uniform(0.0, 2.0 * np.pi, x0.size)
    return x0, y0, speed * np.cos(eta), speed * np.sin(eta)


def insideSquare(n, g, E=ENERGY, seed=0, margin=0.05):
    """n particles in the unit box, clear of the walls by `margin`."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(margin, SIDE - margin, n)
    y0 = rng.uniform(margin, SIDE - margin, n)
    return _launch(x0, y0, g, E, rng)


def insideCircle(n, g, E=ENERGY, R=RADIUS, seed=0, fill=0.9):
    """n particles inside the disc, within `fill` of the radius.

    Area-uniform by the sqrt, which matters not at all for conservation but
    keeps the sample off the centre where every chord is a diameter.
    """
    rng = np.random.default_rng(seed)
    r = R * fill * np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return _launch(r * np.cos(phi), r * np.sin(phi), g, E, rng)


# ------------------------------------------------ the portal map, in numpy
#
# a mirror of asPortal (geometry.h) and teleport (physics.h), written out again
# here so the C++ is compared against something and not against itself


def portalFrames(entry, exitSeg):
    """-> (entryFrame, exitFrame), each (origin, tangent, normal).

    Reproduces the one asymmetry in asPortal: the exit frame takes its ORIGIN
    and DIRECTION from the partner but its LENGTH from the entry, so a
    mismatched pair maps rigidly instead of stretching.
    """
    a = np.array([entry.x1 - entry.x0, entry.y1 - entry.y0])
    length = math.hypot(*a)

    b = np.array([exitSeg.x1 - exitSeg.x0, exitSeg.y1 - exitSeg.y0])
    t = b / math.hypot(*b)

    o = np.array([exitSeg.x0, exitSeg.y0])
    return (_frame(np.array([entry.x0, entry.y0]), np.array([entry.x1, entry.y1])),
            _frame(o, o + length * t))


def _frame(p0, p1):
    d = p1 - p0
    t = d / math.hypot(*d)
    return p0, t, np.array([-t[1], t[0]])   # normal is the tangent turned +90


def referenceTeleport(entry, exitSeg, normalFlip, tangentFlip, s):
    """the state s = (x, y, vx, vy) put through the portal, computed here.

    The position comes from the frames alone and the two signs touch only the
    velocity -- which is why all four sign pairs land on one point.
    """
    (ao, at, an), (bo, bt, bn) = portalFrames(entry, exitSeg)
    p = np.array([s[0], s[1]])
    v = np.array([s[2], s[3]])

    u = (p - ao) @ at
    vt, vn = v @ at, v @ an

    out = bo + u * bt
    vOut = tangentFlip * vt * bt + normalFlip * vn * bn
    return np.array([out[0], out[1], vOut[0], vOut[1]])


def arcLength(frame, x, y):
    """how far along a portal face a point sits, measured in that face's frame"""
    o, t, _ = frame
    return (x - o[0]) * t[0] + (y - o[1]) * t[1]


# ------------------------------------------------------------------ geometry


def distanceToSegment(px, py, x0, y0, x1, y1):
    """|point - segment|, with the projection clamped to the ends.

    The clamp is the point: a bounce is asserted to be on the SEGMENT, and the
    distance to its infinite carrier would call a hit out past the end good.
    """
    dx, dy = x1 - x0, y1 - y0
    span = dx * dx + dy * dy
    u = np.clip(((px - x0) * dx + (py - y0) * dy) / span, 0.0, 1.0)
    return np.hypot(px - (x0 + u * dx), py - (y0 + u * dy))


def squareWalls():
    """the four segments of the unit box, as plain tuples"""
    return [(0.0, 0.0, SIDE, 0.0), (SIDE, 0.0, SIDE, SIDE),
            (SIDE, SIDE, 0.0, SIDE), (0.0, SIDE, 0.0, 0.0)]


def realRows(rec, name):
    """column `name` for every particle, flattened over its own real rows only.

    Rows past counts[i] are the zeros the buffer was allocated with, never
    anything the run wrote: asserting on them tests numpy, not the billiard.
    """
    a = getattr(rec, name)
    return np.concatenate([a[i, :rec.counts[i]] for i in range(a.shape[0])])
