# what every square-with-portals script needs: the table, a picture of it, the
# initial conditions, and the way a run is read back out.
#
# nothing here runs a billiard or draws a figure of its own. Each script owns its
# own parameters and its own output, and takes only the pieces that would
# otherwise be copied between them -- makeTable above all, since a table that
# drifts between two scripts is two different experiments being compared

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

# this file sits two folders down, so python puts PortalExperiments/square/ on
# the path and not the repo root, which is where hpcBilliards lives. Adding it
# here is what lets any script in this folder run from any directory, with no
# PYTHONPATH to remember. parents[1] is PortalExperiments/, parents[2] is the
# root. compileScene is re-exported so a script needs this import and no other
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from hpcBilliards import compileScene       # noqa: E402, re-exported

SIDE = 1.0          # the box is [0, L] x [0, L], so the ground is y = 0


def makeTable(g, dLeft, dRight, h, L=SIDE, normalFlip=+1.0, tangentFlip=+1.0):
    """the box, with a portal in each vertical wall unless h closes them up.

    d is where a portal starts, not where it is centred, so its gap runs from d
    up to d + h. At h = 0 there is no gap: the walls are whole, the portal pair
    is left out of the scene entirely, and what is left is a plain square box.
    Passing a zero-length segment instead would not do -- abBuildScene refuses
    one rather than let it build as NaN and read as a wall that is never hit.
    """
    if h == 0.0:
        return {
            "g": g,
            "deadTime": 1e-10,
            # counterclockwise from the bottom-left corner. Four whole walls, so
            # no two of them share an end other than at a real corner: a pair of
            # collinear pieces meeting mid-wall would bounce a particle twice at
            # the join, which is no bounce at all
            "solidObjects": [
                {"type": "lineSegment", "params": [0.0, 0.0, L, 0.0]},
                {"type": "lineSegment", "params": [L, 0.0, L, L]},
                {"type": "lineSegment", "params": [L, L, 0.0, L]},
                {"type": "lineSegment", "params": [0.0, L, 0.0, 0.0]},
            ],
        }

    l0, l1 = dLeft, dLeft + h           # the gap in the left wall
    r0, r1 = dRight, dRight + h         # the gap in the right wall

    # the portal has to fit on the wall it is cut into. Flush with a corner is
    # allowed -- that is the piece() below -- but hanging off the end is not
    for name, d, lo, hi in (("dLeft", dLeft, l0, l1), ("dRight", dRight, r0, r1)):
        if not (0.0 <= lo and hi <= L):
            raise ValueError(
                f"{name} = {d} puts a portal of length {h} at [{lo}, {hi}], "
                f"which does not fit on a wall of length {L}: "
                f"keep 0 <= {name} <= {L - h}")

    # a portal flush with a corner leaves nothing of the wall beside it, and a
    # zero-length lineSegment is degenerate: abBuildScene refuses it rather than
    # let it build as NaN and read as a wall that is never hit. So that piece is
    # left out. That changes the object count, and so the structure, which means
    # a placement flush with a corner compiles its own library the first time it
    # is seen -- a few compiles across a sweep, all of them cached afterwards
    def piece(x0, y0, x1, y1):
        return [] if (x0 == x1 and y0 == y1) else [
            {"type": "lineSegment", "params": [x0, y0, x1, y1]}]

    left = [0.0, l0, 0.0, l1]           # bottom end first: u grows upwards
    right = [L, r0, L, r1]              # likewise, so the two agree on u

    return {
        "g": g,             # gravity, pulling towards -y
        "deadTime": 1e-10,  # roots below this are the collision just resolved

        # counterclockwise from the bottom-left corner, with the two vertical
        # walls split around the portal each one carries
        "solidObjects": [
            *piece(0.0, 0.0, L, 0.0),
            *piece(L, 0.0, L, r0),
            *piece(L, r1, L, L),
            *piece(L, L, 0.0, L),
            *piece(0.0, L, 0.0, l1),
            *piece(0.0, l0, 0.0, 0.0),
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
    """the table as the solver was given it: walls black, portals coloured.

    Everything is read back off the scene dict rather than redrawn from L, so
    the picture cannot drift out of step with what is being run: the gaps the
    walls were split around are exactly the ones that show here, and at h = 0
    there are no portalObjects to draw and this comes out a plain box.
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(5.5, 5.5))

    for wall in table["solidObjects"]:
        x0, y0, x1, y1 = wall["params"]
        axes.plot([x0, x1], [y0, y1], color="#101010", lw=1.5, zorder=3,
                  solid_capstyle="butt")

    # the pair is written both ways round, so the two entries are the two
    # mouths and drawing the entry of each covers both gaps exactly once
    colours = ["#e8730a", "#1f77b4"]
    for k, portal in enumerate(table.get("portalObjects", [])):
        x0, y0, x1, y1 = portal["entryParams"]
        axes.plot([x0, x1], [y0, y1], color=colours[k % len(colours)], lw=3,
                  zorder=5, solid_capstyle="butt", label=f"portal {k + 1}")

    axes.set_aspect("equal")
    axes.set_xlim(-0.15 * L, 1.15 * L)
    axes.set_ylim(-0.15 * L, 1.15 * L)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    if table.get("portalObjects"):
        axes.legend(loc="upper center", fontsize=9)
    return axes


def seriesToBirkoff(x, y, vx, vy, x0=0.0, y0=0.0):
    """collisions -> the Birkhoff pair (theta, p) on the unit square.

    theta is arclength counterclockwise from the corner at (x0, y0) -- ground,
    right wall, top, left wall, ending at 4 -- and p is the outgoing velocity
    projected on that wall's tangent once it is normalised, so cos of the angle
    to the wall. That pair is the canonical one, area preserving as it stands.

    Every row here is a collision, so the point is on a wall by construction
    and the wall is just the nearest of the four: no tolerance to choose and
    nothing to rule out. A corner ties, and the tie goes to the earlier wall,
    which is the order theta runs in anyway
    """
    u, v = np.asarray(x) - x0, np.asarray(y) - y0   # the box is [0, 1]^2 in u, v

    wall = np.argmin([v, 1.0 - u, 1.0 - v, u], axis=0)
    theta = wall + np.choose(wall, [u, v, 1.0 - u, 1.0 - v])

    # the four tangents are a quarter turn apart in that same order, so the
    # angle to the wall is the velocity's angle less the wall's own: no frame
    # to assemble and no dot product to take, one wall at a time
    alpha = np.arctan2(vy, vx) - wall * (0.5 * np.pi)

    return theta, np.cos(alpha)


def heightAt(theta, L=SIDE):
    """how high the wall is at arclength theta -- the potential's whole story.

    A collision needs E >= g*y there, and y depends only on theta, so this is
    what carves the map into a reachable part and an empty one.
    """
    s = np.asarray(theta, dtype=float)
    return np.select([s <= L, s <= 2.0 * L, s <= 3.0 * L],
                     [np.zeros_like(s), s - L, np.full_like(s, L)],
                     default=4.0 * L - s)


def startWithFixedEnergy(E, N, g, L=SIDE, seed=0):
    """N particles sharing an energy, spread uniformly over the box by area.

    The reachable band is 0 <= y <= E/g, capped by the ceiling. A box has the
    same width at every height, unlike the circle, so uniform by area is just
    uniform in x and uniform in y -- there is no chord to weight against.
    """
    highest = min(E / g, L) if g != 0.0 else L
    if not highest > 0.0:
        raise ValueError(f"nothing to sample: E = {E} does not lift a particle "
                         f"off the ground at g = {g}")

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, L, N)
    y = rng.uniform(0.0, highest, N)

    speed = np.sqrt(2.0 * (E - g * y))
    direction = rng.uniform(0.0, 2.0 * np.pi, N)
    return x, y, speed * np.cos(direction), speed * np.sin(direction)


def fan(E, g, n, x0, y0):
    """n particles from one point, directions spread over the whole turn.

    They share an energy, so they share a speed: E = v^2/2 + g*y is read at the
    launch height and what is left over goes into v. Only the direction tells
    them apart, which is what makes the spread between them the portals' doing
    rather than the initial conditions'.
    """
    speed = np.sqrt(2.0 * (E - g * y0))
    eta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (np.full(n, x0), np.full(n, y0),
            speed * np.cos(eta), speed * np.sin(eta))


def realRows(run):
    """every written row of every particle, flattened.

    counts differ once the event filter has been applied, so the block cannot
    simply be ravelled: the rows past counts[i] are buffer that was never
    written, and they all sit at the origin
    """
    keep = np.arange(run.x.shape[1])[None, :] < run.counts[:, None]
    return run.x[keep], run.y[keep], run.vx[keep], run.vy[keep]


def arcs(rows, g, n=60):
    """the exact parabola leaving each event, up to the next one.

    Nothing is integrated: between two events the flight is a parabola in closed
    form, so this evaluates the one the solver already solved.

    A portal writes only where the particle came OUT, so running row i forward
    over t[i+1] - t[i] lands on the mouth it went IN by. The gap between that
    point and row i+1 is the jump itself, and no arc is drawn across it.
    """
    t, x, y = rows["t"], rows["x"], rows["y"]
    vx, vy = rows["vx"], rows["vy"]
    out = []
    for i in range(len(t) - 1):
        tau = np.linspace(0.0, t[i + 1] - t[i], n)
        out.append((x[i] + vx[i] * tau,
                    y[i] + vy[i] * tau - 0.5 * g * tau * tau))
    return out
