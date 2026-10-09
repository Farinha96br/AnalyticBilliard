"""Particles started inside a closed square box stay in it.

The box is four SEGMENTS, not four infinite lines, which is what gives this
test teeth: a particle that slips past a corner of a segment box meets nothing
ever again. Four infinite lines could not leak however badly the solver broke.

A leak shows up two ways and both are asserted, because neither implies the
other. The cheap one is counts: hitTime answers MAGIC_NO_COLLISION when nothing
is ever hit, the recorder stops there, and counts[i] falls short of the rows it
was given. The direct one is the coordinates. A NaN reads as "never hits" too,
since NaN loses every comparison, so finiteness is its own check.

Run at g = 0 and at g = 1: gravity curves every flight into a parabola and
sends the line solver down a different branch, so containment is a different
claim under each.
"""

import numpy as np
import pytest

import helpers as H
from helpers import GEOM_EPS, push

NPART = 64
ITERATIONS = 500
TF, DT = 50.0, 0.01


@pytest.fixture(params=[0.0, 1.0], ids=["noGravity", "gravity"])
def boxRun(request, squareBox):
    """the event table, one row per bounce, plus the gravity it was run under"""
    g = request.param
    c = push(squareBox, H.squareScene(g=g))
    ic = H.insideSquare(NPART, g=g)
    rec = c.recordWithIterations(
        *ic, iterations=ITERATIONS, save=["t", "x", "y", "vx", "vy", "evType"])
    return g, ic, rec, c


def test_everyParticleRanToTheEnd(boxRun):
    """the cheapest leak detector there is: a particle that got out stops
    collecting rows, because there is nothing left for it to hit"""
    _, _, rec, _ = boxRun
    rows = rec.x.shape[1]
    short = np.flatnonzero(rec.counts != rows)
    assert short.size == 0, (
        f"particles {short.tolist()} stopped early at "
        f"{rec.counts[short].tolist()} of {rows} rows -- they ran out of walls")


def test_nothingLeavesTheBox(boxRun):
    _, _, rec, _ = boxRun
    x, y = H.realRows(rec, "x"), H.realRows(rec, "y")
    assert x.min() >= -GEOM_EPS and x.max() <= H.SIDE + GEOM_EPS, \
        f"x reached [{x.min():.3e}, {x.max():.17g}]"
    assert y.min() >= -GEOM_EPS and y.max() <= H.SIDE + GEOM_EPS, \
        f"y reached [{y.min():.3e}, {y.max():.17g}]"


def test_nothingIsNan(boxRun):
    _, _, rec, _ = boxRun
    for name in ("t", "x", "y", "vx", "vy"):
        bad = ~np.isfinite(H.realRows(rec, name))
        assert not bad.any(), f"{bad.sum()} non-finite values in {name}"


def test_everyBounceIsOnAWall(boxRun):
    """distance to the nearest wall, with the projection clamped to the
    segment's ends -- measuring to the infinite carrier would wave through a
    hit recorded past the end of a wall, which is exactly how a corner leaks"""
    _, _, rec, _ = boxRun
    ev = H.realRows(rec, "evType")
    x, y = H.realRows(rec, "x"), H.realRows(rec, "y")

    bounce = ev == 1                      # row 0 is the initial state, inside
    assert bounce.sum() > NPART * 100, "almost nothing bounced"
    d = np.min([H.distanceToSegment(x[bounce], y[bounce], *w)
                for w in H.squareWalls()], axis=0)
    assert d.max() < GEOM_EPS, f"a bounce {d.max():.3e} away from every wall"


def test_timeIncreasesStrictly(boxRun):
    """what fails if deadTime ever stops suppressing the collision just
    resolved: the solver re-finds it, and the run stops going anywhere"""
    _, _, rec, _ = boxRun
    for i in range(NPART):
        t = rec.t[i, :rec.counts[i]]
        assert (np.diff(t) > 0.0).all(), f"particle {i} did not advance in time"


def test_theGridRunStaysInsideToo(boxRun):
    """events only ever say where a particle touched a wall. Sampling the
    flights in between is what would catch one that flew out and came back."""
    g, ic, _, c = boxRun
    rec = c.recordWithTime(*ic, tf=TF, dt=DT)

    x, y = H.realRows(rec, "x"), H.realRows(rec, "y")
    assert (rec.counts == rec.x.shape[1]).all(), "a grid run stopped short"
    assert x.min() >= -GEOM_EPS and x.max() <= H.SIDE + GEOM_EPS
    assert y.min() >= -GEOM_EPS and y.max() <= H.SIDE + GEOM_EPS
    assert np.isfinite(x).all() and np.isfinite(y).all()
