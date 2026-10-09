"""With g = 0 a flight is a straight line; with g > 0 it is a parabola.

Gravity enters the physics in exactly one place -- trajectory(), where y picks
up -g*t^2/2 and vy picks up -g*t -- and it is a mutable global, so switching it
off is a parameter push and not a rebuild. Two degenerations hang off g = 0: the
quartic root finder drops to a quadratic when its leading coefficient vanishes,
and the line solver's own quadratic drops to a linear one.

Straightness is asserted twice. Once with nothing to collide with, where the
claim is about trajectory() alone and holds exactly; once inside the box, where
it has to survive five hundred collisions. Then it is asserted FALSE at g = 1 --
without that, every test here would pass just as happily against a trajectory
that had forgotten about gravity altogether.
"""

import numpy as np
import pytest

import helpers as H
from helpers import GEOM_EPS, TOL, push

NPART = 64
ITERATIONS = 200
X0, Y0, VX, VY = 0.0, 0.5, 1.0, 0.4
TF, DT = 10.0, 0.01


def test_freeFlightIsExactlyStraight(oneWall):
    """one wall, a thousand units away and never reached.

    When nothing is ever hit the grid recorder coasts the whole window off the
    initial state in one go, so this is trajectory() on its own: no collision,
    no snapping, nothing else to blame a discrepancy on. It should be exact,
    not close.
    """
    c = push(oneWall, H.freeFlightScene(g=0.0))
    rec = c.recordWithTime([X0], [Y0], [VX], [VY], tf=TF, dt=DT)

    n = rec.counts[0]
    assert n == rec.x.shape[1], "the free flight stopped short of the window"
    t = rec.t[0, :n]

    assert rec.x[0, :n] == pytest.approx(X0 + VX * t, abs=TOL)
    assert rec.y[0, :n] == pytest.approx(Y0 + VY * t, abs=TOL)
    assert np.ptp(rec.vx[0, :n]) == 0.0, "vx changed with no force on it"
    assert np.ptp(rec.vy[0, :n]) == 0.0, "vy changed at zero gravity"

    # a straight line sampled on a uniform grid has no second difference
    assert np.abs(np.diff(rec.y[0, :n], 2)).max() < TOL


def _flightCrossProducts(rec):
    """|d x v| / (|d| |v|) for each flight between two consecutive events.

    Zero exactly when the displacement is parallel to the velocity it was made
    with -- i.e. when the flight was a straight line. Normalised, so the number
    means the same for a long flight and a short one.
    """
    dx = rec.x[:, 1:] - rec.x[:, :-1]
    dy = rec.y[:, 1:] - rec.y[:, :-1]
    vx, vy = rec.vx[:, :-1], rec.vy[:, :-1]

    scale = np.hypot(dx, dy) * np.hypot(vx, vy)
    return np.abs(dx * vy - dy * vx) / np.where(scale > 0.0, scale, 1.0)


@pytest.fixture(scope="module")
def boxIC():
    return H.insideSquare(NPART, g=0.0)


def test_everyFlightInABoxIsStraight(squareBox, boxIC):
    """straightness that has to survive 200 collisions, not just one flight"""
    c = push(squareBox, H.squareScene(g=0.0))
    rec = c.recordWithIterations(*boxIC, iterations=ITERATIONS,
                                 save=["t", "x", "y", "vx", "vy"])
    assert (rec.counts == rec.x.shape[1]).all()

    bend = _flightCrossProducts(rec)
    assert bend.max() < GEOM_EPS, (
        f"a flight bent by {bend.max():.3e} with gravity switched off")


def test_speedNeverChangesAtZeroGravity(squareBox, boxIC):
    """g = 0 and restitution 1: |v| is constant for the whole run"""
    c = push(squareBox, H.squareScene(g=0.0))
    rec = c.recordWithIterations(*boxIC, iterations=ITERATIONS,
                                 save=["t", "x", "y", "vx", "vy"])

    speed = np.hypot(rec.vx, rec.vy)
    assert np.ptp(speed, axis=1).max() == pytest.approx(0.0, abs=TOL)


def test_gravityActuallyBendsThem(squareBox):
    """the guard. Same scene, same measure, g = 1 -- and now the flights must
    be visibly curved. Measured: a median of 7e-2 against 0 at g = 0."""
    c = push(squareBox, H.squareScene(g=1.0))
    rec = c.recordWithIterations(*H.insideSquare(NPART, g=1.0),
                                 iterations=ITERATIONS,
                                 save=["t", "x", "y", "vx", "vy"])

    bend = _flightCrossProducts(rec)
    assert np.median(bend) > 1e-3, (
        f"gravity is on but the flights are still straight (median "
        f"{np.median(bend):.3e}): the zero-gravity tests prove nothing")
    assert (bend > GEOM_EPS).mean() > 0.99


def test_aVerticalThrowReturnsWithTheSameSpeed(oneWall):
    """the positive-gravity companion, on events so that it stays exact.

    Thrown straight up from the floor at v0, a particle comes back down at v0
    and does it every 2*v0/g. Asserted on the bounce table rather than on a
    sampled apex, which would only ever be right to within dt.
    """
    g, v0 = 1.0, 1.5
    c = push(oneWall, H.floorScene(g=g))
    rec = c.recordWithIterations([0.0], [0.0], [0.0], [v0], iterations=6,
                                 save=["t", "x", "y", "vx", "vy", "evType"])

    n = rec.counts[0]
    assert n == 7, f"expected 6 floor bounces, got {n - 1}"
    assert (rec.evType[0, 1:n] == 1).all()

    bounces = rec.t[0, 1:n]
    assert np.abs(rec.vy[0, 1:n]) == pytest.approx(v0, abs=TOL)
    assert np.diff(bounces) == pytest.approx(2.0 * v0 / g, abs=TOL)
    assert rec.y[0, 1:n] == pytest.approx(0.0, abs=GEOM_EPS)
