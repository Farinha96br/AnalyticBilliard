"""A round billiard under gravity conserves energy, over 100 particles.

E = |v|^2 / 2 + g*y per unit mass. Nothing integrates it: a flight is a closed
form parabola and a wall is specular with restitution 1, so energy is conserved
by construction and what is left to measure is the arithmetic. The hit time is
where the cost sits -- a circle is an equal-radii ellipse, so every collision is
a quartic solved by Ferrari's method, polished by two Newton steps and accepted
on a residual. That is the sharpest numerics in the library and this is the test
that watches it.

Every particle starts on the same energy shell, by inverting the relation above
for the speed. So the conserved quantity is one number, shared, and a drift
shows up as a spread rather than as a hundred separate baselines.
"""

import numpy as np
import pytest

import helpers as H
from helpers import GEOM_EPS, TOL, push

NPART = 100
ITERATIONS = 300
G = 1.0

# measured, not guessed: the worst relative drift over 100 particles x 300
# bounces is 1.3e-13, median 1.1e-13. This is ~10x that, so it fails on a real
# regression and not on the last bit of a double.
ENERGY_TOL = 1e-12

# the spread over the last events against the spread over the first: ~1.04
# measured, i.e. the error is rounding and not a leak that accumulates
DRIFT_GROWTH = 5.0


@pytest.fixture(scope="module")
def circleRun(roundBilliard):
    c = push(roundBilliard, H.circleScene(g=G))
    ic = H.insideCircle(NPART, g=G)
    return c.recordWithIterations(
        *ic, iterations=ITERATIONS,
        save=["t", "x", "y", "vx", "vy", "Energy"])


def test_everyParticleRanToTheEnd(circleRun):
    rows = circleRun.x.shape[1]
    assert (circleRun.counts == rows).all(), (
        f"{(circleRun.counts != rows).sum()} of {NPART} particles stopped short "
        f"-- a missed quartic root loses the wall entirely")


def test_energyIsConservedForEveryParticle(circleRun):
    rec = circleRun
    worst, who = 0.0, -1
    for i in range(NPART):
        e = rec.energy[i, :rec.counts[i]]
        drift = np.max(np.abs(e - e[0])) / abs(e[0])
        if drift > worst:
            worst, who = drift, i
    assert worst < ENERGY_TOL, (
        f"particle {who} drifted {worst:.3e} relative, over "
        f"{ITERATIONS} bounces (tolerance {ENERGY_TOL:.0e})")


def test_theEnergyColumnAgreesWithTheState(circleRun):
    """the column is the only place the library computes energy, so it is worth
    checking it is the energy of the state it sits next to and not of another"""
    rec = circleRun
    here = 0.5 * (rec.vx ** 2 + rec.vy ** 2) + G * rec.y
    for i in range(NPART):
        n = rec.counts[i]
        assert here[i, :n] == pytest.approx(rec.energy[i, :n], abs=TOL)


def test_everyParticleStartsOnTheSameShell(circleRun):
    """the initial conditions really are one energy shell: without this, a
    conserved-per-particle result could hide a badly sampled launch"""
    first = circleRun.energy[:, 0]
    assert first == pytest.approx(H.ENERGY, abs=1e-12), np.ptp(first)


def test_noParticleLeavesTheCircle(circleRun):
    """a spurious quartic root invents a collision and snaps the particle onto
    the surface from the wrong side, which reads here as a radius over 1"""
    r = np.hypot(H.realRows(circleRun, "x"), H.realRows(circleRun, "y"))
    assert r.max() <= H.RADIUS + GEOM_EPS, f"max radius {r.max():.17g}"


def test_driftDoesNotGrowAcrossTheRun(circleRun):
    """rounding or a leak? A leak accumulates, so its spread over the closing
    events is far wider than over the opening ones. Rounding's is not."""
    rec = circleRun
    half = rec.x.shape[1] // 2
    opening = max(np.ptp(rec.energy[i, :half]) for i in range(NPART))
    closing = max(np.ptp(rec.energy[i, half:rec.counts[i]]) for i in range(NPART))
    assert closing <= DRIFT_GROWTH * opening, (
        f"spread grew from {opening:.3e} to {closing:.3e} across the run: "
        f"that is a secular drift, not rounding")


def test_speedIsPreservedByEveryBounceAtZeroGravity(roundBilliard):
    """the same circle with gravity switched off, where energy is just speed:
    a specular wall with restitution 1 cannot change |v| at all"""
    c = push(roundBilliard, H.circleScene(g=0.0))
    ic = H.insideCircle(NPART, g=0.0)
    rec = c.recordWithIterations(*ic, iterations=ITERATIONS,
                                 save=["t", "x", "y", "vx", "vy"])

    speed = np.hypot(rec.vx, rec.vy)
    for i in range(NPART):
        s = speed[i, :rec.counts[i]]
        assert np.ptp(s) / s[0] < ENERGY_TOL, f"particle {i} changed speed"
