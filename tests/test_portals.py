"""A portal maps a particle rigidly, and the two flips turn only its velocity.

The exit frame takes its origin and direction from the partner face but its
length from the entry, so the map is a rigid motion: distance along the surface
is the only thing that crosses over. The flips multiply the tangential and
normal components of the velocity by +-1 and touch nothing else. Together those
say all four sign pairs land on ONE point, and a level pair conserves energy
exactly.

Every test here runs at g = 0 unless it says otherwise, which is what makes the
comparison exact: with no gravity the velocity arriving at the portal IS the
previous event row's velocity, so the state going in can be reconstructed and
fed to an independent implementation of the map.

The flips and both faces are parameters, not structure, so every orientation
below comes out of the single portalChannel build.
"""

import numpy as np
import pytest

import helpers as H
from helpers import TOL, push

# chosen so the first portal crossing is unmistakable: it lands at u = 0.8625,
# well clear of the midpoint (where reversing the exit would be invisible) and
# of both corners (where a wall would share the event), with the tangential and
# normal velocities both non-zero so either flip has something to act on
START = ([0.25], [0.3], [0.8], [0.6])

FLIPS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

EXITS = {
    "forward": H.PORTAL_EXIT,            # same direction as the entry face
    "reversed": H.PORTAL_EXIT_REVERSED,  # the same face written end for end
    "twiceAsLong": H.PORTAL_EXIT_LONG,   # a mismatched pair
}


def firstCrossing(compiled, scene):
    """-> (arriving state, leaving state) for the run's first portal event.

    A portal row records where the particle came OUT; where it went in is not
    stored. At g = 0 the flight before it is a straight line at constant
    velocity, so the arriving state follows from the row before and the gap
    between the two timestamps.
    """
    push(compiled, scene)
    rec = compiled.recordWithIterations(
        *START, iterations=12, save=["t", "x", "y", "vx", "vy", "evType"])

    ev = rec.evType[0, :rec.counts[0]]
    portals = np.flatnonzero(ev == 2)
    assert portals.size, f"no portal event in this run: evType = {ev.tolist()}"
    k = int(portals[0])
    assert k > 0, "the initial row cannot be a portal event"

    dt = rec.t[0, k] - rec.t[0, k - 1]
    arriving = np.array([rec.x[0, k - 1] + rec.vx[0, k - 1] * dt,
                         rec.y[0, k - 1] + rec.vy[0, k - 1] * dt,
                         rec.vx[0, k - 1], rec.vy[0, k - 1]])
    leaving = np.array([rec.x[0, k], rec.y[0, k], rec.vx[0, k], rec.vy[0, k]])
    return arriving, leaving


@pytest.mark.parametrize("exitName", list(EXITS))
@pytest.mark.parametrize("flips", FLIPS)
def test_matchesTheReferenceTransform(portalChannel, flips, exitName):
    """the whole map at once, against an implementation written independently"""
    nf, tf = flips
    exitSeg = EXITS[exitName]
    arriving, leaving = firstCrossing(
        portalChannel, H.portalScene(nf, tf, exitSeg))

    expected = H.referenceTeleport(H.PORTAL_ENTRY, exitSeg, nf, tf, arriving)
    assert leaving == pytest.approx(expected, abs=TOL), (
        f"nf={nf} tf={tf} exit={exitName}: got {leaving}, expected {expected}")


@pytest.mark.parametrize("exitName", list(EXITS))
@pytest.mark.parametrize("flips", FLIPS)
def test_arcLengthIsPreserved(portalChannel, flips, exitName):
    """the rigid-motion claim, independent of the flips: how far along the entry
    face a particle arrived is how far along the exit face it leaves"""
    nf, tf = flips
    exitSeg = EXITS[exitName]
    arriving, leaving = firstCrossing(
        portalChannel, H.portalScene(nf, tf, exitSeg))

    entryFrame, exitFrame = H.portalFrames(H.PORTAL_ENTRY, exitSeg)
    uIn = H.arcLength(entryFrame, arriving[0], arriving[1])
    uOut = H.arcLength(exitFrame, leaving[0], leaving[1])
    assert uOut == pytest.approx(uIn, abs=TOL)


@pytest.mark.parametrize("flips", FLIPS)
def test_speedIsUnchanged(portalChannel, flips):
    """orthonormal frames and +-1 signs cannot change |v|, whatever the flips"""
    nf, tf = flips
    arriving, leaving = firstCrossing(portalChannel, H.portalScene(nf, tf))
    assert np.hypot(*leaving[2:]) == pytest.approx(np.hypot(*arriving[2:]),
                                                   abs=TOL)


def test_allFourFlipsLandOnTheSamePoint(portalChannel):
    """the cleanest statement of 'the signs act on the velocity alone'"""
    points = {firstCrossing(portalChannel, H.portalScene(nf, tf))[1][:2].tobytes()
              for nf, tf in FLIPS}
    assert len(points) == 1, "the flips moved where the particle came out"


def test_aReversedExitMapsEndForEnd(portalChannel):
    """writing the far face's endpoints the other way round is what moves the
    exit, and it is the orientation the library asks you to pass nf = -1 with"""
    _, forward = firstCrossing(portalChannel, H.portalScene(1.0, 1.0,
                                                            H.PORTAL_EXIT))
    _, reverse = firstCrossing(portalChannel, H.portalScene(1.0, 1.0,
                                                            H.PORTAL_EXIT_REVERSED))
    assert forward[0] == pytest.approx(reverse[0], abs=TOL)   # same face
    # mirrored in the face's own length: u from one end becomes u from the other
    assert reverse[1] == pytest.approx(H.SIDE - forward[1], abs=TOL)


@pytest.mark.parametrize("flips", FLIPS)
def test_theFlipsAreTheSignsOfTheVelocityComponents(portalChannel, flips):
    """the physical reading: nf = -1 turns it back through the face it arrived
    at, tf = -1 reverses the way it was travelling ALONG the face"""
    nf, tf = flips
    arriving, leaving = firstCrossing(portalChannel, H.portalScene(nf, tf))

    (_, aT, aN), (_, bT, bN) = H.portalFrames(H.PORTAL_ENTRY, H.PORTAL_EXIT)
    vIn, vOut = arriving[2:], leaving[2:]

    assert vOut @ bN == pytest.approx(nf * (vIn @ aN), abs=TOL)
    assert vOut @ bT == pytest.approx(tf * (vIn @ aT), abs=TOL)
    # and the launch really does load both components, so neither is vacuous
    assert abs(vIn @ aN) > 0.1 and abs(vIn @ aT) > 0.1


def test_mismatchedLengthsDoNotStretch(portalChannel):
    """an exit face twice as long maps rigidly, not proportionally.

    u = 0.8625 in must be u = 0.8625 out, not 1.725: the exit keeps its own
    direction but borrows the ENTRY's length.
    """
    arriving, leaving = firstCrossing(
        portalChannel, H.portalScene(1.0, 1.0, H.PORTAL_EXIT_LONG))

    entryFrame, exitFrame = H.portalFrames(H.PORTAL_ENTRY, H.PORTAL_EXIT_LONG)
    uIn = H.arcLength(entryFrame, arriving[0], arriving[1])
    uOut = H.arcLength(exitFrame, leaving[0], leaving[1])

    assert uOut == pytest.approx(uIn, abs=TOL)
    assert uOut < H.SIDE, "the particle was stretched along the longer face"


def test_aLevelPortalConservesEnergy(portalChannel):
    """under gravity now. The two faces span the same heights, so the rigid map
    carries y across untouched and U = g*y with it -- and the flips cannot help,
    since they never touch the position."""
    c = push(portalChannel, H.portalScene(1.0, 1.0, g=1.0))
    rec = c.recordWithIterations(
        *START, iterations=400,
        save=["t", "x", "y", "vx", "vy", "Energy", "evType"])

    n = rec.counts[0]
    energy, ev = rec.energy[0, :n], rec.evType[0, :n]
    assert (ev == 2).sum() > 10, "this run barely used the portal"
    assert np.ptp(energy) / abs(energy[0]) < 1e-12, (
        f"energy spread {np.ptp(energy):.3e} over {n} events")
