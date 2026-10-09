"""Two compiled scenes of the same shape do not overwrite each other.

The build cache keys a .so on the scene's STRUCTURE alone -- that is what lets a
parameter change skip the compiler -- and the scene itself lives in globals
inside that .so: objs[], g, deadTime. dlopen hands back one mapping per path, so
two CompiledScenes of the same shape really are sharing one set of globals.

Whoever pushed last used to decide what BOTH of them computed, while each still
reported the gravity it was given. A run now puts its own numbers back first.

Gravity is read back from a particle thrown straight up off the floor: it comes
down at 2*v0/g, so the time of its first bounce says what gravity the library
was actually running. deadTime needs its own trick, below.
"""

import numpy as np
import pytest

from hpcBilliards import Line, LineSegment, Scene, compileScene, updateScene

V0 = 1.0


def floorAt(g):
    return Scene(g=g, solidObjects=[Line(0.0, 1.0, 0.0)])


def firstBounce(compiled):
    """-> the gravity this run actually happened under"""
    rec = compiled.recordWithIterations([0.0], [0.0], [0.0], [V0], iterations=1)
    return 2.0 * V0 / rec.t[0, 1]


def test_twoScenesOfOneShapeShareTheLibrary(backend):
    """the premise. If this ever stops being true the rest is moot, not broken."""
    a = compileScene(floorAt(1.0), backend=backend)
    b = compileScene(floorAt(4.0), backend=backend)
    assert a.so_path == b.so_path
    assert a.lib._handle == b.lib._handle


def test_eachKeepsItsOwnGravity(backend):
    a = compileScene(floorAt(1.0), backend=backend)
    b = compileScene(floorAt(4.0), backend=backend)

    # interleaved, so neither simply happens to be the one pushed last
    assert firstBounce(a) == pytest.approx(1.0)
    assert firstBounce(b) == pytest.approx(4.0)
    assert firstBounce(a) == pytest.approx(1.0)
    assert firstBounce(b) == pytest.approx(4.0)


# deadTime is invisible until a collision time falls INSIDE it: firstRoot takes
# only t > deadTime, reading anything sooner as the bounce it has just resolved.
# So the two values have to disagree about whether the wall is there at all --
# one wall, and a particle close enough to it that they do.
NEAR = 0.05      # distance to the wall, closing at speed 1, so the hit is at t = 0.05
FINE = 1e-12     # under NEAR: the hit is taken
COARSE = 0.1     # over NEAR: the hit is dismissed as already resolved


def wallAt(deadTime):
    """one wall at x = 0. Same single-Line structure as the floor above, so this
    costs no extra build -- and shares its library, which is the point."""
    return Scene(g=0.0, deadTime=deadTime, solidObjects=[Line(1.0, 0.0, 0.0)])


def towardsTheWall(compiled):
    """-> (rows written, the particle's vx at the end)

    Two rows means the initial state and a bounce, leaving vx = +1 as it travels
    away. One row means stepScene found nothing it was willing to hit, so the
    run stopped where it began with vx = -1, still closing.
    """
    rec = compiled.recordWithIterations([NEAR], [0.0], [-1.0], [0.0], iterations=1,
                                        save=["t", "x", "y", "vx", "vy"])
    n = rec.counts[0]
    return n, rec.vx[0, n - 1]


def test_eachKeepsItsOwnDeadTime(backend):
    """the third global that _push restores, and the only test here that says
    what deadTime actually does.

    Same wall, same particle, nothing different but deadTime -- and under one
    value it bounces while under the other it passes straight through.
    """
    fine = compileScene(wallAt(FINE), backend=backend)
    coarse = compileScene(wallAt(COARSE), backend=backend)

    # interleaved, so neither can pass by being the one pushed last
    assert towardsTheWall(fine) == (2, 1.0)
    assert towardsTheWall(coarse) == (1, -1.0)
    assert towardsTheWall(fine) == (2, 1.0)
    assert towardsTheWall(coarse) == (1, -1.0)


def test_theReportedGravityIsTheOneThatRuns(backend):
    """the attribute used to agree with what you asked for and disagree with
    what ran, which is what made this hard to notice at all"""
    a = compileScene(floorAt(1.0), backend=backend)
    compileScene(floorAt(4.0), backend=backend)
    assert firstBounce(a) == pytest.approx(a.g)


def test_eachKeepsItsOwnGeometry(backend):
    """not just the constants: objs[] is a global too"""
    def box(width):
        return Scene(g=0.0, solidObjects=[
            LineSegment(0.0, 0.0, width, 0.0), LineSegment(width, 0.0, width, 1.0),
            LineSegment(width, 1.0, 0.0, 1.0), LineSegment(0.0, 1.0, 0.0, 0.0)])

    narrow = compileScene(box(1.0), backend=backend)
    wide = compileScene(box(9.0), backend=backend)

    def rightWall(c):   # fired along +x, the first bounce is the far wall
        return c.recordWithIterations([0.01], [0.5], [1.0], [0.0],
                                      iterations=1).x[0, 1]

    assert rightWall(narrow) == pytest.approx(1.0)
    assert rightWall(wide) == pytest.approx(9.0)
    assert rightWall(narrow) == pytest.approx(1.0)


def test_theGridRecorderReclaimsToo(backend):
    """both recorders go through _run, but assert it rather than assume it"""
    a = compileScene(floorAt(1.0), backend=backend)
    compileScene(floorAt(4.0), backend=backend)

    rec = a.recordWithTime([0.0], [0.0], [0.0], [V0], tf=1.0, dt=0.25)
    # y = v0*t - g*t^2/2 at g = 1, not at g = 4
    assert rec.y[0, :rec.counts[0]] == pytest.approx(
        V0 * rec.t[0, :rec.counts[0]] - 0.5 * rec.t[0, :rec.counts[0]] ** 2)


def test_escapeRunsReclaimToo(backend):
    """getBasins has its own call site, so it needs its own check.

    Thrown up at v0 = 2 towards a ceiling basin at y = 1: under g = 1 it gets
    there, under g = 4 it cannot reach and reports 0.
    """
    def chute(g):
        return Scene(g=g, solidObjects=[Line(0.0, 1.0, 0.0)],
                     basinObjects={1: [Line(0.0, 1.0, -1.0)]})

    slow = compileScene(chute(1.0), backend=backend)
    fast = compileScene(chute(4.0), backend=backend)

    assert slow.getBasins([0.0], [0.0], [0.0], [2.0], tf=10.0).id[0] == 1
    assert fast.getBasins([0.0], [0.0], [0.0], [2.0], tf=10.0).id[0] == 0
    assert slow.getBasins([0.0], [0.0], [0.0], [2.0], iterations=20).id[0] == 1
    assert fast.getBasins([0.0], [0.0], [0.0], [2.0], iterations=20).id[0] == 0


def test_updateSceneStillClaimsTheLibrary(backend):
    a = compileScene(floorAt(1.0), backend=backend)
    b = compileScene(floorAt(4.0), backend=backend)
    assert firstBounce(b) == pytest.approx(4.0)

    scene = floorAt(0.25)
    updateScene(scene, a)
    assert firstBounce(a) == pytest.approx(0.25)


def test_anEditWithoutAPushStillDoesNotReachTheLibrary(backend):
    """the contract this must not have broken.

    'Edit then push' is how a sweep works, and a re-push on every run would
    have quietly turned an un-pushed edit into a live one. Ownership is taken
    by the first run below, so the second has nothing to reclaim and the edit
    stays where it was written.
    """
    scene = floorAt(1.0)
    a = compileScene(scene, backend=backend)
    assert firstBounce(a) == pytest.approx(1.0)

    scene.g = 4.0                      # edited, deliberately not pushed
    assert firstBounce(a) == pytest.approx(1.0), \
        "an un-pushed edit reached the library"

    updateScene(scene, a)              # now it is asked for
    assert firstBounce(a) == pytest.approx(4.0)
