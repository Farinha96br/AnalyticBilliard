"""Every execution mode builds and runs, off one compiled scene.

A mode is not a build. The four of them -- the event table, the uniform grid and
escape basins under either limit -- are separate entry points into the SAME .so
(abRunIterations, abRunSampled, abRunBasinsTime, abRunBasinsIterations), and
switching between them never runs the compiler. What does decide the build is
the backend and the scene's structure, basin slots included.

So these tests compile one scene that has everything -- walls and a basin -- and
then ask each mode for a well-formed result. test_aModeSwitchNeverRebuilds is
the one that pins the claim above rather than assuming it.
"""

import numpy as np
import pytest

from hpcBilliards import SceneError, compileScene, getBasins, updateScene

import helpers as H
from helpers import push

NPART = 32
ITERATIONS = 200
TF, DT = 20.0, 0.05


@pytest.fixture
def ic():
    return H.insideSquare(NPART, g=1.0)


def test_compilesAndLoads(basinBox):
    assert basinBox.so_path.exists(), "compileScene reported a .so that is not there"
    assert basinBox.lib.abObjectCount() == len(basinBox.types)
    # the basin is what getBasins needs and what changes the generated header
    assert any(t.startswith("basin:") for t in basinBox.types), basinBox.types


def test_recordWithIterations(basinBox, ic):
    c = push(basinBox, H.basinScene(g=1.0))
    rec = c.recordWithIterations(*ic, iterations=ITERATIONS, stride=2,
                                 save=["t", "x", "y", "vx", "vy", "evType"])

    assert rec.x.shape == (NPART, ITERATIONS // 2 + 1)
    assert (rec.counts >= 1).all(), "a run that wrote no row at all"
    for name in ("t", "x", "y", "vx", "vy"):
        assert np.isfinite(H.realRows(rec, name)).all(), name
    assert set(H.realRows(rec, "evType").tolist()) <= {0, 1, 2, 3}


def test_recordWithTime(basinBox, ic):
    c = push(basinBox, H.basinScene(g=1.0))
    rec = c.recordWithTime(*ic, tf=TF, dt=DT)

    rows = int(TF / DT) + 1
    assert rec.x.shape == (NPART, rows)

    # one grid, shared by every particle: row n is the time n*dt, not whatever
    # the n-th event happened to be. Compared per particle over its own real
    # rows, because the basin ends a run wherever it falls and counts says so.
    assert rec.counts.max() > 1, "no particle survived even one sample"
    for i in range(NPART):
        n = rec.counts[i]
        assert rec.t[i, :n] == pytest.approx(DT * np.arange(n), abs=1e-12), i
    for name in ("t", "x", "y", "vx", "vy"):
        assert np.isfinite(H.realRows(rec, name)).all(), name


@pytest.mark.parametrize("limit", [{"tf": 50.0}, {"iterations": 2000}])
def test_escapeBasins(basinBox, ic, limit):
    """both limits: one row per particle, and a label that means something"""
    c = push(basinBox, H.basinScene(g=1.0))
    esc = getBasins(c, *ic, **limit)

    for name in ("t", "x", "y", "vx", "vy", "id"):
        assert getattr(esc, name).shape == (NPART,), name
    assert np.isfinite(esc.t).all()
    # 1 is the ceiling; 0 is reserved for a particle still going at the limit
    assert set(esc.id.tolist()) <= {0, 1}
    assert sum(esc.tally().values()) == NPART


def test_somethingActuallyEscapesAndSomethingDoesNot(basinBox, ic):
    """a tally of all zeros would pass every test above and mean nothing"""
    c = push(basinBox, H.basinScene(g=1.0))
    tally = getBasins(c, *ic, tf=50.0).tally()
    assert tally.get(1, 0) > 0, f"no particle ever reached the basin: {tally}"
    assert tally.get(0, 0) > 0, f"nothing survived to the limit: {tally}"


def test_aModeSwitchNeverRebuilds(basinBox, ic, backend):
    """the structural claim: the mode is a call, not a build.

    All four modes run off the one library, and compiling the same scene again
    is a cache hit. (The first build cannot be asserted to MISS -- .abcache is
    gitignored, so CI arrives cold and a local rerun arrives warm.)
    """
    scene = H.basinScene(g=1.0)
    c = push(basinBox, scene)

    c.recordWithIterations(*ic, iterations=50)
    c.recordWithTime(*ic, tf=5.0, dt=0.1)
    getBasins(c, *ic, tf=5.0)
    getBasins(c, *ic, iterations=50)

    again = compileScene(scene, backend=backend)
    assert again.cached is True, "a second build of one scene is not a cache hit"
    assert again.so_path == basinBox.so_path

    scene.g = 0.25
    assert updateScene(scene, basinBox) is basinBox   # pushes, never compiles
    push(basinBox, H.basinScene(g=1.0))


def test_everyColumnIsAvailable(basinBox, ic):
    c = push(basinBox, H.basinScene(g=1.0))

    rec = c.recordWithIterations(
        *ic, iterations=50, save=["t", "x", "y", "vx", "vy", "Energy", "evType"])
    assert rec.energy.shape == rec.x.shape   # the key is Energy, the attribute energy

    esc = getBasins(c, *ic, tf=5.0, save=["t", "x", "y", "vx", "vy", "Energy", "id"])
    assert esc.energy.shape == (NPART,)


def test_eachModeRefusesWhatItCannotMean(basinBox, ic, squareBox):
    c = push(basinBox, H.basinScene(g=1.0))

    with pytest.raises(ValueError):       # a grid sample is not an event
        c.recordWithTime(*ic, tf=1.0, dt=0.1, save=["t", "evType"])
    with pytest.raises(ValueError):       # neither limit
        getBasins(c, *ic)
    with pytest.raises(ValueError):       # both limits
        getBasins(c, *ic, tf=1.0, iterations=10)
    with pytest.raises(ValueError):       # an iterations= run IS the bound
        getBasins(c, *ic, iterations=10, maxEvents=5)

    # and a scene with nothing to escape through cannot be asked at all
    plain = push(squareBox, H.squareScene(g=1.0))
    with pytest.raises(SceneError):
        getBasins(plain, *ic, tf=1.0)
