"""ctypes binding: CompiledScene, updateScene, the recorders and getBasins."""

import ctypes
import math
import warnings

import numpy as np

from . import codegen, compile as _compile, scene as _scene

# must match enum abColumn in UsableFunctions.cpp
COLUMN_BIT = {"t": 1 << 0, "x": 1 << 1, "y": 1 << 2, "vx": 1 << 3,
              "vy": 1 << 4, "Energy": 1 << 5, "evType": 1 << 6,
              "id": 1 << 7}
COLUMNS = list(COLUMN_BIT)

# an event row knows what kind of event it is; a grid sample is not an event,
# so evType is not merely unset there but meaningless. an escape row is neither:
# it is where a run ENDED, so what it carries instead is which basin ended it
DEFAULT_SAVE = ["t", "x", "y", "vx", "vy", "evType"]
DEFAULT_SAVE_SAMPLED = ["t", "x", "y", "vx", "vy"]
DEFAULT_SAVE_ESCAPE = ["t", "x", "y", "vx", "vy", "id"]

# the columns each kind of run can produce at all
TABLE_COLUMNS = ["t", "x", "y", "vx", "vy", "Energy", "evType"]
ESCAPE_COLUMNS = ["t", "x", "y", "vx", "vy", "Energy", "id"]

EV_INITIAL, EV_BOUNCE, EV_PORTAL = 0, 1, 2
EVENT_CODE = {"initial": EV_INITIAL, "bounce": EV_BOUNCE, "portal": EV_PORTAL}

WARN_BYTES = 256 * 1024 ** 2
MAGIC_NO_COLLISION = 987654321000.0 # types.h: what the solver means by "never"


class _Output(ctypes.Structure):
    _fields_ = [("t", ctypes.POINTER(ctypes.c_double)),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("vx", ctypes.POINTER(ctypes.c_double)),
                ("vy", ctypes.POINTER(ctypes.c_double)),
                ("energy", ctypes.POINTER(ctypes.c_double)),
                ("evType", ctypes.POINTER(ctypes.c_int)),
                ("counts", ctypes.POINTER(ctypes.c_int)),
                ("maxRows", ctypes.c_int),
                ("columns", ctypes.c_int)]


# must match struct abEscape in UsableFunctions.cpp
class _Escape(ctypes.Structure):
    _fields_ = [("t", ctypes.POINTER(ctypes.c_double)),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("vx", ctypes.POINTER(ctypes.c_double)),
                ("vy", ctypes.POINTER(ctypes.c_double)),
                ("energy", ctypes.POINTER(ctypes.c_double)),
                ("basin", ctypes.POINTER(ctypes.c_int)),
                ("columns", ctypes.c_int)]


class Escape:
    """where each particle's run ended: one row per particle, all 1-D.

    No trajectory is kept, only the ending, whichever limit ended it -- there is
    no series here to slice, and .t is when each single row happened.

    `id` is the basin that absorbed it, or **0** for a particle still going when
    the limit ran out: a tf= run reports those at exactly tf, an iterations= run
    at the last event it took. For the rest, the state is the one AT the basin
    surface, with the velocity still pointing into it: the run stops on contact
    rather than responding to it.

    Unpacking yields the saved columns in the order asked for.
    """

    def __init__(self, cols, order):
        self._cols, self._order = cols, order
        for name, a in cols.items():
            setattr(self, "energy" if name == "Energy" else name, a)

    def __iter__(self):
        return iter(self._cols[k] for k in self._order)

    def tally(self):
        """-> {basin label: how many particles ended there}, 0 included"""
        if "id" not in self._cols:
            raise ValueError("this run did not save the 'id' column")
        label, n = np.unique(self._cols["id"], return_counts=True)
        return dict(zip(label.tolist(), n.tolist()))

    def __repr__(self):
        n = next(iter(self._cols.values())).size
        where = f", basins={self.tally()}" if "id" in self._cols else ""
        return f"<Escape {n} particles, columns={self._order}{where}>"


class Record:
    """the arrays a run produced, padded to (nPart, rows).

    Both kinds of run size their own buffer, so nothing is ever cut short for
    want of room -- but a particle can still stop early, by running out of
    things to hit. Only rows [0:counts[i]] of particle i are real; `trim(i)`
    does that slicing for you, and unpacking yields the saved columns in the
    order asked for.
    """

    def __init__(self, cols, counts, order):
        self._cols, self._order = cols, order
        self.counts = counts
        for name, a in cols.items():
            setattr(self, "energy" if name == "Energy" else name, a)

    def __iter__(self):
        return iter(self._cols[k] for k in self._order)

    def trim(self, i):
        """particle i's real rows, as a dict of 1-D arrays"""
        return {k: v[i, :self.counts[i]] for k, v in self._cols.items()}

    def __repr__(self):
        n, m = next(iter(self._cols.values())).shape
        return (f"<Record {n} particles x {m} rows, columns={self._order}, "
                f"counts={self.counts.tolist()}>")


class CompiledScene:
    def __init__(self, scene, backend="openmp"):
        self.backend = backend
        self.struct_hash, self.types = _scene.structure_hash(scene)

        header = codegen.generate(self.types)
        self.so_path, self.cached = _compile.build(header, backend)
        self.lib = ctypes.CDLL(str(self.so_path))
        self._declare()

        # asked before anything is run, because a GPU build that fell back to the
        # host answers every question correctly except how long it took
        self.onDevice = bool(self.lib.abOnDevice())
        if backend == "gpu_openmp" and not self.onDevice:
            raise _scene.SceneError(
                f"{backend} compiled and loaded, but its target region ran on "
                f"the HOST: the device image did not register when "
                f"{self.so_path.name} was opened, and OpenMP fell back silently. "
                f"The known trigger is another library reaching the process "
                f"first -- importing matplotlib.pyplot before compileScene does "
                f"it -- so importing it afterwards is the workaround. Use "
                f"backend='openmp' to ask for the CPU on purpose")

        if self.lib.abObjectCount() != len(self.types):
            raise RuntimeError(
                f"{self.so_path} holds {self.lib.abObjectCount()} objects but "
                f"this scene has {len(self.types)}: stale cache entry")
        self._push(scene)

    def _declare(self):
        L = self.lib
        d, i, P = ctypes.c_double, ctypes.c_int, ctypes.POINTER
        L.abObjectCount.restype = i
        L.abOnDevice.restype = i
        L.abSetConstants.argtypes = [d, d]
        L.abSetConstants.restype = i
        L.abBuildScene.argtypes = [P(d), P(i)]
        L.abBuildScene.restype = i
        L.abRunIterations.argtypes = [P(d)] * 4 + [i, i, i, P(_Output)]
        L.abRunIterations.restype = i
        L.abRunSampled.argtypes = [P(d)] * 4 + [i, d, i, P(_Output)]
        L.abRunSampled.restype = i
        L.abRunBasinsTime.argtypes = [P(d)] * 4 + [i, d, i, P(_Escape)]
        L.abRunBasinsTime.restype = i
        L.abRunBasinsIterations.argtypes = [P(d)] * 4 + [i, i, P(_Escape)]
        L.abRunBasinsIterations.restype = i

    def _push(self, scene):
        """parameters and constants into the running library. no compiler."""
        g, dead = _scene.constants(scene)
        if self.lib.abSetConstants(g, dead) != 0:
            raise _scene.SceneError(f"library rejected deadTime={dead}")
        self.g = g

        # kept so basins= can be a parameter push: replacing one key of the
        # scene needs the other keys, and this is the only copy of them
        self.scene = scene

        _, params, offsets = _scene.flatten(scene)
        p = np.ascontiguousarray(params, dtype=np.float64)
        o = np.ascontiguousarray(offsets, dtype=np.int32)
        rc = self.lib.abBuildScene(
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if rc != 0:
            k = rc - 1
            raise _scene.SceneError(
                f"object {k} ({self.types[k]}) is degenerate -- a zero-length "
                f"segment, a line with a == b == 0, or a zero ellipse radius. "
                f"it would build as NaN and read as an object that is never hit")

    def recordWithIterations(self, x, y, vx, vy, iterations, stride=1,
                             save=None, eventType=None):
        """the event table: one row per event, or per stride-th event.

        Row m is event m * stride and row 0 is always the initial state, so the
        run fills iterations // stride + 1 rows -- known before it starts, which
        is why there is nothing to say about how much room to give it.

        stride decides only what is written down. Every event in between is
        still simulated, so the trajectory is bit for bit the one stride=1
        walks; you are subsampling the record, not the physics.
        """
        iterations, stride = int(iterations), int(stride)
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        save = list(DEFAULT_SAVE if save is None else save)
        if eventType is not None and "evType" not in save:
            save.append("evType")  # needed to apply the filter

        rec = self._run(
            x, y, vx, vy, iterations // stride + 1, save, "recordWithIterations",
            lambda args, nPart, out: self.lib.abRunIterations(
                *args, nPart, iterations, stride, ctypes.byref(out)))
        return self._filter(rec, eventType) if eventType else rec

    def recordWithTime(self, x, y, vx, vy, tf, dt, save=None, maxEvents=10 ** 7):
        """the time series: one row per tick of the grid t = 0, dt, 2*dt, ...

        dt is not a step size and nothing is integrated. Between collisions the
        state is an exact parabola, and this evaluates it wherever you ask, so a
        smaller dt buys more samples of the very same trajectory and costs no
        accuracy -- there is none to lose. Rows are samples, not events, so
        there is no evType: use recordWithIterations for that table.

        Every particle fills the same floor(tf / dt) + 1 rows, so .t is one grid
        shared by all of them. maxEvents bounds the collision walk between two
        samples; a particle that hits the bound stops short, and .counts says so.
        """
        tf, dt = float(tf), float(dt)
        if not tf > 0.0:
            raise ValueError(f"tf must be > 0, got {tf}")
        if not 0.0 < dt <= tf:
            raise ValueError(f"dt must be > 0 and <= tf ({tf}), got {dt}")
        if tf >= MAGIC_NO_COLLISION:
            raise ValueError(
                f"tf must stay well under {MAGIC_NO_COLLISION:g}: that is the "
                f"value the solver reads as 'never hits anything'")

        save = list(DEFAULT_SAVE_SAMPLED if save is None else save)
        if "evType" in save:
            raise ValueError(
                "a grid row is a sample of a flight, not an event, so it has "
                "no evType. use recordWithIterations for the event table")

        return self._run(
            x, y, vx, vy, int(math.floor(tf / dt + 1e-9)) + 1, save,
            "recordWithTime",
            lambda args, nPart, out: self.lib.abRunSampled(
                *args, nPart, dt, int(maxEvents), ctypes.byref(out)))

    def getBasins(self, x, y, vx=None, vy=None, tf=None, iterations=None,
                  basins=None, save=None, maxEvents=None):
        """run each particle until a basin absorbs it, or until you give up on
        it. -> Escape

        Give up either at a time, tf=, or after a number of events,
        iterations= -- exactly one of the two, since they are the same run
        stopped by different clocks. The recorders are two functions because
        they build different tables; this is one because the table is the same
        either way: NOTHING along the way is kept, only where each run ended.

        One row per particle, then, so this costs the same memory whether a
        particle escaped at once or bounced a million times first -- which is
        what makes a 512x512 grid of initial conditions reasonable, and what
        separates iterations= here from recordWithIterations, which stores
        every event it counts.

        `id` is the basin that took it, or 0 for a particle still going when the
        limit ran out. An escaped particle's state is the one AT the basin
        surface, snapped onto it, with the velocity still pointing in: the run
        ends on contact instead of responding to it, so what comes back is the
        escape and not the bounce that would have followed.

        Where an id 0 particle is reported is the one thing the two limits
        disagree about: a tf= run coasts its last flight to exactly t = tf, an
        iterations= run stops at the last event it took and reports t there --
        there being no tf to coast to.

        maxEvents belongs to tf= alone: it bounds a run that would otherwise
        collide unboundedly often before tf, and a particle that hits it reports
        id 0 early, at its last event rather than at tf. An iterations= run is
        already that bound, so it takes no maxEvents.

        `basins=` pushes new numbers into the shapes the scene was compiled
        with, exactly as updateScene does -- so sweeping a basin's position
        never runs the compiler, and changing the shape *list* is refused.
        """
        if (tf is None) == (iterations is None):
            raise ValueError(
                "getBasins needs exactly one limit: tf=, the time to give up "
                "at, or iterations=, the number of events to give up after")

        byTime = tf is not None
        if byTime:
            tf = float(tf)
            if not tf > 0.0:
                raise ValueError(f"tf must be > 0, got {tf}")
            if tf >= MAGIC_NO_COLLISION:
                raise ValueError(
                    f"tf must stay well under {MAGIC_NO_COLLISION:g}: that is "
                    f"the value the solver reads as 'never hits anything'")
            maxEvents = 10 ** 7 if maxEvents is None else int(maxEvents)
            if maxEvents < 1:
                raise ValueError(f"maxEvents must be >= 1, got {maxEvents}: a "
                                 f"run allowed no event at all could only ever "
                                 f"report its initial state")
        else:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(f"iterations must be >= 0, got {iterations}")
            if maxEvents is not None:
                raise ValueError(
                    "maxEvents bounds how long a tf= run may collide for; an "
                    "iterations= run is that bound already, so pass one or the "
                    "other")

        if not any(t.startswith(_scene.BASIN) for t in self.types):
            raise _scene.SceneError(
                "this scene was compiled with no basinObjects, so there is "
                "nothing for a particle to escape through. add a 'basinObjects' "
                "key to the scene dict and compile it")

        if basins is not None:
            # the structure check and the parameter push, both already written
            updateScene({**self.scene, "basinObjects": basins}, self)

        save = list(DEFAULT_SAVE_ESCAPE if save is None else save)
        bad = [s for s in save if s not in ESCAPE_COLUMNS]
        if bad:
            raise ValueError(
                f"unknown column(s) {bad} for an escape run, expected from "
                f"{ESCAPE_COLUMNS}. a row here is where a run ended, not an "
                f"event, so there is no evType")

        x, y = (np.ascontiguousarray(a, dtype=np.float64).ravel() for a in (x, y))
        nPart = x.size
        vx = np.zeros(nPart) if vx is None else vx
        vy = np.zeros(nPart) if vy is None else vy
        vx, vy = (np.ascontiguousarray(a, dtype=np.float64).ravel()
                  for a in (vx, vy))
        if not (y.size == vx.size == vy.size == nPart):
            raise ValueError("x, y, vx, vy must be the same length")

        cols, bits, dummy = {}, 0, np.zeros(1, dtype=np.float64)
        idummy = np.zeros(1, dtype=np.int32)
        for name in ESCAPE_COLUMNS:
            if name in save:
                dt = np.int32 if name == "id" else np.float64
                cols[name] = np.zeros(nPart, dtype=dt)
                bits |= COLUMN_BIT[name]

        def dp(name):
            return cols.get(name, dummy).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        out = _Escape(
            dp("t"), dp("x"), dp("y"), dp("vx"), dp("vy"), dp("Energy"),
            cols.get("id", idummy).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            bits)

        args = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                for a in (x, y, vx, vy)]
        rc = (self.lib.abRunBasinsTime(*args, nPart, tf, maxEvents,
                                       ctypes.byref(out)) if byTime else
              self.lib.abRunBasinsIterations(*args, nPart, iterations,
                                             ctypes.byref(out)))
        if rc != 0:
            raise RuntimeError(f"run failed with code {rc}")

        return Escape(cols, save)

    def _run(self, x, y, vx, vy, rows, save, who, call):
        """allocate the columns, let `call` fill them, wrap them in a Record.

        Everything the two recorders share. They differ only in how many rows
        they work out they need, and which entry point they hand them to.
        """
        bad = [s for s in save if s not in TABLE_COLUMNS]
        if bad:
            raise ValueError(f"unknown column(s) {bad}, expected from "
                             f"{TABLE_COLUMNS}")

        x, y, vx, vy = (np.ascontiguousarray(a, dtype=np.float64).ravel()
                        for a in (x, y, vx, vy))
        nPart = x.size
        if not (y.size == vx.size == vy.size == nPart):
            raise ValueError("x, y, vx, vy must be the same length")

        self._warn_memory(who, nPart, rows, save)

        cols, bits, dummy = {}, 0, np.zeros(1, dtype=np.float64)
        idummy = np.zeros(1, dtype=np.int32)
        for name in TABLE_COLUMNS:
            if name in save:
                dt = np.int32 if name == "evType" else np.float64
                cols[name] = np.zeros((nPart, rows), dtype=dt)
                bits |= COLUMN_BIT[name]

        counts = np.zeros(nPart, dtype=np.int32)

        def dp(name):
            a = cols.get(name, dummy)
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        out = _Output(
            dp("t"), dp("x"), dp("y"), dp("vx"), dp("vy"), dp("Energy"),
            (cols.get("evType", idummy)).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            rows, bits)

        args = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                for a in (x, y, vx, vy)]
        rc = call(args, nPart, out)
        if rc != 0:
            raise RuntimeError(f"run failed with code {rc}")

        return Record(cols, counts, save)

    @staticmethod
    def _filter(rec, eventType):
        want = [EVENT_CODE[e] for e in eventType]
        keep = np.isin(rec.evType, want)
        for i in range(keep.shape[0]):
            keep[i, rec.counts[i]:] = False
        counts = keep.sum(axis=1).astype(np.int32)
        m = int(counts.max()) if counts.size else 0

        cols = {}
        for name, a in rec._cols.items():
            out = np.zeros((a.shape[0], m), dtype=a.dtype)
            for i in range(a.shape[0]):
                out[i, :counts[i]] = a[i][keep[i]]
            cols[name] = out
        return Record(cols, counts, rec._order)

    @staticmethod
    def _warn_memory(who, nPart, rows, save):
        per_row = sum(4 if s == "evType" else 8 for s in save)
        nbytes = nPart * rows * per_row
        if nbytes > WARN_BYTES:
            warnings.warn(
                f"{who} will allocate {nbytes / 1024**2:.0f} MB "
                f"({nbytes / 1024**3:.2f} GB): {nPart} particles x {rows} "
                f"rows x {len(save)} columns. narrow save=, raise stride= or "
                f"raise dt= to shrink it")


def compileScene(scene, backend="openmp"):
    return CompiledScene(scene, backend)


def updateScene(scene, compiled):
    """New parameters into an already-built library. Never compiles.

    Only the structure -- the slot types, in order -- is compared. Parameters,
    the portal flips, g and deadTime all go straight through.
    """
    h, types = _scene.structure_hash(scene)
    if h != compiled.struct_hash:
        old, new = compiled.types, types
        if len(old) != len(new):
            raise _scene.SceneError(
                f"scene has {len(new)} objects, the compiled one has {len(old)}: "
                f"objects cannot be added or removed without recompiling")
        k = next(i for i, (a, b) in enumerate(zip(old, new)) if a != b)
        raise _scene.SceneError(
            f"object {k} changed type from {old[k]!r} to {new[k]!r}: "
            f"that is structure, not a parameter, so it needs a recompile")
    compiled._push(scene)
    return compiled


def getBasins(compiled, x, y, vx=None, vy=None, tf=None, iterations=None,
              basins=None, save=None, maxEvents=None):
    """run each particle until a basin absorbs it, or until tf= / iterations=
    runs out. -> Escape

    The module-level form of CompiledScene.getBasins, matching updateScene's
    call shape. See that method for the two limits and what the columns mean.
    """
    return compiled.getBasins(x, y, vx, vy, tf=tf, iterations=iterations,
                              basins=basins, save=save, maxEvents=maxEvents)
