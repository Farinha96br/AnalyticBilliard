"""ctypes binding: CompiledScene, updateScene, recordScene."""

import ctypes
import warnings

import numpy as np

from . import codegen, compile as _compile, scene as _scene

# must match enum abColumn in UsableFunctions.cpp
COLUMN_BIT = {"t": 1 << 0, "x": 1 << 1, "y": 1 << 2, "vx": 1 << 3,
              "vy": 1 << 4, "Energy": 1 << 5, "evType": 1 << 6}
COLUMNS = list(COLUMN_BIT)
DEFAULT_SAVE = ["t", "x", "y", "vx", "vy", "evType"]

EV_INITIAL, EV_BOUNCE, EV_PORTAL = 0, 1, 2
EVENT_CODE = {"initial": EV_INITIAL, "bounce": EV_BOUNCE, "portal": EV_PORTAL}

WARN_BYTES = 256 * 1024 ** 2


class _Output(ctypes.Structure):
    _fields_ = [("t", ctypes.POINTER(ctypes.c_double)),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("vx", ctypes.POINTER(ctypes.c_double)),
                ("vy", ctypes.POINTER(ctypes.c_double)),
                ("energy", ctypes.POINTER(ctypes.c_double)),
                ("evType", ctypes.POINTER(ctypes.c_int)),
                ("counts", ctypes.POINTER(ctypes.c_int)),
                ("truncated", ctypes.POINTER(ctypes.c_int)),
                ("maxRows", ctypes.c_int),
                ("columns", ctypes.c_int)]


class Record:
    """the arrays a run produced, padded to (nPart, maxRows).

    Only rows [0:counts[i]] of particle i are real. `trim(i)` does that slicing
    for you; unpacking yields the saved columns in the order asked for.
    """

    def __init__(self, cols, counts, truncated, order):
        self._cols, self._order = cols, order
        self.counts, self.truncated = counts, truncated
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

        if self.lib.abObjectCount() != len(self.types):
            raise RuntimeError(
                f"{self.so_path} holds {self.lib.abObjectCount()} objects but "
                f"this scene has {len(self.types)}: stale cache entry")
        self._push(scene)

    def _declare(self):
        L = self.lib
        d, i, P = ctypes.c_double, ctypes.c_int, ctypes.POINTER
        L.abObjectCount.restype = i
        L.abSetConstants.argtypes = [d, d]
        L.abSetConstants.restype = i
        L.abBuildScene.argtypes = [P(d), P(i)]
        L.abBuildScene.restype = i
        for fn, cap in ((L.abRunIterations, i), (L.abRunTime, d)):
            fn.argtypes = [P(d), P(d), P(d), P(d), i, cap, P(_Output)]
            fn.restype = i

    def _push(self, scene):
        """parameters and constants into the running library. no compiler."""
        g, dead = _scene.constants(scene)
        if self.lib.abSetConstants(g, dead) != 0:
            raise _scene.SceneError(f"library rejected deadTime={dead}")
        self.g = g

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

    def recordScene(self, x, y, vx, vy, iterations=None, tf=None,
                    maxRows=None, save=None, eventType=None):
        if (iterations is None) == (tf is None):
            raise ValueError("give exactly one of iterations= or tf= "
                             "(they are mutually exclusive)")

        save = list(DEFAULT_SAVE if save is None else save)
        bad = [s for s in save if s not in COLUMN_BIT]
        if bad:
            raise ValueError(f"unknown column(s) {bad}, expected from {COLUMNS}")
        if eventType is not None and "evType" not in save:
            save.append("evType")  # needed to apply the filter

        x, y, vx, vy = (np.ascontiguousarray(a, dtype=np.float64).ravel()
                        for a in (x, y, vx, vy))
        nPart = x.size
        if not (y.size == vx.size == vy.size == nPart):
            raise ValueError("x, y, vx, vy must be the same length")

        if maxRows is None:
            if iterations is None:
                raise ValueError("tf= runs have no a-priori row count: "
                                 "pass maxRows= (and check .truncated)")
            maxRows = iterations + 1  # one row per event, plus the initial state

        self._warn_memory(nPart, maxRows, save)

        cols, bits, dummy = {}, 0, np.zeros(1, dtype=np.float64)
        idummy = np.zeros(1, dtype=np.int32)
        for name in COLUMNS:
            if name in save:
                dt = np.int32 if name == "evType" else np.float64
                cols[name] = np.zeros((nPart, maxRows), dtype=dt)
                bits |= COLUMN_BIT[name]

        counts = np.zeros(nPart, dtype=np.int32)
        trunc = np.zeros(nPart, dtype=np.int32)

        def dp(name):
            a = cols.get(name, dummy)
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        out = _Output(
            dp("t"), dp("x"), dp("y"), dp("vx"), dp("vy"), dp("Energy"),
            (cols.get("evType", idummy)).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            trunc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            maxRows, bits)

        args = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                for a in (x, y, vx, vy)]
        if iterations is not None:
            rc = self.lib.abRunIterations(*args, nPart, int(iterations), ctypes.byref(out))
        else:
            rc = self.lib.abRunTime(*args, nPart, float(tf), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"run failed with code {rc}")

        if trunc.any():
            warnings.warn(f"particles {np.flatnonzero(trunc).tolist()} filled "
                          f"maxRows={maxRows} before finishing; their records "
                          f"are cut short, not complete")

        rec = Record(cols, counts, trunc, save)
        return self._filter(rec, eventType) if eventType else rec

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
        return Record(cols, counts, rec.truncated, rec._order)

    @staticmethod
    def _warn_memory(nPart, maxRows, save):
        per_row = sum(4 if s == "evType" else 8 for s in save)
        nbytes = nPart * maxRows * per_row
        if nbytes > WARN_BYTES:
            warnings.warn(
                f"recordScene will allocate {nbytes / 1024**2:.0f} MB "
                f"({nbytes / 1024**3:.2f} GB): {nPart} particles x {maxRows} "
                f"rows x {len(save)} columns. narrow save= to shrink it")


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
