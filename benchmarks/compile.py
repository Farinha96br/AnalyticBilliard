"""Compile time of a scene of N objects of one type, g = 1.0.

Each point is a cold build: the cache is pointed at a fresh directory, so the
compiler always runs. Three stages are timed apart --

    codegen   scene structure -> generatedScene.h
    build     g++ (or nvc++) on it
    load      compileScene against the now-warm cache: dlopen and the push

--objType rand draws each object's type at random from the five, portals
included; portals go to portalObjects, the rest to solidObjects.

    python benchmarks/compile.py --objType rand portal --N 10 100 1000
"""

import argparse
import pathlib
import statistics
import tempfile
import time

import benchCommon as bc  # sets AB_LOG and AB_GPUARCH before the library loads

import numpy as np

from analyticBilliards import (Ellipse, EllipseArc, Line, LineSegment, Portal,
                               Scene, codegen, compileScene)
from analyticBilliards import compile as abCompile
from analyticBilliards.compile import BACKENDS
from analyticBilliards.scene import structure_hash

TYPES = ["line", "lineSegment", "ellipse", "ellipseArc", "portal"]
COLUMNS = ["objType", "N", "codegen_s", "build_s", "load_s", "so_KiB", "ok"]
CSV_COLUMNS = ["host", "backend", "objType", "N", "codegen_s", "build_s",
               "build_min_s", "load_s", "so_KiB", "ok"]


def randomSegment(rng):
    x0, y0 = rng.uniform(-5, 5, 2)
    return LineSegment(x0, y0, x0 + rng.uniform(0.1, 2), y0 + rng.uniform(0.1, 2))


def randomObject(kind, rng):
    """any valid object: the numbers never reach the compiler"""
    if kind == "line":
        return Line(*rng.normal(size=2), rng.uniform(-5, 5))
    if kind == "lineSegment":
        return randomSegment(rng)
    if kind == "portal":
        return Portal(randomSegment(rng), randomSegment(rng))
    cx, cy = rng.uniform(-5, 5, 2)
    a, b = rng.uniform(0.5, 2, 2)
    tilt = rng.uniform(0, np.pi)
    if kind == "ellipse":
        return Ellipse(cx, cy, a, b, tilt)
    return EllipseArc(cx, cy, a, b, tilt, *rng.uniform(0, 2 * np.pi, 2))


def makeScene(objType, n, rng):
    kinds = rng.choice(TYPES, n) if objType == "rand" else [objType] * n
    objs = [randomObject(k, rng) for k in kinds]
    return Scene(g=1.0,
                 solidObjects=[o for o in objs if not isinstance(o, Portal)],
                 portalObjects=[o for o in objs if isinstance(o, Portal)])


def coldBuild(scene, backend):
    """-> (codegen_s, build_s, load_s, so KiB, ok), in a cache of its own"""
    with tempfile.TemporaryDirectory() as tmp:
        abCompile.CACHE = pathlib.Path(tmp)

        t0 = time.perf_counter()
        types = structure_hash(scene)[1]
        header = codegen.generate(types)
        tGen = time.perf_counter() - t0

        t0 = time.perf_counter()
        so, cached = abCompile.build(header, backend)
        tBuild = time.perf_counter() - t0

        t0 = time.perf_counter()
        billiard = compileScene(scene, backend=backend)
        tLoad = time.perf_counter() - t0

        ok = (not cached and billiard.cached and billiard.so_path == so
              and billiard.lib.abObjectCount() == len(types))
        return tGen, tBuild, tLoad, so.stat().st_size / 1024, ok


def bench(objType, n, backend, repeat, rng):
    scene = makeScene(objType, n, rng)
    runs = [coldBuild(scene, backend) for _ in range(repeat)]
    gen, build, load, size, ok = zip(*runs)
    return {"host": bc.host(), "backend": backend, "objType": objType, "N": n,
            "codegen_s": statistics.median(gen),
            "build_s": statistics.median(build), "build_min_s": min(build),
            "load_s": statistics.median(load), "so_KiB": size[-1],
            "ok": all(ok)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objType", nargs="+", default=["rand"],
                   choices=TYPES + ["rand"])
    p.add_argument("--N", type=int, nargs="+", default=[10])
    p.add_argument("--backend", default="openmp", choices=sorted(BACKENDS))
    p.add_argument("--repeat", type=int, default=1,
                   help="cold builds per point; the median is reported")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", help="append one row per point to this file")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cache = abCompile.CACHE
    rows = []
    try:
        for t in args.objType:
            for n in args.N:
                rows.append(bench(t, n, args.backend, args.repeat, rng))
                if args.csv:
                    bc.appendCSV(args.csv, rows[-1:], CSV_COLUMNS)
    finally:
        abCompile.CACHE = cache
    bc.printTable([[bc.fmt(r[c]) for c in COLUMNS] for r in rows], COLUMNS)
