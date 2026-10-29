"""What the benchmarks share: the options, the timed loop, the CSV and the table.

Each run scene keeps every particle confined forever, so a run always takes all
`iter` events -- and stepScene still asks every object for its hit time at each
one. The cost of an event therefore grows with Nobj, and ns/(ev*obj) is the
price of a single root solve.

Every event is recorded (x, y, vx, vy), as an ordinary run would. Energy is
recomputed here from those columns, measured from a floor below which no
particle can be, so that it is positive and the relative drift means something.
"""

import argparse
import csv
import os
import socket
import statistics
import time

os.environ.setdefault("AB_LOG", "DEBUG")          # the library logs to stderr
os.environ.setdefault("AB_GPUARCH", "ccnative")   # build for this machine's GPU

import numpy as np

from analyticBilliards import compileScene
from analyticBilliards.binding import _availableRAM, _estimate
from analyticBilliards.compile import BACKENDS

G = 1.0
SAVE = ["x", "y", "vx", "vy"]
BLOCK = 4096  # particles per block when checking, so the checks cost no memory

# Nobj is what was asked for; objects is what the scene holds (Portal adds a
# floor), and it is what ns_ev_obj divides by
COLUMNS = ["Np", "Nobj", "objects", "iter", "events", "build_s", "cached",
           "run_s", "Mev_s", "ns_ev_obj", "drift", "ok"]
CSV_COLUMNS = ["host", "scene", "backend", "threads", "Np", "Nobj", "objects",
               "iter", "events", "build_s", "cached", "run_s", "run_min_s",
               "Mev_s", "ns_ev_obj", "drift", "ok", "mem_MiB"]


def host():
    return socket.gethostname()


def recordBytes(nP, it, save=SAVE):
    """what one run allocates, as the library itself estimates it"""
    return _estimate(nP, it + 1, save)


def availableRAM():
    return _availableRAM()


def threadsFor(backend):
    """recorded, not set: OMP_NUM_THREADS decides, before the library loads"""
    if backend == "linear":
        return 1
    if backend == "gpu_openmp":
        return 0
    return int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))


def parser(doc):
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Np", type=int, nargs="+", default=[1000],
                   help="particles; several values sweep them")
    p.add_argument("--Nobj", type=int, nargs="+", default=[10],
                   help="objects; several values sweep them, one build each")
    p.add_argument("--iter", type=int, default=1000, help="events per particle")
    p.add_argument("--backend", default="openmp", choices=sorted(BACKENDS))
    p.add_argument("--repeat", type=int, default=3,
                   help="timed runs per point; the median is reported")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", help="append one row per point to this file")
    return p


def check(rec, it, floor, confined):
    """-> (every particle took every event, max |dE/E|, all rows confined)"""
    full = bool((rec.counts == it + 1).all())
    drift, inside = 0.0, True
    for b in range(0, rec.counts.size, BLOCK):
        s = slice(b, b + BLOCK)
        x, y, vx, vy = rec.x[s], rec.y[s], rec.vx[s], rec.vy[s]
        E = 0.5 * (vx * vx + vy * vy) + G * (y - floor)
        drift = max(drift, float(np.max(np.abs(E - E[:, :1]) / E[:, :1])))
        inside &= bool(confined(x, y).all())
    return full, drift, inside


def measure(billiard, nP, it, repeat, makeICs, confined, floor, rng):
    """one timed point -> dict of the table's columns"""
    x0, y0, vx0, vy0 = makeICs(nP, rng)
    billiard.recordWithIterations(x0, y0, vx0, vy0, 1)  # warm-up

    times = []
    for _ in range(repeat):
        rec = None  # let the last record go before the next is allocated
        t0 = time.perf_counter()
        rec = billiard.recordWithIterations(x0, y0, vx0, vy0, it, save=SAVE)
        times.append(time.perf_counter() - t0)
    run = statistics.median(times)

    full, drift, inside = check(rec, it, floor, confined)
    nObj, events = len(billiard.types), nP * it
    return {"Np": nP, "objects": nObj, "iter": it, "events": events,
            "run_s": run, "run_min_s": min(times), "Mev_s": events / run / 1e6,
            "ns_ev_obj": run / (events * nObj) * 1e9, "drift": drift,
            "ok": full and inside and drift < 1e-8,
            "mem_MiB": recordBytes(nP, it) / 2 ** 20}


def runBench(scene, makeScene, makeICs, confined, floor, args):
    """makeScene(nObj) -> Scene; makeICs(n, rng) -> (x, y, vx, vy);
    confined(x, y) -> bool array; floor: the lowest height reachable"""
    rng = np.random.default_rng(args.seed)
    rows = []
    for nObj in args.Nobj:
        t0 = time.perf_counter()
        billiard = compileScene(makeScene(nObj), backend=args.backend)
        build = time.perf_counter() - t0
        for nP in args.Np:
            r = measure(billiard, nP, args.iter, args.repeat, makeICs,
                        confined, floor, rng)
            r.update(host=host(), scene=scene, backend=args.backend, Nobj=nObj,
                     threads=threadsFor(args.backend), build_s=build,
                     cached=billiard.cached)
            rows.append(r)
            if args.csv:
                appendCSV(args.csv, [r], CSV_COLUMNS)
    printTable([[fmt(r[c]) for c in COLUMNS] for r in rows], COLUMNS)
    return rows


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-3 and v else f"{v:.4g}"
    return str(v)


def appendCSV(path, rows, cols):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in rows)


def readCSV(path):
    """-> list of dicts, numbers as numbers"""
    def conv(v):
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return {"True": True, "False": False}.get(v, v)
    with open(path, newline="") as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def printTable(rows, cols):
    cells = [cols] + [[str(c) for c in r] for r in rows]
    width = [max(len(r[j]) for r in cells) for j in range(len(cols))]
    line = lambda r: " | ".join(c.rjust(w) for c, w in zip(r, width))
    print()
    print(line(cells[0]))
    print("-+-".join("-" * w for w in width))
    for r in cells[1:]:
        print(line(r))
