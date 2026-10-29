"""Every benchmark sweep, on whatever machine this runs on.

Edit PARAMS below, or override any of it from the command line. Results land in
benchmarks/results/<hostname>/, so the same command on another machine never
mixes with this one's:

    machine.json    what the machine is: CPU, cores, RAM, GPU, compilers, commit
    compile.csv     one cold build per (backend, objType, N)
    scaling.csv     one timed run per (scene, backend, threads, Np)
    logs/           each point's full DEBUG log
    failures.log    what failed or timed out; the sweep carries on past it

Two scaling passes: throughput, every backend x scene x Np with openmp on every
logical core; and strong scaling, openmp x scene x threads at the largest Np.

Before anything runs the memory each run needs is estimated -- every event is
recorded, so it is Np * (iter + 1) * 32 B on the host, and the same again on a
GPU -- and a point over memFraction of what is available is refused unless
--force. Points already in a CSV are skipped, so a sweep that was stopped picks
up where it left off; --fresh starts over.

    python benchmarks/runBenchmarks.py --dry-run          # what would run
    python benchmarks/runBenchmarks.py --plot             # everything
    python benchmarks/runBenchmarks.py --only scaling --Np 1000 100000 --iter 50
"""

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import benchCommon as bc

from analyticBilliards import compile as abCompile

HERE = pathlib.Path(__file__).resolve().parent

PARAMS = {
    "backends": None,  # None: every backend this machine can build
    # compile sweep
    "compileTypes": ["line", "lineSegment", "ellipse", "ellipseArc", "portal",
                     "rand"],
    "compileN": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    "compileRepeat": 1,
    # scaling sweep
    "scenes": ["Ellipse", "ElipseArc", "LineSeg", "Line", "Portal"],
    "Nobj": 10,  # even, for Portal
    "iter": 100,
    "Np": [100, 1000, 10000, 100000],
    "repeat": 3,
    "threads": None,  # None: 1, 2, 4, 8, the physical and the logical cores
    # limits
    "timeout": 1800,      # s, per compile point and per scaling group
    "capAfterTimeout": 5000,  # once a compile point times out, skip N above this
    "memFraction": 0.8,   # refuse a run needing more than this of what is free
}


# ------------------------------------------------------------------- machine

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def physicalCores():
    try:
        pairs, phys = set(), None
        for line in open("/proc/cpuinfo"):
            k, _, v = line.partition(":")
            if k.strip() == "physical id":
                phys = v.strip()
            elif k.strip() == "core id":
                pairs.add((phys, v.strip()))
        return len(pairs) or os.cpu_count()
    except OSError:
        return os.cpu_count()


def cpuModel():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def totalRAM():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None


def gpuInfo():
    """-> [{name, memory_MiB, free_MiB, driver}] or [] without a usable GPU"""
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
               "driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        name, total, free, driver = [s.strip() for s in line.split(",")]
        gpus.append({"name": name, "memory_MiB": int(total),
                     "free_MiB": int(free), "driver": driver})
    return gpus


def availableBackends():
    cxx = os.environ.get("AB_CXX", "g++")
    out = ["linear", "openmp"] if shutil.which(cxx) else []
    if pathlib.Path(abCompile.NVCXX).exists() and gpuInfo():
        out.append("gpu_openmp")
    return out


def machineInfo():
    first = lambda s: s.splitlines()[0] if s else None
    nvcxx = run([abCompile.NVCXX, "--version"])
    return {
        "host": bc.host(), "cpu": cpuModel(), "physicalCores": physicalCores(),
        "logicalCores": os.cpu_count(), "RAM_GiB": round((totalRAM() or 0) / 2 ** 30, 1),
        "gpus": gpuInfo(), "os": platform.platform(), "python": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "cxx": first(run([os.environ.get("AB_CXX", "g++"), "--version"])),
        "nvcxx": nvcxx.strip().splitlines()[0] if nvcxx else None,
        "gpuArch": os.environ.get("AB_GPUARCH"),
        "commit": run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"]),
        "dirty": bool(run(["git", "-C", str(HERE), "status", "--porcelain",
                           "--", "../src"])),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "AB_"))},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# -------------------------------------------------------------------- points

def done(path, keys):
    """the key tuples already in a CSV"""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {tuple(str(r.get(k, "")) for k in keys) for r in bc.readCSV(path)}


def rowCount(path):
    return len(bc.readCSV(path)) if path.exists() and path.stat().st_size else 0


class Timeout(Exception):
    pass


def launch(cmd, env, log, timeout, failures):
    """run one point; -> True if it finished. Its output goes to the log.
    Raises Timeout if it ran past `timeout`"""
    t0 = time.perf_counter()
    try:
        with open(log, "w") as f:
            r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=f,
                               text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        note(failures, f"TIMEOUT after {timeout} s: {' '.join(cmd)}")
        raise Timeout
    print(r.stdout.rstrip("\n"))
    print(f"  [{time.perf_counter() - t0:.1f} s]")
    if r.returncode != 0:
        note(failures, f"rc={r.returncode}: {' '.join(cmd)} (see {log})")
    return r.returncode == 0


def note(failures, msg):
    print("  !!", msg)
    with open(failures, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def envFor(backend, threads=None):
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    if backend == "openmp":
        env["OMP_NUM_THREADS"] = str(threads)
    return env


def compileSweep(P, out, failures):
    csv = out / "compile.csv"
    have = set() if P["fresh"] else done(csv, ["backend", "objType", "N"])
    cap = None  # set by the first timeout: past it, every big N costs the same
    for backend in P["backends"]:
        for objType in P["compileTypes"]:
            for n in P["compileN"]:
                if (backend, objType, str(n)) in have:
                    continue
                if cap is not None and n > cap:
                    break
                print(f"\n== compile {backend} {objType} N={n}")
                try:
                    ok = launch([sys.executable, str(HERE / "compile.py"),
                                 "--backend", backend, "--objType", objType,
                                 "--N", str(n), "--repeat",
                                 str(P["compileRepeat"]), "--csv", str(csv)],
                                envFor(backend, os.cpu_count()),
                                out / "logs" / f"compile_{backend}_{objType}_{n}.log",
                                P["timeout"], failures)
                except Timeout:
                    ok = False
                    if cap is None and P["capAfterTimeout"]:
                        cap = P["capAfterTimeout"]
                        note(failures, f"capping every later compile point "
                                       f"at N <= {cap}")
                if not ok:  # a larger N only takes longer
                    note(failures, f"skipping {backend} {objType} N > {n}")
                    break


def scalingGroups(P):
    """-> [(scene, backend, threads, [Np])]: throughput pass, then threads pass"""
    logical, top = os.cpu_count(), max(P["Np"])
    groups = []
    for backend in P["backends"]:
        t = logical if backend == "openmp" else bc.threadsFor(backend)
        groups += [(s, backend, t, list(P["Np"])) for s in P["scenes"]]
    if "openmp" in P["backends"]:
        groups += [(s, "openmp", t, [top]) for s in P["scenes"]
                   for t in P["threads"] if t != logical]
    return groups


def scalingSweep(P, out, failures, allowed):
    csv = out / "scaling.csv"
    keys = ["scene", "backend", "threads", "Np", "Nobj", "iter"]
    have = set() if P["fresh"] else done(csv, keys)
    for scene, backend, threads, nps in scalingGroups(P):
        nps = [n for n in nps if allowed(backend, n)
               and (scene, backend, str(threads), str(n), str(P["Nobj"]),
                    str(P["iter"])) not in have]
        if not nps:
            continue
        print(f"\n== {scene} {backend} threads={threads} Np={nps}")
        before = rowCount(csv)
        try:
            ok = launch([sys.executable, str(HERE / "scene.py"),
                         "--scene", scene, "--backend", backend, "--Np", *map(str, nps),
                         "--Nobj", str(P["Nobj"]), "--iter", str(P["iter"]),
                         "--repeat", str(P["repeat"]), "--csv", str(csv)],
                        envFor(backend, threads),
                        out / "logs" / f"{scene}_{backend}_{threads}.log",
                        P["timeout"], failures)
        except Timeout:
            continue  # the rows it finished are in the CSV; the rest are not
        got = rowCount(csv) - before
        if ok and got != len(nps):
            note(failures, f"{scene} {backend} threads={threads}: expected "
                           f"{len(nps)} rows, got {got}")


# -------------------------------------------------------------------- memory

def memoryCheck(P, force, save=bc.SAVE):
    """print the estimate for recording `save`; -> allowed(backend, Np)"""
    free = bc.availableRAM()
    gpus = gpuInfo() if "gpu_openmp" in P["backends"] else []
    gpuFree = gpus[0]["free_MiB"] * 2 ** 20 if gpus else None
    frac = P["memFraction"]
    mib = lambda b: f"{b / 2 ** 20:,.1f} MiB"

    perRow = sum(4 if c in ("evType", "id") else 8 for c in save)
    print(f"\nmemory per run, every event recorded: Np x (iter + 1) x "
          f"{perRow} B ({', '.join(save)}), iter={P['iter']}")
    print(f"  host RAM available: {mib(free) if free else 'unknown'}"
          + (f", GPU free: {mib(gpuFree)}" if gpuFree else ""))
    refused = set()
    for n in sorted(P["Np"]):
        need = bc.recordBytes(n, P["iter"], save)
        bad = [w for w, lim in (("host", free), ("GPU", gpuFree))
               if lim and need > frac * lim and (w == "host" or gpus)]
        print(f"  Np={n:>9,}: {mib(need):>12}" + (f"   > {frac:.0%} of "
              f"{' and '.join(bad)} free" if bad else ""))
        if "host" in bad:
            refused.add(("*", n))
        elif "GPU" in bad:
            refused.add(("gpu_openmp", n))
    if refused and not force:
        print("  refusing those points; --force runs them anyway")

    def allowed(backend, n):
        return force or not ({("*", n), (backend, n)} & refused)
    return allowed


# ---------------------------------------------------------------------- main

def options():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    D = PARAMS
    p.add_argument("--backends", nargs="+", default=D["backends"],
                   choices=sorted(abCompile.BACKENDS))
    p.add_argument("--compile-types", dest="compileTypes", nargs="+",
                   default=D["compileTypes"])
    p.add_argument("--compile-N", dest="compileN", type=int, nargs="+",
                   default=D["compileN"])
    p.add_argument("--compile-repeat", dest="compileRepeat", type=int,
                   default=D["compileRepeat"])
    p.add_argument("--scenes", nargs="+", default=D["scenes"])
    p.add_argument("--Nobj", type=int, default=D["Nobj"])
    p.add_argument("--iter", type=int, default=D["iter"])
    p.add_argument("--Np", type=int, nargs="+", default=D["Np"])
    p.add_argument("--repeat", type=int, default=D["repeat"])
    p.add_argument("--threads", type=int, nargs="+", default=D["threads"])
    p.add_argument("--timeout", type=float, default=D["timeout"])
    p.add_argument("--cap-after-timeout", dest="capAfterTimeout", type=int,
                   default=D["capAfterTimeout"],
                   help="after the first compile timeout, skip N above this "
                        "(0: never cap)")
    p.add_argument("--mem-fraction", dest="memFraction", type=float,
                   default=D["memFraction"])
    p.add_argument("--results", type=pathlib.Path,
                   default=HERE / "results" / bc.host())
    p.add_argument("--only", choices=["compile", "scaling"])
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the memory estimate, run nothing")
    p.add_argument("--force", action="store_true",
                   help="run points over the memory limit")
    p.add_argument("--fresh", action="store_true",
                   help="rerun points already in the CSVs")
    p.add_argument("--plot", action="store_true",
                   help="draw the figures and the table at the end")
    return p.parse_args()


def main():
    P = vars(options())
    P["backends"] = P["backends"] or availableBackends()
    if not P["threads"]:
        logical = os.cpu_count()
        P["threads"] = sorted({t for t in (1, 2, 4, 8, physicalCores(), logical)
                               if t <= logical})
    if "Portal" in P["scenes"] and P["Nobj"] % 2:
        raise SystemExit("--Nobj must be even while Portal is in --scenes")

    out = P["results"]
    doCompile, doScaling = P["only"] in (None, "compile"), P["only"] in (None, "scaling")
    nCompile = len(P["backends"]) * len(P["compileTypes"]) * len(P["compileN"])
    groups = scalingGroups(P)
    nScaling = sum(len(g[3]) for g in groups)

    print(f"host {bc.host()}: backends {P['backends']}, threads {P['threads']}")
    print(f"results -> {out}")
    if doCompile:
        print(f"compile: {nCompile} points, N up to {max(P['compileN'])}")
    if doScaling:
        print(f"scaling: {nScaling} points in {len(groups)} runs, "
              f"Nobj={P['Nobj']}, iter={P['iter']}, repeat={P['repeat']}")
    allowed = memoryCheck(P, P["force"]) if doScaling else None
    if P["dry_run"]:
        return

    (out / "logs").mkdir(parents=True, exist_ok=True)
    info = machineInfo()
    info["params"] = {k: P[k] for k in PARAMS}
    (out / "machine.json").write_text(json.dumps(info, indent=2) + "\n")
    failures = out / "failures.log"

    t0 = time.perf_counter()
    if doCompile:
        compileSweep(P, out, failures)
    if doScaling:
        scalingSweep(P, out, failures, allowed)
    print(f"\nsweeps done in {(time.perf_counter() - t0) / 60:.1f} min")

    if P["plot"]:
        for script in ("plotCompile.py", "tableScaling.py", "plotScaling.py"):
            subprocess.run([sys.executable, str(HERE / script),
                            "--results", str(out)])


if __name__ == "__main__":
    main()
