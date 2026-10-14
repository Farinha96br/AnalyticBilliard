"""Build UsableFunctions.cpp into a .so, cached by what actually affects it.

Flags are lifted verbatim from cpp/portals/checkHashPortal.sh -- they are the
ones already proven to hold the three backends bitwise identical.
"""

import hashlib
import logging
import os
import pathlib
import subprocess
import time

log = logging.getLogger(__name__)

SRC = pathlib.Path(__file__).resolve().parent.parent  # src/
CPP = SRC / "cpp"           # every .h and .cpp lives here
CACHE = SRC.parent / ".abcache"  # at the repo root, next to src/

NVCXX = os.environ.get(
    "AB_NVCXX", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvc++")
GPUARCH = os.environ.get("AB_GPUARCH", "cc89")

def _gompRpath(cxx):
    """pin the .so to the libgomp belonging to the compiler that built it.

    An OpenMP build needs a libgomp at least as new as the compiler's. A conda
    or venv interpreter often ships an older one of its own, and since it is the
    host process it wins the search: the library then fails to load with a
    "version GOMP_x.y not found" that has nothing to do with any code here, and
    only for the backend that is the default. --disable-new-dtags makes this a
    DT_RPATH, which is searched before LD_LIBRARY_PATH, so the match is settled
    at build time rather than left to whoever imports it.

    -> [] if the compiler cannot say, in which case nothing is pinned and the
    behaviour is exactly what it was before.
    """
    try:
        r = subprocess.run([cxx, "-print-file-name=libgomp.so.1"],
                           capture_output=True, text=True)
    except OSError:
        return []
    if r.returncode != 0:
        return []
    lib = pathlib.Path(r.stdout.strip())
    if not lib.is_absolute() or not lib.exists():
        return [] # gcc echoes the bare name back when it has no idea
    return [f"-Wl,--disable-new-dtags,-rpath,{lib.resolve().parent}"]


# -ffp-contract=off / -Kieee -Mnofma are what keep the three outputs bit for bit
# identical: no FMA fusion, and no non-IEEE division or sqrt
BACKENDS = {
    "linear":     (lambda: [os.environ.get("AB_CXX", "g++"), "-O2", "-ffp-contract=off",
                            "-fno-gnu-unique"]),
    "openmp":     (lambda: [os.environ.get("AB_CXX", "g++"), "-O2", "-ffp-contract=off",
                            "-fno-gnu-unique", "-fopenmp", "-foffload=disable"]
                           + _gompRpath(os.environ.get("AB_CXX", "g++"))),

    "gpu_openmp": (lambda: [NVCXX, "-O2", "-Kieee", "-Mnofma",
                            "-mp=gpu", f"-gpu={GPUARCH}", "-Minfo=mp",
                            "-static-nvidia"]),
}

LINK_LAST = {}


class BuildError(RuntimeError):
    pass


def build(header_src, backend):
    """-> (so_path, was_cached). header_src is the generatedScene.h text."""
    if backend not in BACKENDS:
        raise BuildError(f"unknown backend {backend!r}, "
                         f"expected one of {sorted(BACKENDS)}")
    flags = BACKENDS[backend]()
    link = LINK_LAST.get(backend, list)()

    ubody = (CPP / "UsableFunctions.cpp").read_bytes()
    key = hashlib.sha1(header_src.encode() + ubody
                       + repr(flags).encode() + repr(link).encode()).hexdigest()[:16]

    CACHE.mkdir(exist_ok=True)
    workdir = CACHE / f"{backend}_{key}"
    so = workdir / "libab.so"
    if so.exists():
        log.info("cached %s build", backend)
        log.debug("so=%s", so)
        return so, True

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "generatedScene.h").write_text(header_src)

    cmd = (flags + ["-shared", "-fPIC", f"-I{CPP}", f"-I{workdir}",
                    str(CPP / "UsableFunctions.cpp")]
           + link + ["-o", str(so)])
    log.debug("cmd: %s", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr.strip():
        log.debug("stderr: %s", r.stderr.strip())
    if r.returncode != 0:
        raise BuildError("compile failed:\n  " + " ".join(cmd) + "\n" + r.stderr)

    # a GPU build with no device kernel is the dangerous case: nvc++ produces a
    # perfectly good host binary, OMP_TARGET_OFFLOAD=MANDATORY has no target
    # region to enforce, and the library runs on the CPU reporting success
    if backend == "gpu_openmp" and "nvkernel" not in r.stderr:
        raise BuildError("gpu_openmp built no device kernel -- the target region "
                         "did not compile for the GPU, so this backend would "
                         "silently run on the host:\n" + r.stderr)
    log.info("compiled %s in %.2f s", backend, time.perf_counter() - t0)
    log.debug("so=%s", so)
    return so, False
