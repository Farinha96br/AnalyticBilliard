# AnalyticBilliard

Analystic Billiar (AB) is a fast semi-symbolic solver for gravitational (or straight) billiards that solve collisions iteratively.

## Tests

Run `pytest tests/` from the repo root; it needs only `numpy`, `pytest` and a
`g++`. The suite compiles its own scenes, so the first run spends a second or
two building and later runs come straight out of `.abcache/`. It uses the
`linear` backend by default — pass `--backend openmp`, or `gpu_openmp` given a
device, to run the same checks against another one. Note that the build cache
does not notice edits to the headers in `cpp/`, so clear it with
`rm -rf .abcache` before testing a change to the physics.

On `gpu_openmp` the whole suite passes and the results are bit-for-bit the ones
the CPU backends give, but the process still **exits with code 1**. The suite
builds one shared library per scene shape, and NVIDIA's OpenMP runtime mishandles
shutdown when more than one offload library is loaded: the first frees its device
memory and destroys the CUDA context, and the rest then fail to free theirs in a
context that is gone, which prints

    Accelerator Fatal Error: call to cuMemFree returned error 709
    (CUDA_ERROR_CONTEXT_IS_DESTROYED)

Two libraries are enough to trigger it and one never does. It happens after every
test has run, so no result is affected -- read the pytest report and ignore the
exit code, or give each test its own process (`pytest --forked`, which needs
pytest-forked) if something downstream depends on that code.
