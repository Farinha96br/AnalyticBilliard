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
