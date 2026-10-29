# Benchmarks

These scripts measure two things about AnalyticBilliard:

- **compile time**: how long a scene takes to build, against the number of
  objects;
- **run time**: how fast a scene runs, and how that scales with threads,
  particles and backend.

Every script works on any machine. Results go to a folder named after the
machine, so runs from different computers never mix.

## Requirements

- The package installed in editable mode, from the repo root: `pip install -e .`
- `g++` with OpenMP, for the `linear` and `openmp` backends.
- Optional: the NVIDIA HPC SDK (`nvc++`) and a GPU, for `gpu_openmp`. Builds
  target the GPU in the machine they run on (`AB_GPUARCH=ccnative`). If `nvc++`
  is not at the default path, point `AB_NVCXX` at it.
- `numpy` and `matplotlib`. LaTeX is optional: the figures use it if `latex` is
  on the PATH, and plain matplotlib text otherwise.

## Quick start

Run everything from the repo root:

```bash
python benchmarks/runBenchmarks.py --dry-run   # what would run, and the memory it needs
python benchmarks/runBenchmarks.py --plot      # run everything, then draw figures and table
```

The full default sweep on a 6-core desktop with a GPU took about 2 hours. Most
of that is building scenes with thousands of objects; see
[Things to know](#things-to-know).

## The driver: `runBenchmarks.py`

This one script runs every sweep. Defaults live in the `PARAMS` dict at the top
of the file. Edit them there, or override them on the command line.

| Option | Default | What it does |
|---|---|---|
| `--backends` | every backend this machine can build | any of `linear openmp gpu_openmp` |
| `--compile-types` | `line lineSegment ellipse ellipseArc portal rand` | object types for the compile sweep |
| `--compile-N` | `1 2 5 … 5000 10000` | object counts for the compile sweep |
| `--compile-repeat` | 1 | cold builds per compile point (median reported) |
| `--scenes` | `Ellipse ElipseArc LineSeg Line Portal` | scenes for the scaling sweep |
| `--Nobj` | 10 | objects per scene; must be even when `Portal` is included |
| `--iter` | 100 | events per particle |
| `--Np` | `100 1000 10000 100000` | particle counts |
| `--repeat` | 3 | timed runs per scaling point (median reported) |
| `--threads` | 1, 2, 4, 8, physical cores, logical cores | OpenMP thread counts for strong scaling |
| `--timeout` | 1800 | seconds per compile point, and per scaling run |
| `--cap-after-timeout` | 5000 | after the first compile timeout, skip every later N above this (0 = never) |
| `--mem-fraction` | 0.8 | refuse a run that needs more than this fraction of free memory |
| `--results` | `benchmarks/results/<hostname>` | where everything is written |
| `--only compile` / `--only scaling` | — | run just one of the two sweeps |
| `--dry-run` | — | print the plan and the memory estimate, run nothing |
| `--force` | — | run points over the memory limit anyway |
| `--fresh` | — | rerun points that are already in the CSVs |
| `--plot` | — | draw the figures and the table at the end |

### What it runs

**Compile sweep.** One cold build per (backend, object type, N). Each build
uses an empty cache, so the compiler always runs. The `rand` type mixes all five
object types at random.

**Scaling sweep.** Two passes:

1. **Throughput:** every backend × scene × `Np`. OpenMP uses every logical core.
2. **Strong scaling:** OpenMP × scene × thread count, at the largest `Np`.

Every point runs in its own process. The OpenMP thread count is set through
`OMP_NUM_THREADS`, and one failing point cannot take the rest of the sweep down.

### Memory

Every event is recorded (x, y, vx, vy), so one run needs `Np × (iter + 1) × 32`
bytes. A GPU run needs the same amount again on the device.

Before starting, the driver prints the estimate for each `Np` against the free
host RAM and free GPU memory. Any point that would need more than
`--mem-fraction` of either is skipped unless you pass `--force`. Use `--dry-run`
to see the estimate without running anything.

### Stopping, resuming and failures

- **Resuming:** points already in `compile.csv` or `scaling.csv` are skipped. If
  a sweep is interrupted, run the same command again and it continues where it
  stopped. Use `--fresh` to start over.
- **Failures and timeouts** go to `failures.log`, and the sweep carries on. When
  a compile point fails, the larger N for that object type and backend are
  skipped, since they would only take longer.
- **The cap:** the first compile timeout caps every later compile point at
  `--cap-after-timeout`, so one slow configuration doesn't cost the timeout over
  and over.

### Examples

```bash
# CPU only, smaller and quicker
python benchmarks/runBenchmarks.py --backends linear openmp --compile-N 1 10 100 1000 --plot

# only the scaling sweep, with more events per particle
python benchmarks/runBenchmarks.py --only scaling --iter 500 --Np 1000 100000

# a quick smoke test into a scratch folder
python benchmarks/runBenchmarks.py --results /tmp/abtest --compile-N 1 10 \
    --Np 100 1000 --threads 1 2 --iter 50 --repeat 1 --plot
```

## Results folder

`benchmarks/results/<hostname>/` holds:

| File | What it is |
|---|---|
| `machine.json` | the machine: CPU, physical and logical cores, RAM, GPU, compiler versions, git commit, `OMP_*`/`AB_*` env, the parameters used |
| `compile.csv` | one row per compile point: codegen, build and load time, `.so` size |
| `scaling.csv` | one row per scaling point: run time, throughput, ns per event per object, energy drift, checks |
| `compile.pdf/png` | build time against N, one panel per backend |
| `scalingThreads.pdf/png` | OpenMP speedup against threads |
| `scalingNp.pdf/png` | throughput against `Np`, one panel per backend |
| `scaling.md`, `scaling.tex` | the scalability table (the `.tex` uses booktabs) |
| `failures.log` | every failure, timeout and skip |
| `logs/` | each point's full DEBUG log |

To compare machines, copy each machine's `results/<hostname>/` folder into this
one. They don't collide.

## Figures and table

The figures and table are drawn from the CSVs alone, in seconds. Each script
takes `--results DIR` and defaults to this machine's folder:

```bash
python benchmarks/plotCompile.py   [--results benchmarks/results/<host>]
python benchmarks/plotScaling.py   [--results benchmarks/results/<host>]
python benchmarks/tableScaling.py  [--results benchmarks/results/<host>]
```

- **Colours:** each object type keeps the same colour and marker in every
  figure. The colours come from the package's Portal palette (`PORTAL_COLORS`
  in `analyticBilliards.utils`). Shared style is in `benchPlot.py`.
- **`MAX_N`:** the compile figure stops at `MAX_N` (5000), set at the top of
  `plotCompile.py`.
- **Table columns:** throughput in Mev/s (millions of events per second) at the
  largest `Np`, for linear, OpenMP on 1 thread, the physical cores and all
  logical cores, and the GPU. It also shows speedups over `linear`, and the
  OpenMP efficiency on the physical cores.

## Single benchmarks

The driver calls these scripts, but each one also runs on its own and prints a
table. `--scene`, `--Np`, `--Nobj`, `--objType` and `--N` take several values
and sweep over them.

### `compile.py`: compile time

```bash
python benchmarks/compile.py --objType rand portal --N 10 100 1000 [--backend openmp] [--repeat 1] [--csv out.csv]
```

`--objType` is one of `line`, `lineSegment`, `ellipse`, `ellipseArc`, `portal`
or `rand`. Gravity is g = 1. It reports three times separately:

- `codegen`: writing the scene header;
- `build`: the compiler;
- `load`: dlopen and pushing the parameters into the library.

### `scene.py`: run time

```bash
python benchmarks/scene.py --scene Ellipse --Np 1000 10000 --Nobj 1 10 100 --iter 1000 [--backend openmp] [--repeat 3] [--csv out.csv]
python benchmarks/scene.py --scene Ellipse ElipseArc LineSeg Line Portal --Nobj 10   # several scenes
```

Each scene keeps every particle trapped forever, so every run makes all `iter`
events. The solver still checks every object at every event, so the cost per
event grows with `Nobj`. Gravity is g = 1.

| `--scene` | Scene | Built by `commonTables.` |
|---|---|---|
| `Ellipse` | nested concentric ellipses, radii 1.5k × k; particles inside the innermost one | `nestedEllipses` |
| `ElipseArc` | the same nested family as arcs, each with an opening of ±0.3 rad at the top; particles are too slow to reach it | `nestedEllipseArcs` |
| `LineSeg` | horizontal segments 2×10⁸ long, stacked at y = 0, −1, −2, …; particles bounce on the top one with vx = 10⁻⁵ | `stackedLineSegs` |
| `Line` | the same as `LineSeg`, with infinite lines | `stackedLines` |
| `Portal` | a box on a floor whose side walls are `Nobj / 2` stacked portal pairs, so the box is periodic in x; `Nobj` must be even, and the floor adds one object | `portalBox` |

The tables live in `analyticBilliards.commonTables` (the `benchmarks`
submodule), with `circle`, `rectangle` and `lorentzGas` used by `GPU_WORTH`.
`benchmarkTablesExample.py`, at the repo root, draws them. The initial
conditions and the confinement checks are in `scene.py`.

**Columns:**

- `Mev_s`: millions of events per second.
- `ns_ev_obj`: nanoseconds per event per object.
- `drift`: the largest relative energy error, recomputed from the recorded
  x, y, vx, vy.
- `ok`: true only if all three checks passed:
  - every particle made every event;
  - every recorded point stayed in its region;
  - the drift is below 10⁻⁸.

## Experiments

Three focused experiments live in their own folders. `IO_EXPERIMENTS/` and
`GPU_WORTH/` have the same three scripts:

- the measurement script, which runs one backend (and one scene) per process;
- `run.py`, the driver, with a `PARAMS` dict at the top and the same `--dry-run`,
  `--fresh`, `--force` and `--plot` options and the same memory check as
  `runBenchmarks.py`;
- `plot.py`, which draws the figure and writes the table.

Results go to `<folder>/results/<hostname>/`.

### `IO_EXPERIMENTS/`: what does recording cost?

```bash
python benchmarks/IO_EXPERIMENTS/run.py --plot      # Np=80000, iter=1000, 15 rounds
```

The scene is a unit circle with g = 1. Only the saved columns change between
runs; the physics is the same every time. It times two sets of runs:

- columns added one at a time, from nothing up to all 7 (`t x y vx vy Energy
  evType`). This is the default;
- each column saved alone. Add it with `--families cumulative single`.

Saving nothing (`save=[]`) is the baseline: the run still makes every event but
writes nothing. All configurations are timed in rounds, in a new random order
each round, so a machine that drifts over time can't favour one of them. The
speedup is taken within each round, against that round's nothing-saved run, and
the median over rounds is reported. OpenMP threads are pinned
(`OMP_PROC_BIND=close`, `OMP_PLACES=threads`) to cut jitter.

The full run takes about 2 hours, most of it on `linear`.

Output: `io.pdf/png` shows the speedup against the number of columns saved
(events/s over events/s saving nothing, median over rounds), and `io.md` is the
table, with the 25th and 75th percentiles.

### `GPU_WORTH/`: when is the GPU worth it?

```bash
python benchmarks/GPU_WORTH/run.py --plot           # Np 10 .. 10^6, iter=50
```

| Scene | What it is |
|---|---|
| `circle` | the unit circle, g = 1 |
| `square` | the box [−1, 1]² made of four `LineSegment`s, g = 1 |
| `lorentz` | the cell of the diffusion experiment (`paper_examples/klagesDellago.py`), gap w = 0.1, g = 0 |

Every run saves `t x y vx vy evType` for every event; `--save` picks other
columns, e.g.

```bash
python benchmarks/GPU_WORTH/run.py --save x y --iter 1000 \
    --Np 10 30 100 300 1000 3000 10000 30000 100000 300000 \
    --results benchmarks/GPU_WORTH/results_xy_iter1000/$(hostname) --plot
```

For each scene, backend
and `Np`, it records:

- `run_s`: the median wall time of one warm call;
- `Mev_s`: events per second;
- `build_s`: the compiler alone, once per process;
- `start_s`: loading the library and starting the device or runtime, plus a
  first one-particle call. This is the one-off cost.

Output:

- `gpu_worth.pdf/png`: run time and events/s against `Np`, one column per scene.
  A dotted line marks where the GPU starts beating OpenMP for good.
- `gpu_worth.md`: the table, with the `Np` from which the GPU beats each CPU
  backend.

### `ENERGY_DRIFT/`: how much energy is lost over long runs?

```bash
python benchmarks/ENERGY_DRIFT/drift.py --iter 1e7 --stride 1e4 --out drift.npz   # ~65 s on OpenMP
```

100 particles in the unit circle with g = 1, 10⁷ events each, every 10⁴-th
recorded. The energy is recomputed from the recorded x, y, vx, vy. This one is a
single script, with no driver or results folder. `energyDrift.md` has the
results, the cause (a bias in the wall normal used by `bounce()`, which makes
the drift grow linearly, about 10⁻¹⁶ per event), and a proposed fix that is not
applied. `drift.png/pdf` is its figure.

## Things to know

- **Large scenes compile slowly.** Build time grows faster than linearly once a
  scene has more than about 100 objects. Mixed types (`rand`) are by far the
  slowest:
  - with g++, `rand` at N = 5000 took about 7 minutes, and N = 10000 did not
    finish in 30 minutes;
  - the compiler (`cc1plus`) can use 3 GB of memory or more on these;
  - nvc++ refuses 5000 portals outright, with "Source file too large to compile
    at this optimization level".
- **`nvc++` is slow to start.** Every GPU build takes about 10 s, even for a
  single object.
- **The GPU needs many particles.** At small `Np` the GPU is slower than the
  CPU. Its throughput was still rising at `Np` = 10⁵.
- **`AB_LOG` sets the log level.** The benchmarks log at DEBUG to stderr, and
  the driver sends that to `logs/`. Set `AB_LOG=WARNING` to quiet a single
  script run by hand.
- **Exit code 1 on the GPU.** On `gpu_openmp`, a process that loaded more than
  one GPU library can exit with code 1 at shutdown even though every result is
  correct (see the main readme). The driver avoids this by running each point
  in its own process.
