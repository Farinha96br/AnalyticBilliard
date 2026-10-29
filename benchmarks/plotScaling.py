"""The scalability figures, from scaling.csv written by runBenchmarks.py.

    scalingThreads   OpenMP strong scaling at the largest Np: speedup over one
                     thread against the thread count, one line per scene
    scalingNp        throughput against Np, one panel per backend, one line per
                     scene -- where each backend stops being starved for work

    python benchmarks/plotScaling.py [--results benchmarks/results/<host>]
"""

import benchCommon as bc
import benchPlot as bp
from benchPlot import plt

args = bp.options(__doc__)
rows = [r for r in bc.readCSV(args.results / "scaling.csv") if r["ok"]]
info = bp.machine(args.results)
top = max(r["Np"] for r in rows)
scenes = [s for s in bp.SCENE_OBJECT if any(r["scene"] == s for r in rows)]
label = lambda s: bp.tex(bp.SCENE_OBJECT[s])

# ---- strong scaling
omp = [r for r in rows if r["backend"] == "openmp" and r["Np"] == top]
if omp:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pMax = max(r["threads"] for r in omp)
    ax.plot([1, pMax], [1, pMax], "--", color=bp.MUTED, lw=0.8, label="ideal")
    phys = info.get("physicalCores")
    if phys and phys < pMax:
        ax.axvline(phys, color=bp.MUTED, lw=0.6, ls=":")
        ax.text(phys, 0.98, " physical cores", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=7, color=bp.MUTED)
    for s in scenes:
        pts = sorted((r["threads"], r["run_s"]) for r in omp if r["scene"] == s)
        base = dict(pts).get(1)
        if base:
            ax.plot([p for p, _ in pts], [base / t for _, t in pts], label=label(s),
                    **bp.style(bp.SCENE_OBJECT[s]))
    ax.set(xlabel="OpenMP threads", ylabel=r"speedup $T_1 / T_p$",
           xlim=(0.5, pMax + 0.5), ylim=(0, pMax + 0.5))
    ax.set_title(bp.tex(f"Np = {top:,}, {info.get('cpu', '')}"), fontsize=7,
                 color=bp.INK)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    bp.save(fig, args.results, "scalingThreads")

# ---- throughput against Np, each backend at its full width
full = {"linear": 1, "openmp": max((r["threads"] for r in rows
                                    if r["backend"] == "openmp"), default=0),
        "gpu_openmp": 0}
backends = [b for b in bp.BACKENDS if any(r["backend"] == b for r in rows)]
fig, axes = plt.subplots(1, len(backends), figsize=(2.4 * len(backends), 2.4),
                         sharey=True, squeeze=False)
for ax, backend in zip(axes[0], backends):
    for s in scenes:
        pts = sorted((r["Np"], r["Mev_s"]) for r in rows
                     if r["backend"] == backend and r["scene"] == s
                     and r["threads"] == full[backend])
        if pts:
            ax.plot(*zip(*pts), label=label(s), **bp.style(bp.SCENE_OBJECT[s]))
    title = bp.BACKEND_NAME[backend]
    if backend == "openmp":
        title += f", {full[backend]} threads"
    ax.set(xscale="log", yscale="log", xlabel="particles $N_p$")
    ax.set_title(title, fontsize=8, color=bp.INK)
axes[0][0].set_ylabel("throughput [Mev/s]")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels),
           bbox_to_anchor=(0.5, 1.0), fontsize=7)
fig.tight_layout()
bp.save(fig, args.results, "scalingNp")
