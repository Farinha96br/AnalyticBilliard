"""Cold build time against the number of objects: one panel per backend, one
line per object type. Reads compile.csv, written by runBenchmarks.py.

    python benchmarks/plotCompile.py [--results benchmarks/results/<host>]
"""

import benchCommon as bc
import benchPlot as bp
from benchPlot import plt

MAX_N = 5000  # the sweep caps every backend here after the first timeout

args = bp.options(__doc__)
rows = [r for r in bc.readCSV(args.results / "compile.csv")
        if r["ok"] and r["N"] <= MAX_N]
info = bp.machine(args.results)
backends = [b for b in bp.BACKENDS if any(r["backend"] == b for r in rows)]
# the compiler's name alone: "g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0" -> g++
name = lambda v, default: v.split()[0] if v else default
compiler = {"linear": name(info.get("cxx"), "g++"),
            "openmp": name(info.get("cxx"), "g++"),
            "gpu_openmp": name(info.get("nvcxx"), "nvc++")}

fig, axes = plt.subplots(1, len(backends), figsize=(2.4 * len(backends), 2.4),
                         sharey=True, squeeze=False)
for ax, backend in zip(axes[0], backends):
    for obj in bp.OBJECTS:
        pts = sorted((r["N"], r["build_s"]) for r in rows
                     if r["backend"] == backend and r["objType"] == obj)
        if pts:
            ax.plot(*zip(*pts), label=bp.tex(obj), **bp.style(obj))
    ax.set(xscale="log", yscale="log", xlabel="$N$")
    # inside the panel, top left: every curve starts low and rises to the right
    ax.text(0.04, 0.95, bp.tex(f"{backend} ({compiler[backend]})"),
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=bp.INK)
axes[0][0].set_ylabel("build time [s]")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.tight_layout()
# above the first panel and flush with its left edge. Anchored to the panel, not
# the figure: bbox_inches="tight" crops the figure when saving, which would
# shift anything placed in figure coordinates
axes[0][0].legend(handles, labels, loc="lower left", ncol=len(labels),
                  bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, borderpad=0.0,
                  handlelength=1.6, columnspacing=1.2)
bp.save(fig, args.results, "compile")

print(f"\ncold build time [s] at the largest N, {info.get('host', '')}")
for backend in backends:
    sub = [r for r in rows if r["backend"] == backend]
    n = max(r["N"] for r in sub)
    print(f"  {backend:>10} N={n}: " + ", ".join(
        f"{r['objType']} {r['build_s']:.1f}" for r in sub if r["N"] == n))
