"""The parallel scalability table: one row per scene, at the largest Np, from
scaling.csv written by runBenchmarks.py. Printed, and written next to it as
scaling.md and scaling.tex (booktabs).

Throughput is Mev/s, millions of events per second. Speedups are against the
linear backend; the OpenMP efficiency is its own speedup over one thread,
divided by the physical core count -- the threads past that are SMT siblings.

    python benchmarks/tableScaling.py [--results benchmarks/results/<host>]
"""

import benchCommon as bc
import benchPlot as bp

args = bp.options(__doc__)
rows = [r for r in bc.readCSV(args.results / "scaling.csv") if r["ok"]]
info = bp.machine(args.results)
phys = info.get("physicalCores")
logical = info.get("logicalCores") or max(r["threads"] for r in rows)
top = max(r["Np"] for r in rows)


def mev(scene, backend, threads=None):
    for r in rows:
        if (r["scene"] == scene and r["backend"] == backend and r["Np"] == top
                and (threads is None or r["threads"] == threads)):
            return r["Mev_s"]
    return None


HEAD = ["scene", "objects", "linear", "OMP 1", f"OMP {phys}", f"OMP {logical}",
        "OMP/lin", f"eff@{phys}", "GPU", "GPU/lin"]
table = []
for scene in bp.SCENE_OBJECT:
    if not any(r["scene"] == scene for r in rows):
        continue
    lin, one = mev(scene, "linear"), mev(scene, "openmp", 1)
    core, alls = mev(scene, "openmp", phys), mev(scene, "openmp", logical)
    gpu = mev(scene, "gpu_openmp")
    ratio = lambda a, b: a / b if a and b else None
    objs = next(r["objects"] for r in rows if r["scene"] == scene)
    table.append([scene, objs, lin, one, core, alls, ratio(alls, lin),
                  ratio(core, one) / phys if core and one and phys else None,
                  gpu, ratio(gpu, lin)])


def cell(v, j):
    if v is None:
        return "--"
    if isinstance(v, str) or isinstance(v, int):
        return str(v)
    return f"{v:.2f}" if HEAD[j].startswith("eff") else (
        f"{v:.1f}x" if "/" in HEAD[j] else f"{v:.3g}")


text = [[cell(v, j) for j, v in enumerate(r)] for r in table]
it = rows[0]["iter"]
caption = (f"Throughput in Mev/s at Np = {top:,}, {it} events per particle, "
           f"every event recorded. {info.get('cpu', '')}"
           + (f"; {info['gpus'][0]['name']}" if info.get("gpus") else "") + ".")
print(caption)
bc.printTable(text, HEAD)

md = ["| " + " | ".join(HEAD) + " |", "|" + "---:|" * len(HEAD)]
md += ["| " + " | ".join(r) + " |" for r in text]
(args.results / "scaling.md").write_text(caption + "\n\n" + "\n".join(md) + "\n")

esc = lambda s: s.replace("%", r"\%").replace("_", r"\_")
tx = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}",
      r"\begin{tabular}{l" + "r" * (len(HEAD) - 1) + "}", r"\toprule",
      " & ".join(esc(h) for h in HEAD) + r" \\", r"\midrule"]
tx += [" & ".join(esc(c).replace("x", r"$\times$") if c.endswith("x") else esc(c)
                  for c in r) + r" \\" for r in text]
tx += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(args.results / "scaling.tex").write_text("\n".join(tx) + "\n")
print(f"wrote {args.results}/scaling.md and scaling.tex")
