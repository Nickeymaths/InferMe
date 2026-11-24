#!/usr/bin/env python3
"""Usage: ./sweep.py <gemm_name>... [--sizes 32,128,1024,4096,8192]   (runs inside the container)
Sweeps N, plots GFLOPS of the named gemm implementations against cuBLAS to plots/<names>.png.
Sizes an implementation cannot run (crash or WRONG result) are left out of its line."""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("names", nargs="+")
parser.add_argument("--sizes", default="32,128,1024,4096,8192")
args = parser.parse_args()
sizes = [int(s) for s in args.sizes.split(",")]

subprocess.run("cmake -S . -B build -G Ninja >/dev/null && cmake --build build --target gemm",
               shell=True, check=True)

series = {name: {} for name in ["cublas", *args.names]}
peak = 0.0
for size in sizes:
    run = subprocess.run(["./build/gemm", "--size", str(size), *args.names],
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(f"N={size}: gemm failed, skipped\n{run.stderr.strip()}")
        continue
    peak = float(re.search(r"peak ([\d.]+) GFLOPS", run.stdout).group(1))
    for line in run.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in series:
            wrong = "WRONG" in line
            print(f"N={size:5d} {fields[0]:24s} {fields[2]:>8s} GFLOPS{'  WRONG, skipped' if wrong else ''}")
            if not wrong:
                series[fields[0]][size] = float(fields[2])

fig, ax = plt.subplots(figsize=(8, 5))
for name, points in series.items():
    if points:
        ax.plot(list(points), list(points.values()), marker="o", label=name,
                linestyle="--" if name == "cublas" else "-", color="black" if name == "cublas" else None)
ax.axhline(peak, color="gray", linestyle=":", label=f"peak {peak:.0f}")
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [str(s) for s in sizes])
ax.set_xlabel("N (M = N = K)")
ax.set_ylabel("GFLOPS")
ax.grid(True, alpha=0.3)
ax.legend()
Path("plots").mkdir(exist_ok=True)
out = f"plots/{'-'.join(args.names)}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
