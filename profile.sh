#!/usr/bin/env bash
# Usage: [KERNEL=regex] ./profile.sh <project> [args...]   (runs inside the container)
# Profiles the first kernel matching KERNEL (default: *_kernel) with the full ncu set and writes to profiles/:
#   <name>.ncu-rep    full report (open in nsight-gui, or ncu --import)
#   <name>.txt        details page, same text printed below
#   <name>.sass.csv   per-instruction source page: samples, stall reasons, executed counts
#   <name>.cubin.sass cuobjdump SASS of the executable
#   <name>.resusage.txt registers, spills, shared memory per kernel
# Quick queries on an existing report:
#   ncu --import R.ncu-rep --page raw --metrics <m1>,<m2>
#   ncu --import R.ncu-rep --page source --print-source sass --csv
#   ncu --query-metrics | grep <word>
set -e
project=$1; shift
cmake -S . -B build -G Ninja >/dev/null
cmake --build build --target "$project"
mkdir -p profiles
name=profiles/$project$(printf -- '-%s' "$@" | tr -c 'A-Za-z0-9\n' '-')
kernel=${KERNEL:-regex:_kernel\$}

ncu --set full --import-source yes -k "$kernel" -c 1 -f -o "$name" ./build/"$project" "$@"
ncu --import "$name.ncu-rep" --page details | tee "$name.txt"
ncu --import "$name.ncu-rep" --page source --print-source sass --csv > "$name.sass.csv"
cuobjdump -sass ./build/"$project" > "$name.cubin.sass"
cuobjdump -res-usage ./build/"$project" > "$name.resusage.txt"
echo "wrote $name.{ncu-rep,txt,sass.csv,cubin.sass,resusage.txt}"
