#!/usr/bin/env bash
# Usage: ./profile.sh <project> [args...]   (runs inside the container)
# Profiles the first kernel named *_kernel; report goes to profiles/.
set -e
project=$1; shift
cmake -S . -B build -G Ninja >/dev/null
cmake --build build --target "$project"
mkdir -p profiles
report=profiles/$project$(printf -- '-%s' "$@" | tr -c 'A-Za-z0-9\n' '-')
ncu --set full -k 'regex:_kernel$' -c 1 -f -o "$report" ./build/"$project" "$@"
ncu --import "$report.ncu-rep" --page details
