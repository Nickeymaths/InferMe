#!/usr/bin/env bash
# Usage: ./run.sh <project> [args...]   (runs inside the container)
set -e
project=$1; shift
cmake -S . -B build -G Ninja >/dev/null
cmake --build build --target "$project"
./build/"$project" "$@"
