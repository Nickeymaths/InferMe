#!/usr/bin/env bash
# Usage: ./new.sh <project>   creates src/<project>/main.cu and NOTES.md
set -e
project=$1
[ -n "$project" ] || { echo "usage: $0 <project>"; exit 1; }
[ ! -e src/$project ] || { echo "src/$project already exists"; exit 1; }
mkdir -p src/$project
cat > src/$project/main.cu <<'TEMPLATE'
#include <cstdio>
#include "benchmark.cuh"
#include "cuda_check.cuh"

int main() {
}
TEMPLATE
cat > src/$project/NOTES.md <<TEMPLATE
# $project

Goal:

## Results

| Implementation | Date | Result | Idea and what we learned |
|---|---|---|---|

## Backlog

## Research and Q&A log
TEMPLATE
echo "created src/$project/main.cu and NOTES.md"
