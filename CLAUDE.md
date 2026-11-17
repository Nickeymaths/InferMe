# Engineering Philosophy

## Comments
- The best code has few comments. Names tell the story.
- Rename a variable, function, or type before writing a comment to explain it.
- Comment only the non-obvious "why" (hardware quirks, numerical tricks), never the "what".

## Simplicity
- Simple is best. A small improvement that makes the implementation messy isn't worth it.
- Prefer removing code over adding code, as long as performance doesn't suffer.
- Don't add abstractions, options, or generality that nothing needs yet.

## Environment
- Develop inside Docker. Don't install toolchains (CUDA, compilers, libraries) on the host.
- Build and run through the container: `docker compose run --rm cuda <command>`.
- Target GPU: RTX 3060 (`sm_86`).

## Layout
- `src/<project>/*.cu` becomes the executable `<project>` automatically (CMake globs it).
- `common/` holds shared headers: `cuda_check.cuh` (CUDA_CHECK, CUBLAS_CHECK), `benchmark.cuh` (average_milliseconds).
- `./new.sh <project>` creates a project; `docker compose run --rm cuda ./run.sh <project> [args]` builds and runs it.

## GEMM
- Row-major FP32. Each implementation is one file `src/gemm/<name>.cu` ending with `REGISTER_GEMM(<name>, <function>);` (see `naive.cu`). No other file needs editing.
- `./run.sh gemm [--size N] [name...]` benchmarks the named implementations (all if none named), checks each against cuBLAS, and prints ms, GFLOPS, % of cuBLAS, max error.

## Workflow
The project grows incrementally: research and Q&A with the user, then one implementation per step.
- Every project keeps `src/<project>/NOTES.md`: goal, results table, backlog, dated research/Q&A log.
- After each implementation: benchmark, add a row to the results table (real measured numbers only, never estimates), note what was learned, then commit.
- One commit per implementation, message like `gemm: shared_memory_tiling, 38% of cublas`. Git history is the progress record.
- Record durable answers from research/Q&A in the NOTES.md log, so a new session can continue without this conversation.
- Start of a session: read `src/<project>/NOTES.md` and `git log` before proposing anything.

## Status
- Nothing has been compiled or run yet. Build the scaffold and `naive` first and fix any errors.
- Docker needs the `docker` group in the shell that launches Claude Code (`newgrp` and `sg` are missing on this Ubuntu).
