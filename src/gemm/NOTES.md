# GEMM

Goal: beat cuBLAS FP32 on RTX 3060 (sm_86). Row-major, square matrices.
Track at `./run.sh gemm --size 4096`.

## Results

| Implementation | Date | GFLOPS | % cuBLAS | Idea and what we learned |
|---|---|---|---|---|
| cublas | 2026-11-17 | 7688 | 100 | baseline (17.9 ms at 4096) |
| naive | 2026-11-17 | 836 | 10.9 | one thread per C element, global memory only. Scaffold builds and runs; max err 3.5e-4 |

## Backlog
- shared memory tiling
- register blocking (each thread computes a tile of C)
- vectorized global loads
- double buffering
- warp tiling

## Research and Q&A log
