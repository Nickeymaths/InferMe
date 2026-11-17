# GEMM

Goal: beat cuBLAS FP32 on RTX 3060 (sm_86). Row-major, square matrices.
Track at `./run.sh gemm --size 4096`.

## Results

| Implementation | Date | GFLOPS | % cuBLAS | Idea and what we learned |
|---|---|---|---|---|
| cublas | | | 100 | baseline |
| naive | | | | one thread per C element, global memory only |

## Backlog
- shared memory tiling
- register blocking (each thread computes a tile of C)
- vectorized global loads
- double buffering
- warp tiling

## Research and Q&A log
