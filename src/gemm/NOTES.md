# GEMM

Goal: beat cuBLAS FP32 on RTX 3060 (sm_86). Row-major, square matrices.
Track at `./run.sh gemm --size 4096`. Peak = SMs x 128 lanes x 2 x max clock = 12953 GFLOPS.

## Results

| Implementation | Date | GFLOPS | % peak | % cuBLAS | Idea and what we learned |
|---|---|---|---|---|---|
| cublas | 2026-11-17 | 7711 | 59.5 | 100 | baseline (17.8 ms at 4096) |
| naive | 2026-11-17 | 840 | 6.5 | 10.9 | one thread per C element, global memory only. Scaffold builds and runs; max err 3.5e-4 |
| register_blocking | 2026-11-19 | 3641 | 28.1 | 47.8 | 8x8 C tile per thread, 16x16 threads per block, no shared memory. 4.4x over naive. ncu: FMA now the top pipeline (38%), issue slots 46% busy, 96 regs/thread, 33% occupancy. Requires m, n multiples of 128 |

## Backlog
- shared memory tiling
- vectorized global loads
- double buffering
- warp tiling

## Research and Q&A log

### 2026-11-18 Why naive is slow, and what fixes it
- ncu on naive: LSU write-back (L1 -> SM) at 98% of peak while FMA is mostly idle. Each FMA needs 2 loads, and every load goes through the LSU even on an L1 hit.
- Register blocking (TMxTN C tile per thread) cuts loads per FMA from 2 to (TM+TN)/(TM*TN) = 0.25 for 8x8. Registers are the only operand source that bypasses the LSU.
- Shared memory does not give more bandwidth than L1: they are the same SRAM, ~128 B/clk/SM. It cuts global/L1/L2 traffic (each block fetches a tile once instead of once per thread) and makes reuse guaranteed, but its loads still use the LSU. Tiling alone keeps 2 loads per FMA, so it should not fix the LSU bottleneck by itself.
- Without shared memory, threads cannot share registers: the 16 threads with the same ty reload the same A values, and the 16 with the same tx reload the same B values.
- Thread organisation: block tile BMxBN, block has (BM/TM)x(BN/TN) threads, grid (N/BN)x(M/BM). Thread (ty,tx) owns a TMxTN C tile.
- Decision: do register blocking before shared memory to test the LSU hypothesis in isolation.

### 2026-11-19 register_blocking result
- LSU hypothesis confirmed: after register blocking, FMA is the highest-utilized pipeline (38.4%) and the kernel is 4.4x faster.
- Still 48% of cuBLAS. Remaining suspects (not yet profiled in detail): global/L1 traffic (A loads are strided by k, redundant 16x across threads), low occupancy (96 regs -> 33%), only 46% issue slots busy. Next: shared memory tiling on top of the register tile.
