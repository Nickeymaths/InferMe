# GEMM

Goal: beat cuBLAS FP32 on RTX 3060 (sm_86). Row-major, square matrices.
Track at `./run.sh gemm --size 4096`. Peak = SMs x 128 lanes x 2 x max clock = 12953 GFLOPS.

## Results

| Implementation | Date | GFLOPS | % peak | % cuBLAS | Idea and what we learned |
|---|---|---|---|---|---|
| cublas | 2026-11-17 | 7711 | 59.5 | 100 | baseline (17.8 ms at 4096) |
| naive | 2026-11-17 | 840 | 6.5 | 10.9 | one thread per C element, global memory only. Scaffold builds and runs; max err 3.5e-4 |
| register_blocking | 2026-11-19 | 3641 | 28.1 | 47.8 | 8x8 C tile per thread, 16x16 threads per block, no shared memory. 4.4x over naive. ncu: FMA now the top pipeline (38%), issue slots 46% busy, 96 regs/thread, 33% occupancy. Requires m, n multiples of 128 |
| shared_memory_tiling (scalar LDS, 75225b8) | 2026-11-21 | 4758 | 36.7 | 62.1 | 128x128 block tile, BK=8, A and B tiles staged in shared memory with coalesced loads, same 8x8 register tile. 1.30x over register_blocking. ncu: LSU pipe 72% (FMA 34%), 16 scalar LDS per 64 FMA, 33.5M of 100.7M shared-load wavefronts are bank conflicts (a_frag, 2-way), global loads 32/32 B per sector. Requires m, n multiples of 128 and k multiple of 8 |
| shared_memory_tiling (vectorized LDS.128) | 2026-11-22 | 7209 | 55.7 | 96.2 | same tiling, but a_tile stored transposed [BK][128] and each thread owns columns tx*4..+3 and 64+tx*4..+3, so fragments load as 4 LDS.128 per 64 FMA instead of 16 LDS.32. 1.52x over the scalar version. 125 regs, no spills. Not yet profiled |
| shared_memory_tiling (conflict-free a_tile, LDS.128 b_frag, STG.128 C) | 2026-11-23 | 5794 | 44.7 | 75.1 | a_tile row padded to BLOCK_M+4 (0 bank conflicts); each thread owns columns tx*4..+3 and 64+tx*4..+3 so b_frag is 2 LDS.128 and C is written with 16 STG.128. 1.22x over the scalar version. SASS has 0 scalar LDS. 119 regs. ncu (2048): 392M instructions issued (was 417M), FMA 48%, LSU 33% (was 61%), issue active 63%, no bank conflicts. Stalls now led by long scoreboard 1.12 (global loads) and short scoreboard 0.93; mio_throttle fell 0.80 to 0.08. Max err 3.5e-4 at 4096 |
| shared_memory_tiling (float4 global tile loads) | 2026-11-24 | 7305 | 56.4 | 96.6 | one float4 global load per thread for A (then 4 scalar transposed STS) and for B (one STS.128), replacing the strided scalar fill loops. 1.26x over the previous step. ncu (2048): 301M instructions issued (was 392M), FMA 58%, LSU 36%, issue active 65%, 125 regs, 0 bank conflicts on loads and stores. Stalls: long scoreboard 0.77 (was 1.12), short scoreboard 0.66, barrier 0.63, mio_throttle 0.18 |

## Backlog
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

### 2026-11-20 to 21
- Naive: LSU write-back 98% busy, FMA 80% free. Each FMA needed 2 L1 loads. Fix is fewer load instructions per FMA, which is what register blocking does (16 loads per 64 FMAs for 8x8).
- L1 and shared memory are the same SRAM with the same peak bandwidth (128 B/clk/SM). Shared memory still helps: no L1 misses (hit rate was 88%, misses cost L2 latency), no tag lookup, guaranteed reuse, and the layout can be chosen so fragments load with vector LDS.
- register_blocking profile (size 2048): nothing saturated (L1 56%, FMA 37%, issue slots 44%). Latency bound: 3.78 active and 1.17 eligible warps per scheduler, 34% of stall cycles are L1TEX scoreboard (global loads used right after issue). Occupancy 33% because 96 regs x 256 threads allows 2 blocks per SM.
- Occupancy numbers: sm_86 holds at most 48 resident warps per SM (12 per scheduler). A scheduler issues 1 warp-instruction per cycle but picks from its resident pool. The kernel gets 16 warps per SM (4 per scheduler) from the register limit.
- Tiling design: a one-k-slice shared buffer would need 2 syncs per 64 FMAs and strided A loads, so the tile is BK=8 deep.
- Possible bank conflicts in shared_memory_tiling: a_tile reads (stride BK between rows, two ty rows per warp) may be 2-way conflicted; padding the row to BK+1 would fix it. Profile before changing.

### 2026-11-22 vectorized shared loads
- shared_memory_tiling (scalar) profile: LSU instruction pipe is the limit again (72%), not latency. Cause is 16 LDS.32 per 64 FMA, so fewer and wider loads are needed, not just conflict-free ones.
- One LDS.128 loads 4 contiguous floats, so the fragment must be contiguous in shared memory. A: transpose the tile to [k][m]; filling it costs one float4 global load (4 consecutive k) plus 4 scalar stores. B: columns tx*4..+3 and 64+tx*4..+3 per thread; 8 contiguous columns would put neighbouring threads 32 B apart and conflict within a quarter-warp.
- Remaining known issues: the transposed A stores are 2-way bank conflicted (minor, 4 stores per tile; padding the row to BLOCK_M+4 fixes it), occupancy still register-limited (125 regs, about 33%), global loads not overlapped with compute (next: double buffering). Profile this version before choosing.

### 2026-11-23 profile of bank-conflict-free scalar version (size 2048, uncommitted a_tile[BK][BLOCK_M+4])
- Shared-load bank conflicts: 0 (was 33.5M of 100.7M wavefronts). Shared loads executed 41.9M, not 67M: the compiler already fuses the 8 contiguous a_frag floats into 2 LDS.128, leaving 8 scalar LDS.32 for b_frag (stride THREADS, cannot be fused). So 10 LDS per 64 FMA.
- Issue slots busy 63%, FMA pipe 44.5%, LSU 61%. 417M instructions issued vs 268M FMA, so 36% of issue slots are not FMA.
- Stalls per issued instruction (6.0 total): short scoreboard 1.7 (waiting on LDS results), mio_throttle 0.8, long scoreboard 0.7 (global loads), wait 0.58, barrier 0.40. Eligible warps 1.71 per scheduler; occupancy 33% (117 regs, 2 blocks per SM).
- Conclusion: still instruction-issue and shared-load-latency bound, not conflict bound. Next: vectorize b_frag (columns tx*4..+3 and 64+tx*4..+3) to remove the 8 scalar LDS.

### 2026-11-23 result of vectorizing b_frag and C stores
- 4096: 5794 GFLOPS, 75.1% of cublas. LSU pipe dropped 61% to 33% and mio_throttle nearly vanished, so the shared-load path is no longer the bottleneck. FMA only reached 48% though.
- Remaining stalls: long scoreboard 1.12 (global loads not overlapped with compute), short scoreboard 0.93, barrier 0.40. Still 33% occupancy (119 regs).
- The earlier 96% version also had float4 global tile loads, so the 75% to 96% gap is probably the global-load side (Step 2) plus its effect on latency. Hypothesis, not measured.
- Next: vectorize the global-to-shared tile fills (one float4 global load per thread for A and B), then double buffering.

### 2026-11-24 result of vectorizing the global tile loads
- 4096: 7305 GFLOPS, 96.6% of cublas, confirming that the 75% to 96% gap was the global-load side (instruction count 392M to 301M).
- Barrier stall rose to 0.63 and long scoreboard is still 0.77: loads are issued then waited on before the same iteration's compute, so double buffering is the natural next step. Occupancy still 33% (125 regs); FMA pipe at 58%.
