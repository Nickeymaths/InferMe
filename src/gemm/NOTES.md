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
| shared_memory_tiling (BK=16) | 2026-11-24 | 8330 | 64.3 | 108 | BK 8 to 16: two float4 loads per thread per tile for A and B, half as many barriers per FMA. 1.14x over the BK=8 version (7305). 4096, three runs 8345, 8325, 8323 vs cublas 7687 to 7727. ncu (2048): 2.97 ms (was 3.37), instructions 300.6M (unchanged), eligible warps 1.82 to 2.37; stall ratios barrier 92k to 62k, long scoreboard 113k to 61k, short scoreboard 97k to 31k, dispatch 78k to 81k. 127 regs, 16.6 KB smem, max err 3.47e-4. Requires k multiple of 16 |

## Backlog
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

### 2026-11-24 double buffering tried, not worth it
- Implemented on top of the float4-load version: a_tile and b_tile doubled, next tile's float4 loads issued into registers before compute and stored into the other buffer after, one `__syncthreads()` per iteration instead of two. Correct (max err 3.47e-4).
- 4096, same session, GFLOPS (% cublas): double buffered 7410 (96.1), 7390 (96.1), 7355 (95.9), 7311 (94.8); single buffered 7461 (97.2), 7133 (94.0), 7427 (96.3). cublas itself moved 7591 to 7714 between runs, so run-to-run noise is about 3%, larger than any difference. No measurable gain, possibly slightly worse.
- Likely reasons (hypotheses, not profiled): occupancy is 33% (125 regs, 2 blocks per SM), so while one block waits on a barrier or a global load the other issues FMAs, which already hides the latency double buffering targets. The prefetch registers (float4 for A and B) add register pressure for little return.
- Decision: code removed, kept simple. Revisit only if warp tiling or lower register use changes occupancy, and profile (barrier and long scoreboard stalls, regs) before trying again.

### 2026-11-24 profile of shared_memory_tiling (size 2048, full ncu set)
- 3.37 ms at 1.32 GHz. Instructions 301M, of which FFMA 269.5M (89%), so instruction count is already near the floor; removing overhead can gain at most ~10%.
- Issue slots busy 65%, FMA pipe 58%, 35% of cycles no warp eligible. 3.79 active warps per scheduler, 1.82 eligible. Occupancy 33% (125 regs, 2 blocks/SM), achieved 31.6%.
- Stall mix (relative warp-latency ratios, not_selected excluded): long scoreboard 113k, short scoreboard 97k, barrier 92k, dispatch 78k, wait 29k, mio throttle 26k, math pipe throttle 5.5k.
- Memory is not the limit: DRAM 54 GB/s, L2 hit 69%, max bandwidth 44%. The kernel is latency and sync bound, not throughput bound.
- Source-level (SASS) stall sampling, share of all stall samples: long scoreboard 13.8%, short scoreboard 11.5%, barrier 10.7%, dispatch 9.0%, wait 3.5%, mio 2.9%.
- Where: `STS [..], R88` right after the two `LDG.E.128` waits on the global load (long scoreboard, 11.4% of samples on that one instruction; the STS.128 adds 3.7%). The two `BAR.SYNC` cost 5.9% (first LDS after the barrier) + 6.4% (loop-end barrier). So the load -> STS -> barrier -> compute -> barrier phase is about 27% of stalls, paid once per BK=8 (512 FMA per thread).
- Inside the compute body the LDS for step i+1 is already interleaved with step i's FFMAs; short scoreboard on FFMAs is spread thin (0.5-2.4% each). Dispatch stalls (9%) are spread over FFMA and LDS with no hotspot and are unexplained.
- Warp tiling is NOT supported by this profile (no bank conflicts, LSU not saturated). Candidates that target the measured hotspot: BK=16 (halves barriers and load-wait per FMA), prefetch/cp.async so LDG latency overlaps compute (double buffering earlier showed no gain: check in SASS that the LDG was not sunk below the compute).

### 2026-11-24 BK=16 result
- Profile-driven: the source-level profile put ~27% of stall samples at the global load -> STS -> barrier phase, paid once per BK=8. Doubling BK halves that cost per FMA with the same instruction count.
- Confirmed: 7305 to 8330 GFLOPS, first version above cublas (108%). Barrier and long-scoreboard stalls both fell by about 1/3 to 1/2, short scoreboard by 2/3. Dispatch stall (unexplained) unchanged.
- Next candidates: BK=32 (diminishing but cheap to test, 33 KB smem still 2 blocks/SM limited by registers anyway), then re-do the SASS source profile to see what the remaining stalls sit on, including the dispatch stall.

### 2026-11-24 smaller thread tile (8x4 and 4x8) tried, both slower
- Goal: cut registers to get more resident blocks. Block threads stay 256, so the block tile shrinks to 128x64 (8x4) or 64x128 (4x8). BK=16, same kernel structure.
- 4096, GFLOPS (% cublas), same-session cublas about 7650-7700: BK16 8x8 (127 regs, 2 blocks/SM) 8330 (108); 8x4 (80 regs, 3 blocks/SM, 50% occupancy) 7290 (95); 4x8 (71 regs) 6830 (89).
- 8x4 profile (2048): 3.18 ms vs 2.97. Occupancy 33% to 50%, issue slots busy 70%, but instructions 315M vs 301M (FFMA is now 85% of them), FMA pipe 59.8%. Shared-load wavefronts 67M, Mem Busy 76%, max bandwidth 63%; stalls moved to mio_throttle (26k to 72k at 8x8 BK8 baseline vs 8x4) and short scoreboard 117k, i.e. the shared-memory path is the limit again. 0 load bank conflicts.
- 4x8 additionally has a 2-way store conflict on a_tile (2.1M conflicts) and 312M instructions.
- Lesson: 3 LDS.128 per 32 FMA (6 per 64) costs more than occupancy gains; 8x8 (4 LDS.128 per 64) is the sweet spot for the LSU. Occupancy is not the lever here. Code reverted to 8x8.

### 2026-11-24 BK=32 tried, slower than BK=16
- 4096: 7140 GFLOPS (93.3% of cublas, three runs 7153, 7136, 7138; cublas 7646-7670) vs 8330 at BK=16. 128 regs, 33.3 KB smem, still 2 blocks/SM, no spills.
- ncu (2048): 3.57 ms (BK16 2.97). Instructions 327M (BK16 301M), issue slots 67%, FMA 57%. Stalls vs BK16: barrier 62k to 27k and long scoreboard 61k to 38k (the targeted stalls did drop), but short scoreboard 31k to 151k and dispatch 81k to 98k.
- Reading (hypothesis, SASS not inspected): with a 32-step unrolled inner loop at the 128-register cap the compiler no longer overlaps LDS for the next k-step with the current FFMAs, and extra address/transposed-store instructions inflate the count. BK=16 is the sweet spot; code stays at BK=16.

### 2026-11-24 size-8192 profile: bottleneck, dispatch stall in SASS, interleaved LDS tried
- 8192 profile: FMA pipe 68%, issue slots 76%, FFMA is 89% of 19.2G instructions, L1 64%, DRAM 34%, L2 28%. The kernel is FP32-issue bound; ceiling by instruction mix is about 85-89% of peak. The 4% excess shared-store wavefronts (2.4-way conflict) come from the transposed a_tile stores (2-way, lanes q=0/2 and q=1/3 share banks; no row padding fixes it since the stride must stay a multiple of 4). Fix would be `a_row = tid/2`, `a_k = (tid%2 + 2*l)*VEC`; gain ceiling about 4% of shared wavefronts, not pursued.
- Stalls per issued instruction (5.22 total): not_selected 2.26, dispatch 0.56, barrier 0.43, long scoreboard 0.41, short scoreboard 0.22, wait 0.17, mio 0.11.
- SASS of dispatch stall (397k samples): FFMA 48%, LDS 30%. Hot spots are the second LDS.128 of back-to-back pairs and the first FFMA consuming fresh LDS data (43% of FFMA samples sit on about 4 FFMAs per k-step). No FFMA carries `.reuse`. FFMAs whose A and B operands have the same register parity average about 270 samples vs about 103 for different parity (2.6x), consistent with register-bank contention but not confirmed (ncu has no counter for it).
- The `l` load loop (LOADS=2) and the k loop are already fully unrolled by the compiler; `#pragma unroll` would be a no-op.
- Tried interleaving the LDS.128 with FFMAs (explicit double-buffered fragments, next step loaded in 4 pieces after each pair of accumulator rows): 4096, 7740 GFLOPS (101.7% of cublas) vs 8222 (108.1%) same session. 127 regs, no spills. ptxas still clusters some loads (runs of LLL, LLLLL), so source order is not binding. Slower, code reverted.
- Also tried splitting the next-step loads into two sub-steps (A loads after rows 0-3, B loads after rows 4-7): 4096, 7674 and 7652 GFLOPS (102.0 and 101.6% of cublas) vs 8222 baseline. The SASS LDS/FFMA pattern is identical to the 4-piece version, so ptxas reschedules by its own rules regardless of source order and the loss comes from the explicit double-buffered fragments themselves. Source-level load placement is not a lever; code reverted.
