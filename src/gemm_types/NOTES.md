# gemm_types

Goal: for BF16, TF32 and INT8, compare our CUDA-core GEMM (the `gemm/shared_memory_tiling` kernel, BK=16, 8x8 thread tile) against cuBLAS on tensor cores. Row-major square, RTX 3060 (sm_86).
Run: `docker compose run --rm cuda ./run.sh gemm_types [--size N] [bf16|tf32|int8 ...]` (default size 4096).
Layout: `tiled_kernel.cuh` (one kernel template, inputs converted to the accumulator type when stored to shared memory), `compare.cuh` (registry, data, timing, error check), one file per type (`bf16.cu`, `tf32.cu`, `int8.cu`) holding its cuBLAS call and `REGISTER_TYPE`.

## Results (4096, Gop/s = 2*m*n*k / time, three runs each)

| Type (in / accumulate) | cuBLAS (tensor core) | cuda_core | % of cuBLAS | max err vs cuBLAS | Notes |
|---|---|---|---|---|---|
| tf32 / fp32 | 13290 | 8252, 8287, 8284 | 62 | 1.9e-3 | cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32`. Our kernel is the FP32 kernel on inputs with the low 13 mantissa bits cleared (8330 in gemm for plain fp32, so the template costs nothing) |
| bf16 / fp32 | 24255, 26265, 26211 | 8706, 8704, 8718 | 33 to 36 | 9.8e-4 | cuBLAS kernel `ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nn`, tensor pipe used. Ours converts bf16 to float on the shared-memory store |
| int8 / int32 | 89410, 88304, 89157 | 6479, 6475, 6495 | 7.2 to 7.3 | 0 (exact) | cuBLAS with B transposed (TN), tensor pipe used. Ours is plain int multiply-add (IMAD), no dp4a |

Run to run, cuBLAS bf16 moved 24255 to 26265; the other rows are within 1%.

## Backlog
- int8 CUDA-core with `__dp4a` (4 MACs per instruction); needs A and B packed along k, so B must be transposed when staged into shared memory. cuBLAS's own dp4a kernel reached 25.9 Tops (below), which is the target to beat.
- Hand-written tensor-core kernels (`mma.sync` m16n8k16 for bf16 and int8, m16n8k8 for tf32) to see how much of cuBLAS a simple kernel reaches.

## Research and Q&A log

### 2026-11-24 cuBLAS int8 only uses tensor cores for the TN layout
- First version called `cublasGemmEx` int8 with OP_N, OP_N (same as bf16 and tf32). ncu showed kernel `ampere_igemm_int8_128x128_ldg4_nn` with `sm__inst_executed_pipe_tensor` = 0: a CUDA-core (dp4a) kernel. It ran 25.9 Tops at 4096, similar to the bf16 tensor-core number, which would have hidden the problem.
- With A as op T (B passed transposed, n x k row-major, ld = k) cuBLAS picks a kernel with tensor instructions (2.1M at 2048) and runs 86 to 89 Tops, 3.4x faster. `compare_with_cublas(..., cublas_wants_transposed_b=true)` does the transpose once outside the timing. Always check `sm__inst_executed_pipe_tensor` when a cuBLAS type is claimed to be tensor-core.
- bf16 (`s1688gemm_bf16`) and tf32 show tensor instructions with plain row-major.

### 2026-11-24 CUDA-core numbers
- int8 cuda_core at 6480 Gop/s is about half of the 12953 FP32 peak (6476). Consistent with INT32 IMAD running at 64 lanes per SM instead of 128 (hypothesis from the arithmetic, not profiled).
- bf16 cuda_core (8706) is a bit above the FP32 kernel (8330, tf32 row 8270): same FMA count but half the global load bytes. Not profiled.
- tf32 and bf16 errors vs cuBLAS (1.9e-3, 9.8e-4 on outputs of magnitude ~20) are accumulation-order differences, since both sides multiply identical rounded inputs.

### 2026-11-24 why bf16 converts to fp32 on CUDA cores
- The only native CUDA-core bf16 FMA is packed `__hfma2_bf16`, whose accumulator is bf16 too. No CUDA-core instruction does bf16 inputs into an fp32 accumulator (that is the tensor core form). cuBLAS bf16 accumulates in fp32, and a bf16 accumulator over k=4096 (partial sums near 20, bf16 step near 0.1) is mostly rounding noise, so fp32 FMA after a once-per-tile convert is the comparable kernel.
- Int8 is the opposite case: `__dp4a` keeps an exact int32 accumulator, so it is a valid CUDA-core improvement (backlog).

### 2026-11-24 harness checks
- Error check keeps NaN and flags WRONG on `!(error <= tolerance)`. Tolerance is `tolerance_per_k * k` with 2e-6 for bf16 and tf32 (error grows slightly faster than linear in k; 8192: tf32 5.8e-3, bf16 2.6e-3, limit 1.6e-2). Size must be a positive multiple of 128; unknown type names and a missing `--size` value exit 1.
