#include <cuda_bf16.h>
#include "compare.cuh"
#include "tiled_kernel.cuh"

void cublas_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m, int n, int k) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              b, CUDA_R_16BF, n, a, CUDA_R_16BF, k, &beta, c, CUDA_R_32F, n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void run_bf16(int size) {
    compare_with_cublas<__nv_bfloat16, float>(
        "bf16 in, fp32 accumulate and out", size,
        [](float x) { return __float2bfloat16(x); }, 1e-2,
        tiled_gemm<__nv_bfloat16, float>, cublas_bf16);
}

REGISTER_TYPE(bf16, run_bf16);
