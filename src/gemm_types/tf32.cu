#include "compare.cuh"
#include "tiled_kernel.cuh"

void cublas_tf32(const float* a, const float* b, float* c, int m, int n, int k) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              b, CUDA_R_32F, n, a, CUDA_R_32F, k, &beta, c, CUDA_R_32F, n,
                              CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
}

// cuBLAS rounds inputs to TF32 internally. Dropping the low 13 mantissa bits up front makes both sides multiply identical values.
float truncate_to_tf32(float x) {
    unsigned bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits &= 0xffffe000u;
    std::memcpy(&x, &bits, sizeof(bits));
    return x;
}

void run_tf32(int size) {
    compare_with_cublas<float, float>("tf32 in, fp32 accumulate and out", size, truncate_to_tf32, 1e-2,
                                      tiled_gemm<float, float>, cublas_tf32);
}

REGISTER_TYPE(tf32, run_tf32);
