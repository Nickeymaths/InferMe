#include <cstdint>
#include "compare.cuh"
#include "tiled_kernel.cuh"

// cuBLAS only runs int8 on tensor cores for the TN layout: with plain row-major B it picks a CUDA-core dp4a kernel.
// Here b is B transposed (n x k row-major), passed as op(X) = T.
void cublas_int8(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k) {
    const int32_t alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              b, CUDA_R_8I, k, a, CUDA_R_8I, k, &beta, c, CUDA_R_32I, n,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}

void run_int8(int size) {
    compare_with_cublas<int8_t, int32_t>(
        "int8 in, int32 accumulate and out", size,
        [](float x) { return static_cast<int8_t>(std::lrintf(x * 127.f)); }, 0,
        tiled_gemm<int8_t, int32_t>, cublas_int8, true);
}

REGISTER_TYPE(int8, run_int8);
