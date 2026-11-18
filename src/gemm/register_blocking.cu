#include "registry.cuh"

constexpr int THREADS = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_M = THREADS * TM;
constexpr int BLOCK_N = THREADS * TN;

// Requires m and n to be multiples of 128.
__global__ void register_blocking_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row0 = blockIdx.y * BLOCK_M + threadIdx.y * TM;
    int col0 = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc[TM][TN] = {};
    for (int i = 0; i < k; i++) {
        float a_frag[TM], b_frag[TN];
        for (int r = 0; r < TM; r++) a_frag[r] = a[(row0 + r) * k + i];
        for (int j = 0; j < TN; j++) b_frag[j] = b[i * n + col0 + j * THREADS];
        for (int r = 0; r < TM; r++)
            for (int j = 0; j < TN; j++) acc[r][j] += a_frag[r] * b_frag[j];
    }
    for (int r = 0; r < TM; r++)
        for (int j = 0; j < TN; j++) c[(row0 + r) * n + col0 + j * THREADS] = acc[r][j];
}

void register_blocking(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(THREADS, THREADS);
    dim3 grid(n / BLOCK_N, m / BLOCK_M);
    register_blocking_kernel<<<grid, block>>>(a, b, c, m, n, k);
}

REGISTER_GEMM(register_blocking, register_blocking);
