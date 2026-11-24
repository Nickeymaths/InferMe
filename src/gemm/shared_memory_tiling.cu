#include "registry.cuh"

constexpr int THREADS = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_M = THREADS * TM;
constexpr int BLOCK_N = THREADS * TN;
constexpr int BK = 8;
constexpr int VEC = 4;
constexpr int BLOCK_THREADS = THREADS * THREADS;

// Requires m and n to be multiples of 128 and k a multiple of 8.
__global__ void shared_memory_tiling_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[BK][BLOCK_M+4];
    __shared__ float b_tile[BK][BLOCK_N];

    int tid = threadIdx.y * THREADS + threadIdx.x;
    const float* a_block = a + blockIdx.y * BLOCK_M * k;
    const float* b_block = b + blockIdx.x * BLOCK_N;

    alignas(16) float acc[TM][TN] = {};
    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int idx = tid; idx < BLOCK_M * BK; idx += BLOCK_THREADS)
            a_tile[idx % BK][idx / BK] = a_block[(idx / BK) * k + k0 + idx % BK];
        for (int idx = tid; idx < BK * BLOCK_N; idx += BLOCK_THREADS)
            b_tile[idx / BLOCK_N][idx % BLOCK_N] = b_block[(k0 + idx / BLOCK_N) * n + idx % BLOCK_N];
        __syncthreads();

        for (int i = 0; i < BK; i++) {
            alignas(16) float a_frag[TM], b_frag[TN];
            for (int r = 0; r < TM; r++) a_frag[r] = a_tile[i][threadIdx.y * TM + r];
            for (int h = 0; h < TN / VEC; h++)
                reinterpret_cast<float4*>(b_frag)[h] =
                    *reinterpret_cast<const float4*>(&b_tile[i][threadIdx.x * VEC + h * VEC * THREADS]);
            for (int r = 0; r < TM; r++)
                for (int j = 0; j < TN; j++) acc[r][j] += a_frag[r] * b_frag[j];
        }
        __syncthreads();
    }

    int row0 = blockIdx.y * BLOCK_M + threadIdx.y * TM;
    int col0 = blockIdx.x * BLOCK_N + threadIdx.x * VEC;
    for (int r = 0; r < TM; r++)
        for (int h = 0; h < TN / VEC; h++)
            *reinterpret_cast<float4*>(&c[(row0 + r) * n + col0 + h * VEC * THREADS]) =
                reinterpret_cast<float4*>(acc[r])[h];
}

void shared_memory_tiling(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(THREADS, THREADS);
    dim3 grid(n / BLOCK_N, m / BLOCK_M);
    shared_memory_tiling_kernel<<<grid, block>>>(a, b, c, m, n, k);
}

REGISTER_GEMM(shared_memory_tiling, shared_memory_tiling);
