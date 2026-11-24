#include "registry.cuh"

constexpr int THREADS = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_M = THREADS * TM;
constexpr int BLOCK_N = THREADS * TN;
constexpr int BK = 16;
constexpr int VEC = 4;
constexpr int BLOCK_THREADS = THREADS * THREADS;

constexpr int LOADS = BLOCK_M * BK / (VEC * BLOCK_THREADS);
static_assert(BLOCK_M * BK == BK * BLOCK_N && BLOCK_M * BK % (VEC * BLOCK_THREADS) == 0,
              "each thread loads LOADS whole float4s of A and of B per tile");

// Requires m and n to be multiples of 128 and k a multiple of 16.
__global__ void shared_memory_tiling_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ __align__(16) float a_tile[BK][BLOCK_M+4];
    __shared__ __align__(16) float b_tile[BK][BLOCK_N];

    int tid = threadIdx.y * THREADS + threadIdx.x;
    const float* a_block = a + blockIdx.y * BLOCK_M * k;
    const float* b_block = b + blockIdx.x * BLOCK_N;


    alignas(16) float acc[TM][TN] = {};
    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int l = 0; l < LOADS; l++) {
            int idx = tid + l * BLOCK_THREADS;
            int a_row = idx / (BK / VEC), a_k = idx % (BK / VEC) * VEC;
            int b_k = idx / (BLOCK_N / VEC), b_col = idx % (BLOCK_N / VEC) * VEC;
            float4 a_vals = *reinterpret_cast<const float4*>(&a_block[a_row * k + k0 + a_k]);
            a_tile[a_k + 0][a_row] = a_vals.x;
            a_tile[a_k + 1][a_row] = a_vals.y;
            a_tile[a_k + 2][a_row] = a_vals.z;
            a_tile[a_k + 3][a_row] = a_vals.w;
            *reinterpret_cast<float4*>(&b_tile[b_k][b_col]) =
                *reinterpret_cast<const float4*>(&b_block[(k0 + b_k) * n + b_col]);
        }
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
