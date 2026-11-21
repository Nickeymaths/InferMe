#include "registry.cuh"

constexpr int THREADS = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_M = THREADS * TM;
constexpr int BLOCK_N = THREADS * TN;
constexpr int BK = 8;
constexpr int BLOCK_THREADS = THREADS * THREADS;
constexpr int HALF_N = BLOCK_N / 2;

static_assert(BLOCK_M * BK == 4 * BLOCK_THREADS && BK * BLOCK_N == 4 * BLOCK_THREADS,
              "each thread loads exactly one float4 of A and one of B per tile");

__device__ float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ void store4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

// Requires m and n to be multiples of 128 and k a multiple of 8.
__global__ void shared_memory_tiling_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ __align__(16) float a_tile[BK][BLOCK_M];  // transposed so a thread's 8 rows are contiguous
    __shared__ __align__(16) float b_tile[BK][BLOCK_N];

    int tid = threadIdx.y * THREADS + threadIdx.x;
    const float* a_block = a + blockIdx.y * BLOCK_M * k;
    const float* b_block = b + blockIdx.x * BLOCK_N;

    int a_row = tid / 2, a_k = tid % 2 * 4;
    int b_k = tid / 32, b_col = tid % 32 * 4;

    alignas(16) float acc[TM][TN] = {};
    for (int k0 = 0; k0 < k; k0 += BK) {
        float4 a_vals = load4(&a_block[a_row * k + k0 + a_k]);
        a_tile[a_k + 0][a_row] = a_vals.x;
        a_tile[a_k + 1][a_row] = a_vals.y;
        a_tile[a_k + 2][a_row] = a_vals.z;
        a_tile[a_k + 3][a_row] = a_vals.w;
        store4(&b_tile[b_k][b_col], load4(&b_block[(k0 + b_k) * n + b_col]));
        __syncthreads();

        for (int i = 0; i < BK; i++) {
            alignas(16) float a_frag[TM], b_frag[TN];
            store4(&a_frag[0], load4(&a_tile[i][threadIdx.y * TM]));
            store4(&a_frag[4], load4(&a_tile[i][threadIdx.y * TM + 4]));
            store4(&b_frag[0], load4(&b_tile[i][threadIdx.x * 4]));
            store4(&b_frag[4], load4(&b_tile[i][HALF_N + threadIdx.x * 4]));
            for (int r = 0; r < TM; r++)
                for (int j = 0; j < TN; j++) acc[r][j] += a_frag[r] * b_frag[j];
        }
        __syncthreads();
    }

    int row0 = blockIdx.y * BLOCK_M + threadIdx.y * TM;
    int col0 = blockIdx.x * BLOCK_N + threadIdx.x * 4;
    for (int r = 0; r < TM; r++) {
        store4(&c[(row0 + r) * n + col0], load4(&acc[r][0]));
        store4(&c[(row0 + r) * n + col0 + HALF_N], load4(&acc[r][4]));
    }
}

void shared_memory_tiling(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(THREADS, THREADS);
    dim3 grid(n / BLOCK_N, m / BLOCK_M);
    shared_memory_tiling_kernel<<<grid, block>>>(a, b, c, m, n, k);
}

REGISTER_GEMM(shared_memory_tiling, shared_memory_tiling);
