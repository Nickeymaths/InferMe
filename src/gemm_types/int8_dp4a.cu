#include <cstdint>
#include "compare.cuh"
#include "tiled_kernel.cuh"

void cublas_int8(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k);

constexpr int K_TILE = 32;
constexpr int K_GROUPS = K_TILE / 4;  // one group is 4 consecutive k packed in an int, the operand of __dp4a
static_assert(BLOCK_M * K_TILE == 16 * BLOCK_THREADS, "each thread loads 16 bytes of A per tile");
static_assert(K_GROUPS * BLOCK_N / 4 == BLOCK_THREADS, "each thread packs 4 columns of one k group of B");

// Requires m and n to be multiples of 128 and k a multiple of 32.
__global__ void dp4a_kernel(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k) {
    __shared__ __align__(16) int a_tile[K_GROUPS][BLOCK_M + 4];
    __shared__ __align__(16) int b_tile[K_GROUPS][BLOCK_N];

    int tid = threadIdx.y * THREADS + threadIdx.x;
    const int8_t* a_block = a + (size_t)blockIdx.y * BLOCK_M * k;
    const int8_t* b_block = b + blockIdx.x * BLOCK_N;
    int a_row = tid / 2, a_group = tid % 2 * 4;
    int b_group = tid / (BLOCK_N / 4), b_col = tid % (BLOCK_N / 4) * 4;

    alignas(16) int acc[TM][TN] = {};
    for (int k0 = 0; k0 < k; k0 += K_TILE) {
        int4 a_words = *reinterpret_cast<const int4*>(&a_block[(size_t)a_row * k + k0 + a_group * 4]);
        a_tile[a_group + 0][a_row] = a_words.x;
        a_tile[a_group + 1][a_row] = a_words.y;
        a_tile[a_group + 2][a_row] = a_words.z;
        a_tile[a_group + 3][a_row] = a_words.w;

        // B is [k][n]; transpose the 4x4 bytes (4 k rows by 4 columns) so each column's 4 k values share an int.
        int row[4];
        for (int q = 0; q < 4; q++)
            row[q] = *reinterpret_cast<const int*>(&b_block[(size_t)(k0 + b_group * 4 + q) * n + b_col]);
        int low_k01 = __byte_perm(row[0], row[1], 0x5140), high_k01 = __byte_perm(row[0], row[1], 0x7362);
        int low_k23 = __byte_perm(row[2], row[3], 0x5140), high_k23 = __byte_perm(row[2], row[3], 0x7362);
        *reinterpret_cast<int4*>(&b_tile[b_group][b_col]) =
            make_int4(__byte_perm(low_k01, low_k23, 0x5410), __byte_perm(low_k01, low_k23, 0x7632),
                      __byte_perm(high_k01, high_k23, 0x5410), __byte_perm(high_k01, high_k23, 0x7632));
        __syncthreads();

        for (int g = 0; g < K_GROUPS; g++) {
            alignas(16) int a_frag[TM], b_frag[TN];
            for (int r = 0; r < TM; r++) a_frag[r] = a_tile[g][threadIdx.y * TM + r];
            for (int h = 0; h < TN / VEC; h++)
                reinterpret_cast<int4*>(b_frag)[h] =
                    *reinterpret_cast<const int4*>(&b_tile[g][threadIdx.x * VEC + h * VEC * THREADS]);
            for (int r = 0; r < TM; r++)
                for (int j = 0; j < TN; j++) acc[r][j] = __dp4a(a_frag[r], b_frag[j], acc[r][j]);
        }
        __syncthreads();
    }

    int row0 = blockIdx.y * BLOCK_M + threadIdx.y * TM;
    int col0 = blockIdx.x * BLOCK_N + threadIdx.x * VEC;
    for (int r = 0; r < TM; r++)
        for (int h = 0; h < TN / VEC; h++)
            *reinterpret_cast<int4*>(&c[(size_t)(row0 + r) * n + col0 + h * VEC * THREADS]) =
                reinterpret_cast<int4*>(acc[r])[h];
}

void dp4a_gemm(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k) {
    dim3 block(THREADS, THREADS);
    dim3 grid(n / BLOCK_N, m / BLOCK_M);
    dp4a_kernel<<<grid, block>>>(a, b, c, m, n, k);
}

void run_int8_dp4a(int size) {
    compare_with_cublas<int8_t, int32_t>(
        "int8 dp4a in, int32 accumulate and out", size,
        [](float x) { return static_cast<int8_t>(std::lrintf(x * 127.f)); }, 0,
        dp4a_gemm, cublas_int8, true);
}

REGISTER_TYPE(int8_dp4a, run_int8_dp4a);
