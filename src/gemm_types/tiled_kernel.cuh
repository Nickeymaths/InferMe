#pragma once
#include <cuda_runtime.h>

constexpr int THREADS = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_M = THREADS * TM;
constexpr int BLOCK_N = THREADS * TN;
constexpr int BK = 16;
constexpr int VEC = 4;
constexpr int BLOCK_THREADS = THREADS * THREADS;
constexpr int LOADS = BLOCK_M * BK / (VEC * BLOCK_THREADS);

template <typename T>
struct alignas(VEC * sizeof(T)) Vec4 { T v[VEC]; };

// Same kernel as gemm/shared_memory_tiling, on CUDA cores. Inputs are converted to the accumulator
// type when they enter shared memory, so the inner loop is identical for every type.
// Requires m and n to be multiples of 128 and k a multiple of 16.
template <typename In, typename Acc>
__global__ void tiled_kernel(const In* a, const In* b, Acc* c, int m, int n, int k) {
    __shared__ __align__(16) Acc a_tile[BK][BLOCK_M + 4];
    __shared__ __align__(16) Acc b_tile[BK][BLOCK_N];

    int tid = threadIdx.y * THREADS + threadIdx.x;
    const In* a_block = a + (size_t)blockIdx.y * BLOCK_M * k;
    const In* b_block = b + blockIdx.x * BLOCK_N;

    alignas(16) Acc acc[TM][TN] = {};
    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int l = 0; l < LOADS; l++) {
            int idx = tid + l * BLOCK_THREADS;
            int a_row = idx / (BK / VEC), a_k = idx % (BK / VEC) * VEC;
            int b_k = idx / (BLOCK_N / VEC), b_col = idx % (BLOCK_N / VEC) * VEC;
            Vec4<In> a_vals = *reinterpret_cast<const Vec4<In>*>(&a_block[(size_t)a_row * k + k0 + a_k]);
            Vec4<In> b_vals = *reinterpret_cast<const Vec4<In>*>(&b_block[(size_t)(k0 + b_k) * n + b_col]);
            Vec4<Acc> b_converted;
            for (int q = 0; q < VEC; q++) {
                a_tile[a_k + q][a_row] = static_cast<Acc>(a_vals.v[q]);
                b_converted.v[q] = static_cast<Acc>(b_vals.v[q]);
            }
            *reinterpret_cast<Vec4<Acc>*>(&b_tile[b_k][b_col]) = b_converted;
        }
        __syncthreads();

        for (int i = 0; i < BK; i++) {
            alignas(16) Acc a_frag[TM], b_frag[TN];
            for (int r = 0; r < TM; r++) a_frag[r] = a_tile[i][threadIdx.y * TM + r];
            for (int h = 0; h < TN / VEC; h++)
                reinterpret_cast<Vec4<Acc>*>(b_frag)[h] =
                    *reinterpret_cast<const Vec4<Acc>*>(&b_tile[i][threadIdx.x * VEC + h * VEC * THREADS]);
            for (int r = 0; r < TM; r++)
                for (int j = 0; j < TN; j++) acc[r][j] += a_frag[r] * b_frag[j];
        }
        __syncthreads();
    }

    int row0 = blockIdx.y * BLOCK_M + threadIdx.y * TM;
    int col0 = blockIdx.x * BLOCK_N + threadIdx.x * VEC;
    for (int r = 0; r < TM; r++)
        for (int h = 0; h < TN / VEC; h++)
            *reinterpret_cast<Vec4<Acc>*>(&c[(size_t)(row0 + r) * n + col0 + h * VEC * THREADS]) =
                reinterpret_cast<Vec4<Acc>*>(acc[r])[h];
}

template <typename In, typename Acc>
void tiled_gemm(const In* a, const In* b, Acc* c, int m, int n, int k) {
    dim3 block(THREADS, THREADS);
    dim3 grid(n / BLOCK_N, m / BLOCK_M);
    tiled_kernel<In, Acc><<<grid, block>>>(a, b, c, m, n, k);
}
