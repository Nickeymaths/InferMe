#include "registry.cuh"

__global__ void naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.f;
    for (int i = 0; i < k; i++) sum += a[row * k + i] * b[i * n + col];
    c[row * n + col] = sum;
}

void naive(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    naive_kernel<<<grid, block>>>(a, b, c, m, n, k);
}

REGISTER_GEMM(naive, naive);
