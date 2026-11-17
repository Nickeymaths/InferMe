#include <cstdio>
#include "benchmark.cuh"
#include "cuda_check.cuh"

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("GPU: %s (sm_%d%d)\n", properties.name, properties.major, properties.minor);

    const int n = 1 << 20;
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, n * sizeof(float)));
    for (int i = 0; i < n; i++) { a[i] = 1.f; b[i] = 2.f; }

    auto launch = [&] { add<<<(n + 255) / 256, 256>>>(a, b, c, n); };
    float milliseconds = average_milliseconds(launch);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("c[0]=%.1f c[n-1]=%.1f %s, %.3f ms\n", c[0], c[n - 1],
                (c[0] == 3.f && c[n - 1] == 3.f) ? "OK" : "FAIL", milliseconds);
    cudaFree(a); cudaFree(b); cudaFree(c);
}
