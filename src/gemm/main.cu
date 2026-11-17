#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "benchmark.cuh"
#include "cuda_check.cuh"
#include "registry.cuh"

constexpr float tolerance = 1e-3f;

cublasHandle_t cublas_handle;

void cublas_reference(const float* a, const float* b, float* c, int m, int n, int k) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, b, n, a, k, &beta, c, n));
}

float max_abs_difference(const std::vector<float>& x, const std::vector<float>& y) {
    float largest = 0.f;
    for (size_t i = 0; i < x.size(); i++) largest = std::max(largest, std::fabs(x[i] - y[i]));
    return largest;
}

std::vector<float> random_matrix(size_t count, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    return values;
}

int main(int argc, char** argv) {
    int size = 2048;
    std::vector<std::string> requested_names;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = std::atoi(argv[++i]);
        else requested_names.push_back(argv[i]);
    }

    std::vector<GemmEntry> entries = gemm_registry();
    std::sort(entries.begin(), entries.end(),
              [](const GemmEntry& x, const GemmEntry& y) { return x.name < y.name; });
    if (!requested_names.empty()) {
        std::vector<GemmEntry> selected;
        for (const std::string& name : requested_names) {
            auto found = std::find_if(entries.begin(), entries.end(),
                                      [&](const GemmEntry& entry) { return entry.name == name; });
            if (found == entries.end()) {
                std::fprintf(stderr, "unknown gemm '%s'. available:", name.c_str());
                for (const GemmEntry& entry : entries) std::fprintf(stderr, " %s", entry.name.c_str());
                std::fprintf(stderr, "\n");
                return EXIT_FAILURE;
            }
            selected.push_back(*found);
        }
        entries = selected;
    }

    const int m = size, n = size, k = size;
    const double flops = 2.0 * m * n * k;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    std::vector<float> host_a = random_matrix((size_t)m * k, 1);
    std::vector<float> host_b = random_matrix((size_t)k * n, 2);
    std::vector<float> host_c((size_t)m * n), host_reference((size_t)m * n);

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, host_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, host_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, host_c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublas_reference(a, b, c, m, n, k);
    CUDA_CHECK(cudaMemcpy(host_reference.data(), c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float cublas_ms = average_milliseconds([&] { cublas_reference(a, b, c, m, n, k); });
    double cublas_gflops = flops / (cublas_ms * 1e6);

    std::printf("size %dx%dx%d\n", m, n, k);
    std::printf("%-16s %10s %10s %10s %10s\n", "name", "ms", "GFLOPS", "%cublas", "max_err");
    std::printf("%-16s %10.3f %10.1f %10.1f %10s\n", "cublas", cublas_ms, cublas_gflops, 100.0, "-");

    for (const GemmEntry& entry : entries) {
        CUDA_CHECK(cudaMemset(c, 0, host_c.size() * sizeof(float)));
        entry.function(a, b, c, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_c.data(), c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float error = max_abs_difference(host_c, host_reference);

        float ms = average_milliseconds([&] { entry.function(a, b, c, m, n, k); });
        double gflops = flops / (ms * 1e6);
        std::printf("%-16s %10.3f %10.1f %10.1f %10.2e%s\n", entry.name.c_str(), ms, gflops,
                    100.0 * gflops / cublas_gflops, error, error > tolerance ? "  WRONG" : "");
    }

    cudaFree(a); cudaFree(b); cudaFree(c);
    cublasDestroy(cublas_handle);
}
