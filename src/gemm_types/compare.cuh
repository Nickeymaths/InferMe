#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "benchmark.cuh"
#include "cuda_check.cuh"

inline cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t created;
        CUBLAS_CHECK(cublasCreate(&created));
        return created;
    }();
    return handle;
}

struct TypeEntry {
    std::string name;
    void (*run)(int size);
};

inline std::vector<TypeEntry>& type_registry() {
    static std::vector<TypeEntry> entries;
    return entries;
}

struct TypeRegistration {
    TypeRegistration(const char* name, void (*run)(int size)) { type_registry().push_back({name, run}); }
};

#define REGISTER_TYPE(name, function) \
    static TypeRegistration registration_##name(#name, function)

template <typename In, typename Out>
using GemmFunction = void (*)(const In* a, const In* b, Out* c, int m, int n, int k);

// Row-major C = A * B with A m x k and B k x n, our kernel against a cuBLAS call on the same data.
// With cublas_wants_transposed_b, the cuBLAS function gets B as n x k row-major instead.
template <typename In, typename Out>
void compare_with_cublas(const char* name, int size, In (*make_input)(float), double tolerance_per_k,
                         GemmFunction<In, Out> ours, GemmFunction<In, Out> cublas,
                         bool cublas_wants_transposed_b = false) {
    const int m = size, n = size, k = size;
    const double ops = 2.0 * m * n * k;
    const double tolerance = tolerance_per_k * k;

    std::mt19937 generator(1);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    auto random_matrix = [&](size_t count) {
        std::vector<In> values(count);
        for (In& value : values) value = make_input(distribution(generator));
        return values;
    };
    std::vector<In> host_a = random_matrix((size_t)m * k), host_b = random_matrix((size_t)k * n);
    std::vector<Out> host_ours((size_t)m * n), host_cublas((size_t)m * n);

    In *a, *b;
    Out* c;
    CUDA_CHECK(cudaMalloc(&a, host_a.size() * sizeof(In)));
    CUDA_CHECK(cudaMalloc(&b, host_b.size() * sizeof(In)));
    CUDA_CHECK(cudaMalloc(&c, host_ours.size() * sizeof(Out)));
    CUDA_CHECK(cudaMemcpy(a, host_a.data(), host_a.size() * sizeof(In), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host_b.data(), host_b.size() * sizeof(In), cudaMemcpyHostToDevice));

    In* cublas_b = b;
    if (cublas_wants_transposed_b) {
        std::vector<In> host_b_transposed((size_t)n * k);
        for (int row = 0; row < k; row++)
            for (int col = 0; col < n; col++) host_b_transposed[(size_t)col * k + row] = host_b[(size_t)row * n + col];
        CUDA_CHECK(cudaMalloc(&cublas_b, host_b_transposed.size() * sizeof(In)));
        CUDA_CHECK(cudaMemcpy(cublas_b, host_b_transposed.data(), host_b_transposed.size() * sizeof(In), cudaMemcpyHostToDevice));
    }

    cublas(a, cublas_b, c, m, n, k);
    CUDA_CHECK(cudaMemcpy(host_cublas.data(), c, host_cublas.size() * sizeof(Out), cudaMemcpyDeviceToHost));
    float cublas_ms = average_milliseconds([&] { cublas(a, cublas_b, c, m, n, k); });
    double cublas_rate = ops / (cublas_ms * 1e6);

    CUDA_CHECK(cudaMemset(c, 0, host_ours.size() * sizeof(Out)));
    ours(a, b, c, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(host_ours.data(), c, host_ours.size() * sizeof(Out), cudaMemcpyDeviceToHost));
    double error = 0;
    for (size_t i = 0; i < host_ours.size(); i++) {
        double difference = std::fabs((double)host_ours[i] - (double)host_cublas[i]);
        if (!(difference <= error)) error = difference;  // also keeps NaN
    }
    float ours_ms = average_milliseconds([&] { ours(a, b, c, m, n, k); });
    double ours_rate = ops / (ours_ms * 1e6);

    std::printf("%s, size %dx%dx%d\n", name, m, n, k);
    std::printf("  %-10s %10s %12s %10s %10s\n", "", "ms", "Gop/s", "%cublas", "max_err");
    std::printf("  %-10s %10.3f %12.1f %10.1f %10s\n", "cublas", cublas_ms, cublas_rate, 100.0, "-");
    std::printf("  %-10s %10.3f %12.1f %10.1f %10.2e%s\n", "cuda_core", ours_ms, ours_rate,
                100.0 * ours_rate / cublas_rate, error, error <= tolerance ? "" : "  WRONG");

    cudaFree(a); cudaFree(b); cudaFree(c);
    if (cublas_wants_transposed_b) cudaFree(cublas_b);
}
