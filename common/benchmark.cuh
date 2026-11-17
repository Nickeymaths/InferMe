#pragma once
#include <cuda_runtime.h>
#include "cuda_check.cuh"

template <typename Launch>
float average_milliseconds(Launch launch, int warmup_runs = 3, int timed_runs = 20) {
    for (int i = 0; i < warmup_runs; i++) launch();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; i++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_milliseconds / timed_runs;
}
