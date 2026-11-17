#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(status));                           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "%s:%d cublas error %d\n", __FILE__, __LINE__, \
                         static_cast<int>(status));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
