#pragma once
#include <string>
#include <vector>

using GemmFunction = void (*)(const float* a, const float* b, float* c, int m, int n, int k);

struct GemmEntry {
    std::string name;
    GemmFunction function;
};

inline std::vector<GemmEntry>& gemm_registry() {
    static std::vector<GemmEntry> entries;
    return entries;
}

struct GemmRegistration {
    GemmRegistration(const char* name, GemmFunction function) {
        gemm_registry().push_back({name, function});
    }
};

#define REGISTER_GEMM(name, function) \
    static GemmRegistration registration_##name(#name, function)
