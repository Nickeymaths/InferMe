#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "compare.cuh"

int main(int argc, char** argv) {
    int size = 4096;
    std::vector<std::string> requested_names;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--size") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--size needs a value\n");
                return EXIT_FAILURE;
            }
            size = std::atoi(argv[++i]);
        } else {
            requested_names.push_back(argv[i]);
        }
    }
    if (size <= 0 || size % 128 != 0) {
        std::fprintf(stderr, "size must be a positive multiple of 128\n");
        return EXIT_FAILURE;
    }

    std::vector<TypeEntry> entries = type_registry();
    std::sort(entries.begin(), entries.end(),
              [](const TypeEntry& x, const TypeEntry& y) { return x.name < y.name; });
    auto is_registered = [&](const std::string& name) {
        return std::any_of(entries.begin(), entries.end(), [&](const TypeEntry& entry) { return entry.name == name; });
    };
    for (const std::string& name : requested_names) {
        if (is_registered(name)) continue;
        std::fprintf(stderr, "unknown type '%s'. available:", name.c_str());
        for (const TypeEntry& entry : entries) std::fprintf(stderr, " %s", entry.name.c_str());
        std::fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }

    for (const TypeEntry& entry : entries) {
        bool requested = requested_names.empty() ||
                         std::find(requested_names.begin(), requested_names.end(), entry.name) != requested_names.end();
        if (requested) entry.run(size);
    }
}
