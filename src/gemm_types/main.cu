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
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = std::atoi(argv[++i]);
        else requested_names.push_back(argv[i]);
    }

    std::vector<TypeEntry> entries = type_registry();
    std::sort(entries.begin(), entries.end(),
              [](const TypeEntry& x, const TypeEntry& y) { return x.name < y.name; });
    for (const TypeEntry& entry : entries) {
        bool requested = requested_names.empty() ||
                         std::find(requested_names.begin(), requested_names.end(), entry.name) != requested_names.end();
        if (requested) entry.run(size);
    }
}
