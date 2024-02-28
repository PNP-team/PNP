#pragma once
#include "exception.hpp"

using cuda_error = sppark_error;

#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess) {                              \
        auto file = std::strstr(__FILE__, "sppark");        \
        auto str = fmt("%s@%s:%d failed: \"%s\"", #expr,    \
                       file ? file : __FILE__, __LINE__,    \
                       cudaGetErrorString(code));           \
        throw cuda_error{-code, str};                       \
    }                                                       \
} while(0)


