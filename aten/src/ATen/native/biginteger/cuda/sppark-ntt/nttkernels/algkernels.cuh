#pragma once
#include "ATen/native/biginteger/cuda/CurveDef.cuh"
#include "ATen/native/biginteger/cuda/sppark-ntt/ntt_config.h"
#include <cooperative_groups.h>
#pragma diag_suppress 607
namespace at { 
namespace native {

__device__ __forceinline__
index_t bit_rev(index_t i, unsigned int nbits)
{
    if (sizeof(i) == 4 || nbits <= 32)
        return __brev(i) >> (8*sizeof(unsigned int) - nbits);
    else
        return __brevll(i) >> (8*sizeof(unsigned long long) - nbits);
}

template <typename fr_t>
__device__ __forceinline__
void shfl_bfly(fr_t& r, int laneMask)
{
    #pragma unroll
    for (int iter = 0; iter < r.len(); iter++)
        r[iter] = __shfl_xor_sync(0xFFFFFFFF, r[iter], laneMask);
}

template <typename fr_t>
__device__ __forceinline__
void get_intermediate_roots(fr_t& root0, fr_t& root1,
                            index_t idx0, index_t idx1,
                            const fr_t *roots)
{
    int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
    int off = (WINDOW_NUM - 1);

    root0 = roots[off * WINDOW_SIZE + idx0 >> win];
    root1 = roots[off * WINDOW_SIZE + idx1 >> win];
    #pragma unroll 1
    while (off--) {
        win -= LG_WINDOW_SIZE;
        root0 *= roots[off * WINDOW_SIZE + (idx0 >> win) % WINDOW_SIZE];
        root1 *= roots[off * WINDOW_SIZE + (idx1 >> win) % WINDOW_SIZE];
    }
}

}//namespace native
}//namespace at
