#pragma once
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "ATen/native/biginteger/cuda/sppark-ntt/parameters.cuh"
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


__device__ __forceinline__
void shfl_bfly(BLS12_381_Fr_G1& r, int laneMask)
{
    #pragma unroll
    for (int iter = 0; iter < r.len(); iter++)
        r[iter] = __shfl_xor_sync(0xFFFFFFFF, r[iter], laneMask);
}

template<typename T>
__device__ __forceinline__
void swap(T& u1, T& u2)
{
    T temp = u1;
    u1 = u2;
    u2 = temp;
}

__device__ __forceinline__
void get_intermediate_roots(BLS12_381_Fr_G1& root0, BLS12_381_Fr_G1& root1,
                            index_t idx0, index_t idx1,
                            const BLS12_381_Fr_G1 *roots)
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
