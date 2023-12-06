#pragma once
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "ATen/native/biginteger/cuda/sppark-ntt/parameters.cuh"
#include <cooperative_groups.h>
#pragma diag_suppress 607
namespace at { 
namespace native {
// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
__launch_bounds__(1024) __global__
void bit_rev_permutation(BLS12_381_Fr_G1* d_out, const BLS12_381_Fr_G1 *d_in, uint32_t lg_domain_size);

__launch_bounds__(1024) __global__
void bit_rev_permutation_aux(BLS12_381_Fr_G1* out, const BLS12_381_Fr_G1* in, uint32_t lg_domain_size);

__device__ __forceinline__
BLS12_381_Fr_G1 get_intermediate_root(index_t pow, const BLS12_381_Fr_G1 (*roots)[WINDOW_SIZE],
                           unsigned int nbits = MAX_LG_DOMAIN_SIZE);

__launch_bounds__(1024) __global__
void LDE_distribute_powers(BLS12_381_Fr_G1* d_inout, uint32_t lg_blowup, bool bitrev,
                           const BLS12_381_Fr_G1 (*gen_powers)[WINDOW_SIZE],
                           bool ext_pow = false);

__launch_bounds__(1024) __global__
void LDE_spread_distribute_powers(BLS12_381_Fr_G1* out, BLS12_381_Fr_G1* in,
                                  const BLS12_381_Fr_G1 (*gen_powers)[WINDOW_SIZE],
                                  uint32_t lg_domain_size, uint32_t lg_blowup);

}//namespace native
}//namespace at
