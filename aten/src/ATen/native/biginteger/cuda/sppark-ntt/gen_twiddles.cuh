#pragma once
#include <cassert>
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
namespace at { 
namespace native {

#define ONE BLS12_381_Fr_G1::one()

#define MAX_LG_DOMAIN_SIZE 28 // tested only up to 2^31 for now
typedef unsigned int index_t; //for MAX_LG_DOMAIN_SIZE <= 32, otherwise use size_t

#if MAX_LG_DOMAIN_SIZE <= 28
# define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2) 
#else
# define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 2) / 3)
#endif
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

__global__ 
void generate_partial_twiddles( BLS12_381_Fr_G1* roots,
                               const BLS12_381_Fr_G1* root_of_unity);

__global__
void generate_all_twiddles(BLS12_381_Fr_G1* d_radixX_twiddles, const BLS12_381_Fr_G1 *root6,
                                                    const BLS12_381_Fr_G1 *root7,
                                                    const BLS12_381_Fr_G1 *root8,
                                                    const BLS12_381_Fr_G1 *root9,
                                                    const BLS12_381_Fr_G1 *root10);

__launch_bounds__(512) __global__
void generate_radixX_twiddles_X(BLS12_381_Fr_G1* d_radixX_twiddles_X, int n,
                                const BLS12_381_Fr_G1 *root_of_unity);


}//namespace native
}//namespace at                                                                              