#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "ATen/native/biginteger/cuda/sppark-ntt/parameters.cuh"
#include "ATen/native/biginteger/cuda/sppark-ntt/kernels.cuh"
namespace at { 
namespace native {

void CT_NTT(BLS12_381_Fr_G1* d_inout, const int lg_domain_size, const bool is_intt,
        const cudaStream_t& stream,
        BLS12_381_Fr_G1* partial_twiddles,
        BLS12_381_Fr_G1* radix_twiddles,
        BLS12_381_Fr_G1* radix_middles,
        BLS12_381_Fr_G1* partial_group_gen_powers,
        BLS12_381_Fr_G1* Domain_size_inverse);

void GS_NTT(BLS12_381_Fr_G1* d_inout, const int lg_domain_size, const bool is_intt,
        const cudaStream_t& stream,
        BLS12_381_Fr_G1* partial_twiddles,
        BLS12_381_Fr_G1* radix_twiddles,
        BLS12_381_Fr_G1* radix_middles,
        BLS12_381_Fr_G1* partial_group_gen_powers,
        BLS12_381_Fr_G1* Domain_size_inverse);

}//namespace native
}//namespace at