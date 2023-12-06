#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "ATen/native/biginteger/cuda/sppark-ntt/parameters.cuh"
#include "ATen/native/biginteger/cuda/sppark-ntt/kernels.cuh"
namespace at { 
namespace native {

void CT_NTT(BLS12_381_Fr_G1* d_inout, const int lg_domain_size, const bool is_intt,
            const NTTParameters& ntt_parameters, const cudaStream_t& stream);

void GS_NTT(BLS12_381_Fr_G1* d_inout, const int lg_domain_size, const bool is_intt,
    const NTTParameters& ntt_parameters, const cudaStream_t& stream);

}//namespace native
}//namespace at