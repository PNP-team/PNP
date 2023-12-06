#pragma once

#include <cassert>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/sppark-util/gpu_t.cuh"
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "kernels/algorithm.cuh"
#include <cuda.h>

namespace at { 
namespace native {

enum class InputOutputOrder { NN, NR, RN, RR };
enum class Direction { forward, inverse };
enum class Type { standard, coset };
enum class Algorithm { GS, CT };


static void bit_rev(BLS12_381_Fr_G1* d_out, const BLS12_381_Fr_G1* d_inp,
                    uint32_t lg_domain_size, stream_t& stream);


static void LDE_powers(BLS12_381_Fr_G1* inout, bool innt, bool bitrev,
                        uint32_t lg_domain_size, uint32_t lg_blowup,
                        stream_t& stream, bool ext_pow = false);


static void NTT_internal(BLS12_381_Fr_G1* d_inout, uint32_t lg_domain_size,
                            InputOutputOrder order, Direction direction,
                            Type type, stream_t& stream,
                            bool coset_ext_pow = false);


static void Base(const gpu_t& gpu, BLS12_381_Fr_G1* inout, uint32_t lg_domain_size,
                        InputOutputOrder order, Direction direction,
                        Type type, bool coset_ext_pow = false);


void compute_ntt(size_t device_id, BLS12_381_Fr_G1* inout, uint32_t lg_domain_size,
                      InputOutputOrder ntt_order,
                      Direction ntt_direction,
                      Type ntt_type);
}//namespace native
}//namespace at