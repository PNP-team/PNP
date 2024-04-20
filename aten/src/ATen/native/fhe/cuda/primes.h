#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

// #define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

// static __device__ __constant__ __align__(16) const uint32_t PRIME1_r[2] = {
//     TO_CUDA_T(0x30644e72e131a029)};
// static __device__ __constant__ __align__(16) const uint32_t PRIME1_rRR[2] = {
//     TO_CUDA_T(0x0216d0b17f4e44a5)};
// static __device__ __constant__ __align__(16) const uint32_t PRIME1_rone[2] = {
//     TO_CUDA_T(0x0e0a77c19a07df2f)};
// static __device__ __constant__ __align__(16) const uint32_t PRIME1_rx4[2] = {
//     TO_CUDA_T(0xc19139cb84c680a6)};
// static __device__ __constant__ const uint32_t PRIME1_m0 = 0xefffffff;




} // namespace native
} // namespace at
