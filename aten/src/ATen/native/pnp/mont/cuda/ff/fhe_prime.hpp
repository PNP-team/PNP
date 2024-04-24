#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

#define TO_CUDA_T(limb64)                    \
  (uint32_t)(uint64_t(limb64) & 0xffffffff), \
      (uint32_t)((uint64_t(limb64) >> 32) & 0xffffffff)

#define PRIME_PARAM(name, r, rRR, rone, rx32, m0)         \
  static __device__ __constant__ __align__(16)           \
      const uint32_t name##_r[2] = {TO_CUDA_T(r)};       \
  static __device__ __constant__ __align__(16)           \
      const uint32_t name##_rRR[2] = {TO_CUDA_T(rRR)};   \
  static __device__ __constant__ __align__(16)           \
      const uint32_t name##_rone[2] = {TO_CUDA_T(rone)}; \
  static __device__ __constant__ __align__(16)           \
      const uint32_t name##_rx32[2] = {TO_CUDA_T(rx32)};   \
  static __device__ __constant__ /*const*/ uint32_t name##_m0 = m0;

#define TEMP_PRIME_PARAM(name) \
  PRIME_PARAM(                 \
      name,                    \
      0x0ffffffffffc0001,      \
      0x00000ffff8000100,      \
      0x00000000003ffff0,      \
      0xffffffffff800020,      \
      0xfffbffff)

// static __device__ __constant__ __align__(8) const uint32_t FHE_PRIME_0_r[2] =
// {TO_CUDA_T(0x0)}; static __device__ __constant__ __align__(8) const uint32_t
// FHE_PRIME_0_rRR[2] = {TO_CUDA_T(0x0)}; static __device__ __constant__
// __align__(8) const uint32_t FHE_PRIME_0_rone[2] = {TO_CUDA_T(0x0)}; static
// __device__ __constant__ __align__(8) const uint32_t FHE_PRIME_0_rx8[2] =
// {TO_CUDA_T(0x0)}; static __device__ __constant__ /*const*/ uint32_t
// FHE_PRIME_0_m0 = 0xffffffff;

#define DEF_FHE_PRIME(name)                                                    \
  typedef mont_t<59, name##_r, name##_m0, name##_rRR, name##_rone, name##_rx32> \
      name##_mont;                                                             \
  struct name : public name##_mont {                                           \
    using mem_t = name;                                                        \
    __device__ __forceinline__ name() {}                                       \
    __device__ __forceinline__ name(const name##_mont& a) : name##_mont(a) {}  \
  };

APPLY_ALL_FHE_PRIME(TEMP_PRIME_PARAM);
APPLY_ALL_FHE_PRIME(DEF_FHE_PRIME);

} // namespace native
} // namespace at
