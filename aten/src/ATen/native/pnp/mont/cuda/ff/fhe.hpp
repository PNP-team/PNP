#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

static __device__ __constant__ __align__(8) const uint32_t fhe_moduli1_r[2] = {
    TO_CUDA_T(0x0ffffffffffc0001)
    };
static __device__ __constant__ __align__(8) const uint32_t
    fhe_moduli1_rRR[2] = {/* (1<<512)%P */
                        TO_CUDA_T(0x0000003fffe00004)
    };
static __device__ __constant__ __align__(8) const uint32_t
    fhe_moduli1_rone[2] = {/* (1<<256)%P */
                         TO_CUDA_T(0x000000000007fffe)
    };
static __device__ __constant__ __align__(8) const uint32_t fhe_moduli1_rx3[2] = {
    /* left-aligned value of the modulus */
    TO_CUDA_T(0x00000000001ffff8)
    };
static __device__ __constant__ /*const*/ uint32_t fhe_moduli1_m0 = 0xfffbffff;
typedef mont_t<
    59,
    fhe_moduli1_r,
    fhe_moduli1_m0,
    fhe_moduli1_rRR,
    fhe_moduli1_rone,
    fhe_moduli1_rx2>
    fhe_moduli1_mont;
struct fhe_moduli1 : public fhe_moduli1_mont {
  using mem_t = fhe_moduli1;
  __device__ __forceinline__ fhe_moduli1() {}
  __device__ __forceinline__ fhe_moduli1(const fhe_moduli1_mont& a)
      : fhe_moduli1_mont(a) {}
};

} // namespace native
} // namespace at
