#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[8] = {
    TO_CUDA_T(0x43e1f593f0000001),
    TO_CUDA_T(0x2833e84879b97091),
    TO_CUDA_T(0xb85045b68181585d),
    TO_CUDA_T(0x30644e72e131a029)};
static __device__ __constant__ __align__(16) const uint32_t
    ALT_BN128_rRR[8] = {/* (1<<512)%P */
                        TO_CUDA_T(0x1bb8e645ae216da7),
                        TO_CUDA_T(0x53fe3ab1e35c59e3),
                        TO_CUDA_T(0x8c49833d53bb8085),
                        TO_CUDA_T(0x0216d0b17f4e44a5)};
static __device__ __constant__ __align__(16) const uint32_t
    ALT_BN128_rone[8] = {/* (1<<256)%P */
                         TO_CUDA_T(0xac96341c4ffffffb),
                         TO_CUDA_T(0x36fc76959f60cd29),
                         TO_CUDA_T(0x666ea36f7879462e),
                         TO_CUDA_T(0x0e0a77c19a07df2f)};
static __device__ __constant__ __align__(16) const uint32_t
    ALT_BN128_rx4[8] = {/* left-aligned value of the modulus */
                        TO_CUDA_T(0x0f87d64fc0000004),
                        TO_CUDA_T(0xa0cfa121e6e5c245),
                        TO_CUDA_T(0xe14116da06056174),
                        TO_CUDA_T(0xc19139cb84c680a6)};
static __device__ __constant__ const uint32_t ALT_BN128_m0 = 0xefffffff;
typedef mont_t<
    254,
    ALT_BN128_r,
    ALT_BN128_m0,
    ALT_BN128_rRR,
    ALT_BN128_rone,
    ALT_BN128_rx4>
    alt_bn128_fr_mont;
struct ALT_BN128_Fr_G1 : public alt_bn128_fr_mont {
  using mem_t = ALT_BN128_Fr_G1;
  __device__ __forceinline__ ALT_BN128_Fr_G1() {}
  __device__ __forceinline__ ALT_BN128_Fr_G1(const alt_bn128_fr_mont& a)
      : alt_bn128_fr_mont(a) {}
};

static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[8] = {
    TO_CUDA_T(0x3c208c16d87cfd47),
    TO_CUDA_T(0x97816a916871ca8d),
    TO_CUDA_T(0xb85045b68181585d),
    TO_CUDA_T(0x30644e72e131a029)};
static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[8] = {
    /* (1<<512)%P */
    TO_CUDA_T(0xf32cfc5b538afa89),
    TO_CUDA_T(0xb5e71911d44501fb),
    TO_CUDA_T(0x47ab1eff0a417ff6),
    TO_CUDA_T(0x06d89f71cab8351f),
};
static __device__ __constant__ __align__(16) const uint32_t
    ALT_BN128_one[8] = {/* (1<<256)%P */
                        TO_CUDA_T(0xd35d438dc58f0d9d),
                        TO_CUDA_T(0x0a78eb28f5c70b3d),
                        TO_CUDA_T(0x666ea36f7879462c),
                        TO_CUDA_T(0x0e0a77c19a07df2f)};
static __device__ __constant__ __align__(16) const uint32_t
    ALT_BN128_Px4[8] = {/* left-aligned value of the modulus */
                        TO_CUDA_T(0xf082305b61f3f51c),
                        TO_CUDA_T(0x5e05aa45a1c72a34),
                        TO_CUDA_T(0xe14116da06056176),
                        TO_CUDA_T(0xc19139cb84c680a6)};
static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0xe4866389;
typedef mont_t<
    254,
    ALT_BN128_P,
    ALT_BN128_M0,
    ALT_BN128_RR,
    ALT_BN128_one,
    ALT_BN128_Px4>
    alt_bn128_fq_mont;

struct ALT_BN128_Fq_G1 : public alt_bn128_fq_mont {
  using mem_t = ALT_BN128_Fq_G1;
  using coeff_t = ALT_BN128_Fr_G1;
  __device__ __forceinline__ ALT_BN128_Fq_G1() {}
  __device__ __forceinline__ ALT_BN128_Fq_G1(const alt_bn128_fq_mont& a)
      : alt_bn128_fq_mont(a) {}
};

} // namespace native
} // namespace at
