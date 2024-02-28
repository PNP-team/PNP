#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

static __device__ __constant__ __align__(16) const uint32_t BLS12_381_r[8] = {
    TO_CUDA_T(0xffffffff00000001),
    TO_CUDA_T(0x53bda402fffe5bfe),
    TO_CUDA_T(0x3339d80809a1d805),
    TO_CUDA_T(0x73eda753299d7d48)};
static __device__ __constant__ __align__(16) const uint32_t
    BLS12_381_rRR[8] = {/* (1<<512)%P */
                        TO_CUDA_T(0xc999e990f3f29c6d),
                        TO_CUDA_T(0x2b6cedcb87925c23),
                        TO_CUDA_T(0x05d314967254398f),
                        TO_CUDA_T(0x0748d9d99f59ff11)};
static __device__ __constant__ __align__(16) const uint32_t
    BLS12_381_rone[8] = {/* (1<<256)%P */
                         TO_CUDA_T(0x00000001fffffffe),
                         TO_CUDA_T(0x5884b7fa00034802),
                         TO_CUDA_T(0x998c4fefecbc4ff5),
                         TO_CUDA_T(0x1824b159acc5056f)};
static __device__ __constant__ __align__(16) const uint32_t BLS12_381_rx2[8] = {
    /* left-aligned value of the modulus */
    TO_CUDA_T(0xfffffffe00000002),
    TO_CUDA_T(0xa77b4805fffcb7fd),
    TO_CUDA_T(0x6673b0101343b00a),
    TO_CUDA_T(0xe7db4ea6533afa90),
};
static __device__ __constant__ /*const*/ uint32_t BLS12_381_m0 = 0xffffffff;
typedef mont_t<
    255,
    BLS12_381_r,
    BLS12_381_m0,
    BLS12_381_rRR,
    BLS12_381_rone,
    BLS12_381_rx2>
    bls12_381_fr_mont;
struct BLS12_381_Fr_G1 : public bls12_381_fr_mont {
  using mem_t = BLS12_381_Fr_G1;
  __device__ __forceinline__ BLS12_381_Fr_G1() {}
  __device__ __forceinline__ BLS12_381_Fr_G1(const bls12_381_fr_mont& a)
      : bls12_381_fr_mont(a) {}
};

static __device__ __constant__ __align__(16) const uint32_t BLS12_381_P[12] = {
    TO_CUDA_T(0xb9feffffffffaaab),
    TO_CUDA_T(0x1eabfffeb153ffff),
    TO_CUDA_T(0x6730d2a0f6b0f624),
    TO_CUDA_T(0x64774b84f38512bf),
    TO_CUDA_T(0x4b1ba7b6434bacd7),
    TO_CUDA_T(0x1a0111ea397fe69a)};
static __device__ __constant__ __align__(16) const uint32_t
    BLS12_381_RR[12] = {/* (1<<768)%P */
                        TO_CUDA_T(0xf4df1f341c341746),
                        TO_CUDA_T(0x0a76e6a609d104f1),
                        TO_CUDA_T(0x8de5476c4c95b6d5),
                        TO_CUDA_T(0x67eb88a9939d83c0),
                        TO_CUDA_T(0x9a793e85b519952d),
                        TO_CUDA_T(0x11988fe592cae3aa)};
static __device__ __constant__ __align__(16) const uint32_t
    BLS12_381_one[12] = {/* (1<<384)%P */
                         TO_CUDA_T(0x760900000002fffd),
                         TO_CUDA_T(0xebf4000bc40c0002),
                         TO_CUDA_T(0x5f48985753c758ba),
                         TO_CUDA_T(0x77ce585370525745),
                         TO_CUDA_T(0x5c071a97a256ec6d),
                         TO_CUDA_T(0x15f65ec3fa80e493)};
static __device__ __constant__ __align__(16) const uint32_t
    BLS12_381_Px8[12] = {/* left-aligned value of the modulus */
                         TO_CUDA_T(0xcff7fffffffd5558),
                         TO_CUDA_T(0xf55ffff58a9ffffd),
                         TO_CUDA_T(0x39869507b587b120),
                         TO_CUDA_T(0x23ba5c279c2895fb),
                         TO_CUDA_T(0x58dd3db21a5d66bb),
                         TO_CUDA_T(0xd0088f51cbff34d2)};
static __device__ __constant__ const uint32_t BLS12_381_M0 = 0xfffcfffd;
typedef mont_t<
    381,
    BLS12_381_P,
    BLS12_381_M0,
    BLS12_381_RR,
    BLS12_381_one,
    BLS12_381_Px8>
    bls12_381_fq_mont;
struct BLS12_381_Fq_G1 : public bls12_381_fq_mont {
  using mem_t = BLS12_381_Fq_G1;
  using coeff_t = BLS12_381_Fr_G1;
  __device__ __forceinline__ BLS12_381_Fq_G1() {}
  __device__ __forceinline__ BLS12_381_Fq_G1(const bls12_381_fq_mont& a)
      : bls12_381_fq_mont(a) {}
};

} // namespace native
} // namespace at
