#pragma once

#include "mont_t.cuh"

namespace at {
namespace native {

static __device__ __constant__ __align__(16) const uint32_t Pallas_P[8] = {
    0x00000001,
    0x992d30ed,
    0x094cf91b,
    0x224698fc,
    0x00000000,
    0x00000000,
    0x00000000,
    0x40000000};
static __device__ __constant__
    __align__(16) const uint32_t Pallas_RR[8] = {/* (1<<512)%P */
                                                 0x0000000f,
                                                 0x8c78ecb3,
                                                 0x8b0de0e7,
                                                 0xd7d30dbd,
                                                 0xc3c95d18,
                                                 0x7797a99b,
                                                 0x7b9cb714,
                                                 0x096d41af};
static __device__ __constant__
    __align__(16) const uint32_t Pallas_one[8] = {/* (1<<256)%P */
                                                  0xfffffffd,
                                                  0x34786d38,
                                                  0xe41914ad,
                                                  0x992c350b,
                                                  0xffffffff,
                                                  0xffffffff,
                                                  0xffffffff,
                                                  0x3fffffff};
static __device__ __constant__
    __align__(16) const uint32_t Pallas_Px2[8] = {/* left-aligned modulus */
                                                  0x00000002,
                                                  0x325a61da,
                                                  0x1299f237,
                                                  0x448d31f8,
                                                  0x00000000,
                                                  0x00000000,
                                                  0x00000000,
                                                  0x80000000};

static __device__ __constant__ __align__(16) const uint32_t Vesta_P[8] = {
    0x00000001,
    0x8c46eb21,
    0x0994a8dd,
    0x224698fc,
    0x00000000,
    0x00000000,
    0x00000000,
    0x40000000};
static __device__ __constant__
    __align__(16) const uint32_t Vesta_RR[8] = {/* (1<<512)%P */
                                                0x0000000f,
                                                0xfc9678ff,
                                                0x891a16e3,
                                                0x67bb433d,
                                                0x04ccf590,
                                                0x7fae2310,
                                                0x7ccfdaa9,
                                                0x096d41af};
static __device__ __constant__
    __align__(16) const uint32_t Vesta_one[8] = {/* (1<<256)%P */
                                                 0xfffffffd,
                                                 0x5b2b3e9c,
                                                 0xe3420567,
                                                 0x992c350b,
                                                 0xffffffff,
                                                 0xffffffff,
                                                 0xffffffff,
                                                 0x3fffffff};
static __device__ __constant__
    __align__(16) const uint32_t Vesta_Px2[8] = {/* left-aligned modulus */
                                                 0x00000002,
                                                 0x188dd642,
                                                 0x132951bb,
                                                 0x448d31f8,
                                                 0x00000000,
                                                 0x00000000,
                                                 0x00000000,
                                                 0x80000000};

static __device__ __constant__ /*const*/ uint32_t Pasta_M0 = 0xffffffff;
typedef mont_t<255, Pallas_P, Pasta_M0, Pallas_RR, Pallas_one, Pallas_Px2>
    pallas_fr_mont;
struct PALLAS_Fr_G1 : public pallas_fr_mont {
  using mem_t = PALLAS_Fr_G1;
  __device__ __forceinline__ PALLAS_Fr_G1() = default;
  __device__ __forceinline__ PALLAS_Fr_G1(const pallas_fr_mont& a)
      : pallas_fr_mont(a) {}
};

typedef mont_t<255, Vesta_P, Pasta_M0, Vesta_RR, Vesta_one, Vesta_Px2>
    pallas_fq_mont;
struct PALLAS_Fq_G1 : public pallas_fq_mont {
  using mem_t = PALLAS_Fq_G1;
  using coeff_t = PALLAS_Fr_G1;
  __device__ __forceinline__ PALLAS_Fq_G1() = default;
  __device__ __forceinline__ PALLAS_Fq_G1(const pallas_fq_mont& a)
      : pallas_fq_mont(a) {}
};

typedef mont_t<255, Vesta_P, Pasta_M0, Vesta_RR, Vesta_one, Vesta_Px2>
    pallas_fq_mont;
struct VESTA_Fr_G1 : public pallas_fq_mont {
  using mem_t = VESTA_Fr_G1;
  __device__ __forceinline__ VESTA_Fr_G1() = default;
  __device__ __forceinline__ VESTA_Fr_G1(const pallas_fq_mont& a)
      : pallas_fq_mont(a) {}
};

typedef mont_t<255, Pallas_P, Pasta_M0, Pallas_RR, Pallas_one, Pallas_Px2>
    vesta_fq_mont;
struct VESTA_Fq_G1 : public vesta_fq_mont {
  using mem_t = VESTA_Fq_G1;
  using coeff_t = VESTA_Fr_G1;
  __device__ __forceinline__ VESTA_Fq_G1() = default;
  __device__ __forceinline__ VESTA_Fq_G1(const vesta_fq_mont& a)
      : vesta_fq_mont(a) {}
};

} // namespace native
} // namespace at
