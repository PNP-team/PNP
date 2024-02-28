#pragma once

#include <third_party/blst/include/blst_t.hpp>

namespace at {
namespace native {

static const vec256 Pallas_P = {
    TO_LIMB_T(0x992d30ed00000001),
    TO_LIMB_T(0x224698fc094cf91b),
    TO_LIMB_T(0x0000000000000000),
    TO_LIMB_T(0x4000000000000000)};
static const vec256 Pallas_RR = {/* (1<<512)%P */
                                 TO_LIMB_T(0x8c78ecb30000000f),
                                 TO_LIMB_T(0xd7d30dbd8b0de0e7),
                                 TO_LIMB_T(0x7797a99bc3c95d18),
                                 TO_LIMB_T(0x096d41af7b9cb714)};
static const vec256 Pallas_one = {/* (1<<256)%P */
                                  TO_LIMB_T(0x34786d38fffffffd),
                                  TO_LIMB_T(0x992c350be41914ad),
                                  TO_LIMB_T(0xffffffffffffffff),
                                  TO_LIMB_T(0x3fffffffffffffff)};
static const vec256 Pallas_Px2 = {/* left-aligned modulus */
                                  TO_LIMB_T(0x325a61da00000002),
                                  TO_LIMB_T(0x448d31f81299f237),
                                  TO_LIMB_T(0x0000000000000000),
                                  TO_LIMB_T(0x8000000000000000)};

static const vec256 Vesta_P = {
    TO_LIMB_T(0x8c46eb2100000001),
    TO_LIMB_T(0x224698fc0994a8dd),
    TO_LIMB_T(0x0000000000000000),
    TO_LIMB_T(0x4000000000000000)};
static const vec256 Vesta_RR = {/* (1<<512)%P */
                                TO_LIMB_T(0xfc9678ff0000000f),
                                TO_LIMB_T(0x67bb433d891a16e3),
                                TO_LIMB_T(0x7fae231004ccf590),
                                TO_LIMB_T(0x096d41af7ccfdaa9)};
static const vec256 Vesta_one = {/* (1<<256)%P */
                                 TO_LIMB_T(0x5b2b3e9cfffffffd),
                                 TO_LIMB_T(0x992c350be3420567),
                                 TO_LIMB_T(0xffffffffffffffff),
                                 TO_LIMB_T(0x3fffffffffffffff)};
static const vec256 Vesta_Px2 = {/* left-aligned modulus */
                                 TO_LIMB_T(0x188dd64200000002),
                                 TO_LIMB_T(0x448d31f8132951bb),
                                 TO_LIMB_T(0x0000000000000000),
                                 TO_LIMB_T(0x8000000000000000)};

typedef blst_256_t<255, Pallas_P, 0x992d30ecffffffff, Pallas_RR, Pallas_one>
    pallas_fr_mont;
struct PALLAS_Fr_G1 : public pallas_fr_mont {
  using mem_t = PALLAS_Fr_G1;
  inline PALLAS_Fr_G1() = default;
  inline PALLAS_Fr_G1(const pallas_fr_mont& a) : pallas_fr_mont(a) {}
};

typedef blst_256_t<255, Vesta_P, 0x8c46eb20ffffffff, Vesta_RR, Vesta_one>
    pallas_fq_mont;
struct PALLAS_Fq_G1 : public pallas_fq_mont {
  using mem_t = PALLAS_Fq_G1;
  using coeff_t = PALLAS_Fr_G1;
  inline PALLAS_Fq_G1() = default;
  inline PALLAS_Fq_G1(const pallas_fq_mont& a) : pallas_fq_mont(a) {}
};

typedef blst_256_t<255, Vesta_P, 0x8c46eb20ffffffff, Vesta_RR, Vesta_one>
    vesta_fr_mont;
struct VESTA_Fr_G1 : public vesta_fr_mont {
  using mem_t = VESTA_Fr_G1;
  inline VESTA_Fr_G1() = default;
  inline VESTA_Fr_G1(const vesta_fr_mont& a) : vesta_fr_mont(a) {}
};

typedef blst_256_t<255, Pallas_P, 0x992d30ecffffffff, Pallas_RR, Pallas_one>
    vesta_fq_mont;
struct VESTA_Fq_G1 : public vesta_fq_mont {
  using mem_t = VESTA_Fq_G1;
  using coeff_t = VESTA_Fr_G1;
  inline VESTA_Fq_G1() = default;
  inline VESTA_Fq_G1(const vesta_fq_mont& a) : vesta_fq_mont(a) {}
};

} // namespace native
} // namespace at
