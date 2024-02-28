#pragma once

#include <third_party/blst/include/blst_t.hpp>

namespace at {
namespace native {

static const vec256 BLS12_381_r = {
    TO_LIMB_T(0xffffffff00000001),
    TO_LIMB_T(0x53bda402fffe5bfe),
    TO_LIMB_T(0x3339d80809a1d805),
    TO_LIMB_T(0x73eda753299d7d48)};
static const vec256 BLS12_381_rRR = {/* (1<<512)%r */
                                     TO_LIMB_T(0xc999e990f3f29c6d),
                                     TO_LIMB_T(0x2b6cedcb87925c23),
                                     TO_LIMB_T(0x05d314967254398f),
                                     TO_LIMB_T(0x0748d9d99f59ff11)};
static const vec256 BLS12_381_rONE = {/* (1<<256)%r */
                                      TO_LIMB_T(0x00000001fffffffe),
                                      TO_LIMB_T(0x5884b7fa00034802),
                                      TO_LIMB_T(0x998c4fefecbc4ff5),
                                      TO_LIMB_T(0x1824b159acc5056f)};
typedef blst_256_t<
    255,
    BLS12_381_r,
    0xfffffffeffffffff,
    BLS12_381_rRR,
    BLS12_381_rONE>
    bls12_381_fr_mont;
struct BLS12_381_Fr_G1 : public bls12_381_fr_mont {
  using mem_t = BLS12_381_Fr_G1;
  inline BLS12_381_Fr_G1() = default;
  inline BLS12_381_Fr_G1(const bls12_381_fr_mont& a) : bls12_381_fr_mont(a) {}
};

static const vec384 BLS12_381_P = {
    TO_LIMB_T(0xb9feffffffffaaab),
    TO_LIMB_T(0x1eabfffeb153ffff),
    TO_LIMB_T(0x6730d2a0f6b0f624),
    TO_LIMB_T(0x64774b84f38512bf),
    TO_LIMB_T(0x4b1ba7b6434bacd7),
    TO_LIMB_T(0x1a0111ea397fe69a)};
static const vec384 BLS12_381_RR = {/* (1<<768)%P */
                                    TO_LIMB_T(0xf4df1f341c341746),
                                    TO_LIMB_T(0x0a76e6a609d104f1),
                                    TO_LIMB_T(0x8de5476c4c95b6d5),
                                    TO_LIMB_T(0x67eb88a9939d83c0),
                                    TO_LIMB_T(0x9a793e85b519952d),
                                    TO_LIMB_T(0x11988fe592cae3aa)};
static const vec384 BLS12_381_ONE = {/* (1<<384)%P */
                                     TO_LIMB_T(0x760900000002fffd),
                                     TO_LIMB_T(0xebf4000bc40c0002),
                                     TO_LIMB_T(0x5f48985753c758ba),
                                     TO_LIMB_T(0x77ce585370525745),
                                     TO_LIMB_T(0x5c071a97a256ec6d),
                                     TO_LIMB_T(0x15f65ec3fa80e493)};
typedef blst_384_t<
    381,
    BLS12_381_P,
    0x89f3fffcfffcfffd,
    BLS12_381_RR,
    BLS12_381_ONE>
    bls12_381_fq_mont;
struct BLS12_381_Fq_G1 : public bls12_381_fq_mont {
  using mem_t = BLS12_381_Fq_G1;
  using coeff_t = BLS12_381_Fr_G1;
  inline BLS12_381_Fq_G1() = default;
  inline BLS12_381_Fq_G1(const bls12_381_fq_mont& a) : bls12_381_fq_mont(a) {}
};

} // namespace native
} // namespace at
