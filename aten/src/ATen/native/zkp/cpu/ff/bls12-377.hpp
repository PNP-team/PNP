#pragma once

#include <third_party/blst/include/blst_t.hpp>

namespace at {
namespace native {

static const vec256 BLS12_377_r = {
    TO_LIMB_T(0x0a11800000000001),
    TO_LIMB_T(0x59aa76fed0000001),
    TO_LIMB_T(0x60b44d1e5c37b001),
    TO_LIMB_T(0x12ab655e9a2ca556)};
static const vec256 BLS12_377_rRR = {/* (1<<512)%r */
                                     TO_LIMB_T(0x25d577bab861857b),
                                     TO_LIMB_T(0xcc2c27b58860591f),
                                     TO_LIMB_T(0xa7cc008fe5dc8593),
                                     TO_LIMB_T(0x011fdae7eff1c939)};
static const vec256 BLS12_377_rONE = {/* (1<<256)%r */
                                      TO_LIMB_T(0x7d1c7ffffffffff3),
                                      TO_LIMB_T(0x7257f50f6ffffff2),
                                      TO_LIMB_T(0x16d81575512c0fee),
                                      TO_LIMB_T(0x0d4bda322bbb9a9d)};
typedef blst_256_t<
    253,
    BLS12_377_r,
    0xa117fffffffffffu,
    BLS12_377_rRR,
    BLS12_377_rONE>
    bls12_377_fr_mont;
struct BLS12_377_Fr_G1 : public bls12_377_fr_mont {
  using mem_t = BLS12_377_Fr_G1;
  inline BLS12_377_Fr_G1() = default;
  inline BLS12_377_Fr_G1(const bls12_377_fr_mont& a) : bls12_377_fr_mont(a) {}
};

static const vec384 BLS12_377_P = {
    TO_LIMB_T(0x8508c00000000001),
    TO_LIMB_T(0x170b5d4430000000),
    TO_LIMB_T(0x1ef3622fba094800),
    TO_LIMB_T(0x1a22d9f300f5138f),
    TO_LIMB_T(0xc63b05c06ca1493b),
    TO_LIMB_T(0x01ae3a4617c510ea)};
static const vec384 BLS12_377_RR = {/* (1<<768)%P */
                                    TO_LIMB_T(0xb786686c9400cd22),
                                    TO_LIMB_T(0x0329fcaab00431b1),
                                    TO_LIMB_T(0x22a5f11162d6b46d),
                                    TO_LIMB_T(0xbfdf7d03827dc3ac),
                                    TO_LIMB_T(0x837e92f041790bf9),
                                    TO_LIMB_T(0x006dfccb1e914b88)};
static const vec384 BLS12_377_ONE = {/* (1<<384)%P */
                                     TO_LIMB_T(0x02cdffffffffff68),
                                     TO_LIMB_T(0x51409f837fffffb1),
                                     TO_LIMB_T(0x9f7db3a98a7d3ff2),
                                     TO_LIMB_T(0x7b4e97b76e7c6305),
                                     TO_LIMB_T(0x4cf495bf803c84e8),
                                     TO_LIMB_T(0x008d6661e2fdf49a)};
typedef blst_384_t<
    377,
    BLS12_377_P,
    0x8508bfffffffffffu,
    BLS12_377_RR,
    BLS12_377_ONE>
    bls12_377_fq_mont;
struct BLS12_377_Fq_G1 : public bls12_377_fq_mont {
  using mem_t = BLS12_377_Fq_G1;
  using coeff_t = BLS12_377_Fr_G1;
  inline BLS12_377_Fq_G1() = default;
  inline BLS12_377_Fq_G1(const bls12_377_fq_mont& a) : bls12_377_fq_mont(a) {}
};

} // namespace native
} // namespace at
