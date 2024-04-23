#pragma once

#include <third_party/blst/include/blst_t.hpp>

namespace at {
namespace native {

static const vec256 MNT4753_r = {
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0)};
static const vec256 MNT4753_rRR = {/* (1<<512)%r */
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0)};
static const vec256 MNT4753_rONE = {/* (1<<256)%r */
                                    TO_LIMB_T(0x0),
                                    TO_LIMB_T(0x0),
                                    TO_LIMB_T(0x0),
                                    TO_LIMB_T(0x0)};
typedef blst_256_t<253, MNT4753_r, 0x0u, MNT4753_rRR, MNT4753_rONE>
    mnt4753_fr_mont;
struct MNT4753_Fr_G1 : public mnt4753_fr_mont {
  using mem_t = MNT4753_Fr_G1;
  inline MNT4753_Fr_G1() = default;
  inline MNT4753_Fr_G1(const mnt4753_fr_mont& a) : mnt4753_fr_mont(a) {}
};

static const vec384 MNT4753_P = {
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0),
    TO_LIMB_T(0x0)};
static const vec384 MNT4753_RR = {/* (1<<768)%P */
                                  TO_LIMB_T(0x0),
                                  TO_LIMB_T(0x0),
                                  TO_LIMB_T(0x0),
                                  TO_LIMB_T(0x0),
                                  TO_LIMB_T(0x0),
                                  TO_LIMB_T(0x0)};
static const vec384 MNT4753_ONE = {/* (1<<384)%P */
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0),
                                   TO_LIMB_T(0x0)};
typedef blst_384_t<377, MNT4753_P,
 0x0u, MNT4753_RR, MNT4753_ONE>
    mnt4753_fq_mont;
struct MNT4753_Fq_G1 : public mnt4753_fq_mont {
  using mem_t = MNT4753_Fq_G1;
  using coeff_t = MNT4753_Fr_G1;
  inline MNT4753_Fq_G1() = default;
  inline MNT4753_Fq_G1(const mnt4753_fq_mont& a) : mnt4753_fq_mont(a) {}
};

} // namespace native
} // namespace at
