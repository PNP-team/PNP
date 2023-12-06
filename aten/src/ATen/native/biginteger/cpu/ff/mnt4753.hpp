#pragma once

#include <third_party/blst/include/blst_t.hpp>

namespace at { 
namespace native {

static const vec768 MNT4753_r = { 
    TO_LIMB_T(0xd90776e240000001), TO_LIMB_T(0x4ea099170fa13a4f),
    TO_LIMB_T(0xd6c381bc3f005797), TO_LIMB_T(0xb9dff97634993aa4),
    TO_LIMB_T(0x3eebca9429212636), TO_LIMB_T(0xb26c5c28c859a99b),
    TO_LIMB_T(0x99d124d9a15af79d), TO_LIMB_T(0x07fdb925e8a0ed8d),
    TO_LIMB_T(0x5eb7e8f96c97d873), TO_LIMB_T(0xb7f997505b8fafed),
    TO_LIMB_T(0x10229022eee2cdad), TO_LIMB_T(0x0001c4c62d92c411)
};
static const vec768 MNT4753_rRR = {   /* (1<<1536)%r */
    TO_LIMB_T(0x3f9c69c7b7f4c8d1), TO_LIMB_T(0x70a50fa9ee48d127),
    TO_LIMB_T(0xcdbe6702009569cb), TO_LIMB_T(0x6bd8c6c6c49edc38),
    TO_LIMB_T(0x7955876cc35ee94e), TO_LIMB_T(0xc7285529be54a3f4),
    TO_LIMB_T(0xded52121ecec77cf), TO_LIMB_T(0x99be80f2ee12ee8e),
    TO_LIMB_T(0xc8a0ff01493bdcef), TO_LIMB_T(0xacc27988f3d9a316),
    TO_LIMB_T(0xd9e817a8fb44b3c9), TO_LIMB_T(0x000005b58037e0e4)
};
static const vec768 MNT4753_rONE = {  /* (1<<768)%r */
    TO_LIMB_T(0xb99680147fff6f42), TO_LIMB_T(0x4eb16817b589cea8),
    TO_LIMB_T(0xa1ebd2d90c79e179), TO_LIMB_T(0x0f725caec549c0da),
    TO_LIMB_T(0xab0c4ee6d3e6dad4), TO_LIMB_T(0x9fbca908de0ccb62),
    TO_LIMB_T(0x320c3bb713338498), TO_LIMB_T(0x598b4302d2f00a62),
    TO_LIMB_T(0x4074c9cbfd8ca621), TO_LIMB_T(0x0fa47edb3865e88c),
    TO_LIMB_T(0x95455fb31ff9a195), TO_LIMB_T(0x00007b479ec8e242)
};
// typedef blst_768_t<753, MNT4753_r, 0xc90776e23fffffff,
//                         MNT4753_rRR, MNT4753_rONE> mnt4753_fr_mont;
// struct MNT4753_Fr_G1 : public mnt4753_fr_mont {
//     using mem_t = MNT4753_Fr_G1;
//     inline MNT4753_Fr_G1() = default;
//     inline MNT4753_Fr_G1(const mnt4753_fr_mont& a) : mnt4753_fr_mont(a) {}
// };

static const vec768 MNT4753_P = {
    TO_LIMB_T(0x5e9063de245e8001), TO_LIMB_T(0xe39d54522cdd119f),
    TO_LIMB_T(0x638810719ac425f0), TO_LIMB_T(0x685acce9767254a4),
    TO_LIMB_T(0xb80f0da5cb537e38), TO_LIMB_T(0xb117e776f218059d),
    TO_LIMB_T(0x99d124d9a15af79d), TO_LIMB_T(0x07fdb925e8a0ed8d),
    TO_LIMB_T(0x5eb7e8f96c97d873), TO_LIMB_T(0xb7f997505b8fafed),
    TO_LIMB_T(0x10229022eee2cdad), TO_LIMB_T(0x0001c4c62d92c411)
};
static const vec768 MNT4753_RR = {    /* (1<<1536)%P */
    TO_LIMB_T(0x84717088cfd190c8), TO_LIMB_T(0xc7d9ff8e7df03c0a),
    TO_LIMB_T(0xa24bea56242b3507), TO_LIMB_T(0xa896a656a0714c7d),
    TO_LIMB_T(0x80a46659ff6f3ddf), TO_LIMB_T(0x2f47839ef88d7ce8),
    TO_LIMB_T(0xa8c86d4604a3b597), TO_LIMB_T(0xe03c79cac4f7ef07),
    TO_LIMB_T(0x2505daf1f4a81245), TO_LIMB_T(0x8e4605754c381723),
    TO_LIMB_T(0xb081f15bcbfdacaf), TO_LIMB_T(0x00002a33e89cb485)
};
static const vec768 MNT4753_ONE = {   /* (1<768)%P */
    TO_LIMB_T(0x98a8ecabd9dc6f42), TO_LIMB_T(0x91cd31c65a034686),
    TO_LIMB_T(0x97c3e4a0cd14572e), TO_LIMB_T(0x79589819c788b601),
    TO_LIMB_T(0xed269c942108976f), TO_LIMB_T(0x1e0f4d8acf031d68),
    TO_LIMB_T(0x320c3bb713338559), TO_LIMB_T(0x598b4302d2f00a62),
    TO_LIMB_T(0x4074c9cbfd8ca621), TO_LIMB_T(0x0fa47edb3865e88c), 
    TO_LIMB_T(0x95455fb31ff9a195), TO_LIMB_T(0x00007b479ec8e242)
};
// typedef blst_768_t<753, MNT4753_P, 0xf2044cfbe45e7fff,
//                         MNT4753_RR, MNT4753_ONE> mnt4753_fq_mont;
// struct MNT4753_Fq_G1 : public mnt4753_fq_mont {
//     using mem_t = MNT4753_Fq_G1;
//     inline MNT4753_Fq_G1() = default;
//     inline MNT4753_Fq_G1(const mnt4753_fq_mont& a) : mnt4753_fq_mont(a) {}
// };

} // namespace native
} // namespace at
