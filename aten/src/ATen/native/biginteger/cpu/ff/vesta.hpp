#pragma once

# include "mont_t.cuh"

namespace at { 
namespace native {

static const uint32_t Vesta_P[8] = {
    0x00000001, 0x8c46eb21, 0x0994a8dd, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000
};
static const uint32_t Vesta_RR[8] = { /* (1<<512)%P */
    0x0000000f, 0xfc9678ff, 0x891a16e3, 0x67bb433d,
    0x04ccf590, 0x7fae2310, 0x7ccfdaa9, 0x096d41af
};
static const uint32_t Vesta_one[8] = { /* (1<<256)%P */
    0xfffffffd, 0x5b2b3e9c, 0xe3420567, 0x992c350b,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
};
static const uint32_t Vesta_Px2[8] = { /* left-aligned modulus */
    0x00000002, 0x188dd642, 0x132951bb, 0x448d31f8,
    0x00000000, 0x00000000, 0x00000000, 0x80000000
};

static const uint32_t Pallas_P[8] = {
    0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
    0x00000000, 0x00000000, 0x00000000, 0x40000000
};
static const uint32_t Pallas_RR[8] = { /* (1<<512)%P */
    0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
    0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af
};
static const uint32_t Pallas_one[8] = { /* (1<<256)%P */
    0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
    0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
};
static const uint32_t Pallas_Px2[8] = { /* left-aligned modulus */
    0x00000002, 0x325a61da, 0x1299f237, 0x448d31f8,
    0x00000000, 0x00000000, 0x00000000, 0x80000000
};

static  uint32_t Pasta_M0 = 0xffffffff;
typedef mont_t<255, Vesta_P, Pasta_M0,
                Vesta_RR, Vesta_one,
                Vesta_Px2> vesta_fr_mont;
struct Vesta_Fr_G1 : public vesta_fr_mont {
    using mem_t = Vesta_Fr_G1;
    inline Vesta_Fr_G1() = default;
    inline Vesta_Fr_G1(const vesta_fr_mont& a) : vesta_fr_mont(a) {}
};
typedef mont_t<255, Pallas_P, Pasta_M0,
                    Pallas_RR, Pallas_one,
                    Pallas_Px2> vesta_fq_mont;
struct Vesta_Fq_G1 : public vesta_fq_mont {
    using mem_t = Vesta_Fq_G1;
    inline Vesta_Fq_G1() = default;
    inline Vesta_Fq_G1(const vesta_fq_mont& a) : vesta_fq_mont(a) {}
};
}
}




