// Copyright Spranational LLC
// Licensed nder the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#pragma warning(disable : 63)
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
namespace at { 
namespace native {

const BLS12_381_Fr_G1 group_gen = BLS12_381_Fr_G1{(uint32_t[]){0xfffffff1, 0x0000000e, 0x00189c0f, 0x17e363d3, 0x6f8457b0, 0xff9c5787, 0x8fc5a8c4, 0x35133220}};
const BLS12_381_Fr_G1 group_gen_inverse = BLS12_381_Fr_G1{(uint32_t[]){0xdb6db6dc, 0xdb6db6da, 0xdb6cc6da, 0xe6b5824a, 0x05810db9, 0xf8b356e0, 0x60ec4796, 0x66d0f1e6}};

const int S = 32;

const uint32_t forward_roots_of_unity[(S + 1)*8] = {
    0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159,
    0x00000003, 0xfffffffd, 0xfffb13fc, 0xfb38ec08, 0x1ce5880f, 0x99ad8818, 0x7cd877d8, 0x5bc8f5f9,
    0xaa89cfb1, 0xf3b05674, 0x6006b9fe, 0x072f0140, 0x25667a26, 0xce9a0dbf, 0x2d598374, 0x4d2ce405,
    0x2c42eff8, 0x6ecf4995, 0x4d5dfa57, 0x6ec1fa83, 0x5656b15c, 0x3ff5cf38, 0x229e031b, 0x54b80267,
    0x88202b65, 0x1acf64e4, 0x819f004f, 0x5b18b637, 0x3107fdf5, 0xee649cab, 0xce9012a0, 0x27a8543d,
    0xdd04d52f, 0x30c78850, 0xadaddbcb, 0xe27f8794, 0xae22c531, 0x0a59f321, 0xba7f8ca3, 0x1958b127,
    0xe8d6f162, 0xbcf7e31a, 0x662fc7a4, 0x15d1c587, 0x71278a57, 0xb7ed5ecd, 0x40ccecb3, 0x329d0d93,
    0x4aed2336, 0xdaf6401c, 0x41de4f6a, 0xa14c0cba, 0xfc7f2e5b, 0xdfdc7d04, 0x296cff18, 0x6f61ff4e,
    0xa9d4956c, 0xb4b87dd6, 0xd1482f03, 0x6a51d0bc, 0xe0dbcb1e, 0x7e8146f7, 0xcccb85ae, 0x21f4b238,
    0xb651888b, 0x1ff70724, 0xe56a5af4, 0xcd8166cb, 0x98e1e256, 0xe4a992a3, 0x86fd3b03, 0x5705ea36,
    0x36d7822a, 0xa4a6e37a, 0x8787e5f1, 0xfc83a2ce, 0xe7ea8f80, 0xe89553da, 0x2a92ff1a, 0x6f6d38bb,
    0x8dc619e5, 0xf4a46fbb, 0xb8e77487, 0x82fa97dc, 0xf198f018, 0xaf909c5b, 0x83a038fa, 0x00edba7d,
    0x09458a39, 0xf2df262c, 0x99dff177, 0x048cdf5b, 0xc7cce57b, 0x16857bc5, 0xa4a915ae, 0x043b3dbc,
    0xf0ccffc9, 0xa33d279f, 0x59e91972, 0x41fac79f, 0xead1139b, 0x065d227f, 0xda03e055, 0x71db41ab,
    0xf2a9a5d6, 0x9e468a30, 0xca6bcf1b, 0x1fe34726, 0x87eb73ac, 0x54a621ae, 0xbfd61a47, 0x100a9d2e,
    0xf4e11f70, 0x6ebe497b, 0x3c55cafc, 0x1533f482, 0x7eb4d017, 0x66a7f2da, 0xb07f3e81, 0x5b6be79a,
    0x11c947f9, 0x4526e48b, 0xc614d3aa, 0xccc36140, 0x03cd5f24, 0xf2e56f1a, 0x159e0248, 0x57db5a4c,
    0x2d964320, 0x860cb52e, 0x9678fcc9, 0x4c5059aa, 0xcfe87e16, 0xd92f98d7, 0xd283e2d3, 0x40329fd2,
    0xcb39b4a6, 0x56e7145c, 0xdcbe3def, 0x79825dea, 0xf02abbe0, 0x3e516f05, 0xbef5bd2d, 0x532595f8,
    0x90f9b375, 0x3b801e7d, 0x524be4d6, 0xaada3338, 0xbdbd3f25, 0x5f0b11a6, 0x5adb38c4, 0x04063ff8,
    0x784685ce, 0x146391f8, 0x72dbcd2c, 0x6975fe21, 0xedc848d6, 0x7bfaf2b2, 0xafb23c0c, 0x13c5c6b3,
    0x1f88b5f3, 0x245f9cbc, 0x8ef4a333, 0x5dfbb5c6, 0xc68f208f, 0xa9ac7cdb, 0x38255605, 0x44448a98,
    0xbf7c975b, 0x0db3e6cc, 0xaf4d41ec, 0x9d463b98, 0x9ab7a3e7, 0x555e1ff5, 0x2a31f630, 0x6154be20,
    0x9538df40, 0xbcd2a6e0, 0x95e6fafd, 0x290e2f0c, 0xe7fe2f1d, 0x37b754af, 0xd11e6548, 0x500b5365,
    0x35b444fa, 0x6a4fac54, 0xf8a367da, 0xa4c9d25d, 0x4c0632ff, 0xa33932bd, 0xcfb507b5, 0x038a3b43,
    0xd37328ca, 0x25acfadb, 0x36ffb229, 0xa2157fa3, 0x1ac4d49e, 0x432bc6f2, 0xb0894969, 0x60eb5b8d,
    0x9fc1ac27, 0x8a0719bd, 0x29716a74, 0x744144bb, 0x541fcb9a, 0xffea9e0b, 0x88bb635f, 0x4459ada3,
    0xfbd35ce8, 0xe7b5ec2b, 0x35e37e7b, 0xa99a0f8a, 0xa70cacc4, 0x312c2051, 0xc9407c7d, 0x415444df,
    0x8d27bcb7, 0x2c75ff0b, 0x4413b9d7, 0x00fc392e, 0x8752020f, 0xef31d1fd, 0xe7a75dc5, 0x6d5074c8,
    0x4b092216, 0x250a9d0e, 0x81be1b5e, 0x525a6537, 0x889cdfb0, 0xe4381082, 0x1d9f1a1c, 0x02536cde,
    0x7bb12408, 0x8ea72d68, 0xb02387bc, 0xa9ed127c, 0xe7669430, 0x9c59d748, 0xe9ced390, 0x4e720080,
    0x3141e583, 0x1f2fda6f, 0x27fe5b46, 0x8d445922, 0x81e59a5f, 0xe33cee1b, 0x1ad8ae31, 0x22c66625,
    0x5f0e466a, 0xb9b58d8c, 0x1819d7ec, 0x5b1b4c80, 0x52a31e64, 0x0af53ae3, 0x19e9b27b, 0x5bf3adda
};

const uint32_t inverse_roots_of_unity[(S + 1)*8] = {
    0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159,
    0x00000003, 0xfffffffd, 0xfffb13fc, 0xfb38ec08, 0x1ce5880f, 0x99ad8818, 0x7cd877d8, 0x5bc8f5f9,
    0x55763050, 0x0c4fa98a, 0x9ff7a200, 0x4c8ea2c2, 0xe43b5ddf, 0x649fca48, 0xfc43f9d3, 0x26c0c34d,
    0xc14d5724, 0xeee067bc, 0x90c5a3be, 0x34467c7e, 0x7b6536c9, 0x80b27f64, 0x3c033457, 0x1a78ad5b,
    0x234112ae, 0xf222d1b8, 0xcc227ea9, 0x74262683, 0x485acc4a, 0x5596249d, 0x73235a95, 0x2bd4ab5c,
    0x718c6944, 0x64db788d, 0xd9ad2c34, 0x60072eb5, 0xed4e3d6d, 0xfe642f9d, 0xb4134c95, 0x5b5ebc8d,
    0x35357cee, 0xf0e6ca26, 0xec44a018, 0xe8078a41, 0x25aa3530, 0xa9bcb9a2, 0xc7f02aeb, 0x12fe8733,
    0x8e9f64e7, 0x17172e1f, 0xf14b6496, 0x36cbbaa0, 0x72393444, 0x938df195, 0x62d2cf16, 0x6b00e7ac,
    0x9b6b31b6, 0xa41af272, 0xf1b43302, 0x34bd6b2e, 0xaf334b5f, 0xb7451539, 0x6fa46d9f, 0x16921f60,
    0x1f27dee5, 0x7e90c8e2, 0x2ddb1430, 0x20fe5879, 0xbf51b9ec, 0x9612ab09, 0xca80979f, 0x65af656f,
    0xc3b1c080, 0x2b56cea3, 0x8718275c, 0x09831c84, 0x4a7434e8, 0x16722b72, 0xf07adbbd, 0x4a6b4e6b,
    0x9ac9ca75, 0xc22aee4c, 0x4bdf67d7, 0x5af8909b, 0x5d493090, 0xc41f7280, 0x4f460a37, 0x02fb078f,
    0x7e7e766d, 0x5e13f913, 0x1fe9c19f, 0xb8f1f8b3, 0x4ce1d4cf, 0x7c49eacd, 0x4e21e419, 0x19d94266,
    0xa1355e75, 0x507dbef9, 0x9b69cc1a, 0xeb71fd25, 0xee0557f8, 0x6042c8db, 0x9dbce162, 0x2415d770,
    0xb565301e, 0x92c42815, 0x4ddd94bf, 0x55d93cac, 0xef7ea500, 0xa35dff64, 0x9312dc0c, 0x1407943e,
    0x6b2c5dea, 0x8d5ac109, 0x1574d90f, 0x2faa86e3, 0xe5c2d788, 0xbb34c14a, 0xcdc0b4e0, 0x6b3bd790,
    0x32233a37, 0x9da11b8f, 0x649253a6, 0x78cd3e7a, 0x5e782970, 0x47b227da, 0xa95125a3, 0x4ffe8088,
    0x71d5e317, 0x820fbdda, 0x0485951f, 0x56c6bb33, 0xa04f827f, 0x4879070c, 0x52cef69c, 0x4434ce21,
    0x08ea085b, 0x61c7a118, 0x7a9ea4d0, 0xb40fb541, 0x5c0cc736, 0x4c57483a, 0x14aa0589, 0x2bf6e1fb,
    0x13a32f88, 0x39d16575, 0xd9e1e454, 0xcc3112e3, 0x54770c18, 0x8074cbd9, 0xb70a7051, 0x0b25db2f,
    0xb64957f6, 0x68ee7cb8, 0x0a2b8baf, 0x8eeda81e, 0x256c4bec, 0xd749ce3d, 0x2ae8d689, 0x48230b86,
    0xe3c30b08, 0x9aa2ba63, 0xd67b21bc, 0xc3a07402, 0x29af6349, 0x64a4e583, 0xf6ae051c, 0x64bffb89,
    0xf6421a00, 0x52c40fb8, 0x5ec95569, 0xbf33428e, 0x0f71a346, 0x5ecc78dd, 0x9336bf44, 0x5bb66cdb,
    0x3f3f7413, 0x62a17c8a, 0xbb17edba, 0xf60c53c0, 0x0c4e2817, 0x4a9ea2d0, 0xf7c560f3, 0x13b632ee,
    0x990adaca, 0xad970bf0, 0x35d536ac, 0xfcc75b77, 0x776dcbae, 0x8dce37c6, 0x0b26861d, 0x39f39d66,
    0xae32dcde, 0xbd56d684, 0x1409bd44, 0x2db07b79, 0x76a3e44e, 0xf6fc95fc, 0x24ed7b31, 0x5adc0a48,
    0x27c562bb, 0xf6d82dbf, 0x2b57637f, 0x4ddc7290, 0x06eab0a9, 0xc17c34cd, 0x2b59d621, 0x068dfbae,
    0x61b4a6df, 0x3bbb6d5f, 0xb34eca42, 0x029e0c54, 0x3ffd89d1, 0x1d4e3056, 0xd07ec38b, 0x6b3cb83c,
    0x7b483997, 0x25909fe2, 0x3f40368f, 0xe6c3383b, 0xdfbf158f, 0xe7a72791, 0x90ca2226, 0x2bccc288,
    0x54eb1d75, 0x79675917, 0xd07fc645, 0x1d44be7d, 0x47873890, 0x1fddcce9, 0xd4a502f9, 0x1c9bd17e,
    0x70de2c18, 0x0491bceb, 0x87989fc9, 0xfe592990, 0xb064bd3c, 0x687fc87a, 0xfbd5ad87, 0x184857c8,
    0xb510e57e, 0x1e4739c8, 0x184b8322, 0xcc6032ec, 0x984f6527, 0x67262f63, 0x87e5e5d4, 0x5f5b959a,
    0xdcf3219a, 0x4256481a, 0x96b6cad3, 0x45f37b7f, 0x5f7a3b27, 0xf9c3f1d7, 0x658afd43, 0x2d2fc049
};

const uint32_t domain_size_inverse[(S + 1)*8] = {
    0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159,
    0xffffffff, 0x00000000, 0x0001a401, 0xac425bfd, 0xf65e27fa, 0xccc627f7, 0xd66282b7, 0x0c1258ac,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x40000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x20000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x10000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x08000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x04000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x02000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x01000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00800000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00400000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00200000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00100000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00080000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00040000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00020000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00010000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00008000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00004000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00002000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00001000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000800,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000400,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000100,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000080,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000020,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000010,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000008,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000004,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000002,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001
};

#undef BLS12_381_Fr_G1
}//native
}//at