// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ATen/native/biginteger/cuda/ff/alt_bn128.hpp"
#pragma warning(disable : 63)
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
namespace at { 
namespace native {
const uint32_t group_gen[8] = {0x9fffffe6, 0x1b0d0ef9, 0xa32a913f, 0xeaba68a3, 0xd8dd0689, 0x47d8eb76, 0x20f5bbc3, 0x15d00855};
const uint32_t group_gen_inverse[8] = {0x09999999, 0xd7453974, 0x83c3efa8, 0xb4ada7d4, 0xe57f3161, 0xc49ca2f8, 0xac156cb3, 0x162a3754};

const int S = 28;

const uint32_t forward_roots_of_unity[(S + 1)*8] = {
    0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1,
    0xa0000006, 0x974bc177, 0xda58a367, 0xf13771b2, 0x0908122e, 0x51e1a247, 0x4729c0fa, 0x2259d6b1,
    0x9edcef8b, 0x7f753d97, 0xb1479120, 0x5f3f172c, 0x74096c6e, 0x8db16279, 0x39c108cf, 0x2b377b35,
    0xb5ad7c3f, 0xf8ad4ae2, 0x83cb85be, 0x2d60c6ff, 0x5d9429f7, 0xd976fd2b, 0x3f9ad9a9, 0x24407ce7,
    0x742f8f03, 0xf4b67d7c, 0x63d068cc, 0x681b2ddc, 0x1bfb576a, 0x8ce5bcef, 0xd2b63cfe, 0x167c2951,
    0xbf574c64, 0x2214f7b1, 0xf7317df2, 0x28f9232f, 0xb0ad75cd, 0xe57584a8, 0xdc176d03, 0x2b81fb59,
    0x818f61bf, 0x8e9003e5, 0x9bf8fec2, 0x8c9bbf34, 0x3f01534e, 0x53dceecd, 0xe529aa3c, 0x2690966b,
    0x26817bb1, 0x0a79c430, 0x99537df0, 0x7bb2cc7c, 0x0241e6de, 0xb6ca27d5, 0x3632f04c, 0x007ab33f,
    0x83a24acc, 0xa2afb83f, 0x525d536e, 0x92f255d9, 0x0286dd19, 0x5e756608, 0xc52d2549, 0x187bb1a6,
    0x22a423de, 0x9c533be7, 0x7cdf6e0d, 0x642a9d12, 0x409ac005, 0x0dbc7546, 0xb23d5082, 0x00f04c8d,
    0x280c1184, 0x1ef4b3b4, 0xae5e2a2c, 0xcf7ad4c2, 0xc5a36518, 0xb8063b6c, 0x65dfc08c, 0x2348c4b9,
    0xae4fcfb2, 0x48e72189, 0x8df85a07, 0x0a03fb3c, 0xea9b2e0a, 0xff4d8a35, 0xcd9c1d77, 0x28a98c2e,
    0xb3ecdbd3, 0x1dd4522f, 0xd055f3ad, 0x68222a93, 0xb3d555e8, 0xbe9c7d66, 0x6194f846, 0x1b92f6b8,
    0x79a04ed6, 0x894cdcbe, 0x44d30787, 0x956cde6a, 0xd7dbc15f, 0x59a1b62b, 0x9a806f4e, 0x12ebe410,
    0x4d05eab8, 0xba13a0c7, 0x11ab3116, 0x2e015d63, 0x8ca5a05a, 0xb503922c, 0xfee394da, 0x06be15d7,
    0x5bfdb854, 0x804ef705, 0x40ceeaf2, 0x7aa76b71, 0xb2fe89cb, 0xfcc95a68, 0xf1c406c7, 0x1d461c35,
    0x39397433, 0x47b3e759, 0x0d1c24d1, 0x6d3a3a92, 0x74f75f43, 0xa1341251, 0xee6ad556, 0x1b821f01,
    0x1204dc7c, 0xda05b8d8, 0x06308d41, 0x48322ae6, 0x849e892c, 0x35358e27, 0xd62dd592, 0x040fcafb,
    0x09700b84, 0x2566c62f, 0xa0bf8660, 0x33183a76, 0x575058f1, 0xd9398f59, 0x39d1cd34, 0x056d2ece,
    0xe6a27a36, 0x049ea3b7, 0x053edbbc, 0xefebe603, 0x3ace9ed4, 0x8424b45a, 0xa688795e, 0x287c8390,
    0xaa8d931a, 0xda32d465, 0x61808f9c, 0x2669f685, 0xe4c8b085, 0x247bab46, 0x81d6021a, 0x0d3b6687,
    0xf5322f3c, 0x27ea2192, 0x658fe9a7, 0xb11884e9, 0xa053c069, 0x3a8623bc, 0x25e139a6, 0x128ff3f0,
    0xbcef1af2, 0x9b8e226e, 0xdf406b60, 0x9e45f1ab, 0xd5a7bb3d, 0x538dd257, 0x7882a3bc, 0x0a389303,
    0xc09e9100, 0x6e482404, 0x590025b2, 0x0d7591c7, 0x3a5ebe11, 0xa4022779, 0xc1a94ca8, 0x0164a6c3,
    0xd84fd030, 0xc722bd69, 0xcf52162c, 0x600e4a26, 0x45f3a7e9, 0xfb727ed7, 0x69fb275c, 0x1652a7b2,
    0x575c07e2, 0x400efaff, 0x4b8f9ac5, 0x55237349, 0x81e7ad37, 0xaa79abed, 0x084d2e39, 0x1ac6e5b8,
    0xc98a20fe, 0xa0a29422, 0x65935c9d, 0x73d462ca, 0xd44582f7, 0xe1ba4a6e, 0x0c3a82b6, 0x28fc14c0,
    0x80890267, 0x87596414, 0xe4c00349, 0x4a3a78b5, 0x52a6b17e, 0x49004fdd, 0x65e6ea12, 0x284517dd,
    0x80d13d9c, 0x636e7355, 0x2445ffd6, 0xa22bf374, 0x1eb203d8, 0x56452ac0, 0x2963f9e7, 0x1860ef94
};

const uint32_t inverse_roots_of_unity[(S + 1)*8] = {
    0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1,
    0xa0000006, 0x974bc177, 0xda58a367, 0xf13771b2, 0x0908122e, 0x51e1a247, 0x4729c0fa, 0x2259d6b1,
    0x51231076, 0xc46cb7fc, 0xc871df70, 0xc8f4d11b, 0x0d77ebee, 0x2a9ee33d, 0xa770975a, 0x052cd33d,
    0xdda29e9b, 0x3207cbdd, 0x262fbefc, 0x77ca35a6, 0xb059de38, 0x3d7e07df, 0x32bd1157, 0x12a91d5f,
    0xa5d2b60e, 0x302fa787, 0xb54036b7, 0x80cc914f, 0xd235eb3b, 0x4f65c0b3, 0x5108a15e, 0x0adb1208,
    0x001fcfb4, 0x283ae711, 0xfeb4f32a, 0x05ead253, 0xfa1144ca, 0xcbc338ab, 0xdb78fc8e, 0x300e4c10,
    0xc5757884, 0x91c0d6fa, 0xe6bf4559, 0xa782859b, 0xb987f6cd, 0xc3ce14ed, 0x6d591f67, 0x0be9c8c5,
    0xd38119f0, 0x8ee513b2, 0x6bab8280, 0x87ee89f4, 0xc4cc2cf4, 0x7fab75cb, 0xd8704b2c, 0x1bef2752,
    0xa5a90b53, 0x9bf52a88, 0x523899e2, 0x39ccb835, 0x3be1b610, 0x6a3cc419, 0xccac2b38, 0x134c2780,
    0x8adb9263, 0x7e1bb7c1, 0x1c1e4a45, 0x08278277, 0xa915deb6, 0xc3b2b36d, 0xee822f9a, 0x0b9db5e6,
    0x4388ee4f, 0x9a42b93a, 0x56abb91e, 0x16c835b1, 0xa3b49b70, 0x5fcd5d77, 0xe5e81852, 0x1d24a254,
    0x36fa814f, 0xa2be1e2a, 0x0dd79739, 0xe25ec761, 0xd6cafca5, 0xd0725623, 0xc117d364, 0x068d1538,
    0x6c9fc2d9, 0x422d9ca4, 0x4b73a08d, 0xec33febb, 0xe4da54b6, 0x69366b9e, 0x30dc6b9f, 0x2b22e3d4,
    0x2ba9466a, 0xf2fe10e9, 0x67af44ce, 0x77a9e3c1, 0x841b37c9, 0x723cad05, 0xb0a2cbb1, 0x1768636e,
    0x708c94ca, 0xea9351a4, 0xa5f17c35, 0xac912ec7, 0x23d9760a, 0x920b7696, 0x6cf0bc39, 0x2c3adf7c,
    0x503045fa, 0x39635a33, 0x30d9f86e, 0x4f4b5b8d, 0x3979eba3, 0xefba9fb0, 0xa7cd1545, 0x2d4d2d2d,
    0x50573894, 0x489362b6, 0x43d35763, 0x0a19b790, 0xbbc04fa8, 0xda6a5ff6, 0x81855e62, 0x2f6a7c28,
    0x8da73c33, 0x8c193fa1, 0x6d0221ba, 0x478e6f45, 0x329bae82, 0x6817e7a3, 0xeba7a65e, 0x041fd6a3,
    0x13cbc9c3, 0x645bee00, 0x1d4ad1a7, 0xa81ac501, 0xb54ea975, 0x100ef5ed, 0xdad08cdd, 0x04d3bb8b,
    0x2886e2a5, 0x1f296f02, 0x1e34eb1c, 0x550e4e0b, 0xb4688775, 0xdf87a2c1, 0x3ba99c65, 0x2707e80c,
    0x6ed9e349, 0x29a73028, 0xd9689b7f, 0x1ff90914, 0x251e133c, 0x24bbb6ac, 0x6283dfd6, 0x022d7574,
    0xc88809a8, 0x169b4cba, 0x67e4fbe3, 0x457d00f8, 0xb96932b3, 0x6801ddc2, 0x914e975b, 0x048e0c9a,
    0x2fb90b14, 0xd8bb9583, 0xa0c38601, 0xb02515b9, 0x182c7fb7, 0x5e77f76e, 0xd8c1deb4, 0x2d000c05,
    0xf50e3d58, 0x0b095acd, 0xa99eae65, 0x52ef5716, 0x1b9421f0, 0x9b866525, 0x1e965c85, 0x23c1ca18,
    0xd8e96c48, 0x667f0e72, 0x4948a4fa, 0x94109e0f, 0xafe7d786, 0xf8504de3, 0x85a4f030, 0x2cd33fae,
    0xa60bd4de, 0x5fde22a0, 0x273daee6, 0xc2222a37, 0x8e437749, 0x577b5a27, 0xae1d702f, 0x0e8a14ab,
    0x70784d69, 0xf90ad83d, 0x90089c33, 0xc98f3964, 0x6a63be44, 0x538bf7a4, 0xcf5c3c58, 0x1f2fab5c,
    0x2c80552b, 0x9b457955, 0xe04e7089, 0x5821ebc4, 0x2f55cc8d, 0x052c529f, 0x8e3fa253, 0x2d2ec870,
    0x584bb683, 0x89bcc016, 0x0164a50c, 0xe8d9887f, 0x795eda3d, 0x755e95cb, 0x1323b130, 0x0f572b87
};

const uint32_t domain_size_inverse[(S + 1)*8] = {
    0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1,
    0x1ffffffe, 0x783c14d8, 0x0c8d1edd, 0xaf982f6f, 0xfcfd4f45, 0x8f5f7492, 0x3d9cbfac, 0x1f37631a,
    0x0fffffff, 0xbc1e0a6c, 0x86468f6e, 0xd7cc17b7, 0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0x0f9bb18d,
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
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000010
};
}//native
}//at
