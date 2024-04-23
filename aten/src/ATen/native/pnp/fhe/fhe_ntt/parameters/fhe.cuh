#pragma once
#include <array>
#include <cstdint>

namespace at {
namespace native {

template <>
struct NTTHyperParam<fhe_moduli1> {
  constexpr static const std::array<uint32_t, 2> group_gen = {
    0x004fffec, 0x00000000
};
  constexpr static const std::array<uint32_t, 2> group_gen_inverse = {
    0xccca6667, 0x0ccccccc
};

  constexpr static const std::size_t S = 18;

  constexpr static const std::array<uint32_t, (S + 1)* 8>
      forward_roots_of_unity = {
            0x0007fffe, 0x00000000,
            0xfff40003, 0x0fffffff,
            0x7ffc8041, 0x0bfdfeff,
            0x59676d8a, 0x06b8e2e3,
            0x5f15115c, 0x0e08c4a2,
            0x0d2d9c8a, 0x07efec09,
            0xb02b4e39, 0x0a5c4d5c,
            0xe7ab7dac, 0x0bab364d,
            0x3569f178, 0x0c48c7c8,
            0x7e6e398c, 0x06952721,
            0x11c01e3c, 0x0adfdd76,
            0xab2658f5, 0x0a651b37,
            0xabafc98c, 0x093d1def,
            0x2ee1acf9, 0x096bccb2,
            0x667688b3, 0x0659bf4e,
            0xe18aa553, 0x0d2dcf95,
            0x0ff45466, 0x05dcf399,
            0x31e1c5af, 0x0a7f3684,
            0xad44c3c2, 0x0d55c9bd
        };

  constexpr static const std::array<uint32_t, (S + 1)* 8>
      inverse_roots_of_unity = {
            0x0007fffe, 0x00000000,
            0xfff40003, 0x0fffffff,
            0x7fff7fc0, 0x04020100,
            0xc8ac8ede, 0x039f4fc3,
            0xc028fd6e, 0x016a6223,
            0xa34a7871, 0x0dd263b2,
            0xc383984b, 0x0dd6726a,
            0xc9c65175, 0x0aa5af59,
            0x10f2f57a, 0x04f556ad,
            0xf7d63da6, 0x02c50b58,
            0x6828738e, 0x0df06026,
            0x710b1124, 0x028381be,
            0xf9a1dcc8, 0x0f0b8285,
            0x41fe24e2, 0x0ba4b60f,
            0x8df30f2e, 0x0f17d882,
            0xa9b37fe4, 0x0358e2d6,
            0x1f717e9a, 0x05a0d7a5,
            0xb894022b, 0x0375c32e,
            0x380c8797, 0x09977bfd
        };

  constexpr static const std::array<uint32_t, (S + 1)* 8> domain_size_inverse =
      {
            0x0007fffe, 0x00000000,
            0x0003ffff, 0x00000000,
            0x00000000, 0x08000000,
            0x00000000, 0x04000000,
            0x00000000, 0x02000000,
            0x00000000, 0x01000000,
            0x00000000, 0x00800000,
            0x00000000, 0x00400000,
            0x00000000, 0x00200000,
            0x00000000, 0x00100000,
            0x00000000, 0x00080000,
            0x00000000, 0x00040000,
            0x00000000, 0x00020000,
            0x00000000, 0x00010000,
            0x00000000, 0x00008000,
            0x00000000, 0x00004000,
            0x00000000, 0x00002000,
            0x00000000, 0x00001000,
            0x00000000, 0x00000800
      };
};

} // namespace native
} // namespace at
