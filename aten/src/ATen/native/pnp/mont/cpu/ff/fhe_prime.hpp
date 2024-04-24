#pragma once

#include <c10/util/BigInteger.h>

namespace at {
namespace native {

#define DEF_FHE_PRIME(name)                               \
  struct name {                                           \
    using mem_t = name;                                   \
    inline name() = default;                              \
    inline name(const name& a) {}                         \
    name& operator=(const name& other) {                  \
      std::runtime_error("Not implemented");              \
      return *this;                                       \
    }                                                     \
    inline name& operator+=(const name& b) {              \
      return *this;                                       \
    }                                                     \
    friend inline name operator+(name a, const name& b) { \
      return a += b;                                      \
    }                                                     \
    inline name& operator-=(const name& b) {              \
      return *this;                                       \
    }                                                     \
    friend inline name operator-(name a, const name& b) { \
      return a -= b;                                      \
    }                                                     \
    inline name& operator*=(const name& b) {              \
      return *this;                                       \
    }                                                     \
    friend inline name operator*(name a, const name& b) { \
      return a *= b;                                      \
    }                                                     \
    inline name& operator/=(const name& b) {              \
      return *this;                                       \
    }                                                     \
    friend inline name operator/(name a, const name& b) { \
      return a /= b;                                      \
    }                                                     \
    inline void from() {}                                 \
    inline void to() {}                                   \
  };

APPLY_ALL_FHE_PRIME(DEF_FHE_PRIME)
#undef DEF_FHE_PRIME

} // namespace native
} // namespace at
