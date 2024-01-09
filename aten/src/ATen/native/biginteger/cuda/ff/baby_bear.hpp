// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "bb31_t.cuh"

#ifdef __NVCC__
# ifdef __CUDA_ARCH__   // device-side field types
typedef bb31_t fr_t;
# endif
#endif

#ifndef __CUDA_ARCH__   // host-side stand-in to make CUDA code compile
#include <cstdint>      // currently only used as a stand-in and should
class fr_t {            // not be used for any other purpose
    uint32_t val;
    static const uint32_t M = 0x77ffffff;
public:
    using mem_t = fr_t;
    static const uint32_t degree = 1;
    static const uint32_t nbits = 31;
    static const uint32_t MOD = 0x78000001;
    // this is used in constant declaration, e.g. as bb31_t{11}
    inline constexpr fr_t(int a) : val(((uint64_t)a << 32) % MOD) {}

    inline fr_t()                       {}
    inline fr_t(uint32_t a) : val(a)    {}
    inline operator uint32_t() const    { return val; }
    static inline const fr_t one()      { return 1; }
    inline fr_t operator+=(fr_t b)      { return val; }
    inline fr_t operator-=(fr_t b)      { return val; }
    inline fr_t operator*=(fr_t b)      { return val; }
    
    inline fr_t sqr()                   { return val; }
    inline void zero()                  { val = 0; }
    inline bool is_zero() const         { return val==0; }
};
#endif
