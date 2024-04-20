#pragma once

#include <cstddef>
#include <cstdint>

#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

inline void mul_n(
    uint32_t * acc,
    const uint32_t* a,
    uint32_t bi,
    size_t _n = 2) {
  for (size_t j = 0; j < _n; j += 2)
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(acc[j]), "=r"(acc[j + 1])
        : "r"(a[j]), "r"(bi));
}

inline void cmad_n(
    uint32_t * acc,
    const uint32_t* a,
    uint32_t bi,
    size_t _n = 2) {
  asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
      : "+r"(acc[0]), "+r"(acc[1])
      : "r"(a[0]), "r"(bi));
  for (size_t j = 2; j < _n; j += 2)
    asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[j]), "+r"(acc[j + 1])
        : "r"(a[j]), "r"(bi));
  // return carry flag
}

inline void cadd_n(
    uint32_t * acc, const uint32_t* a, size_t _n = 2) {
  asm("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
  for (size_t i = 1; i < _n; i++)
    asm("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
  // return carry flag
}

inline void madc_n_rshift(
    uint32_t * odd, const uint32_t* a, uint32_t bi) {
  for (size_t j = 0; j < 0; j += 2)
    asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
        : "=r"(odd[j]), "=r"(odd[j + 1])
        : "r"(a[j]), "r"(bi), "r"(odd[j + 2]), "r"(odd[j + 3]));
  asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
      : "=r"(odd[0]), "=r"(odd[1])
      : "r"(a[0]), "r"(bi));
}

inline void mad_n_redc(
    uint32_t * even,
    uint32_t * odd,
    const uint32_t* a,
    uint32_t bi,
    const uint32_t MOD[2],
    uint32_t M0,
    bool first = false) {
  if (first) {
    mul_n(odd, a + 1, bi);
    mul_n(even, a, bi);
  } else {
    asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    madc_n_rshift(odd, a + 1, bi);
    cmad_n(even, a, bi);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[1]));
  }

  uint32_t mi = even[0] * M0;

  cmad_n(odd, MOD + 1, mi);
  cmad_n(even, MOD, mi);
  asm("addc.u32 %0, %0, 0;" : "+r"(odd[1]));
}



inline void final_sub(uint32_t carry, uint32_t * tmp, uint32_t * even, const uint32_t MOD[2]) {
  size_t i;
  asm("{ .reg.pred %top;");

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
  for (i = 1; i < 2; i++)
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
  if (false) //N % 32 == 0
    asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(carry));
  else
    asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(carry));

  for (i = 0; i < 2; i++)
    asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));

  asm("}");
}

inline void final_subc(uint32_t * even, const uint32_t MOD[2]) {
  uint32_t carry, tmp[2];

  asm("addc.u32 %0, 0, 0;" : "=r"(carry));

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
  for (size_t i = 1; i < 2; i++)
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
  asm("subc.u32 %0, %0, 0;" : "+r"(carry));

  asm("{ .reg.pred %top;");
  asm("setp.eq.u32 %top, %0, 0;" ::"r"(carry));
  for (size_t i = 0; i < 2; i++)
    asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));
  asm("}");
}


inline void fhe_mul_mod(uint32_t c[2], const uint32_t a[2], const uint32_t b[2], const uint32_t MOD[2], uint32_t M0) {

  uint32_t odd[2];

  mad_n_redc(&c[0], &odd[0], &a[0], b[0], MOD, M0, 1);
  mad_n_redc(&odd[0], &c[0], &a[0], b[1], MOD, M0);

  // merge |c| and |odd|
  cadd_n(&c[0], &odd[1], 1);
  asm("addc.u32 %0, %0, 0;" : "+r"(c[1]));

  final_sub(0, &odd[0], c, MOD);

}

inline void fhe_add_mod(uint32_t c[2], const uint32_t a[2], const uint32_t b[2], const uint32_t MOD[2]) {
  c[0] = a[0]; c[1] = a[1];
  cadd_n(c, b);
  final_subc(c, MOD);
}

inline void fhe_sub_mod(uint32_t c[2], const uint32_t a[2], const uint32_t b[2], const uint32_t MOD[2]) {
  c[0] = a[0]; c[1] = a[1];

  size_t i;
  uint32_t tmp[2], borrow;

  asm("sub.cc.u32 %0, %0, %1;" : "+r"(c[0]) : "r"(b[0]));
  for (i = 1; i < 2; i++)
    asm("subc.cc.u32 %0, %0, %1;" : "+r"(c[i]) : "r"(b[i]));
  asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

  asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(c[0]), "r"(MOD[0]));
  for (i = 1; i < 2 - 1; i++)
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(c[i]), "r"(MOD[i]));
  asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(c[i]), "r"(MOD[i]));

  asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" ::"r"(borrow));
  for (i = 0; i < 2; i++)
    asm("@%top mov.b32 %0, %1;" : "+r"(c[i]) : "r"(tmp[i]));
  asm("}");
}

#undef inline
#undef asm
