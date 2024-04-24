#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/BigInteger.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>
#include <c10/util/bits.h>
#include <c10/util/complex.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint2x4.h>
#include <c10/util/quint4x2.h>
#include <c10/util/quint8.h>

#include <complex>
#include <cstdint>
#include <ostream>

namespace c10 {

// For the macros below:
// NB: If you want to macro some code for all non-QInt scalar types (i.e. types
// with complete information, you probably want one of the
// AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND
// macros below, which are designed to behave similarly to the Dispatch macros
// with the same name.

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)        \
  _(uint8_t, Byte) /* 0 */                                      \
  _(int8_t, Char) /* 1 */                                       \
  _(int16_t, Short) /* 2 */                                     \
  _(int, Int) /* 3 */                                           \
  _(int64_t, Long) /* 4 */                                      \
  _(at::Half, Half) /* 5 */                                     \
  _(float, Float) /* 6 */                                       \
  _(double, Double) /* 7 */                                     \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */               \
  _(c10::complex<float>, ComplexFloat) /* 9 */                  \
  _(c10::complex<double>, ComplexDouble) /* 10 */               \
  _(bool, Bool) /* 11 */                                        \
  _(c10::qint8, QInt8) /* 12 */                                 \
  _(c10::quint8, QUInt8) /* 13 */                               \
  _(c10::qint32, QInt32) /* 14 */                               \
  _(at::BFloat16, BFloat16) /* 15 */                            \
  _(c10::quint4x2, QUInt4x2) /* 16 */                           \
  _(c10::quint2x4, QUInt2x4) /* 17 */                           \
  _(c10::bits1x8, Bits1x8) /* 18 */                             \
  _(c10::bits2x4, Bits2x4) /* 19 */                             \
  _(c10::bits4x2, Bits4x2) /* 20 */                             \
  _(c10::bits8, Bits8) /* 21 */                                 \
  _(c10::bits16, Bits16) /* 22 */                               \
  _(c10::Float8_e5m2, Float8_e5m2) /* 23 */                     \
  _(c10::Float8_e4m3fn, Float8_e4m3fn) /* 24 */                 \
  _(uint64_t, ULong) /* 25 */                                   \
  _(c10::Field64, Field64) /* 26 */                             \
  _(c10::BigInteger, BigInteger) /* 27 */                       \
  _(c10::BigInteger_Mont, BigInteger_Mont) /* 28 */             \
  _(c10::FiniteField, FiniteField) /* 29 */                     \
  _(c10::NOT_CURVE, NOT_CURVE) /* 30 */                         \
  _(c10::ALT_BN128_Fr_G1_Base, ALT_BN128_Fr_G1_Base) /* 31.1 */ \
  _(c10::ALT_BN128_Fr_G2_Base, ALT_BN128_Fr_G2_Base) /* 31.2 */ \
  _(c10::ALT_BN128_Fq_G1_Base, ALT_BN128_Fq_G1_Base) /* 31.3 */ \
  _(c10::ALT_BN128_Fq_G2_Base, ALT_BN128_Fq_G2_Base) /* 31.4 */ \
  _(c10::ALT_BN128_Fr_G1_Mont, ALT_BN128_Fr_G1_Mont) /* 31.5 */ \
  _(c10::ALT_BN128_Fr_G2_Mont, ALT_BN128_Fr_G2_Mont) /* 31.6 */ \
  _(c10::ALT_BN128_Fq_G1_Mont, ALT_BN128_Fq_G1_Mont) /* 31.7 */ \
  _(c10::ALT_BN128_Fq_G2_Mont, ALT_BN128_Fq_G2_Mont) /* 31.8 */ \
  _(c10::BLS12_377_Fr_G1_Base, BLS12_377_Fr_G1_Base) /* 32.1 */ \
  _(c10::BLS12_377_Fr_G2_Base, BLS12_377_Fr_G2_Base) /* 32.2 */ \
  _(c10::BLS12_377_Fq_G1_Base, BLS12_377_Fq_G1_Base) /* 32.3 */ \
  _(c10::BLS12_377_Fq_G2_Base, BLS12_377_Fq_G2_Base) /* 32.4 */ \
  _(c10::BLS12_377_Fr_G1_Mont, BLS12_377_Fr_G1_Mont) /* 32.5 */ \
  _(c10::BLS12_377_Fr_G2_Mont, BLS12_377_Fr_G2_Mont) /* 32.6 */ \
  _(c10::BLS12_377_Fq_G1_Mont, BLS12_377_Fq_G1_Mont) /* 32.7 */ \
  _(c10::BLS12_377_Fq_G2_Mont, BLS12_377_Fq_G2_Mont) /* 32.8 */ \
  _(c10::BLS12_381_Fr_G1_Base, BLS12_381_Fr_G1_Base) /* 33.1 */ \
  _(c10::BLS12_381_Fr_G2_Base, BLS12_381_Fr_G2_Base) /* 33.2 */ \
  _(c10::BLS12_381_Fq_G1_Base, BLS12_381_Fq_G1_Base) /* 33.3 */ \
  _(c10::BLS12_381_Fq_G2_Base, BLS12_381_Fq_G2_Base) /* 33.4 */ \
  _(c10::BLS12_381_Fr_G1_Mont, BLS12_381_Fr_G1_Mont) /* 33.5 */ \
  _(c10::BLS12_381_Fr_G2_Mont, BLS12_381_Fr_G2_Mont) /* 33.6 */ \
  _(c10::BLS12_381_Fq_G1_Mont, BLS12_381_Fq_G1_Mont) /* 33.7 */ \
  _(c10::BLS12_381_Fq_G2_Mont, BLS12_381_Fq_G2_Mont) /* 33.8 */ \
  _(c10::MNT4753_Fr_G1_Base, MNT4753_Fr_G1_Base) /* 34.1 */     \
  _(c10::MNT4753_Fr_G2_Base, MNT4753_Fr_G2_Base) /* 34.2 */     \
  _(c10::MNT4753_Fq_G1_Base, MNT4753_Fq_G1_Base) /* 34.3 */     \
  _(c10::MNT4753_Fq_G2_Base, MNT4753_Fq_G2_Base) /* 34.4 */     \
  _(c10::MNT4753_Fr_G1_Mont, MNT4753_Fr_G1_Mont) /* 34.5 */     \
  _(c10::MNT4753_Fr_G2_Mont, MNT4753_Fr_G2_Mont) /* 34.6 */     \
  _(c10::MNT4753_Fq_G1_Mont, MNT4753_Fq_G1_Mont) /* 34.7 */     \
  _(c10::MNT4753_Fq_G2_Mont, MNT4753_Fq_G2_Mont) /* 34.8 */     \
  _(c10::PALLAS_Fr_G1_Base, PALLAS_Fr_G1_Base) /* 35.1 */       \
  _(c10::PALLAS_Fr_G2_Base, PALLAS_Fr_G2_Base) /* 35.2 */       \
  _(c10::PALLAS_Fq_G1_Base, PALLAS_Fq_G1_Base) /* 35.3 */       \
  _(c10::PALLAS_Fq_G2_Base, PALLAS_Fq_G2_Base) /* 35.4 */       \
  _(c10::PALLAS_Fr_G1_Mont, PALLAS_Fr_G1_Mont) /* 35.5 */       \
  _(c10::PALLAS_Fr_G2_Mont, PALLAS_Fr_G2_Mont) /* 35.6 */       \
  _(c10::PALLAS_Fq_G1_Mont, PALLAS_Fq_G1_Mont) /* 35.7 */       \
  _(c10::PALLAS_Fq_G2_Mont, PALLAS_Fq_G2_Mont) /* 35.8 */       \
  _(c10::VESTA_Fr_G1_Base, VESTA_Fr_G1_Base) /* 36.1 */         \
  _(c10::VESTA_Fr_G2_Base, VESTA_Fr_G2_Base) /* 36.2 */         \
  _(c10::VESTA_Fq_G1_Base, VESTA_Fq_G1_Base) /* 36.3 */         \
  _(c10::VESTA_Fq_G2_Base, VESTA_Fq_G2_Base) /* 36.4 */         \
  _(c10::VESTA_Fr_G1_Mont, VESTA_Fr_G1_Mont) /* 36.5 */         \
  _(c10::VESTA_Fr_G2_Mont, VESTA_Fr_G2_Mont) /* 36.6 */         \
  _(c10::VESTA_Fq_G1_Mont, VESTA_Fq_G1_Mont) /* 36.7 */         \
  _(c10::VESTA_Fq_G2_Mont, VESTA_Fq_G2_Mont) /* 36.8 */         \
  _(c10::FHE_PRIME_0, FHE_PRIME_0) /* 37.0 */                   \
  _(c10::FHE_PRIME_1, FHE_PRIME_1) /* 37.1 */                   \
  _(c10::FHE_PRIME_2, FHE_PRIME_2) /* 37.2 */                   \
  _(c10::FHE_PRIME_3, FHE_PRIME_3) /* 37.3 */                   \
  _(c10::FHE_PRIME_4, FHE_PRIME_4) /* 37.4 */                   \
  _(c10::FHE_PRIME_5, FHE_PRIME_5) /* 37.5 */                   \
  _(c10::FHE_PRIME_6, FHE_PRIME_6) /* 37.6 */                   \
  _(c10::FHE_PRIME_7, FHE_PRIME_7) /* 37.7 */                   \
  _(c10::FHE_PRIME_8, FHE_PRIME_8) /* 37.8 */                   \
  _(c10::FHE_PRIME_9, FHE_PRIME_9) /* 37.9 */                   \
  _(c10::FHE_PRIME_10, FHE_PRIME_10) /* 37.10 */                \
  _(c10::FHE_PRIME_11, FHE_PRIME_11) /* 37.11 */                \
  _(c10::FHE_PRIME_12, FHE_PRIME_12) /* 37.12 */                \
  _(c10::FHE_PRIME_13, FHE_PRIME_13) /* 37.13 */                \
  _(c10::FHE_PRIME_14, FHE_PRIME_14) /* 37.14 */                \
  _(c10::FHE_PRIME_15, FHE_PRIME_15) /* 37.15 */                \
  _(c10::FHE_PRIME_16, FHE_PRIME_16) /* 37.16 */                \
  _(c10::FHE_PRIME_17, FHE_PRIME_17) /* 37.17 */                \
  _(c10::FHE_PRIME_18, FHE_PRIME_18) /* 37.18 */                \
  _(c10::FHE_PRIME_19, FHE_PRIME_19) /* 37.19 */                \
  _(c10::FHE_PRIME_20, FHE_PRIME_20) /* 37.20 */                \
  _(c10::FHE_PRIME_21, FHE_PRIME_21) /* 37.21 */                \
  _(c10::FHE_PRIME_22, FHE_PRIME_22) /* 37.22 */                \
  _(c10::FHE_PRIME_23, FHE_PRIME_23) /* 37.23 */                \
  _(c10::FHE_PRIME_24, FHE_PRIME_24) /* 37.24 */                \
  _(c10::FHE_PRIME_25, FHE_PRIME_25) /* 37.25 */                \
  _(c10::FHE_PRIME_26, FHE_PRIME_26) /* 37.26 */                \
  _(c10::FHE_PRIME_27, FHE_PRIME_27) /* 37.27 */                \
  _(c10::FHE_PRIME_28, FHE_PRIME_28) /* 37.28 */                \
  _(c10::FHE_PRIME_29, FHE_PRIME_29) /* 37.29 */                \
  _(c10::FHE_PRIME_30, FHE_PRIME_30) /* 37.30 */                \
  _(c10::FHE_PRIME_31, FHE_PRIME_31) /* 37.31 */                \
  _(c10::FHE_PRIME_32, FHE_PRIME_32) /* 37.32 */                \
  _(c10::FHE_PRIME_33, FHE_PRIME_33) /* 37.33 */                \
  _(c10::FHE_PRIME_34, FHE_PRIME_34) /* 37.34 */                \
  _(c10::FHE_PRIME_35, FHE_PRIME_35) /* 37.35 */                \
  _(c10::FHE_PRIME_36, FHE_PRIME_36) /* 37.36 */                \
  _(c10::FHE_PRIME_37, FHE_PRIME_37) /* 37.37 */                \
  _(c10::FHE_PRIME_38, FHE_PRIME_38) /* 37.38 */                \
  _(c10::FHE_PRIME_39, FHE_PRIME_39) /* 37.39 */                \
  _(c10::FHE_PRIME_40, FHE_PRIME_40) /* 37.40 */                \
  _(c10::FHE_PRIME_41, FHE_PRIME_41) /* 37.41 */                \
  _(c10::FHE_PRIME_42, FHE_PRIME_42) /* 37.42 */                \
  _(c10::FHE_PRIME_43, FHE_PRIME_43) /* 37.43 */                \
  _(c10::FHE_PRIME_44, FHE_PRIME_44) /* 37.44 */                \
  _(c10::FHE_PRIME_45, FHE_PRIME_45) /* 37.45 */                \
  _(c10::FHE_PRIME_46, FHE_PRIME_46) /* 37.46 */                \
  _(c10::FHE_PRIME_47, FHE_PRIME_47) /* 37.47 */                \
  _(c10::FHE_PRIME_48, FHE_PRIME_48) /* 37.48 */                \
  _(c10::FHE_PRIME_49, FHE_PRIME_49) /* 37.49 */                \
  _(c10::FHE_PRIME_50, FHE_PRIME_50) /* 37.50 */                \
  _(c10::FHE_PRIME_51, FHE_PRIME_51) /* 37.51 */                \
  _(c10::FHE_PRIME_52, FHE_PRIME_52) /* 37.52 */                \
  _(c10::FHE_PRIME_53, FHE_PRIME_53) /* 37.53 */                \
  _(c10::FHE_PRIME_54, FHE_PRIME_54) /* 37.54 */                \
  _(c10::FHE_PRIME_55, FHE_PRIME_55) /* 37.55 */                \
  _(c10::FHE_PRIME_56, FHE_PRIME_56) /* 37.56 */                \
  _(c10::FHE_PRIME_57, FHE_PRIME_57) /* 37.57 */                \
  _(c10::FHE_PRIME_58, FHE_PRIME_58) /* 37.58 */                \
  _(c10::FHE_PRIME_59, FHE_PRIME_59) /* 37.59 */                \
  _(c10::FHE_PRIME_60, FHE_PRIME_60) /* 37.60 */                \
  _(c10::FHE_PRIME_61, FHE_PRIME_61) /* 37.61 */                \
  _(c10::FHE_PRIME_62, FHE_PRIME_62) /* 37.62 */                \
  _(c10::FHE_PRIME_63, FHE_PRIME_63) /* 37.63 */                \
  _(c10::FHE_PRIME_64, FHE_PRIME_64) /* 37.64 */                \
  _(c10::FHE_PRIME_65, FHE_PRIME_65) /* 37.65 */                \
  _(c10::FHE_PRIME_66, FHE_PRIME_66) /* 37.66 */                \
  _(c10::FHE_PRIME_67, FHE_PRIME_67) /* 37.67 */                \
  _(c10::FHE_PRIME_68, FHE_PRIME_68) /* 37.68 */                \
  _(c10::FHE_PRIME_69, FHE_PRIME_69) /* 37.69 */                \
  _(c10::FHE_PRIME_70, FHE_PRIME_70) /* 37.70 */                \
  _(c10::FHE_PRIME_71, FHE_PRIME_71) /* 37.71 */                \
  _(c10::FHE_PRIME_72, FHE_PRIME_72) /* 37.72 */                \
  _(c10::FHE_PRIME_73, FHE_PRIME_73) /* 37.73 */                \
  _(c10::FHE_PRIME_74, FHE_PRIME_74) /* 37.74 */                \
  _(c10::FHE_PRIME_75, FHE_PRIME_75) /* 37.75 */                \
  _(c10::FHE_PRIME_76, FHE_PRIME_76) /* 37.76 */                \
  _(c10::FHE_PRIME_77, FHE_PRIME_77) /* 37.77 */                \
  _(c10::FHE_PRIME_78, FHE_PRIME_78) /* 37.78 */                \
  _(c10::FHE_PRIME_79, FHE_PRIME_79) /* 37.79 */                \
  _(c10::FHE_PRIME_80, FHE_PRIME_80) /* 37.80 */                \
  _(c10::FHE_PRIME_81, FHE_PRIME_81) /* 37.81 */                \
  _(c10::FHE_PRIME_82, FHE_PRIME_82) /* 37.82 */                \
  _(c10::FHE_PRIME_83, FHE_PRIME_83) /* 37.83 */                \
  _(c10::FHE_PRIME_84, FHE_PRIME_84) /* 37.84 */                \
  _(c10::FHE_PRIME_85, FHE_PRIME_85) /* 37.85 */                \
  _(c10::FHE_PRIME_86, FHE_PRIME_86) /* 37.86 */                \
  _(c10::FHE_PRIME_87, FHE_PRIME_87) /* 37.87 */                \
  _(c10::FHE_PRIME_88, FHE_PRIME_88) /* 37.88 */                \
  _(c10::FHE_PRIME_89, FHE_PRIME_89) /* 37.89 */                \
  _(c10::FHE_PRIME_90, FHE_PRIME_90) /* 37.90 */                \
  _(c10::FHE_PRIME_91, FHE_PRIME_91) /* 37.91 */                \
  _(c10::FHE_PRIME_92, FHE_PRIME_92) /* 37.92 */                \
  _(c10::FHE_PRIME_93, FHE_PRIME_93) /* 37.93 */                \
  _(c10::FHE_PRIME_94, FHE_PRIME_94) /* 37.94 */                \
  _(c10::FHE_PRIME_95, FHE_PRIME_95) /* 37.95 */                \
  _(c10::FHE_PRIME_96, FHE_PRIME_96) /* 37.96 */                \
  _(c10::FHE_PRIME_97, FHE_PRIME_97) /* 37.97 */                \
  _(c10::FHE_PRIME_98, FHE_PRIME_98) /* 37.98 */                \
  _(c10::FHE_PRIME_99, FHE_PRIME_99) /* 37.99 */                \
  _(c10::FHE_PRIME_100, FHE_PRIME_100) /* 37.100 */             \
  _(c10::FHE_PRIME_101, FHE_PRIME_101) /* 37.101 */             \
  _(c10::FHE_PRIME_102, FHE_PRIME_102) /* 37.102 */             \
  _(c10::FHE_PRIME_103, FHE_PRIME_103) /* 37.103 */             \
  _(c10::FHE_PRIME_104, FHE_PRIME_104) /* 37.104 */             \
  _(c10::FHE_PRIME_105, FHE_PRIME_105) /* 37.105 */             \
  _(c10::FHE_PRIME_106, FHE_PRIME_106) /* 37.106 */             \
  _(c10::FHE_PRIME_107, FHE_PRIME_107) /* 37.107 */

// If you want to support ComplexHalf for real, add ComplexHalf
// into this macro (and change the name).  But beware: convert()
// doesn't work for all the conversions you need...
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(_) \
  _(uint8_t, Byte)                                                 \
  _(int8_t, Char)                                                  \
  _(int16_t, Short)                                                \
  _(int, Int)                                                      \
  _(int64_t, Long)                                                 \
  _(at::Half, Half)                                                \
  _(float, Float)                                                  \
  _(double, Double)                                                \
  _(c10::complex<float>, ComplexFloat)                             \
  _(c10::complex<double>, ComplexDouble)                           \
  _(bool, Bool)                                                    \
  _(at::BFloat16, BFloat16)                                        \
  _(at::Float8_e5m2, Float8_e5m2)                                  \
  _(at::Float8_e4m3fn, Float8_e4m3fn)

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(at::Half, Half)                            \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<c10::Half>, ComplexHalf)      \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(at::BFloat16, BFloat16)                    \
  _(at::Float8_e5m2, Float8_e5m2)              \
  _(at::Float8_e4m3fn, Float8_e4m3fn)

enum class ScalarType : int16_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

namespace impl {

// These are used to map ScalarTypes to C++ types.

template <c10::ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
  template <>                                                                \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
                                                                             \
    /* This is a workaround for the CUDA bug which prevents */               \
    /* ::detail::ScalarTypeToCType<T>::type being used directly due to */    \
    /* ambiguous reference which can't to be resolved. For some reason it */ \
    /* can't pick between at::detail and at::cuda::detail. */                \
    /* For repro example, please see: */                                     \
    /* https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba */    \
    /* TODO: remove once the bug is fixed. */                                \
    static type t;                                                           \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

} // namespace impl

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)                  \
  template <>                                                                  \
  struct CppTypeToScalarType<cpp_type>                                         \
      : std::                                                                  \
            integral_constant<c10::ScalarType, c10::ScalarType::scalar_type> { \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType)

#undef SPECIALIZE_CppTypeToScalarType

#define AT_FORALL_INT_TYPES(_) \
  _(uint8_t, Byte)             \
  _(int8_t, Char)              \
  _(int16_t, Short)            \
  _(int, Int)                  \
  _(int64_t, Long)

#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(double, Double)

#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _)                            \
  _(uint8_t, Byte)                                                           \
  _(int8_t, Char)                                                            \
  _(int16_t, Short)                                                          \
  _(int, Int)                                                                \
  _(int64_t, Long)                                                           \
  _(float, Float)                                                            \
  _(double, Double)                                                          \
  _(decltype(                                                                \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE>::t), \
    SCALARTYPE)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _)              \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)

#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)

#define AT_FORALL_SCALAR_TYPES_AND4(                                          \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, _)                    \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)

#define AT_FORALL_SCALAR_TYPES_AND5(                                          \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, _)       \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)                                                              \
  _(decltype(                                                                 \
        ::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE5>::t), \
    SCALARTYPE5)

#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define AT_FORALL_FIELD_TYPES(_)                     \
  _(uint64_t, ULong)                                 \
  _(c10::Field64, Field64)                           \
  _(c10::BigInteger, BigInteger)                     \
  _(c10::FiniteField, FiniteField)                   \
  _(c10::ALT_BN128_Fr_G1_Base, ALT_BN128_Fr_G1_Base) \
  _(c10::ALT_BN128_Fr_G2_Base, ALT_BN128_Fr_G2_Base) \
  _(c10::ALT_BN128_Fq_G1_Base, ALT_BN128_Fq_G1_Base) \
  _(c10::ALT_BN128_Fq_G2_Base, ALT_BN128_Fq_G2_Base) \
  _(c10::ALT_BN128_Fr_G1_Mont, ALT_BN128_Fr_G1_Mont) \
  _(c10::ALT_BN128_Fr_G2_Mont, ALT_BN128_Fr_G2_Mont) \
  _(c10::ALT_BN128_Fq_G1_Mont, ALT_BN128_Fq_G1_Mont) \
  _(c10::ALT_BN128_Fq_G2_Mont, ALT_BN128_Fq_G2_Mont) \
  _(c10::BLS12_377_Fr_G1_Base, BLS12_377_Fr_G1_Base) \
  _(c10::BLS12_377_Fr_G2_Base, BLS12_377_Fr_G2_Base) \
  _(c10::BLS12_377_Fq_G1_Base, BLS12_377_Fq_G1_Base) \
  _(c10::BLS12_377_Fq_G2_Base, BLS12_377_Fq_G2_Base) \
  _(c10::BLS12_377_Fr_G1_Mont, BLS12_377_Fr_G1_Mont) \
  _(c10::BLS12_377_Fr_G2_Mont, BLS12_377_Fr_G2_Mont) \
  _(c10::BLS12_377_Fq_G1_Mont, BLS12_377_Fq_G1_Mont) \
  _(c10::BLS12_377_Fq_G2_Mont, BLS12_377_Fq_G2_Mont) \
  _(c10::BLS12_381_Fr_G1_Base, BLS12_381_Fr_G1_Base) \
  _(c10::BLS12_381_Fr_G2_Base, BLS12_381_Fr_G2_Base) \
  _(c10::BLS12_381_Fq_G1_Base, BLS12_381_Fq_G1_Base) \
  _(c10::BLS12_381_Fq_G2_Base, BLS12_381_Fq_G2_Base) \
  _(c10::BLS12_381_Fr_G1_Mont, BLS12_381_Fr_G1_Mont) \
  _(c10::BLS12_381_Fr_G2_Mont, BLS12_381_Fr_G2_Mont) \
  _(c10::BLS12_381_Fq_G1_Mont, BLS12_381_Fq_G1_Mont) \
  _(c10::BLS12_381_Fq_G2_Mont, BLS12_381_Fq_G2_Mont) \
  _(c10::MNT4753_Fr_G1_Base, MNT4753_Fr_G1_Base)     \
  _(c10::MNT4753_Fr_G2_Base, MNT4753_Fr_G2_Base)     \
  _(c10::MNT4753_Fq_G1_Base, MNT4753_Fq_G1_Base)     \
  _(c10::MNT4753_Fq_G2_Base, MNT4753_Fq_G2_Base)     \
  _(c10::MNT4753_Fr_G1_Mont, MNT4753_Fr_G1_Mont)     \
  _(c10::MNT4753_Fr_G2_Mont, MNT4753_Fr_G2_Mont)     \
  _(c10::MNT4753_Fq_G1_Mont, MNT4753_Fq_G1_Mont)     \
  _(c10::MNT4753_Fq_G2_Mont, MNT4753_Fq_G2_Mont)     \
  _(c10::PALLAS_Fr_G1_Base, PALLAS_Fr_G1_Base)       \
  _(c10::PALLAS_Fr_G2_Base, PALLAS_Fr_G2_Base)       \
  _(c10::PALLAS_Fq_G1_Base, PALLAS_Fq_G1_Base)       \
  _(c10::PALLAS_Fq_G2_Base, PALLAS_Fq_G2_Base)       \
  _(c10::PALLAS_Fr_G1_Mont, PALLAS_Fr_G1_Mont)       \
  _(c10::PALLAS_Fr_G2_Mont, PALLAS_Fr_G2_Mont)       \
  _(c10::PALLAS_Fq_G1_Mont, PALLAS_Fq_G1_Mont)       \
  _(c10::PALLAS_Fq_G2_Mont, PALLAS_Fq_G2_Mont)       \
  _(c10::VESTA_Fr_G1_Base, VESTA_Fr_G1_Base)         \
  _(c10::VESTA_Fr_G2_Base, VESTA_Fr_G2_Base)         \
  _(c10::VESTA_Fq_G1_Base, VESTA_Fq_G1_Base)         \
  _(c10::VESTA_Fq_G2_Base, VESTA_Fq_G2_Base)         \
  _(c10::VESTA_Fr_G1_Mont, VESTA_Fr_G1_Mont)         \
  _(c10::VESTA_Fr_G2_Mont, VESTA_Fr_G2_Mont)         \
  _(c10::VESTA_Fq_G1_Mont, VESTA_Fq_G1_Mont)         \
  _(c10::VESTA_Fq_G2_Mont, VESTA_Fq_G2_Mont)         \
  _(c10::FHE_PRIME_0, FHE_PRIME_0)                   \
  _(c10::FHE_PRIME_1, FHE_PRIME_1)                   \
  _(c10::FHE_PRIME_2, FHE_PRIME_2)                   \
  _(c10::FHE_PRIME_3, FHE_PRIME_3)                   \
  _(c10::FHE_PRIME_4, FHE_PRIME_4)                   \
  _(c10::FHE_PRIME_5, FHE_PRIME_5)                   \
  _(c10::FHE_PRIME_6, FHE_PRIME_6)                   \
  _(c10::FHE_PRIME_7, FHE_PRIME_7)                   \
  _(c10::FHE_PRIME_8, FHE_PRIME_8)                   \
  _(c10::FHE_PRIME_9, FHE_PRIME_9)                   \
  _(c10::FHE_PRIME_10, FHE_PRIME_10)                 \
  _(c10::FHE_PRIME_11, FHE_PRIME_11)                 \
  _(c10::FHE_PRIME_12, FHE_PRIME_12)                 \
  _(c10::FHE_PRIME_13, FHE_PRIME_13)                 \
  _(c10::FHE_PRIME_14, FHE_PRIME_14)                 \
  _(c10::FHE_PRIME_15, FHE_PRIME_15)                 \
  _(c10::FHE_PRIME_16, FHE_PRIME_16)                 \
  _(c10::FHE_PRIME_17, FHE_PRIME_17)                 \
  _(c10::FHE_PRIME_18, FHE_PRIME_18)                 \
  _(c10::FHE_PRIME_19, FHE_PRIME_19)                 \
  _(c10::FHE_PRIME_20, FHE_PRIME_20)                 \
  _(c10::FHE_PRIME_21, FHE_PRIME_21)                 \
  _(c10::FHE_PRIME_22, FHE_PRIME_22)                 \
  _(c10::FHE_PRIME_23, FHE_PRIME_23)                 \
  _(c10::FHE_PRIME_24, FHE_PRIME_24)                 \
  _(c10::FHE_PRIME_25, FHE_PRIME_25)                 \
  _(c10::FHE_PRIME_26, FHE_PRIME_26)                 \
  _(c10::FHE_PRIME_27, FHE_PRIME_27)                 \
  _(c10::FHE_PRIME_28, FHE_PRIME_28)                 \
  _(c10::FHE_PRIME_29, FHE_PRIME_29)                 \
  _(c10::FHE_PRIME_30, FHE_PRIME_30)                 \
  _(c10::FHE_PRIME_31, FHE_PRIME_31)                 \
  _(c10::FHE_PRIME_32, FHE_PRIME_32)                 \
  _(c10::FHE_PRIME_33, FHE_PRIME_33)                 \
  _(c10::FHE_PRIME_34, FHE_PRIME_34)                 \
  _(c10::FHE_PRIME_35, FHE_PRIME_35)                 \
  _(c10::FHE_PRIME_36, FHE_PRIME_36)                 \
  _(c10::FHE_PRIME_37, FHE_PRIME_37)                 \
  _(c10::FHE_PRIME_38, FHE_PRIME_38)                 \
  _(c10::FHE_PRIME_39, FHE_PRIME_39)                 \
  _(c10::FHE_PRIME_40, FHE_PRIME_40)                 \
  _(c10::FHE_PRIME_41, FHE_PRIME_41)                 \
  _(c10::FHE_PRIME_42, FHE_PRIME_42)                 \
  _(c10::FHE_PRIME_43, FHE_PRIME_43)                 \
  _(c10::FHE_PRIME_44, FHE_PRIME_44)                 \
  _(c10::FHE_PRIME_45, FHE_PRIME_45)                 \
  _(c10::FHE_PRIME_46, FHE_PRIME_46)                 \
  _(c10::FHE_PRIME_47, FHE_PRIME_47)                 \
  _(c10::FHE_PRIME_48, FHE_PRIME_48)                 \
  _(c10::FHE_PRIME_49, FHE_PRIME_49)                 \
  _(c10::FHE_PRIME_50, FHE_PRIME_50)                 \
  _(c10::FHE_PRIME_51, FHE_PRIME_51)                 \
  _(c10::FHE_PRIME_52, FHE_PRIME_52)                 \
  _(c10::FHE_PRIME_53, FHE_PRIME_53)                 \
  _(c10::FHE_PRIME_54, FHE_PRIME_54)                 \
  _(c10::FHE_PRIME_55, FHE_PRIME_55)                 \
  _(c10::FHE_PRIME_56, FHE_PRIME_56)                 \
  _(c10::FHE_PRIME_57, FHE_PRIME_57)                 \
  _(c10::FHE_PRIME_58, FHE_PRIME_58)                 \
  _(c10::FHE_PRIME_59, FHE_PRIME_59)                 \
  _(c10::FHE_PRIME_60, FHE_PRIME_60)                 \
  _(c10::FHE_PRIME_61, FHE_PRIME_61)                 \
  _(c10::FHE_PRIME_62, FHE_PRIME_62)                 \
  _(c10::FHE_PRIME_63, FHE_PRIME_63)                 \
  _(c10::FHE_PRIME_64, FHE_PRIME_64)                 \
  _(c10::FHE_PRIME_65, FHE_PRIME_65)                 \
  _(c10::FHE_PRIME_66, FHE_PRIME_66)                 \
  _(c10::FHE_PRIME_67, FHE_PRIME_67)                 \
  _(c10::FHE_PRIME_68, FHE_PRIME_68)                 \
  _(c10::FHE_PRIME_69, FHE_PRIME_69)                 \
  _(c10::FHE_PRIME_70, FHE_PRIME_70)                 \
  _(c10::FHE_PRIME_71, FHE_PRIME_71)                 \
  _(c10::FHE_PRIME_72, FHE_PRIME_72)                 \
  _(c10::FHE_PRIME_73, FHE_PRIME_73)                 \
  _(c10::FHE_PRIME_74, FHE_PRIME_74)                 \
  _(c10::FHE_PRIME_75, FHE_PRIME_75)                 \
  _(c10::FHE_PRIME_76, FHE_PRIME_76)                 \
  _(c10::FHE_PRIME_77, FHE_PRIME_77)                 \
  _(c10::FHE_PRIME_78, FHE_PRIME_78)                 \
  _(c10::FHE_PRIME_79, FHE_PRIME_79)                 \
  _(c10::FHE_PRIME_80, FHE_PRIME_80)                 \
  _(c10::FHE_PRIME_81, FHE_PRIME_81)                 \
  _(c10::FHE_PRIME_82, FHE_PRIME_82)                 \
  _(c10::FHE_PRIME_83, FHE_PRIME_83)                 \
  _(c10::FHE_PRIME_84, FHE_PRIME_84)                 \
  _(c10::FHE_PRIME_85, FHE_PRIME_85)                 \
  _(c10::FHE_PRIME_86, FHE_PRIME_86)                 \
  _(c10::FHE_PRIME_87, FHE_PRIME_87)                 \
  _(c10::FHE_PRIME_88, FHE_PRIME_88)                 \
  _(c10::FHE_PRIME_89, FHE_PRIME_89)                 \
  _(c10::FHE_PRIME_90, FHE_PRIME_90)                 \
  _(c10::FHE_PRIME_91, FHE_PRIME_91)                 \
  _(c10::FHE_PRIME_92, FHE_PRIME_92)                 \
  _(c10::FHE_PRIME_93, FHE_PRIME_93)                 \
  _(c10::FHE_PRIME_94, FHE_PRIME_94)                 \
  _(c10::FHE_PRIME_95, FHE_PRIME_95)                 \
  _(c10::FHE_PRIME_96, FHE_PRIME_96)                 \
  _(c10::FHE_PRIME_97, FHE_PRIME_97)                 \
  _(c10::FHE_PRIME_98, FHE_PRIME_98)                 \
  _(c10::FHE_PRIME_99, FHE_PRIME_99)                 \
  _(c10::FHE_PRIME_100, FHE_PRIME_100)               \
  _(c10::FHE_PRIME_101, FHE_PRIME_101)               \
  _(c10::FHE_PRIME_102, FHE_PRIME_102)               \
  _(c10::FHE_PRIME_103, FHE_PRIME_103)               \
  _(c10::FHE_PRIME_104, FHE_PRIME_104)               \
  _(c10::FHE_PRIME_105, FHE_PRIME_105)               \
  _(c10::FHE_PRIME_106, FHE_PRIME_106)               \
  _(c10::FHE_PRIME_107, FHE_PRIME_107)

#define AT_FORALL_COMPLEX_TYPES(_)     \
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

#define DEFINE_CONSTANT(_, name) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
  case ScalarType::name:                   \
    return sizeof(ctype);

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
    default:
      TORCH_CHECK(false, "Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

static inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short);

  return isIntegral || (includeBool && t == ScalarType::Bool);
}

C10_DEPRECATED_MESSAGE(
    "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")
static inline bool isIntegralType(ScalarType t) {
  return isIntegralType(t, /*includeBool=*/false);
}

static inline bool isFloatingType(ScalarType t) {
  return (
      t == ScalarType::Double || t == ScalarType::Float ||
      t == ScalarType::Half || t == ScalarType::BFloat16 ||
      t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e4m3fn);
}

static inline bool isFloat8Type(ScalarType t) {
  return t == ScalarType::Float8_e4m3fn || t == ScalarType::Float8_e5m2;
}

static inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 || isFloat8Type(t);
}

static inline bool isComplexType(ScalarType t) {
  return (
      t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);
}

static inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
      t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
      t == ScalarType::QUInt2x4;
}

static inline bool isEllipticCurveType(ScalarType t) {
  return t == ScalarType::ALT_BN128_Fr_G1_Base ||
      t == ScalarType::ALT_BN128_Fr_G2_Base ||
      t == ScalarType::ALT_BN128_Fq_G1_Base ||
      t == ScalarType::ALT_BN128_Fq_G2_Base ||
      t == ScalarType::ALT_BN128_Fr_G1_Mont ||
      t == ScalarType::ALT_BN128_Fr_G2_Mont ||
      t == ScalarType::ALT_BN128_Fq_G1_Mont ||
      t == ScalarType::ALT_BN128_Fq_G2_Mont ||
      t == ScalarType::BLS12_377_Fr_G1_Base ||
      t == ScalarType::BLS12_377_Fr_G2_Base ||
      t == ScalarType::BLS12_377_Fq_G1_Base ||
      t == ScalarType::BLS12_377_Fq_G2_Base ||
      t == ScalarType::BLS12_377_Fr_G1_Mont ||
      t == ScalarType::BLS12_377_Fr_G2_Mont ||
      t == ScalarType::BLS12_377_Fq_G1_Mont ||
      t == ScalarType::BLS12_377_Fq_G2_Mont ||
      t == ScalarType::BLS12_381_Fr_G1_Base ||
      t == ScalarType::BLS12_381_Fr_G2_Base ||
      t == ScalarType::BLS12_381_Fq_G1_Base ||
      t == ScalarType::BLS12_381_Fq_G2_Base ||
      t == ScalarType::BLS12_381_Fr_G1_Mont ||
      t == ScalarType::BLS12_381_Fr_G2_Mont ||
      t == ScalarType::BLS12_381_Fq_G1_Mont ||
      t == ScalarType::BLS12_381_Fq_G2_Mont ||
      t == ScalarType::MNT4753_Fr_G1_Base ||
      t == ScalarType::MNT4753_Fr_G2_Base ||
      t == ScalarType::MNT4753_Fq_G1_Base ||
      t == ScalarType::MNT4753_Fq_G2_Base ||
      t == ScalarType::MNT4753_Fr_G1_Mont ||
      t == ScalarType::MNT4753_Fr_G2_Mont ||
      t == ScalarType::MNT4753_Fq_G1_Mont ||
      t == ScalarType::MNT4753_Fq_G2_Mont ||
      t == ScalarType::PALLAS_Fr_G1_Base ||
      t == ScalarType::PALLAS_Fr_G2_Base ||
      t == ScalarType::PALLAS_Fq_G1_Base ||
      t == ScalarType::PALLAS_Fq_G2_Base ||
      t == ScalarType::PALLAS_Fr_G1_Mont ||
      t == ScalarType::PALLAS_Fr_G2_Mont ||
      t == ScalarType::PALLAS_Fq_G1_Mont ||
      t == ScalarType::MNT4753_Fq_G2_Mont ||
      t == ScalarType::VESTA_Fr_G1_Base || t == ScalarType::VESTA_Fr_G2_Base ||
      t == ScalarType::VESTA_Fq_G1_Base || t == ScalarType::VESTA_Fq_G2_Base ||
      t == ScalarType::VESTA_Fr_G1_Mont || t == ScalarType::VESTA_Fr_G2_Mont ||
      t == ScalarType::VESTA_Fq_G1_Mont || t == ScalarType::VESTA_Fq_G2_Mont;
}

static inline bool isFHEPrimeType(ScalarType t) {
  return t == ScalarType::FHE_PRIME_0 || t == ScalarType::FHE_PRIME_1 ||
t == ScalarType::FHE_PRIME_2 || t == ScalarType::FHE_PRIME_3 ||
t == ScalarType::FHE_PRIME_4 || t == ScalarType::FHE_PRIME_5 ||
t == ScalarType::FHE_PRIME_6 || t == ScalarType::FHE_PRIME_7 ||
t == ScalarType::FHE_PRIME_8 || t == ScalarType::FHE_PRIME_9 ||
t == ScalarType::FHE_PRIME_10 || t == ScalarType::FHE_PRIME_11 ||
t == ScalarType::FHE_PRIME_12 || t == ScalarType::FHE_PRIME_13 ||
t == ScalarType::FHE_PRIME_14 || t == ScalarType::FHE_PRIME_15 ||
t == ScalarType::FHE_PRIME_16 || t == ScalarType::FHE_PRIME_17 ||
t == ScalarType::FHE_PRIME_18 || t == ScalarType::FHE_PRIME_19 ||
t == ScalarType::FHE_PRIME_20 || t == ScalarType::FHE_PRIME_21 ||
t == ScalarType::FHE_PRIME_22 || t == ScalarType::FHE_PRIME_23 ||
t == ScalarType::FHE_PRIME_24 || t == ScalarType::FHE_PRIME_25 ||
t == ScalarType::FHE_PRIME_26 || t == ScalarType::FHE_PRIME_27 ||
t == ScalarType::FHE_PRIME_28 || t == ScalarType::FHE_PRIME_29 ||
t == ScalarType::FHE_PRIME_30 || t == ScalarType::FHE_PRIME_31 ||
t == ScalarType::FHE_PRIME_32 || t == ScalarType::FHE_PRIME_33 ||
t == ScalarType::FHE_PRIME_34 || t == ScalarType::FHE_PRIME_35 ||
t == ScalarType::FHE_PRIME_36 || t == ScalarType::FHE_PRIME_37 ||
t == ScalarType::FHE_PRIME_38 || t == ScalarType::FHE_PRIME_39 ||
t == ScalarType::FHE_PRIME_40 || t == ScalarType::FHE_PRIME_41 ||
t == ScalarType::FHE_PRIME_42 || t == ScalarType::FHE_PRIME_43 ||
t == ScalarType::FHE_PRIME_44 || t == ScalarType::FHE_PRIME_45 ||
t == ScalarType::FHE_PRIME_46 || t == ScalarType::FHE_PRIME_47 ||
t == ScalarType::FHE_PRIME_48 || t == ScalarType::FHE_PRIME_49 ||
t == ScalarType::FHE_PRIME_50 || t == ScalarType::FHE_PRIME_51 ||
t == ScalarType::FHE_PRIME_52 || t == ScalarType::FHE_PRIME_53 ||
t == ScalarType::FHE_PRIME_54 || t == ScalarType::FHE_PRIME_55 ||
t == ScalarType::FHE_PRIME_56 || t == ScalarType::FHE_PRIME_57 ||
t == ScalarType::FHE_PRIME_58 || t == ScalarType::FHE_PRIME_59 ||
t == ScalarType::FHE_PRIME_60 || t == ScalarType::FHE_PRIME_61 ||
t == ScalarType::FHE_PRIME_62 || t == ScalarType::FHE_PRIME_63 ||
t == ScalarType::FHE_PRIME_64 || t == ScalarType::FHE_PRIME_65 ||
t == ScalarType::FHE_PRIME_66 || t == ScalarType::FHE_PRIME_67 ||
t == ScalarType::FHE_PRIME_68 || t == ScalarType::FHE_PRIME_69 ||
t == ScalarType::FHE_PRIME_70 || t == ScalarType::FHE_PRIME_71 ||
t == ScalarType::FHE_PRIME_72 || t == ScalarType::FHE_PRIME_73 ||
t == ScalarType::FHE_PRIME_74 || t == ScalarType::FHE_PRIME_75 ||
t == ScalarType::FHE_PRIME_76 || t == ScalarType::FHE_PRIME_77 ||
t == ScalarType::FHE_PRIME_78 || t == ScalarType::FHE_PRIME_79 ||
t == ScalarType::FHE_PRIME_80 || t == ScalarType::FHE_PRIME_81 ||
t == ScalarType::FHE_PRIME_82 || t == ScalarType::FHE_PRIME_83 ||
t == ScalarType::FHE_PRIME_84 || t == ScalarType::FHE_PRIME_85 ||
t == ScalarType::FHE_PRIME_86 || t == ScalarType::FHE_PRIME_87 ||
t == ScalarType::FHE_PRIME_88 || t == ScalarType::FHE_PRIME_89 ||
t == ScalarType::FHE_PRIME_90 || t == ScalarType::FHE_PRIME_91 ||
t == ScalarType::FHE_PRIME_92 || t == ScalarType::FHE_PRIME_93 ||
t == ScalarType::FHE_PRIME_94 || t == ScalarType::FHE_PRIME_95 ||
t == ScalarType::FHE_PRIME_96 || t == ScalarType::FHE_PRIME_97 ||
t == ScalarType::FHE_PRIME_98 || t == ScalarType::FHE_PRIME_99 ||
t == ScalarType::FHE_PRIME_100 || t == ScalarType::FHE_PRIME_101 ||
t == ScalarType::FHE_PRIME_102 || t == ScalarType::FHE_PRIME_103 ||
t == ScalarType::FHE_PRIME_104 || t == ScalarType::FHE_PRIME_105 ||
t == ScalarType::FHE_PRIME_106 || t == ScalarType::FHE_PRIME_107;

}

static inline bool isBigIntegerType(ScalarType t) {
  // Don't forget to extend this when adding new BigInteger types
  return t == ScalarType::ULong || t == ScalarType::Field64 ||
      t == ScalarType::BigInteger || t == ScalarType::BigInteger_Mont ||
      t == ScalarType::FiniteField || isEllipticCurveType(t);
}

static inline bool isMontgomeryField(ScalarType t) {
  // Don't forget to extend this when adding new BigInteger types
  return t == ScalarType::BigInteger_Mont ||
      t == ScalarType::ALT_BN128_Fr_G1_Mont ||
      t == ScalarType::ALT_BN128_Fr_G2_Mont ||
      t == ScalarType::ALT_BN128_Fq_G1_Mont ||
      t == ScalarType::ALT_BN128_Fq_G2_Mont ||
      t == ScalarType::BLS12_377_Fr_G1_Mont ||
      t == ScalarType::BLS12_377_Fr_G2_Mont ||
      t == ScalarType::BLS12_377_Fq_G1_Mont ||
      t == ScalarType::BLS12_377_Fq_G2_Mont ||
      t == ScalarType::BLS12_381_Fr_G1_Mont ||
      t == ScalarType::BLS12_381_Fr_G2_Mont ||
      t == ScalarType::BLS12_381_Fq_G1_Mont ||
      t == ScalarType::BLS12_381_Fq_G2_Mont ||
      t == ScalarType::MNT4753_Fr_G1_Mont ||
      t == ScalarType::MNT4753_Fr_G2_Mont ||
      t == ScalarType::MNT4753_Fq_G1_Mont ||
      t == ScalarType::MNT4753_Fq_G2_Mont ||
      t == ScalarType::PALLAS_Fr_G1_Mont ||
      t == ScalarType::PALLAS_Fr_G2_Mont ||
      t == ScalarType::PALLAS_Fq_G1_Mont ||
      t == ScalarType::PALLAS_Fq_G2_Mont || t == ScalarType::VESTA_Fr_G1_Mont ||
      t == ScalarType::VESTA_Fr_G2_Mont || t == ScalarType::VESTA_Fq_G1_Mont ||
      t == ScalarType::VESTA_Fq_G2_Mont;
}

static inline uint16_t bit_length(ScalarType t) {
  switch (t) {
    case ScalarType::ALT_BN128_Fr_G1_Base:
    case ScalarType::ALT_BN128_Fr_G1_Mont:
    case ScalarType::ALT_BN128_Fr_G2_Base:
    case ScalarType::ALT_BN128_Fr_G2_Mont:
    case ScalarType::ALT_BN128_Fq_G1_Base:
    case ScalarType::ALT_BN128_Fq_G1_Mont:
    case ScalarType::ALT_BN128_Fq_G2_Base:
    case ScalarType::ALT_BN128_Fq_G2_Mont:
      return 254;
    case ScalarType::BLS12_377_Fr_G1_Base:
    case ScalarType::BLS12_377_Fr_G1_Mont:
    case ScalarType::BLS12_377_Fr_G2_Base:
    case ScalarType::BLS12_377_Fr_G2_Mont:
      return 253;
    case ScalarType::BLS12_377_Fq_G1_Base:
    case ScalarType::BLS12_377_Fq_G1_Mont:
    case ScalarType::BLS12_377_Fq_G2_Base:
    case ScalarType::BLS12_377_Fq_G2_Mont:
      return 377;
    case ScalarType::BLS12_381_Fr_G1_Base:
    case ScalarType::BLS12_381_Fr_G1_Mont:
    case ScalarType::BLS12_381_Fr_G2_Base:
    case ScalarType::BLS12_381_Fr_G2_Mont:
      return 255;
    case ScalarType::BLS12_381_Fq_G1_Base:
    case ScalarType::BLS12_381_Fq_G1_Mont:
    case ScalarType::BLS12_381_Fq_G2_Base:
    case ScalarType::BLS12_381_Fq_G2_Mont:
      return 381;
    case ScalarType::MNT4753_Fr_G1_Base:
    case ScalarType::MNT4753_Fr_G1_Mont:
    case ScalarType::MNT4753_Fr_G2_Base:
    case ScalarType::MNT4753_Fr_G2_Mont:
      return 753;
    case ScalarType::MNT4753_Fq_G1_Base:
    case ScalarType::MNT4753_Fq_G1_Mont:
    case ScalarType::MNT4753_Fq_G2_Base:
    case ScalarType::MNT4753_Fq_G2_Mont:
      return 753;
    case ScalarType::PALLAS_Fr_G1_Base:
    case ScalarType::PALLAS_Fr_G1_Mont:
    case ScalarType::PALLAS_Fr_G2_Base:
    case ScalarType::PALLAS_Fr_G2_Mont:
    case ScalarType::PALLAS_Fq_G1_Base:
    case ScalarType::PALLAS_Fq_G1_Mont:
    case ScalarType::PALLAS_Fq_G2_Base:
    case ScalarType::PALLAS_Fq_G2_Mont:
      return 255;
    case ScalarType::VESTA_Fr_G1_Base:
    case ScalarType::VESTA_Fr_G1_Mont:
    case ScalarType::VESTA_Fr_G2_Base:
    case ScalarType::VESTA_Fr_G2_Mont:
    case ScalarType::VESTA_Fq_G1_Base:
    case ScalarType::VESTA_Fq_G1_Mont:
    case ScalarType::VESTA_Fq_G2_Base:
    case ScalarType::VESTA_Fq_G2_Mont:
      return 255;
    ALL_FHE_PRIME_CASE
      return 61;
    default:
      TORCH_CHECK(false, "not a elliptic curve type");
  }
}

static inline uint8_t num_uint64(ScalarType t) {
  return ((bit_length(t) + 63) / 64);
}

static inline bool isBitsType(ScalarType t) {
  return t == ScalarType::Bits1x8 || t == ScalarType::Bits2x4 ||
      t == ScalarType::Bits4x2 || t == ScalarType::Bits8 ||
      t == ScalarType::Bits16;
}

static inline ScalarType toQIntType(ScalarType t) {
  switch (t) {
    case ScalarType::Byte:
      return ScalarType::QUInt8;
    case ScalarType::Char:
      return ScalarType::QInt8;
    case ScalarType::Int:
      return ScalarType::QInt32;
    default:
      return t;
  }
}

static inline ScalarType toUnderlying(ScalarType t) {
  switch (t) {
    case ScalarType::QUInt8:
      return ScalarType::Byte;
    case ScalarType::QInt8:
      return ScalarType::Char;
    case ScalarType::QInt32:
      return ScalarType::Int;
    case ScalarType::QUInt4x2:
      return ScalarType::Byte;
    case ScalarType::QUInt2x4:
      return ScalarType::Byte;
    default:
      return t;
  }
}

static inline bool isSignedType(ScalarType t) {
  TORCH_CHECK(!isQIntType(t), "isSignedType not supported for quantized types");
#define CASE_SIGNED(ctype, name) \
  case ScalarType::name:         \
    return std::numeric_limits<ctype>::is_signed;

  switch (t) {
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK(false, "Bits types are undefined");
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      return true;
      AT_FORALL_SCALAR_TYPES_AND5(
          Half, Bool, BFloat16, Float8_e5m2, Float8_e4m3fn, CASE_SIGNED)
    default:
      TORCH_CHECK(false, "Unknown ScalarType");
  }
#undef CASE_SIGNED
}

static inline bool isUnderlying(ScalarType type, ScalarType qtype) {
  return type == toUnderlying(qtype);
}

static inline ScalarType toRealValueType(ScalarType t) {
  switch (t) {
    case ScalarType::ComplexHalf:
      return ScalarType::Half;
    case ScalarType::ComplexFloat:
      return ScalarType::Float;
    case ScalarType::ComplexDouble:
      return ScalarType::Double;
    default:
      return t;
  }
}

static inline ScalarType toComplexType(ScalarType t) {
  switch (t) {
    case ScalarType::BFloat16:
      // BFloat16 has range equivalent to Float,
      // so we map it to ComplexFloat.
      return ScalarType::ComplexFloat;
    case ScalarType::Half:
      return ScalarType::ComplexHalf;
    case ScalarType::Float:
      return ScalarType::ComplexFloat;
    case ScalarType::Double:
      return ScalarType::ComplexDouble;
    case ScalarType::ComplexHalf:
      return ScalarType::ComplexHalf;
    case ScalarType::ComplexFloat:
      return ScalarType::ComplexFloat;
    case ScalarType::ComplexDouble:
      return ScalarType::ComplexDouble;
    default:
      TORCH_CHECK(false, "Unknown Complex ScalarType for ", t);
  }
}

// see tensor_attributes.rst for detailed explanation and examples
// of casting rules.
static inline bool canCast(const ScalarType from, const ScalarType to) {
  // We disallow complex -> non complex, e.g., float_tensor *= complex is
  // disallowed.
  if (isComplexType(from) && !isComplexType(to)) {
    return false;
  }
  // We disallow float -> integral, e.g., int_tensor *= float is disallowed.
  if (isFloatingType(from) && isIntegralType(to, false)) {
    return false;
  }

  // Treat bool as a distinct "category," to be consistent with type promotion
  // rules (e.g. `bool_tensor + 5 -> int64_tensor`). If `5` was in the same
  // category as `bool_tensor`, we would not promote. Differing categories
  // implies `bool_tensor += 5` is disallowed.
  //
  // NB: numpy distinguishes "unsigned" as a category to get the desired
  // `bool_tensor + 5 -> int64_tensor` behavior. We don't, because:
  // * We don't want the performance hit of checking the runtime sign of
  // Scalars.
  // * `uint8_tensor + 5 -> int64_tensor` would be undesirable.
  if (from != ScalarType::Bool && to == ScalarType::Bool) {
    return false;
  }
  return true;
}

static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f2 = ScalarType::Half;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto c2 = ScalarType::ComplexHalf;
  constexpr auto c4 = ScalarType::ComplexFloat;
  constexpr auto c8 = ScalarType::ComplexDouble;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto bf = ScalarType::BFloat16;
  constexpr auto b8 = ScalarType::Float8_e5m2;
  constexpr auto h8 = ScalarType::Float8_e4m3fn;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // For QInt and BigInteger types, we only allow exact match
  if ((isQIntType(a) || isBigIntegerType(a)) && a == b) {
    return a;
  }

  if (isQIntType(a) || isQIntType(b) || isBigIntegerType(a) ||
      isBigIntegerType(b)) {
    TORCH_CHECK(
        false,
        "promoteTypes with quantized numbers and big integers is not handled yet; figure out what the correct rules should be, offending types: ",
        toString(a),
        " ",
        toString(b));
  }

  if (isBitsType(a) && a == b) {
    return a;
  } else if (isBitsType(a) || isBitsType(b)) {
    return ScalarType::Undefined;
  }

  // Bits and Quantized are 7 dtypes already handled and not included
  // in the promotion table below. Therefore every dtype above them
  // needs to be shifted down to account for that.
  static constexpr int shift_distance =
      static_cast<int>(ScalarType::Float8_e5m2) -
      static_cast<int>(ScalarType::QUInt4x2);
  static constexpr int shift_threshold = static_cast<int>(ScalarType::Bits16);

  if (static_cast<int>(a) > shift_threshold) {
    a = static_cast<ScalarType>(static_cast<int>(a) - shift_distance);
  }

  if (static_cast<int>(b) > shift_threshold) {
    b = static_cast<ScalarType>(static_cast<int>(b) - shift_distance);
  }

  // For the same reason the size of promotion table is decreased by 7
  // comparing to the number of all dtypes.
  static constexpr int NUM_PROMOTE_TYPES =
      static_cast<int>(ScalarType::NumOptions) - shift_distance;

  // this matrix has to be consistent with
  // AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS undefined is used where we
  // are not sure about the correct value for type promotion.
  // clang-format off
  static constexpr ScalarType _promoteTypesLookup[
      NUM_PROMOTE_TYPES][NUM_PROMOTE_TYPES] = {
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf  b8  h8*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, ud, ud, ud, bf, b8, h8},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, ud, ud, ud, bf, b8, h8},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, ud, ud, ud, bf, b8, h8},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, ud, ud, ud, bf, b8, h8},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, ud, ud, ud, bf, b8, h8},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, ud, ud, ud, f4, f4, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, ud, ud, ud, f4, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, ud, ud, ud, f8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, ud, ud, ud, c4, c4, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, ud, ud, ud, c4, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, ud, ud, ud, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, ud, ud, ud, bf, b8, h8},
      /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q3 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, ud, ud, ud, bf, bf, bf},
      /* b8 */ {b8, b8, b8, b8, b8, f4, f4, f8, c4, c4, c8, b8, ud, ud, ud, bf, b8, ud},
      /* h8 */ {h8, h8, h8, h8, h8, f4, f4, f8, c4, c4, c8, h8, ud, ud, ud, bf, ud, h8},
  };

  // clang-format on
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace c10
