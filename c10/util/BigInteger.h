#pragma once
#include <c10/macros/Macros.h>
#include <cstdint>

#define APPLY_ALL_CURVE(FUNC) \
  FUNC(ALT_BN128_Fr_G1)       \
  FUNC(ALT_BN128_Fr_G2)       \
  FUNC(ALT_BN128_Fq_G1)       \
  FUNC(ALT_BN128_Fq_G2)       \
  FUNC(BLS12_377_Fr_G1)       \
  FUNC(BLS12_377_Fr_G2)       \
  FUNC(BLS12_377_Fq_G1)       \
  FUNC(BLS12_377_Fq_G2)       \
  FUNC(BLS12_381_Fr_G1)       \
  FUNC(BLS12_381_Fr_G2)       \
  FUNC(BLS12_381_Fq_G1)       \
  FUNC(BLS12_381_Fq_G2)       \
  FUNC(MNT4753_Fr_G1)         \
  FUNC(MNT4753_Fr_G2)         \
  FUNC(MNT4753_Fq_G1)         \
  FUNC(MNT4753_Fq_G2)         \
  FUNC(PALLAS_Fr_G1)          \
  FUNC(PALLAS_Fr_G2)          \
  FUNC(PALLAS_Fq_G1)          \
  FUNC(PALLAS_Fq_G2)          \
  FUNC(VESTA_Fr_G1)           \
  FUNC(VESTA_Fr_G2)           \
  FUNC(VESTA_Fq_G1)           \
  FUNC(VESTA_Fq_G2)

#define APPLY_ALL_FHE_PRIME(FUNC) \
  FUNC(FHE_PRIME_0)               \
  FUNC(FHE_PRIME_1)               \
  FUNC(FHE_PRIME_2)               \
  FUNC(FHE_PRIME_3)               \
  FUNC(FHE_PRIME_4)               \
  FUNC(FHE_PRIME_5)               \
  FUNC(FHE_PRIME_6)               \
  FUNC(FHE_PRIME_7)               \
  FUNC(FHE_PRIME_8)               \
  FUNC(FHE_PRIME_9)               \
  FUNC(FHE_PRIME_10)              \
  FUNC(FHE_PRIME_11)              \
  FUNC(FHE_PRIME_12)              \
  FUNC(FHE_PRIME_13)              \
  FUNC(FHE_PRIME_14)              \
  FUNC(FHE_PRIME_15)              \
  FUNC(FHE_PRIME_16)              \
  FUNC(FHE_PRIME_17)              \
  FUNC(FHE_PRIME_18)              \
  FUNC(FHE_PRIME_19)              \
  FUNC(FHE_PRIME_20)              \
  FUNC(FHE_PRIME_21)              \
  FUNC(FHE_PRIME_22)              \
  FUNC(FHE_PRIME_23)              \
  FUNC(FHE_PRIME_24)              \
  FUNC(FHE_PRIME_25)              \
  FUNC(FHE_PRIME_26)              \
  FUNC(FHE_PRIME_27)              \
  FUNC(FHE_PRIME_28)              \
  FUNC(FHE_PRIME_29)              \
  FUNC(FHE_PRIME_30)              \
  FUNC(FHE_PRIME_31)              \
  FUNC(FHE_PRIME_32)              \
  FUNC(FHE_PRIME_33)              \
  FUNC(FHE_PRIME_34)              \
  FUNC(FHE_PRIME_35)              \
  FUNC(FHE_PRIME_36)              \
  FUNC(FHE_PRIME_37)              \
  FUNC(FHE_PRIME_38)              \
  FUNC(FHE_PRIME_39)              \
  FUNC(FHE_PRIME_40)              \
  FUNC(FHE_PRIME_41)              \
  FUNC(FHE_PRIME_42)              \
  FUNC(FHE_PRIME_43)              \
  FUNC(FHE_PRIME_44)              \
  FUNC(FHE_PRIME_45)              \
  FUNC(FHE_PRIME_46)              \
  FUNC(FHE_PRIME_47)              \
  FUNC(FHE_PRIME_48)              \
  FUNC(FHE_PRIME_49)              \
  FUNC(FHE_PRIME_50)              \
  FUNC(FHE_PRIME_51)              \
  FUNC(FHE_PRIME_52)              \
  FUNC(FHE_PRIME_53)              \
  FUNC(FHE_PRIME_54)              \
  FUNC(FHE_PRIME_55)              \
  FUNC(FHE_PRIME_56)              \
  FUNC(FHE_PRIME_57)              \
  FUNC(FHE_PRIME_58)              \
  FUNC(FHE_PRIME_59)              \
  FUNC(FHE_PRIME_60)              \
  FUNC(FHE_PRIME_61)              \
  FUNC(FHE_PRIME_62)              \
  FUNC(FHE_PRIME_63)              \
  FUNC(FHE_PRIME_64)              \
  FUNC(FHE_PRIME_65)              \
  FUNC(FHE_PRIME_66)              \
  FUNC(FHE_PRIME_67)              \
  FUNC(FHE_PRIME_68)              \
  FUNC(FHE_PRIME_69)              \
  FUNC(FHE_PRIME_70)              \
  FUNC(FHE_PRIME_71)              \
  FUNC(FHE_PRIME_72)              \
  FUNC(FHE_PRIME_73)              \
  FUNC(FHE_PRIME_74)              \
  FUNC(FHE_PRIME_75)              \
  FUNC(FHE_PRIME_76)              \
  FUNC(FHE_PRIME_77)              \
  FUNC(FHE_PRIME_78)              \
  FUNC(FHE_PRIME_79)              \
  FUNC(FHE_PRIME_80)              \
  FUNC(FHE_PRIME_81)              \
  FUNC(FHE_PRIME_82)              \
  FUNC(FHE_PRIME_83)              \
  FUNC(FHE_PRIME_84)              \
  FUNC(FHE_PRIME_85)              \
  FUNC(FHE_PRIME_86)              \
  FUNC(FHE_PRIME_87)              \
  FUNC(FHE_PRIME_88)              \
  FUNC(FHE_PRIME_89)              \
  FUNC(FHE_PRIME_90)              \
  FUNC(FHE_PRIME_91)              \
  FUNC(FHE_PRIME_92)              \
  FUNC(FHE_PRIME_93)              \
  FUNC(FHE_PRIME_94)              \
  FUNC(FHE_PRIME_95)              \
  FUNC(FHE_PRIME_96)              \
  FUNC(FHE_PRIME_97)              \
  FUNC(FHE_PRIME_98)              \
  FUNC(FHE_PRIME_99)              \
  FUNC(FHE_PRIME_100)             \
  FUNC(FHE_PRIME_101)             \
  FUNC(FHE_PRIME_102)             \
  FUNC(FHE_PRIME_103)             \
  FUNC(FHE_PRIME_104)             \
  FUNC(FHE_PRIME_105)             \
  FUNC(FHE_PRIME_106)             \
  FUNC(FHE_PRIME_107)

namespace at {
namespace native {

#define DEF_STRUCT(name) struct name;

APPLY_ALL_CURVE(DEF_STRUCT);
APPLY_ALL_FHE_PRIME(DEF_STRUCT);

#undef DEF_STRUCT

} // namespace native
} // namespace at

namespace c10 {

struct NOT_CURVE {
  NOT_CURVE() = delete;
};

struct alignas(8) Field64 {
  uint64_t val_;
  Field64() = default;
  C10_HOST_DEVICE explicit Field64(uint64_t val) : val_(val) {}
  C10_HOST_DEVICE operator uint64_t() const {
    return val_;
  }
};

struct alignas(8) BigInteger {
  uint64_t val_;
  BigInteger() = default;
  C10_HOST_DEVICE explicit BigInteger(uint64_t val) : val_(val) {}
  C10_HOST_DEVICE operator uint64_t() const {
    return val_;
  }
};

struct alignas(8) BigInteger_Mont {
  uint64_t val_;
  BigInteger_Mont() = default;
  C10_HOST_DEVICE explicit BigInteger_Mont(uint64_t val) : val_(val) {}
  C10_HOST_DEVICE operator uint64_t() const {
    return val_;
  }
};

struct alignas(8) FiniteField {
  uint64_t val_;
  FiniteField() = default;
  C10_HOST_DEVICE explicit FiniteField(uint64_t val) : val_(val) {}
  C10_HOST_DEVICE operator uint64_t() const {
    return val_;
  }
};

#define DEF_CURVE(name)                                               \
  struct alignas(8) name##_Base {                                     \
    using compute_type = at::native::name;                            \
    uint64_t val_;                                                    \
    name##_Base() = default;                                          \
    C10_HOST_DEVICE explicit name##_Base(uint64_t val) : val_(val) {} \
    C10_HOST_DEVICE operator uint64_t() const {                       \
      return val_;                                                    \
    }                                                                 \
  };                                                                  \
  struct alignas(8) name##_Mont {                                     \
    using compute_type = at::native::name;                            \
    uint64_t val_;                                                    \
    name##_Mont() = default;                                          \
    C10_HOST_DEVICE explicit name##_Mont(uint64_t val) : val_(val) {} \
    C10_HOST_DEVICE operator uint64_t() const {                       \
      return val_;                                                    \
    }                                                                 \
  };



APPLY_ALL_CURVE(DEF_CURVE);
#undef DEF_CURVE

#define DEF_FHE_PRIME(name)                                    \
  struct alignas(8) name {                                     \
    using compute_type = at::native::name;                     \
    uint64_t val_;                                             \
    name() = default;                                          \
    C10_HOST_DEVICE explicit name(uint64_t val) : val_(val) {} \
    C10_HOST_DEVICE operator uint64_t() const {                \
      return val_;                                             \
    }                                                          \
  };

APPLY_ALL_FHE_PRIME(DEF_FHE_PRIME);
#undef DEF_FHE_PRIME

#define APPLY_ALL_BIGINTEGER_CASE(FUNC) \
  FUNC(Field64)                         \
  FUNC(BigInteger)                      \
  FUNC(BigInteger_Mont)                 \
  FUNC(FiniteField)                     \
  FUNC(ALT_BN128_Fr_G1_Base)            \
  FUNC(ALT_BN128_Fr_G2_Base)            \
  FUNC(ALT_BN128_Fq_G1_Base)            \
  FUNC(ALT_BN128_Fq_G2_Base)            \
  FUNC(BLS12_377_Fr_G1_Base)            \
  FUNC(BLS12_377_Fr_G2_Base)            \
  FUNC(BLS12_377_Fq_G1_Base)            \
  FUNC(BLS12_377_Fq_G2_Base)            \
  FUNC(BLS12_381_Fr_G1_Base)            \
  FUNC(BLS12_381_Fr_G2_Base)            \
  FUNC(BLS12_381_Fq_G1_Base)            \
  FUNC(BLS12_381_Fq_G2_Base)            \
  FUNC(MNT4753_Fr_G1_Base)              \
  FUNC(MNT4753_Fr_G2_Base)              \
  FUNC(MNT4753_Fq_G1_Base)              \
  FUNC(MNT4753_Fq_G2_Base)              \
  FUNC(PALLAS_Fr_G1_Base)               \
  FUNC(PALLAS_Fr_G2_Base)               \
  FUNC(PALLAS_Fq_G1_Base)               \
  FUNC(PALLAS_Fq_G2_Base)               \
  FUNC(VESTA_Fr_G1_Base)                \
  FUNC(VESTA_Fr_G2_Base)                \
  FUNC(VESTA_Fq_G1_Base)                \
  FUNC(VESTA_Fq_G2_Base)                \
  FUNC(ALT_BN128_Fr_G1_Mont)            \
  FUNC(ALT_BN128_Fr_G2_Mont)            \
  FUNC(ALT_BN128_Fq_G1_Mont)            \
  FUNC(ALT_BN128_Fq_G2_Mont)            \
  FUNC(BLS12_377_Fr_G1_Mont)            \
  FUNC(BLS12_377_Fr_G2_Mont)            \
  FUNC(BLS12_377_Fq_G1_Mont)            \
  FUNC(BLS12_377_Fq_G2_Mont)            \
  FUNC(BLS12_381_Fr_G1_Mont)            \
  FUNC(BLS12_381_Fr_G2_Mont)            \
  FUNC(BLS12_381_Fq_G1_Mont)            \
  FUNC(BLS12_381_Fq_G2_Mont)            \
  FUNC(MNT4753_Fr_G1_Mont)              \
  FUNC(MNT4753_Fr_G2_Mont)              \
  FUNC(MNT4753_Fq_G1_Mont)              \
  FUNC(MNT4753_Fq_G2_Mont)              \
  FUNC(PALLAS_Fr_G1_Mont)               \
  FUNC(PALLAS_Fr_G2_Mont)               \
  FUNC(PALLAS_Fq_G1_Mont)               \
  FUNC(PALLAS_Fq_G2_Mont)               \
  FUNC(VESTA_Fr_G1_Mont)                \
  FUNC(VESTA_Fr_G2_Mont)                \
  FUNC(VESTA_Fq_G1_Mont)                \
  FUNC(VESTA_Fq_G2_Mont)                \
  FUNC(FHE_PRIME_0)                     \
  FUNC(FHE_PRIME_1)                     \
  FUNC(FHE_PRIME_2)                     \
  FUNC(FHE_PRIME_3)                     \
  FUNC(FHE_PRIME_4)                     \
  FUNC(FHE_PRIME_5)                     \
  FUNC(FHE_PRIME_6)                     \
  FUNC(FHE_PRIME_7)                     \
  FUNC(FHE_PRIME_8)                     \
  FUNC(FHE_PRIME_9)                     \
  FUNC(FHE_PRIME_10)                    \
  FUNC(FHE_PRIME_11)                    \
  FUNC(FHE_PRIME_12)                    \
  FUNC(FHE_PRIME_13)                    \
  FUNC(FHE_PRIME_14)                    \
  FUNC(FHE_PRIME_15)                    \
  FUNC(FHE_PRIME_16)                    \
  FUNC(FHE_PRIME_17)                    \
  FUNC(FHE_PRIME_18)                    \
  FUNC(FHE_PRIME_19)                    \
  FUNC(FHE_PRIME_20)                    \
  FUNC(FHE_PRIME_21)                    \
  FUNC(FHE_PRIME_22)                    \
  FUNC(FHE_PRIME_23)                    \
  FUNC(FHE_PRIME_24)                    \
  FUNC(FHE_PRIME_25)                    \
  FUNC(FHE_PRIME_26)                    \
  FUNC(FHE_PRIME_27)                    \
  FUNC(FHE_PRIME_28)                    \
  FUNC(FHE_PRIME_29)                    \
  FUNC(FHE_PRIME_30)                    \
  FUNC(FHE_PRIME_31)                    \
  FUNC(FHE_PRIME_32)                    \
  FUNC(FHE_PRIME_33)                    \
  FUNC(FHE_PRIME_34)                    \
  FUNC(FHE_PRIME_35)                    \
  FUNC(FHE_PRIME_36)                    \
  FUNC(FHE_PRIME_37)                    \
  FUNC(FHE_PRIME_38)                    \
  FUNC(FHE_PRIME_39)                    \
  FUNC(FHE_PRIME_40)                    \
  FUNC(FHE_PRIME_41)                    \
  FUNC(FHE_PRIME_42)                    \
  FUNC(FHE_PRIME_43)                    \
  FUNC(FHE_PRIME_44)                    \
  FUNC(FHE_PRIME_45)                    \
  FUNC(FHE_PRIME_46)                    \
  FUNC(FHE_PRIME_47)                    \
  FUNC(FHE_PRIME_48)                    \
  FUNC(FHE_PRIME_49)                    \
  FUNC(FHE_PRIME_50)                    \
  FUNC(FHE_PRIME_51)                    \
  FUNC(FHE_PRIME_52)                    \
  FUNC(FHE_PRIME_53)                    \
  FUNC(FHE_PRIME_54)                    \
  FUNC(FHE_PRIME_55)                    \
  FUNC(FHE_PRIME_56)                    \
  FUNC(FHE_PRIME_57)                    \
  FUNC(FHE_PRIME_58)                    \
  FUNC(FHE_PRIME_59)                    \
  FUNC(FHE_PRIME_60)                    \
  FUNC(FHE_PRIME_61)                    \
  FUNC(FHE_PRIME_62)                    \
  FUNC(FHE_PRIME_63)                    \
  FUNC(FHE_PRIME_64)                    \
  FUNC(FHE_PRIME_65)                    \
  FUNC(FHE_PRIME_66)                    \
  FUNC(FHE_PRIME_67)                    \
  FUNC(FHE_PRIME_68)                    \
  FUNC(FHE_PRIME_69)                    \
  FUNC(FHE_PRIME_70)                    \
  FUNC(FHE_PRIME_71)                    \
  FUNC(FHE_PRIME_72)                    \
  FUNC(FHE_PRIME_73)                    \
  FUNC(FHE_PRIME_74)                    \
  FUNC(FHE_PRIME_75)                    \
  FUNC(FHE_PRIME_76)                    \
  FUNC(FHE_PRIME_77)                    \
  FUNC(FHE_PRIME_78)                    \
  FUNC(FHE_PRIME_79)                    \
  FUNC(FHE_PRIME_80)                    \
  FUNC(FHE_PRIME_81)                    \
  FUNC(FHE_PRIME_82)                    \
  FUNC(FHE_PRIME_83)                    \
  FUNC(FHE_PRIME_84)                    \
  FUNC(FHE_PRIME_85)                    \
  FUNC(FHE_PRIME_86)                    \
  FUNC(FHE_PRIME_87)                    \
  FUNC(FHE_PRIME_88)                    \
  FUNC(FHE_PRIME_89)                    \
  FUNC(FHE_PRIME_90)                    \
  FUNC(FHE_PRIME_91)                    \
  FUNC(FHE_PRIME_92)                    \
  FUNC(FHE_PRIME_93)                    \
  FUNC(FHE_PRIME_94)                    \
  FUNC(FHE_PRIME_95)                    \
  FUNC(FHE_PRIME_96)                    \
  FUNC(FHE_PRIME_97)                    \
  FUNC(FHE_PRIME_98)                    \
  FUNC(FHE_PRIME_99)                    \
  FUNC(FHE_PRIME_100)                   \
  FUNC(FHE_PRIME_101)                   \
  FUNC(FHE_PRIME_102)                   \
  FUNC(FHE_PRIME_103)                   \
  FUNC(FHE_PRIME_104)                   \
  FUNC(FHE_PRIME_105)                   \
  FUNC(FHE_PRIME_106)                   \
  FUNC(FHE_PRIME_107)

#define DEF_IS_FIELD(name) \
  template <>              \
  struct is_field<name> : public std::true_type {};

template <typename T>
struct is_field : public std::false_type {};

APPLY_ALL_BIGINTEGER_CASE(DEF_IS_FIELD);

#define DEF_CASE(name) case at::k##name:

#define ALL_BIGINTEGER_CASE APPLY_ALL_BIGINTEGER_CASE(DEF_CASE)
#define ALL_FHE_PRIME_CASE APPLY_ALL_FHE_PRIME(DEF_CASE)

} // namespace c10
