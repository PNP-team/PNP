#pragma once
#include <cstdint>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

struct ALT_BN128_Fr_G1;
struct ALT_BN128_Fr_G2;
struct ALT_BN128_Fq_G1;
struct ALT_BN128_Fq_G2;
struct BLS12_377_Fr_G1;
struct BLS12_377_Fr_G2;
struct BLS12_377_Fq_G1;
struct BLS12_377_Fq_G2;
struct BLS12_381_Fr_G1;
struct BLS12_381_Fr_G2;
struct BLS12_381_Fq_G1;
struct BLS12_381_Fq_G2;
struct MNT4753_Fr_G1;
struct MNT4753_Fr_G2;
struct MNT4753_Fq_G1;
struct MNT4753_Fq_G2;
struct PALLAS_Fr_G1;
struct PALLAS_Fr_G2;
struct PALLAS_Fq_G1;
struct PALLAS_Fq_G2;
struct VESTA_Fr_G1;
struct VESTA_Fr_G2;
struct VESTA_Fq_G1;
struct VESTA_Fq_G2;

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
  operator uint64_t() const { return val_; }
};

struct alignas(8) BigInteger {
  uint64_t val_;
  BigInteger() = default;
  C10_HOST_DEVICE explicit BigInteger(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

struct alignas(8) BigInteger_Mont {
  uint64_t val_;
  BigInteger_Mont() = default;
  C10_HOST_DEVICE explicit BigInteger_Mont(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

struct alignas(8) FiniteField {
  uint64_t val_;
  FiniteField() = default;
  C10_HOST_DEVICE explicit FiniteField(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};


#define DEF_BIGINTEGER(name)                                        \
struct alignas(8) name##_Base {                                     \
  using compute_type = at::native::name;                            \
  uint64_t val_;                                                    \
  name##_Base() = default;                                          \
  C10_HOST_DEVICE explicit name##_Base(uint64_t val) : val_(val) {} \
  operator uint64_t() const { return val_; }                        \
};                                                                  \
struct alignas(8) name##_Mont {                                     \
  using compute_type = at::native::name;                            \
  uint64_t val_;                                                    \
  name##_Mont() = default;                                          \
  C10_HOST_DEVICE explicit name##_Mont(uint64_t val) : val_(val) {} \
  operator uint64_t() const { return val_; }                        \
};

#define APPLY_ALL_CURVE(FUNC) \
FUNC(ALT_BN128_Fr_G1)         \
FUNC(ALT_BN128_Fr_G2)         \
FUNC(ALT_BN128_Fq_G1)         \
FUNC(ALT_BN128_Fq_G2)         \
FUNC(BLS12_377_Fr_G1)         \
FUNC(BLS12_377_Fr_G2)         \
FUNC(BLS12_377_Fq_G1)         \
FUNC(BLS12_377_Fq_G2)         \
FUNC(BLS12_381_Fr_G1)         \
FUNC(BLS12_381_Fr_G2)         \
FUNC(BLS12_381_Fq_G1)         \
FUNC(BLS12_381_Fq_G2)         \
FUNC(MNT4753_Fr_G1)           \
FUNC(MNT4753_Fr_G2)           \
FUNC(MNT4753_Fq_G1)           \
FUNC(MNT4753_Fq_G2)           \
FUNC(PALLAS_Fr_G1)            \
FUNC(PALLAS_Fr_G2)            \
FUNC(PALLAS_Fq_G1)            \
FUNC(PALLAS_Fq_G2)            \
FUNC(VESTA_Fr_G1)            \
FUNC(VESTA_Fr_G2)            \
FUNC(VESTA_Fq_G1)            \
FUNC(VESTA_Fq_G2)



APPLY_ALL_CURVE(DEF_BIGINTEGER);

#define APPLY_ALL_BIGINTEGER_CASE(FUNC) \
FUNC(Field64)                           \
FUNC(BigInteger)                        \
FUNC(BigInteger_Mont)                   \
FUNC(FiniteField)                       \
FUNC(ALT_BN128_Fr_G1_Base)              \
FUNC(ALT_BN128_Fr_G2_Base)              \
FUNC(ALT_BN128_Fq_G1_Base)              \
FUNC(ALT_BN128_Fq_G2_Base)              \
FUNC(BLS12_377_Fr_G1_Base)              \
FUNC(BLS12_377_Fr_G2_Base)              \
FUNC(BLS12_377_Fq_G1_Base)              \
FUNC(BLS12_377_Fq_G2_Base)              \
FUNC(BLS12_381_Fr_G1_Base)              \
FUNC(BLS12_381_Fr_G2_Base)              \
FUNC(BLS12_381_Fq_G1_Base)              \
FUNC(BLS12_381_Fq_G2_Base)              \
FUNC(MNT4753_Fr_G1_Base)                \
FUNC(MNT4753_Fr_G2_Base)                \
FUNC(MNT4753_Fq_G1_Base)                \
FUNC(MNT4753_Fq_G2_Base)                \
FUNC(PALLAS_Fr_G1_Base)                 \
FUNC(PALLAS_Fr_G2_Base)                 \
FUNC(PALLAS_Fq_G1_Base)                 \
FUNC(PALLAS_Fq_G2_Base)                 \
FUNC(VESTA_Fr_G1_Base)                  \
FUNC(VESTA_Fr_G2_Base)                  \
FUNC(VESTA_Fq_G1_Base)                  \
FUNC(VESTA_Fq_G2_Base)                  \
FUNC(ALT_BN128_Fr_G1_Mont)              \
FUNC(ALT_BN128_Fr_G2_Mont)              \
FUNC(ALT_BN128_Fq_G1_Mont)              \
FUNC(ALT_BN128_Fq_G2_Mont)              \
FUNC(BLS12_377_Fr_G1_Mont)              \
FUNC(BLS12_377_Fr_G2_Mont)              \
FUNC(BLS12_377_Fq_G1_Mont)              \
FUNC(BLS12_377_Fq_G2_Mont)              \
FUNC(BLS12_381_Fr_G1_Mont)              \
FUNC(BLS12_381_Fr_G2_Mont)              \
FUNC(BLS12_381_Fq_G1_Mont)              \
FUNC(BLS12_381_Fq_G2_Mont)              \
FUNC(MNT4753_Fr_G1_Mont)                \
FUNC(MNT4753_Fr_G2_Mont)                \
FUNC(MNT4753_Fq_G1_Mont)                \
FUNC(MNT4753_Fq_G2_Mont)                \
FUNC(PALLAS_Fr_G1_Mont)                 \
FUNC(PALLAS_Fr_G2_Mont)                 \
FUNC(PALLAS_Fq_G1_Mont)                 \
FUNC(PALLAS_Fq_G2_Mont)                 \
FUNC(VESTA_Fr_G1_Mont)                  \
FUNC(VESTA_Fr_G2_Mont)                  \
FUNC(VESTA_Fq_G1_Mont)                  \
FUNC(VESTA_Fq_G2_Mont)


#define DEF_IS_FIELD(name)                                   \
template<>                                                   \
struct is_field<name> : public std::true_type {};

template <typename T>
struct is_field : public std::false_type {};

APPLY_ALL_BIGINTEGER_CASE(DEF_IS_FIELD);


#define DEF_CASE(name) \
  case at::k##name:

#define ALL_BIGINTEGER_CASE         \
APPLY_ALL_BIGINTEGER_CASE(DEF_CASE)

} // namespace c10
