#pragma once

#include <_types/_uint8_t.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>

namespace c10 {

enum class CurveFamily : int8_t {
  BN254 = 0,
  ALT_BN128 = 1,
  BLS12_377 = 2,
  BLS12_381 = 3,
  MNT4753 = 4,
  COMPILE_TIME_MAX_CURVE_FAMILY_TYPES = 5,
};

enum class CurveGroup : int8_t {
  G1 = 0,
  G2 = 1,
  COMPILE_TIME_MAX_CURVE_GROUP_TYPES = 2,
};

enum class CurveField : int8_t {
  Fr = 0,
  Fq = 1,
  COMPILE_TIME_MAX_CURVE_FIELD_TYPES = 2,
};

enum class CurveType : int8_t {
  NOT_CURVE = 0,
  BN254_Fr_G1 = 1,
  BN254_Fr_G2 = 2,
  BN254_Fq_G1 = 3,
  BN254_Fq_G2 = 4,
  ALT_BN128_Fr_G1 = 5,
  ALT_BN128_Fr_G2 = 6,
  ALT_BN128_Fq_G1 = 7,
  ALT_BN128_Fq_G2 = 8,
  BLS12_377_Fr_G1 = 9,
  BLS12_377_Fr_G2 = 10,
  BLS12_377_Fq_G1 = 11,
  BLS12_377_Fq_G2 = 12,
  BLS12_381_Fr_G1 = 13,
  BLS12_381_Fr_G2 = 14,
  BLS12_381_Fq_G1 = 15,
  BLS12_381_Fq_G2 = 16,
  MNT4753_Fr_G1 = 17,
  MNT4753_Fr_G2 = 18,
  MNT4753_Fq_G1 = 19,
  MNT4753_Fq_G2 = 20,
  COMPILE_TIME_MAX_CURVE_TYPES = 21,
};

struct C10_API Curve final {

  /// Constructs a new `Curve` from a `CurveFamily`
  /* implicit */ Curve(CurveFamily curve_family, CurveField curve_field, CurveGroup curve_group) 
  : curve_family_(curve_family), curve_field_(curve_field), curve_group_(curve_group) {}

  /// Constructs a `Curve` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Curve(const std::string& curve_family_string, const std::string& curve_field_string, const std::string& curve_group_string);

  /// Returns true if the type and index of this `Device` matches that of
  /// `other`.
  bool operator==(const Curve& other) const noexcept {
    return this->curve_family_ == other.curve_family_ && this->curve_field_ == other.curve_field_ && this->curve_group_ == other.curve_group_;
  }

  /// Returns true if the type or index of this `Device` differs from that of
  /// `other`.
  bool operator!=(const Curve& other) const noexcept {
    return !(*this == other);
  }

  /// Returns the family of curve this is.
  CurveFamily curve_family() const noexcept {
    return curve_family_;
  }

  /// Returns the field of curve this is.
  CurveField curve_field() const noexcept {
    return curve_field_;
  }
  
  /// Returns the group of curve this is.
  CurveGroup curve_group() const noexcept {
    return curve_group_;
  }

  CurveType type() const noexcept {
    return static_cast<CurveType>(static_cast<int8_t>(curve_family_) * static_cast<int8_t>(CurveField::COMPILE_TIME_MAX_CURVE_FIELD_TYPES) * static_cast<int8_t>(CurveGroup::COMPILE_TIME_MAX_CURVE_GROUP_TYPES) + static_cast<int8_t>(curve_field_) * static_cast<int8_t>(CurveGroup::COMPILE_TIME_MAX_CURVE_GROUP_TYPES) + static_cast<int8_t>(curve_group_) + 1);
  }

  uint8_t expected_last_dim() const noexcept {
    switch (type()) {
      case CurveType::BN254_Fr_G1 : return 4;
      case CurveType::BN254_Fr_G2 : return 4;
      case CurveType::BN254_Fq_G1 : return 4;
      case CurveType::BN254_Fq_G2 : return 4;
      case CurveType::ALT_BN128_Fr_G1 : return 4;
      case CurveType::ALT_BN128_Fr_G2 : return 4;
      case CurveType::ALT_BN128_Fq_G1 : return 4;
      case CurveType::ALT_BN128_Fq_G2 : return 4;
      case CurveType::BLS12_377_Fr_G1 : return 4;
      case CurveType::BLS12_377_Fr_G2 : return 4;
      case CurveType::BLS12_377_Fq_G1 : return 6;
      case CurveType::BLS12_377_Fq_G2 : return 6;
      case CurveType::BLS12_381_Fr_G1 : return 4;
      case CurveType::BLS12_381_Fr_G2 : return 4;
      case CurveType::BLS12_381_Fq_G1 : return 6;
      case CurveType::BLS12_381_Fq_G2 : return 6;
      case CurveType::MNT4753_Fr_G1 : return 12;
      case CurveType::MNT4753_Fr_G2 : return 12;
      case CurveType::MNT4753_Fq_G1 : return 12;
      case CurveType::MNT4753_Fq_G2 : return 12;
      default:
        TORCH_CHECK(false, "Unsupported curve group");
    }
  }

  /// Same string as returned from operator<<.
  std::string str() const;

 private:
  CurveFamily curve_family_;
  CurveField curve_field_;
  CurveGroup curve_group_;
};

C10_API std::ostream& operator<<(std::ostream& stream, const Curve& device);

} // namespace c10
