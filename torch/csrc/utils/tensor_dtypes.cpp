#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_types.h>

namespace torch {
namespace utils {

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType) {
  switch (scalarType) {
    case at::ScalarType::Byte:
      // no "byte" because byte is signed in numpy and we overload
      // byte to mean bool often
      return std::make_pair("uint8", "");
    case at::ScalarType::Char:
      // no "char" because it is not consistently signed or unsigned; we want
      // to move to int8
      return std::make_pair("int8", "");
    case at::ScalarType::Double:
      return std::make_pair("float64", "double");
    case at::ScalarType::Float:
      return std::make_pair("float32", "float");
    case at::ScalarType::Int:
      return std::make_pair("int32", "int");
    case at::ScalarType::Long:
      return std::make_pair("int64", "long");
    case at::ScalarType::ULong:
      return std::make_pair("uint64", "unsigned long");
    case at::ScalarType::Short:
      return std::make_pair("int16", "short");
    case at::ScalarType::Half:
      return std::make_pair("float16", "half");
    case at::ScalarType::ComplexHalf:
      return std::make_pair("complex32", "chalf");
    case at::ScalarType::ComplexFloat:
      return std::make_pair("complex64", "cfloat");
    case at::ScalarType::ComplexDouble:
      return std::make_pair("complex128", "cdouble");
    case at::ScalarType::Bool:
      return std::make_pair("bool", "");
    case at::ScalarType::QInt8:
      return std::make_pair("qint8", "");
    case at::ScalarType::QUInt8:
      return std::make_pair("quint8", "");
    case at::ScalarType::QInt32:
      return std::make_pair("qint32", "");
    case at::ScalarType::BFloat16:
      return std::make_pair("bfloat16", "");
    case at::ScalarType::QUInt4x2:
      return std::make_pair("quint4x2", "");
    case at::ScalarType::QUInt2x4:
      return std::make_pair("quint2x4", "");
    case at::ScalarType::Bits1x8:
      return std::make_pair("bits1x8", "");
    case at::ScalarType::Bits2x4:
      return std::make_pair("bits2x4", "");
    case at::ScalarType::Bits4x2:
      return std::make_pair("bits4x2", "");
    case at::ScalarType::Bits8:
      return std::make_pair("bits8", "");
    case at::ScalarType::Bits16:
      return std::make_pair("bits16", "");
    case at::ScalarType::Float8_e5m2:
      return std::make_pair("float8_e5m2", "");
    case at::ScalarType::Float8_e4m3fn:
      return std::make_pair("float8_e4m3fn", "");
    case at::ScalarType::Field64:
      return std::make_pair("field64", "");
    case at::ScalarType::BigInteger:
      return std::make_pair("big_integer", "");
    case at::ScalarType::BigInteger_Mont:
      return std::make_pair("big_integer_mont", "");
    case at::ScalarType::FiniteField:
      return std::make_pair("finite_field", "");
    case at::ScalarType::NOT_CURVE:
      return std::make_pair("should_never_use", "");
    case at::ScalarType::ALT_BN128_Fr_G1_Base:
      return std::make_pair("ALT_BN128_Fr_G1_Base", "");
    case at::ScalarType::ALT_BN128_Fr_G2_Base:
      return std::make_pair("ALT_BN128_Fr_G2_Base", "");
    case at::ScalarType::ALT_BN128_Fq_G1_Base:
      return std::make_pair("ALT_BN128_Fq_G1_Base", "");
    case at::ScalarType::ALT_BN128_Fq_G2_Base:
      return std::make_pair("ALT_BN128_Fq_G2_Base", "");
    case at::ScalarType::BLS12_377_Fr_G1_Base:
      return std::make_pair("BLS12_377_Fr_G1_Base", "");
    case at::ScalarType::BLS12_377_Fr_G2_Base:
      return std::make_pair("BLS12_377_Fr_G2_Base", "");
    case at::ScalarType::BLS12_377_Fq_G1_Base:
      return std::make_pair("BLS12_377_Fq_G1_Base", "");
    case at::ScalarType::BLS12_377_Fq_G2_Base:
      return std::make_pair("BLS12_377_Fq_G2_Base", "");
    case at::ScalarType::BLS12_381_Fr_G1_Base:
      return std::make_pair("BLS12_381_Fr_G1_Base", "");
    case at::ScalarType::BLS12_381_Fr_G2_Base:
      return std::make_pair("BLS12_381_Fr_G2_Base", "");
    case at::ScalarType::BLS12_381_Fq_G1_Base:
      return std::make_pair("BLS12_381_Fq_G1_Base", "");
    case at::ScalarType::BLS12_381_Fq_G2_Base:
      return std::make_pair("BLS12_381_Fq_G2_Base", "");
    case at::ScalarType::MNT4753_Fr_G1_Base:
      return std::make_pair("MNT4753_Fr_G1_Base", "");
    case at::ScalarType::MNT4753_Fr_G2_Base:
      return std::make_pair("MNT4753_Fr_G2_Base", "");
    case at::ScalarType::MNT4753_Fq_G1_Base:
      return std::make_pair("MNT4753_Fq_G1_Base", "");
    case at::ScalarType::MNT4753_Fq_G2_Base:
      return std::make_pair("MNT4753_Fq_G2_Base", "");
    case at::ScalarType::PALLAS_Fr_G1_Base:
      return std::make_pair("PALLAS_Fr_G1_Base", "");
    case at::ScalarType::PALLAS_Fr_G2_Base:
      return std::make_pair("PALLAS_Fr_G2_Base", "");
    case at::ScalarType::PALLAS_Fq_G1_Base:
      return std::make_pair("PALLAS_Fq_G1_Base", "");
    case at::ScalarType::PALLAS_Fq_G2_Base:
      return std::make_pair("PALLAS_Fq_G2_Base", "");
    case at::ScalarType::VESTA_Fr_G1_Base:
      return std::make_pair("VESTA_Fr_G1_Base", "");
    case at::ScalarType::VESTA_Fr_G2_Base:
      return std::make_pair("VESTA_Fr_G2_Base", "");
    case at::ScalarType::VESTA_Fq_G1_Base:
      return std::make_pair("VESTA_Fq_G1_Base", "");
    case at::ScalarType::VESTA_Fq_G2_Base:
      return std::make_pair("VESTA_Fq_G2_Base", "");
    case at::ScalarType::ALT_BN128_Fr_G1_Mont:
      return std::make_pair("ALT_BN128_Fr_G1_Mont", "");
    case at::ScalarType::ALT_BN128_Fr_G2_Mont:
      return std::make_pair("ALT_BN128_Fr_G2_Mont", "");
    case at::ScalarType::ALT_BN128_Fq_G1_Mont:
      return std::make_pair("ALT_BN128_Fq_G1_Mont", "");
    case at::ScalarType::ALT_BN128_Fq_G2_Mont:
      return std::make_pair("ALT_BN128_Fq_G2_Mont", "");
    case at::ScalarType::BLS12_377_Fr_G1_Mont:
      return std::make_pair("BLS12_377_Fr_G1_Mont", "");
    case at::ScalarType::BLS12_377_Fr_G2_Mont:
      return std::make_pair("BLS12_377_Fr_G2_Mont", "");
    case at::ScalarType::BLS12_377_Fq_G1_Mont:
      return std::make_pair("BLS12_377_Fq_G1_Mont", "");
    case at::ScalarType::BLS12_377_Fq_G2_Mont:
      return std::make_pair("BLS12_377_Fq_G2_Mont", "");
    case at::ScalarType::BLS12_381_Fr_G1_Mont:
      return std::make_pair("BLS12_381_Fr_G1_Mont", "");
    case at::ScalarType::BLS12_381_Fr_G2_Mont:
      return std::make_pair("BLS12_381_Fr_G2_Mont", "");
    case at::ScalarType::BLS12_381_Fq_G1_Mont:
      return std::make_pair("BLS12_381_Fq_G1_Mont", "");
    case at::ScalarType::BLS12_381_Fq_G2_Mont:
      return std::make_pair("BLS12_381_Fq_G2_Mont", "");
    case at::ScalarType::MNT4753_Fr_G1_Mont:
      return std::make_pair("MNT4753_Fr_G1_Mont", "");
    case at::ScalarType::MNT4753_Fr_G2_Mont:
      return std::make_pair("MNT4753_Fr_G2_Mont", "");
    case at::ScalarType::MNT4753_Fq_G1_Mont:
      return std::make_pair("MNT4753_Fq_G1_Mont", "");
    case at::ScalarType::MNT4753_Fq_G2_Mont:
      return std::make_pair("MNT4753_Fq_G2_Mont", "");
    case at::ScalarType::PALLAS_Fr_G1_Mont:
      return std::make_pair("PALLAS_Fr_G1_Mont", "");
    case at::ScalarType::PALLAS_Fr_G2_Mont:
      return std::make_pair("PALLAS_Fr_G2_Mont", "");
    case at::ScalarType::PALLAS_Fq_G1_Mont:
      return std::make_pair("PALLAS_Fq_G1_Mont", "");
    case at::ScalarType::PALLAS_Fq_G2_Mont:
      return std::make_pair("PALLAS_Fq_G2_Mont", "");
    case at::ScalarType::VESTA_Fr_G1_Mont:
      return std::make_pair("VESTA_Fr_G1_Mont", "");
    case at::ScalarType::VESTA_Fr_G2_Mont:
      return std::make_pair("VESTA_Fr_G2_Mont", "");
    case at::ScalarType::VESTA_Fq_G1_Mont:
      return std::make_pair("VESTA_Fq_G1_Mont", "");
    case at::ScalarType::VESTA_Fq_G2_Mont:
      return std::make_pair("VESTA_Fq_G2_Mont", "");
    case at::ScalarType::FHE_PRIME_0:
      return std::make_pair("FHE_PRIME_0", "");
    case at::ScalarType::FHE_PRIME_1:
      return std::make_pair("FHE_PRIME_1", "");
    case at::ScalarType::FHE_PRIME_2:
      return std::make_pair("FHE_PRIME_2", "");
    case at::ScalarType::FHE_PRIME_3:
      return std::make_pair("FHE_PRIME_3", "");
    case at::ScalarType::FHE_PRIME_4:
      return std::make_pair("FHE_PRIME_4", "");
    case at::ScalarType::FHE_PRIME_5:
      return std::make_pair("FHE_PRIME_5", "");
    case at::ScalarType::FHE_PRIME_6:
      return std::make_pair("FHE_PRIME_6", "");
    case at::ScalarType::FHE_PRIME_7:
      return std::make_pair("FHE_PRIME_7", "");
    case at::ScalarType::FHE_PRIME_8:
      return std::make_pair("FHE_PRIME_8", "");
    case at::ScalarType::FHE_PRIME_9:
      return std::make_pair("FHE_PRIME_9", "");
    case at::ScalarType::FHE_PRIME_10:
      return std::make_pair("FHE_PRIME_10", "");
    case at::ScalarType::FHE_PRIME_11:
      return std::make_pair("FHE_PRIME_11", "");
    case at::ScalarType::FHE_PRIME_12:
      return std::make_pair("FHE_PRIME_12", "");
    case at::ScalarType::FHE_PRIME_13:
      return std::make_pair("FHE_PRIME_13", "");
    case at::ScalarType::FHE_PRIME_14:
      return std::make_pair("FHE_PRIME_14", "");
    case at::ScalarType::FHE_PRIME_15:
      return std::make_pair("FHE_PRIME_15", "");
    case at::ScalarType::FHE_PRIME_16:
      return std::make_pair("FHE_PRIME_16", "");
    case at::ScalarType::FHE_PRIME_17:
      return std::make_pair("FHE_PRIME_17", "");
    case at::ScalarType::FHE_PRIME_18:
      return std::make_pair("FHE_PRIME_18", "");
    case at::ScalarType::FHE_PRIME_19:
      return std::make_pair("FHE_PRIME_19", "");
    case at::ScalarType::FHE_PRIME_20:
      return std::make_pair("FHE_PRIME_20", "");
    case at::ScalarType::FHE_PRIME_21:
      return std::make_pair("FHE_PRIME_21", "");
    case at::ScalarType::FHE_PRIME_22:
      return std::make_pair("FHE_PRIME_22", "");
    case at::ScalarType::FHE_PRIME_23:
      return std::make_pair("FHE_PRIME_23", "");
    case at::ScalarType::FHE_PRIME_24:
      return std::make_pair("FHE_PRIME_24", "");
    case at::ScalarType::FHE_PRIME_25:
      return std::make_pair("FHE_PRIME_25", "");
    case at::ScalarType::FHE_PRIME_26:
      return std::make_pair("FHE_PRIME_26", "");
    case at::ScalarType::FHE_PRIME_27:
      return std::make_pair("FHE_PRIME_27", "");
    case at::ScalarType::FHE_PRIME_28:
      return std::make_pair("FHE_PRIME_28", "");
    case at::ScalarType::FHE_PRIME_29:
      return std::make_pair("FHE_PRIME_29", "");
    case at::ScalarType::FHE_PRIME_30:
      return std::make_pair("FHE_PRIME_30", "");
    case at::ScalarType::FHE_PRIME_31:
      return std::make_pair("FHE_PRIME_31", "");
    case at::ScalarType::FHE_PRIME_32:
      return std::make_pair("FHE_PRIME_32", "");
    case at::ScalarType::FHE_PRIME_33:
      return std::make_pair("FHE_PRIME_33", "");
    case at::ScalarType::FHE_PRIME_34:
      return std::make_pair("FHE_PRIME_34", "");
    case at::ScalarType::FHE_PRIME_35:
      return std::make_pair("FHE_PRIME_35", "");
    case at::ScalarType::FHE_PRIME_36:
      return std::make_pair("FHE_PRIME_36", "");
    case at::ScalarType::FHE_PRIME_37:
      return std::make_pair("FHE_PRIME_37", "");
    case at::ScalarType::FHE_PRIME_38:
      return std::make_pair("FHE_PRIME_38", "");
    case at::ScalarType::FHE_PRIME_39:
      return std::make_pair("FHE_PRIME_39", "");
    case at::ScalarType::FHE_PRIME_40:
      return std::make_pair("FHE_PRIME_40", "");
    case at::ScalarType::FHE_PRIME_41:
      return std::make_pair("FHE_PRIME_41", "");
    case at::ScalarType::FHE_PRIME_42:
      return std::make_pair("FHE_PRIME_42", "");
    case at::ScalarType::FHE_PRIME_43:
      return std::make_pair("FHE_PRIME_43", "");
    case at::ScalarType::FHE_PRIME_44:
      return std::make_pair("FHE_PRIME_44", "");
    case at::ScalarType::FHE_PRIME_45:
      return std::make_pair("FHE_PRIME_45", "");
    case at::ScalarType::FHE_PRIME_46:
      return std::make_pair("FHE_PRIME_46", "");
    case at::ScalarType::FHE_PRIME_47:
      return std::make_pair("FHE_PRIME_47", "");
    case at::ScalarType::FHE_PRIME_48:
      return std::make_pair("FHE_PRIME_48", "");
    case at::ScalarType::FHE_PRIME_49:
      return std::make_pair("FHE_PRIME_49", "");
    case at::ScalarType::FHE_PRIME_50:
      return std::make_pair("FHE_PRIME_50", "");
    case at::ScalarType::FHE_PRIME_51:
      return std::make_pair("FHE_PRIME_51", "");
    case at::ScalarType::FHE_PRIME_52:
      return std::make_pair("FHE_PRIME_52", "");
    case at::ScalarType::FHE_PRIME_53:
      return std::make_pair("FHE_PRIME_53", "");
    case at::ScalarType::FHE_PRIME_54:
      return std::make_pair("FHE_PRIME_54", "");
    case at::ScalarType::FHE_PRIME_55:
      return std::make_pair("FHE_PRIME_55", "");
    case at::ScalarType::FHE_PRIME_56:
      return std::make_pair("FHE_PRIME_56", "");
    case at::ScalarType::FHE_PRIME_57:
      return std::make_pair("FHE_PRIME_57", "");
    case at::ScalarType::FHE_PRIME_58:
      return std::make_pair("FHE_PRIME_58", "");
    case at::ScalarType::FHE_PRIME_59:
      return std::make_pair("FHE_PRIME_59", "");
    case at::ScalarType::FHE_PRIME_60:
      return std::make_pair("FHE_PRIME_60", "");
    case at::ScalarType::FHE_PRIME_61:
      return std::make_pair("FHE_PRIME_61", "");
    case at::ScalarType::FHE_PRIME_62:
      return std::make_pair("FHE_PRIME_62", "");
    case at::ScalarType::FHE_PRIME_63:
      return std::make_pair("FHE_PRIME_63", "");
    case at::ScalarType::FHE_PRIME_64:
      return std::make_pair("FHE_PRIME_64", "");
    case at::ScalarType::FHE_PRIME_65:
      return std::make_pair("FHE_PRIME_65", "");
    case at::ScalarType::FHE_PRIME_66:
      return std::make_pair("FHE_PRIME_66", "");
    case at::ScalarType::FHE_PRIME_67:
      return std::make_pair("FHE_PRIME_67", "");
    case at::ScalarType::FHE_PRIME_68:
      return std::make_pair("FHE_PRIME_68", "");
    case at::ScalarType::FHE_PRIME_69:
      return std::make_pair("FHE_PRIME_69", "");
    case at::ScalarType::FHE_PRIME_70:
      return std::make_pair("FHE_PRIME_70", "");
    case at::ScalarType::FHE_PRIME_71:
      return std::make_pair("FHE_PRIME_71", "");
    case at::ScalarType::FHE_PRIME_72:
      return std::make_pair("FHE_PRIME_72", "");
    case at::ScalarType::FHE_PRIME_73:
      return std::make_pair("FHE_PRIME_73", "");
    case at::ScalarType::FHE_PRIME_74:
      return std::make_pair("FHE_PRIME_74", "");
    case at::ScalarType::FHE_PRIME_75:
      return std::make_pair("FHE_PRIME_75", "");
    case at::ScalarType::FHE_PRIME_76:
      return std::make_pair("FHE_PRIME_76", "");
    case at::ScalarType::FHE_PRIME_77:
      return std::make_pair("FHE_PRIME_77", "");
    case at::ScalarType::FHE_PRIME_78:
      return std::make_pair("FHE_PRIME_78", "");
    case at::ScalarType::FHE_PRIME_79:
      return std::make_pair("FHE_PRIME_79", "");
    case at::ScalarType::FHE_PRIME_80:
      return std::make_pair("FHE_PRIME_80", "");
    case at::ScalarType::FHE_PRIME_81:
      return std::make_pair("FHE_PRIME_81", "");
    case at::ScalarType::FHE_PRIME_82:
      return std::make_pair("FHE_PRIME_82", "");
    case at::ScalarType::FHE_PRIME_83:
      return std::make_pair("FHE_PRIME_83", "");
    case at::ScalarType::FHE_PRIME_84:
      return std::make_pair("FHE_PRIME_84", "");
    case at::ScalarType::FHE_PRIME_85:
      return std::make_pair("FHE_PRIME_85", "");
    case at::ScalarType::FHE_PRIME_86:
      return std::make_pair("FHE_PRIME_86", "");
    case at::ScalarType::FHE_PRIME_87:
      return std::make_pair("FHE_PRIME_87", "");
    case at::ScalarType::FHE_PRIME_88:
      return std::make_pair("FHE_PRIME_88", "");
    case at::ScalarType::FHE_PRIME_89:
      return std::make_pair("FHE_PRIME_89", "");
    case at::ScalarType::FHE_PRIME_90:
      return std::make_pair("FHE_PRIME_90", "");
    case at::ScalarType::FHE_PRIME_91:
      return std::make_pair("FHE_PRIME_91", "");
    case at::ScalarType::FHE_PRIME_92:
      return std::make_pair("FHE_PRIME_92", "");
    case at::ScalarType::FHE_PRIME_93:
      return std::make_pair("FHE_PRIME_93", "");
    case at::ScalarType::FHE_PRIME_94:
      return std::make_pair("FHE_PRIME_94", "");
    case at::ScalarType::FHE_PRIME_95:
      return std::make_pair("FHE_PRIME_95", "");
    case at::ScalarType::FHE_PRIME_96:
      return std::make_pair("FHE_PRIME_96", "");
    case at::ScalarType::FHE_PRIME_97:
      return std::make_pair("FHE_PRIME_97", "");
    case at::ScalarType::FHE_PRIME_98:
      return std::make_pair("FHE_PRIME_98", "");
    case at::ScalarType::FHE_PRIME_99:
      return std::make_pair("FHE_PRIME_99", "");
    case at::ScalarType::FHE_PRIME_100:
      return std::make_pair("FHE_PRIME_100", "");
    case at::ScalarType::FHE_PRIME_101:
      return std::make_pair("FHE_PRIME_101", "");
    case at::ScalarType::FHE_PRIME_102:
      return std::make_pair("FHE_PRIME_102", "");
    case at::ScalarType::FHE_PRIME_103:
      return std::make_pair("FHE_PRIME_103", "");
    case at::ScalarType::FHE_PRIME_104:
      return std::make_pair("FHE_PRIME_104", "");
    case at::ScalarType::FHE_PRIME_105:
      return std::make_pair("FHE_PRIME_105", "");
    case at::ScalarType::FHE_PRIME_106:
      return std::make_pair("FHE_PRIME_106", "");
    case at::ScalarType::FHE_PRIME_107:
      return std::make_pair("FHE_PRIME_107", "");
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

void initializeDtypes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  at::ScalarType all_scalar_types[] = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

  for (at::ScalarType scalarType : all_scalar_types) {
    auto [primary_name, legacy_name] = getDtypeNames(scalarType);
    PyObject* dtype = THPDtype_New(scalarType, primary_name);
    torch::registerDtypeObject((THPDtype*)dtype, scalarType);
    Py_INCREF(dtype);
    if (PyModule_AddObject(torch_module.get(), primary_name.c_str(), dtype) !=
        0) {
      throw python_error();
    }
    if (!legacy_name.empty()) {
      Py_INCREF(dtype);
      if (PyModule_AddObject(torch_module.get(), legacy_name.c_str(), dtype) !=
          0) {
        throw python_error();
      }
    }
  }
}

} // namespace utils
} // namespace torch
