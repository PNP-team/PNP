
#include <cstdint>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <ATen/core/DistributionsHelper.h>

#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/celu_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu_backward_native.h>
#include <ATen/ops/elu_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/hardshrink_backward_native.h>
#include <ATen/ops/hardshrink_native.h>
#include <ATen/ops/hardsigmoid_backward_native.h>
#include <ATen/ops/hardsigmoid_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/hardswish_native.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward_native.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/log_sigmoid_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/relu6_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rrelu_native.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/rrelu_with_noise_backward_native.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#include <ATen/ops/selu_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/softshrink_backward_native.h>
#include <ATen/ops/softshrink_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>
#include <ATen/ops/zeros_like.h>

#include <utility>
#include <vector>
#endif

#include <execinfo.h>


struct alignas(8) BLS12_377_Fq_G2 {
  using underlying = uint64_t;
  uint64_t val_;
  BLS12_377_Fq_G2() = default;
  C10_HOST_DEVICE explicit BLS12_377_Fq_G2(uint64_t val) : val_(val) {}
};

struct alignas(8) BN254_Fr_G1 {
  using underlying = uint64_t;
  uint64_t val_;
  BN254_Fr_G1() = default;
  C10_HOST_DEVICE explicit BN254_Fr_G1(uint64_t val) : val_(val) {}
};


namespace at {
namespace native {

namespace {

template <typename T>
static void to_mont(const c10::EllipticCurve* input, const int64_t num, c10::EllipticCurve* output) {
  auto input_ = reinterpret_cast<const T*>(input);
  auto output_ = reinterpret_cast<T*>(output);
  auto num_ = num / (sizeof(T)/sizeof(c10::EllipticCurve));
  for(auto i = 0; i < num_; i++) {
    output_[i].val_ = input_[i].val_ + 3;
  }
}




template <typename T>
static void add(const c10::EllipticCurve* input, const int64_t num, c10::EllipticCurve* output) {
  auto input_ = reinterpret_cast<const T*>(input);
  auto output_ = reinterpret_cast<T*>(output);
  auto num_ = num / (sizeof(T)/sizeof(c10::EllipticCurve));
  for(auto i = 0; i < num_; i++) {
    output_[i].val_ += input_[i].val_ + 3;
  }
}




#define CURVE_DISPATCH_SWITCH(TYPE, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    c10::CurveType _st = the_type;                                          \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported curve type");                       \
    }                                                                       \
  }()

#define CURVE_DISPATCH_CASE(enum_type, NAME, ...)  \
  case CurveType::enum_type: {                     \
    NAME<enum_type>(__VA_ARGS__);                  \
    break;                                         \
  }

#define CURVE_DISPATCH_CASE_TYPES(NAME, ...)      \
  CURVE_DISPATCH_CASE(BN254_Fr_G1, NAME, __VA_ARGS__)     \
  CURVE_DISPATCH_CASE(BLS12_377_Fq_G2, NAME, __VA_ARGS__)

#define CURVE_DISPATCH_TYPES(TYPE, NAME, ...) \
  CURVE_DISPATCH_SWITCH(TYPE, CURVE_DISPATCH_CASE_TYPES(NAME, __VA_ARGS__))


static void to_mount_cpu_template(
    Tensor& output,
    const Tensor& input_) {

  //check whether it is in normal space
  
  AT_DISPATCH_CURVE_TYPES(input_.scalar_type(), "to_mont_cpu", [&] {
    CURVE_DISPATCH_TYPES(input_.curve().type(), 
                        to_mont, 
                        input_.const_data_ptr<scalar_t>(), 
                        input_.numel(), 
                        output.mutable_data_ptr<scalar_t>());
  });
}

} // namespace

Tensor to_mont_cpu(const Tensor& input) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  to_mount_cpu_template(output, input);
  return output;
}

Tensor& to_mont_cpu_(Tensor& self) {
  to_mount_cpu_template(self, self);
  return self;
}

Tensor& to_mont_out_cpu(const Tensor& input, Tensor& output) {
  to_mount_cpu_template(output, input);
  return output;
}

}}  // namespace at::native
