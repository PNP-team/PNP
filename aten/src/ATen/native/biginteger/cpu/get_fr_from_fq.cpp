#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include "CurveDef.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

static void msm(Tensor& self) {
  AT_DISPATCH_FQ_MONT_TYPES(self.scalar_type(), "msm_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    auto fr_ptr = reinterpret_cast<scalar_t::compute_type::coeff_t*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].to();
      fr_ptr[i].to();
    }
  });
}

} // namespace

} // namespace native
} // namespace at
