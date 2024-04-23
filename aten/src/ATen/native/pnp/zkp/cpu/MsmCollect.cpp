#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/native/pnp/mont/cpu/curve_def.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <iostream>
#include "collect.h"
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wmissing-prototypes")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wmissing-prototypes")
#endif

namespace at {
namespace native {

static void pippenger_collect(
    Tensor& self,
    const Tensor& step1res,
    size_t npoints) {
  AT_DISPATCH_FQ_MONT_TYPES(self.scalar_type(), "msm_cpu", [&] {
    using point_t = jacobian_t<scalar_t::compute_type>;
    using bucket_t = xyzz_t<scalar_t::compute_type>;
    using affine_t = bucket_t::affine_t;
    auto wbits = 17;
    if (npoints > 192) {
      wbits = std::min(lg2(npoints + npoints / 2) - 8, 18);
      if (wbits < 10)
        wbits = 10;
    } else if (npoints > 0) {
      wbits = 10;
    }
    auto nbits = scalar_t::compute_type::coeff_t::bit_length();
    auto nwins = (nbits - 1) / wbits + 1;
    auto lenofres = nwins * MSM_NTHREADS / 1 * 2;
    auto lenofone =
        step1res.numel() / (num_uint64(step1res.scalar_type()) * 4) - lenofres;

    auto self_ptr =
        reinterpret_cast<point_t*>(self.mutable_data_ptr<scalar_t>());
    auto res_ptr =
        reinterpret_cast<bucket_t*>(step1res.mutable_data_ptr<scalar_t>());
    auto ones_ptr =
        reinterpret_cast<bucket_t*>(step1res.mutable_data_ptr<scalar_t>()) +
        lenofres;
    collect_t<bucket_t, point_t, affine_t, scalar_t::compute_type::coeff_t>
        msm_collect{npoints};
    msm_collect.collect(self_ptr, res_ptr, ones_ptr, lenofone);
  });
}

Tensor msm_collect_cpu(const Tensor& step1res, int64_t npoints) {
  Tensor out =
      at::empty({3, num_uint64(step1res.scalar_type())}, step1res.options());
  pippenger_collect(out, step1res, npoints);
  return out;
}

} // namespace native
} // namespace at