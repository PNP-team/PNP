#include <stddef.h>
#include <stdint.h>
#include <iostream>

#include "ec/xyzz_t.hpp"
#include "sppark-msm/pippenger.cuh"

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>

#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "CurveDef.cuh"
#include <math.h>

namespace at {
namespace native {

static void mult_pippenger_inf(Tensor& self, const Tensor& points, const Tensor& scalars)
{
    AT_DISPATCH_FQ_MONT_TYPES(self.scalar_type(), "msm_cuda", [&] {
        using point_t = jacobian_t<scalar_t::compute_type>;
        using bucket_t = xyzz_t<scalar_t::compute_type>;
        using affine_t = bucket_t::affine_t;
        auto npoints = points.numel() / num_uint64(points.scalar_type());
        auto ffi_affine_sz = sizeof(affine_t); //affine mode (X,Y)
       auto self_ptr = reinterpret_cast<point_t*>(self.mutable_data_ptr<scalar_t>());
       auto point_ptr = reinterpret_cast<affine_t*>(points.mutable_data_ptr<scalar_t>());
       auto scalar_ptr = reinterpret_cast<scalar_t::compute_type::coeff_t*>(scalars.mutable_data_ptr());
       mult_pippenger<bucket_t>(self_ptr, point_ptr, npoints, scalar_ptr, true, ffi_affine_sz);
    });
}

Tensor msm_zkp_cuda(const Tensor& points, const Tensor& scalars) {
    std::cout << points.scalar_type() << std::endl;
    Tensor out = at::empty({3, num_uint64(points.scalar_type())}, points.options());
    std::cout << out.scalar_type() << std::endl;
    mult_pippenger_inf(out, points, scalars);
    return out;
}


}//namespace native
}//namespace at