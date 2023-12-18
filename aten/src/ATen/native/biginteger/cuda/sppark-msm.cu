#include <stddef.h>
#include <stdint.h>

#include "ec/jacobian_t.hpp"
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

#include <math.h>

namespace at {
namespace native {

// template <fp_t>
// struct MSMType {
//     using point_t = jacobian_t<fp_t>;
// }
// typedef jacobian_t<fp_t> point_t;
// typedef xyzz_t<fp_t> bucket_t;
// typedef bucket_t::affine_inf_t affine_t;
// typedef fr_t scalar_t;

template <typename fq_t>
static void mult_pippenger_inf(Tensor& out, const Tensor& points, const Tensor& scalars)
{
    using point_t = jacobian_t<fq_t>;
    using bucket_t = xyzz_t<fq_t>;
    auto npoints = points.numel() / num_uint64(points.scalar_type());
    auto ffi_affine_sz = sizeof(fq_t) * 2; //affine mode (X,Y)
    // AT_DISPATCH_FQ_MONT_TYPES(out.scalar_type(), "msm_cuda", [&] {
    //    auto point_ptr = reinterpret_cast<scalar_t::compute_type*>(points.mutable_data_ptr<scalar_t>());
    //    auto scalar_ptr = reinterpret_cast<scalar_t::compute_type::scalar_type*>(scalars.mutable_data_ptr<scalar_t>());
    //    mult_pippenger<bucket_t>(out, point_ptr, npoints, scalar_ptr, false, ffi_affine_sz);
    // });
}

}//namespace native
}//namespace at