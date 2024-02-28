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
        using bucket_h = bucket_t::mem_t;
        using affine_t = bucket_t::affine_t;
        auto npoints = points.numel() / num_uint64(points.scalar_type());
        auto ffi_affine_sz = sizeof(affine_t); //affine mode (X,Y)
        auto self_ptr = reinterpret_cast<bucket_h*>(self.mutable_data_ptr<scalar_t>());
        auto point_ptr = reinterpret_cast<affine_t*>(points.mutable_data_ptr<scalar_t>());
        auto scalar_ptr = reinterpret_cast<scalar_t::compute_type::coeff_t*>(scalars.mutable_data_ptr());
        mult_pippenger<point_t, bucket_t>(self_ptr, point_ptr, npoints, scalar_ptr, true, ffi_affine_sz); //template选择问题
    });
}

Tensor msm_zkp_cuda(const Tensor& points, const Tensor& scalars) {

    std::cout << points.scalar_type() << std::endl;
    // Tensor out = at::empty({3, num_uint64(points.scalar_type())}, points.options());
    auto wbits = 17;
    auto nbits = bit_length(scalars.scalar_type());
    auto nwins = (nbits - 1) / wbits + 1;
    auto smcount = 34;
    //需要在这里自动获取bit length
    std::cout << "zhiyuan's nwins: " << nwins * MSM_NTHREADS/1 * 2 << std::endl;
    std::cout << "zhiyuan's ones: " << smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ << std::endl;

    Tensor out = at::empty({(nwins * MSM_NTHREADS/1 * 2 + smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) * 3, num_uint64(points.scalar_type())}, points.options());
    std::cout<<out.numel()<<std::endl;
    // std::cout << out.scalar_type() << std::endl;
    mult_pippenger_inf(out, points, scalars);
    std::cout<< " lalala " <<std::endl;
    return out;
}


}//namespace native
}//namespace at