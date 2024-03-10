#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include <cstdio>
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

constexpr static int lg2(size_t n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static void mult_pippenger_inf(Tensor& self, const Tensor& points, const Tensor& scalars)
{
    AT_DISPATCH_FQ_MONT_TYPES(self.scalar_type(), "msm_cuda", [&] {
        using point_t = jacobian_t<scalar_t::compute_type>;
        using bucket_t = xyzz_t<scalar_t::compute_type>;
        using affine_t = bucket_t::affine_t;
        auto npoints = points.numel() / (num_uint64(points.scalar_type()) * 2);
        auto ffi_affine_sz = sizeof(affine_t); //affine mode (X,Y)
        auto self_ptr = reinterpret_cast<bucket_t*>(self.mutable_data_ptr<scalar_t>());
        auto point_ptr = reinterpret_cast<affine_t*>(points.mutable_data_ptr<scalar_t>());
        auto scalar_ptr = reinterpret_cast<scalar_t::compute_type::coeff_t*>(scalars.mutable_data_ptr());
        mult_pippenger<point_t>(self_ptr, point_ptr, npoints, scalar_ptr, false, ffi_affine_sz); 
    });
}

Tensor msm_zkp_cuda(const Tensor& points, const Tensor& scalars) {

    auto wbits = 17;
    auto npoints = points.numel() / (num_uint64(points.scalar_type()) *2);
    if (npoints > 192) {
        wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
        if (wbits < 10)
            wbits = 10;
    } else if (npoints > 0) {
        wbits = 10;
    }
    auto nbits = bit_length(scalars.scalar_type());
    auto nwins = (nbits - 1) / wbits + 1;
    auto smcount = 34;
    std::cout << "nwins: " << nwins << std::endl;
    std::cout << "ones: " << smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ << std::endl;
    Tensor out = at::empty({(nwins * MSM_NTHREADS/1 * 2 + smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) * 4, num_uint64(points.scalar_type())},
      points.options());
    mult_pippenger_inf(out, points, scalars);
    return out;
}


}//namespace native
}//namespace at