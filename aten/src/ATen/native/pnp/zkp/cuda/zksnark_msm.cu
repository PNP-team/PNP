#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <math.h>
#include <stdint.h>
#include <ATen/native/pnp/mont/cuda/curve_def.cuh>
#include <cstdio>
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"
#include "sppark_msm/pippenger.cuh"

namespace at {
namespace native {

constexpr static int lg2(size_t n) {
  int ret = 0;
  while (n >>= 1)
    ret++;
  return ret;
}

static void mult_pippenger_inf(
    Tensor& self,
    const Tensor& points,
    const Tensor& scalars,
    Tensor& workspace,
    int64_t smcount,
    int64_t blob_u64) {
  AT_DISPATCH_FQ_MONT_TYPES(points.scalar_type(), "msm_cuda", [&] {
    using point_t = jacobian_t<scalar_t::compute_type>;
    using bucket_t = xyzz_t<scalar_t::compute_type>;
    using bucket_h = bucket_t::mem_t;
    using affine_t = bucket_t::affine_t;
    auto npoints = points.numel() / (num_uint64(points.scalar_type()) * 2);
    auto ffi_affine_sz = sizeof(affine_t); // affine mode (X,Y)
    auto bucket_ptr =
        reinterpret_cast<bucket_h*>(workspace.mutable_data_ptr<uint64_t>());
    auto temp_ptr = reinterpret_cast<uint8_t*>(
        workspace.mutable_data_ptr<uint64_t>() + blob_u64);
    auto self_ptr =
        reinterpret_cast<bucket_t*>(self.mutable_data_ptr<scalar_t>());
    auto point_ptr = reinterpret_cast<affine_t*>(points.mutable_data_ptr());
    auto scalar_ptr = reinterpret_cast<scalar_t::compute_type::coeff_t*>(
        scalars.mutable_data_ptr());
    mult_pippenger<point_t>(
        self_ptr,
        point_ptr,
        npoints,
        scalar_ptr,
        bucket_ptr,
        temp_ptr,
        smcount,
        false,
        ffi_affine_sz);
  });
}

Tensor msm_zkp_cuda(
    const Tensor& points,
    const Tensor& scalars,
    int64_t smcount,
    c10::optional<Device> device,
    c10::optional<Layout> layout,
    c10::optional<bool> pin_memory) {
  auto wbits = 17;
  auto npoints = points.numel() / (num_uint64(points.scalar_type()) * 2);
  if (npoints > 192) {
    wbits = std::min(lg2(npoints + npoints / 2) - 8, 18);
    if (wbits < 10)
      wbits = 10;
  } else if (npoints > 0) {
    wbits = 10;
  }
  auto nbits = bit_length(scalars.scalar_type());
  auto nwins = (nbits - 1) / wbits + 1;
  uint32_t row_sz = 1U << (wbits - 1);
  size_t d_buckets_sz =
      (nwins * row_sz) + (smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
  size_t d_blob_sz =
      d_buckets_sz * sizeof(uint64_t) * num_uint64(points.scalar_type()) * 4 +
      (nwins * row_sz * sizeof(uint32_t));
  uint32_t blob_u64 = d_blob_sz / sizeof(uint64_t);

  size_t digits_sz = nwins * npoints * sizeof(uint32_t);
  uint32_t digit_u64 = digits_sz / sizeof(uint64_t);

  auto workspace = at::empty(
      {blob_u64 + digit_u64},
      ScalarType::ULong,
      points.options().layout(),
      points.options().device(),
      points.options().pinned_memory(),
      c10::nullopt);

  auto out = at::empty(
      {(nwins * MSM_NTHREADS / 1 * 2 +
        smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ) *
           4,
       num_uint64(points.scalar_type())},
      points.options());

  mult_pippenger_inf(out, points, scalars, workspace, smcount, blob_u64);
  return out;
}

} // namespace native
} // namespace at