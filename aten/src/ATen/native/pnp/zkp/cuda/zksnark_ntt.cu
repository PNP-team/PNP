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
#include <stddef.h>
#include <stdint.h>
#include "zksnark_ntt/ntt_kernel/ntt.cuh"
#include "zksnark_ntt/parameters/parameters.cuh"

// temporarily set device_id to 0, set InputOutputOrder to NN
// TODO: optimize memory copy for inout data
namespace at {
namespace native {

static void params_zkp_template(
    Tensor& self,
    Tensor& local_params,
    bool is_intt) {
  AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "load_ntt_params_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    auto local_ptr = reinterpret_cast<scalar_t::compute_type*>(
        local_params.mutable_data_ptr<scalar_t>());
    NTTParameters(is_intt, self_ptr, local_ptr);
  });
}

Tensor params_zkp_cuda(
    int64_t domain_size,
    bool is_intt,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  auto partial_sz = WINDOW_NUM * WINDOW_SIZE;
  auto S1 = 2 * partial_sz;
  auto S2 = 32 + 64 + 128 + 256 + 512;
  auto S3 = 64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512;
  auto S4 = domain_size + 1;

  auto double_roots = 2 * domain_size; // forward and inverse
  auto double_group_gen = 2; // group generator and inverse group generator

  auto local_params = at::empty(
      {double_group_gen + double_roots, num_uint64(*dtype)},
      dtype,
      layout,
      device,
      pin_memory,
      c10::nullopt);
  auto params = at::empty(
      {S1 + S2 + S3 + S4, num_uint64(*dtype)},
      dtype,
      layout,
      device,
      pin_memory,
      c10::nullopt);
  params_zkp_template(params, local_params, is_intt);
  return params;
}

static void ntt_zkp(
    Tensor& self,
    const Tensor& params,
    bool is_intt,
    bool is_coset) {
  auto len = self.numel() / num_uint64(self.scalar_type());
  uint32_t lg_domain_size = log2(len);
  TORCH_CHECK(len == 1 << lg_domain_size, "NTT Length check!");
  AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "ntt_cuda", [&] {
    auto L1 = WINDOW_NUM * WINDOW_SIZE;
    auto L2 = 2 * L1;
    auto L3 = L2 + (32 + 64 + 128 + 256 + 512);
    auto L4 = L3 + (64 * 64 + 4096 * 64 + 128 * 128 + 256 * 256 + 512 * 512);

    auto pt_ptr = reinterpret_cast<scalar_t::compute_type*>(
        params.mutable_data_ptr<scalar_t>());
    auto pggp_ptr = reinterpret_cast<scalar_t::compute_type*>(
                        params.mutable_data_ptr<scalar_t>()) +
        L1;
    auto rp_ptr = reinterpret_cast<scalar_t::compute_type*>(
                      params.mutable_data_ptr<scalar_t>()) +
        L2;
    auto rpm_ptr = reinterpret_cast<scalar_t::compute_type*>(
                       params.mutable_data_ptr<scalar_t>()) +
        L3;
    auto size_inverse_ptr = reinterpret_cast<scalar_t::compute_type*>(
                                params.mutable_data_ptr<scalar_t>()) +
        L4;
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());

    compute_ntt(
        0,
        self_ptr,
        pt_ptr,
        rp_ptr,
        rpm_ptr,
        pggp_ptr,
        size_inverse_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        is_intt ? Direction::inverse : Direction::forward,
        is_coset ? Type::coset : Type::standard);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

Tensor ntt_zkp_cuda(
    const Tensor& inout,
    const Tensor& params,
    bool is_intt,
    bool is_coset) {
  Tensor output = inout.clone();
  ntt_zkp(output, params, is_intt, is_coset);
  return output;
}

Tensor& ntt_zkp_cuda_(
    Tensor& inout,
    const Tensor& params,
    bool is_intt,
    bool is_coset) {
  ntt_zkp(inout, params, is_intt, is_coset);
  return inout;
}

Tensor& ntt_zkp_out_cuda(
    const Tensor& inout,
    const Tensor& params,
    bool is_intt,
    bool is_coset,
    Tensor& output) {
  copy(output, inout);
  ntt_zkp(output, params, is_intt, is_coset);
  return output;
}

} // namespace native
} // namespace at