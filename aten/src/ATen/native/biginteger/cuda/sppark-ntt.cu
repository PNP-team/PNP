#include <stddef.h>
#include <stdint.h>

#include <ATen/native/biginteger/cuda/sppark-ntt/ntt.cuh>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>

#include <math.h>

//temporarily set device_id to 0, set InputOutputOrder to NN
namespace at {
namespace native {


static void ntt_zkp(Tensor& inout) {
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    compute_ntt(
        0,
        self_ptr,
        lg_domain_size,  
        InputOutputOrder::NN,
        Direction::forward,
        Type::standard
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_zkp(Tensor& inout) {
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    compute_ntt(
        0,
        self_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        Direction::inverse,
        Type::standard
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void ntt_coset_zkp(Tensor& inout) {
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    compute_ntt(
        0,
        self_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        Direction::forward,
        Type::coset
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_coset_zkp(Tensor& inout) {
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    compute_ntt(
        0,
        self_ptr,
        lg_domain_size,
        InputOutputOrder::NN,
        Direction::inverse,
        Type::coset
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

Tensor ntt_zkp_cuda(const Tensor& inout) {
  Tensor output = inout.clone();
  ntt_zkp(output);
  return output;
}

Tensor& ntt_zkp_cuda_(Tensor& inout) {
  ntt_zkp(inout);
  return inout;
}

Tensor& ntt_zkp_out_cuda(const Tensor& inout, Tensor& output) {
  copy(output, inout);
  ntt_zkp(output);
  return output;
}

Tensor intt_zkp_cuda(const Tensor& inout) {
  Tensor output = inout.clone();
  intt_zkp(output);
  return output;
}

Tensor& intt_zkp_cuda_(Tensor& inout) {
  intt_zkp(inout);
  return inout;
}

Tensor& intt_zkp_out_cuda(const Tensor& inout, Tensor& output) {
  copy(output, inout);
  intt_zkp(output);
  return output;
}

Tensor ntt_coset_zkp_cuda(const Tensor& inout) {
  Tensor output = inout.clone();
  ntt_coset_zkp(output);
  return output;
}

Tensor& ntt_coset_zkp_cuda_(Tensor& inout) {
  ntt_coset_zkp(inout);
  return inout;
}

Tensor& ntt_coset_zkp_out_cuda(const Tensor& inout, Tensor& output) {
  copy(output, inout);
  ntt_coset_zkp(output);
  return output;
}

Tensor intt_coset_zkp_cuda(const Tensor& inout) {
  Tensor output = inout.clone();
  intt_coset_zkp(output);
  return output;
}

Tensor& intt_coset_zkp_cuda_(Tensor& inout) {
  intt_coset_zkp(inout);
  return inout;
}

Tensor& intt_coset_zkp_out_cuda(const Tensor& inout, Tensor& output) {
  copy(output, inout);
  intt_coset_zkp(output);
  return output;
}

}//namespace native
}//namespace at