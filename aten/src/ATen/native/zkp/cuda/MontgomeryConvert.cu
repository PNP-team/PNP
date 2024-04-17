#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>

#include "CurveDef.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

template <typename T>
__global__ void to_mont_kernel(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].to();
  }
}

template <typename T>
__global__ void to_base_kernel(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].from();
  }
}

template <typename T>
__global__ void add_mont_kernel(const int64_t N, T* a, T* b) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] += b[i];
  }
}

template <typename T>
__global__ void sub_mont_kernel(const int64_t N, T* a, T* b) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] -= b[i];
  }
}

template <typename T>
__global__ void mul_mont_kernel(const int64_t N, T* a, T* b) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] *= b[i];
  }
}

template <typename T>
__global__ void div_mont_kernel(const int64_t N, T* a, T* b) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] /= b[i];
  }
}

#define CONVERT_ELEM(name)                        \
  else if (type == ScalarType::name##_Base) {     \
    return caffe2::TypeMeta::Make<name##_Mont>(); \
  }                                               \
  else if (type == ScalarType::name##_Mont) {     \
    return caffe2::TypeMeta::Make<name##_Base>(); \
  }

caffe2::TypeMeta get_corresponding_type(const ScalarType type) {
  if (false) {
    ;
  }
  APPLY_ALL_CURVE(CONVERT_ELEM)
  else {
    throw std::runtime_error("Unsupported curve type");
  }
}
#undef CONVERT_ELEM

static void to_mont_cuda_template(Tensor& self) {
  AT_DISPATCH_BASE_TYPES(self.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    to_mont_kernel<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}
static void add_cuda_template(Tensor& a, const Tensor& b) {
  TORCH_CHECK(a.numel() == b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(a.scalar_type(), "add_mod_cuda", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        b.mutable_data_ptr<scalar_t>());
    int64_t N = a.numel() / num_uint64(a.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    add_mont_kernel<<<grid, block_work_size(), 0, stream>>>(N, a_ptr, b_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void sub_cuda_template(Tensor& a, const Tensor& b) {
  TORCH_CHECK(a.numel() == b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(a.scalar_type(), "sub_mod_cuda", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        b.mutable_data_ptr<scalar_t>());
    int64_t N = a.numel() / num_uint64(a.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    sub_mont_kernel<<<grid, block_work_size(), 0, stream>>>(N, a_ptr, b_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void mul_cuda_template(Tensor& a, const Tensor& b) {
  TORCH_CHECK(a.numel() == b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(a.scalar_type(), "mul_mod_cuda", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        b.mutable_data_ptr<scalar_t>());
    int64_t N = a.numel() / num_uint64(a.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    mul_mont_kernel<<<grid, block_work_size(), 0, stream>>>(N, a_ptr, b_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
static void div_cuda_template(Tensor& a, const Tensor& b) {
  TORCH_CHECK(a.numel() == b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(a.scalar_type(), "div_mod_cuda", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        b.mutable_data_ptr<scalar_t>());
    int64_t N = a.numel() / num_uint64(a.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    div_mont_kernel<<<grid, block_work_size(), 0, stream>>>(N, a_ptr, b_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void to_base_cuda_template(Tensor& self) {
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "to_base_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    to_base_kernel<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

} // namespace

Tensor to_mont_cuda(const Tensor& input) {
  Tensor output = input.clone();
  to_mont_cuda_template(output);
  return output;
}

Tensor& to_mont_cuda_(Tensor& self) {
  to_mont_cuda_template(self);
  return self;
}

Tensor& to_mont_out_cuda(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_mont_cuda_template(output);
  return output;
}

Tensor to_base_cuda(const Tensor& input) {
  Tensor output = input.clone();
  to_base_cuda_template(output);
  return output;
}

Tensor& to_base_cuda_(Tensor& self) {
  to_base_cuda_template(self);
  return self;
}

Tensor& to_base_out_cuda(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_base_cuda_template(output);
  return output;
}

Tensor add_mod_cuda(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  add_cuda_template(c, b);
  return c;
}

Tensor& add_mod_cuda_(Tensor& a, const Tensor& b) {
  add_cuda_template(a, b);
  return a;
}

Tensor& add_mod_cuda_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  add_cuda_template(c, b);
  return c;
}

Tensor sub_mod_cuda(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  sub_cuda_template(c, b);
  return c;
}

Tensor& sub_mod_cuda_(Tensor& a, const Tensor& b) {
  sub_cuda_template(a, b);
  return a;
}
Tensor& sub_mod_cuda_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  sub_cuda_template(c, b);
  return c;
}

Tensor mul_mod_cuda(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  mul_cuda_template(c, b);
  return c;
}
Tensor& mul_mod_cuda_(Tensor& a, const Tensor& b) {
  mul_cuda_template(a, b);
  return a;
}
Tensor& mul_mod_cuda_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  mul_cuda_template(c, b);
  return c;
}

Tensor div_mod_cuda(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  div_cuda_template(c, b);
  return c;
}
Tensor& div_mod_cuda_(Tensor& a, const Tensor& b) {
  div_cuda_template(a, b);
  return a;
}
Tensor& div_mod_cuda_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  div_cuda_template(c, b);
  return c;
}

} // namespace native
} // namespace at
