#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/native/pnp/mont/cuda/curve_def.cuh>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define BIN_KERNEL(name, op)                           \
  template <typename T>                                \
  __global__ void mont_##name##_mod_kernel(            \
      const int64_t N, T* c, const T* a, const T* b) { \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < N) {                                       \
      c[i] = a[i] op b[i];                             \
    }                                                  \
  }                                                    \
  template <typename T>                                \
  __global__ void mont_##name##_mod_kernel_(           \
      const int64_t N, T* self, const T* other) {      \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < N) {                                       \
      self[i] op## = other[i];                         \
    }                                                  \
  }

#define BIN_OP_TEMPLATE(name)                                                \
  static void name##_template(Tensor& c, const Tensor& a, const Tensor& b) { \
    TORCH_CHECK(                                                             \
        a.numel() == b.numel(), "The number of elements must be the same!"); \
    AT_DISPATCH_MONT_TYPES(a.scalar_type(), "mont_" #name "_mod_cuda", [&] { \
      auto a_ptr =                                                           \
          reinterpret_cast<scalar_t::compute_type*>(a.data_ptr<scalar_t>()); \
      auto b_ptr =                                                           \
          reinterpret_cast<scalar_t::compute_type*>(b.data_ptr<scalar_t>()); \
      auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(                \
          c.mutable_data_ptr<scalar_t>());                                   \
      int64_t N = a.numel() / num_uint64(a.scalar_type());                   \
      int64_t grid = (N + block_work_size() - 1) / block_work_size();        \
      auto stream = at::cuda::getCurrentCUDAStream();                        \
      mont_##name##_mod_kernel<<<grid, block_work_size(), 0, stream>>>(      \
          N, c_ptr, a_ptr, b_ptr);                                           \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    });                                                                      \
  }                                                                          \
  static void name##_template_(Tensor& self, const Tensor& other) {          \
    TORCH_CHECK(                                                             \
        self.numel() == other.numel(),                                       \
        "The number of elements must be the same!");                         \
    AT_DISPATCH_MONT_TYPES(                                                  \
        self.scalar_type(), "mont_" #name "_mod_cuda", [&] {                 \
          auto other_ptr = reinterpret_cast<scalar_t::compute_type*>(        \
              other.data_ptr<scalar_t>());                                   \
          auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(         \
              self.mutable_data_ptr<scalar_t>());                            \
          int64_t N = self.numel() / num_uint64(self.scalar_type());         \
          int64_t grid = (N + block_work_size() - 1) / block_work_size();    \
          auto stream = at::cuda::getCurrentCUDAStream();                    \
          mont_##name##_mod_kernel_<<<grid, block_work_size(), 0, stream>>>( \
              N, self_ptr, other_ptr);                                       \
          C10_CUDA_KERNEL_LAUNCH_CHECK();                                    \
        });                                                                  \
  }

#define BIN_OP(name)                                                         \
  Tensor name##_mod_cuda(const Tensor& a, const Tensor& b) {                 \
    Tensor c = at::empty_like(a);                                            \
    name##_template(c, a, b);                                                \
    return c;                                                                \
  }                                                                          \
  Tensor& name##_mod_cuda_(Tensor& self, const Tensor& other) {              \
    name##_template_(self, other);                                           \
    return self;                                                             \
  }                                                                          \
  Tensor& name##_mod_cuda_out(const Tensor& a, const Tensor& b, Tensor& c) { \
    name##_template(c, a, b);                                                \
    return c;                                                                \
  }

namespace at {
namespace native {

namespace {

template <typename T>
__global__ void to_mont_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].to();
  }
}

template <typename T>
__global__ void to_base_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i].from();
  }
}

BIN_KERNEL(add, +);
BIN_KERNEL(sub, -);
BIN_KERNEL(mul, *);
BIN_KERNEL(div, /);

#define CONVERT_CURVE(name)                       \
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
  APPLY_ALL_CURVE(CONVERT_CURVE)
  else {
    throw std::runtime_error("Unsupported curve type");
  }
}
#undef CONVERT_CURVE

static void to_mont_cuda_template(Tensor& self) {
  AT_DISPATCH_BASE_TYPES(self.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    to_mont_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  if (!c10::isFHEPrimeType(self.scalar_type())) {
    self.set_dtype(get_corresponding_type(self.scalar_type()));
  }
}

static void to_base_cuda_template(Tensor& self) {
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "to_base_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    to_base_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  if (!c10::isFHEPrimeType(self.scalar_type())) {
    self.set_dtype(get_corresponding_type(self.scalar_type()));
  }
}

BIN_OP_TEMPLATE(add);
BIN_OP_TEMPLATE(sub);
BIN_OP_TEMPLATE(mul);
BIN_OP_TEMPLATE(div);

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

BIN_OP(add);
BIN_OP(sub);
BIN_OP(mul);
BIN_OP(div);

} // namespace native
} // namespace at
