#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/native/pnp/mont/cuda/curve_def.cuh>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define BLOCK_SIZE (512)
#define MAX_NUM_BLOCKS (BLOCK_SIZE)

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

#define SCALAR_KERNEL(name, op)                             \
  template <typename T>                                     \
  __global__ void mont_##name##_scalar_mod_kernel(          \
      const int64_t N, T* c, const T* a, const T* scalar) { \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;      \
    if (i < N) {                                            \
      c[i] = a[i] op scalar[0];                             \
    }                                                       \
  }                                                         \
  template <typename T>                                     \
  __global__ void mont_##name##_scalar_mod_kernel_(         \
      const int64_t N, T* self, const T* scalar) {          \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;      \
    if (i < N) {                                            \
      self[i] op## = scalar[0];                             \
    }                                                       \
  }

#define BIN_OP_TEMPLATE(name)                                                  \
  static void name##_template(Tensor& c, const Tensor& a, const Tensor& b) {   \
    TORCH_CHECK(                                                               \
        a.numel() == b.numel() && a.numel() == c.numel(),                      \
        "The number of elements must be the same!");                           \
    AT_DISPATCH_MONT_TYPES(a.scalar_type(), "mont_##name##_mod_cuda", [&] {    \
      auto a_ptr =                                                             \
          reinterpret_cast<scalar_t::compute_type*>(a.data_ptr<scalar_t>());   \
      auto b_ptr =                                                             \
          reinterpret_cast<scalar_t::compute_type*>(b.data_ptr<scalar_t>());   \
      auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(                  \
          c.mutable_data_ptr<scalar_t>());                                     \
      int64_t N = a.numel() / num_uint64(a.scalar_type());                     \
      int64_t grid = (N + block_work_size() - 1) / block_work_size();          \
      auto stream = at::cuda::getCurrentCUDAStream();                          \
      mont_##name##_mod_kernel<<<grid, block_work_size(), 0, stream>>>(        \
          N, c_ptr, a_ptr, b_ptr);                                             \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    });                                                                        \
  }                                                                            \
  static void name##_template_(Tensor& self, const Tensor& other) {            \
    TORCH_CHECK(                                                               \
        self.numel() == other.numel(),                                         \
        "The number of elements must be the same!");                           \
    AT_DISPATCH_MONT_TYPES(self.scalar_type(), "mont_##name##_mod_cuda", [&] { \
      auto other_ptr = reinterpret_cast<scalar_t::compute_type*>(              \
          other.data_ptr<scalar_t>());                                         \
      auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(               \
          self.mutable_data_ptr<scalar_t>());                                  \
      int64_t N = self.numel() / num_uint64(self.scalar_type());               \
      int64_t grid = (N + block_work_size() - 1) / block_work_size();          \
      auto stream = at::cuda::getCurrentCUDAStream();                          \
      mont_##name##_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(       \
          N, self_ptr, other_ptr);                                             \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    });                                                                        \
  }

#define SCALAR_OP_TEMPLATE(name)                                           \
  static void name##_scalar_template(                                      \
      Tensor& c, const Tensor& a, const Tensor& b) {                       \
    TORCH_CHECK(                                                           \
        b.numel() == num_uint64(b.scalar_type()),                          \
        "The second tensor must be a scalar!");                            \
    AT_DISPATCH_MONT_TYPES(                                                \
        a.scalar_type(), "mont_##name##_scalar_mod_cuda", [&] {            \
          auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(          \
              a.data_ptr<scalar_t>());                                     \
          auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(          \
              b.data_ptr<scalar_t>());                                     \
          auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(          \
              c.mutable_data_ptr<scalar_t>());                             \
          int64_t N = a.numel() / num_uint64(a.scalar_type());             \
          int64_t grid = (N + block_work_size() - 1) / block_work_size();  \
          auto stream = at::cuda::getCurrentCUDAStream();                  \
          mont_##name##_scalar_mod_kernel<<<                               \
              grid,                                                        \
              block_work_size(),                                           \
              0,                                                           \
              stream>>>(N, c_ptr, a_ptr, b_ptr);                           \
          C10_CUDA_KERNEL_LAUNCH_CHECK();                                  \
        });                                                                \
  }                                                                        \
  static void name##_scalar_template_(Tensor& self, const Tensor& other) { \
    TORCH_CHECK(                                                           \
        other.numel() == num_uint64(other.scalar_type()),                  \
        "The second tensor must be a scalar!");                            \
    AT_DISPATCH_MONT_TYPES(                                                \
        self.scalar_type(), "mont_##name##_scalar_mod_cuda", [&] {         \
          auto other_ptr = reinterpret_cast<scalar_t::compute_type*>(      \
              other.data_ptr<scalar_t>());                                 \
          auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(       \
              self.mutable_data_ptr<scalar_t>());                          \
          int64_t N = self.numel() / num_uint64(self.scalar_type());       \
          int64_t grid = (N + block_work_size() - 1) / block_work_size();  \
          auto stream = at::cuda::getCurrentCUDAStream();                  \
          mont_##name##_scalar_mod_kernel_<<<                              \
              grid,                                                        \
              block_work_size(),                                           \
              0,                                                           \
              stream>>>(N, self_ptr, other_ptr);                           \
          C10_CUDA_KERNEL_LAUNCH_CHECK();                                  \
        });                                                                \
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
  Tensor& name##_mod_out_cuda(const Tensor& a, const Tensor& b, Tensor& c) { \
    name##_template(c, a, b);                                                \
    return c;                                                                \
  }

#define SCALAR_OP(name)                                                \
  Tensor name##_mod_scalar_cuda(const Tensor& a, const Tensor& b) {    \
    Tensor c = at::empty_like(a);                                      \
    name##_scalar_template(c, a, b);                                   \
    return c;                                                          \
  }                                                                    \
  Tensor& name##_mod_scalar_cuda_(Tensor& self, const Tensor& other) { \
    name##_scalar_template_(self, other);                              \
    return self;                                                       \
  }                                                                    \
  Tensor& name##_mod_scalar_out_cuda(                                  \
      const Tensor& a, const Tensor& b, Tensor& c) {                   \
    name##_scalar_template(c, a, b);                                   \
    return c;                                                          \
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

template <typename T>
__global__ void inv_mod_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = data[i].reciprocal();
  }
}

template <typename T>
__global__ void exp_mod_kernel_(const int64_t N, T* data, int exp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = data[i] ^ exp;
  }
}

template <typename T>
__global__ void one_kernel_(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = T::one();
  }
}

template <typename T>
__global__ void poly_eval_kernel(const int64_t N, const T* x, T* data) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    data[tid] = T::one();
  } else if (tid == 1) {
    data[tid] = *x;
  } else if (tid < N) {
    data[tid] = (*x) ^ tid;
  }
}

template <typename T>
__global__ void poly_reduce_kernel_first(
    const int64_t N,
    const T* x,
    const T* coff,
    T* temp) {
  int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  T sum;
  sum.zero();
  for (size_t i = tid; i < N; i += BLOCK_SIZE * gridDim.x) {
    sum += x[i] * coff[i];
  }
  __shared__ T shared_sum[BLOCK_SIZE];
  shared_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    temp[blockIdx.x] = shared_sum[0];
  }
}

template <typename T>
__global__ void poly_reduce_kernel_second(
    const int64_t N,
    const T* temp,
    T* y) {
  int64_t tid = threadIdx.x;
  __shared__ T shared_sum[BLOCK_SIZE];
  if (tid < N) {
    shared_sum[threadIdx.x] = temp[tid];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[0] = shared_sum[0];
  }
}

BIN_KERNEL(add, +);
BIN_KERNEL(sub, -);
BIN_KERNEL(mul, *);
BIN_KERNEL(div, /);
SCALAR_KERNEL(add, +);
SCALAR_KERNEL(sub, -);
SCALAR_KERNEL(mul, *);
SCALAR_KERNEL(div, /);

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
    to_mont_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
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
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

static void inv_mod_cuda_template(Tensor& self) {
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "inv_mod_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    inv_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void exp_mod_cuda_template(Tensor& self, int exp) {
  if (exp == 1) {
    return;
  }
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "exp_mod_cuda", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t N = self.numel() / num_uint64(self.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (exp == 0) {
      one_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr);
    } else {
      exp_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(N, self_ptr, exp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void poly_eval_cuda_template(const Tensor& x, Tensor& poly) {
  AT_DISPATCH_MONT_TYPES(poly.scalar_type(), "poly_eval_cuda", [&] {
    auto poly_ptr = reinterpret_cast<scalar_t::compute_type*>(
        poly.mutable_data_ptr<scalar_t>());
    auto x_ptr = reinterpret_cast<scalar_t::compute_type*>(
        x.mutable_data_ptr<scalar_t>());
    int64_t N = poly.numel() / num_uint64(poly.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    poly_eval_kernel<<<grid, block_work_size(), 0, stream>>>(
        N, x_ptr, poly_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void poly_reduce_cuda_template(
    const Tensor& x,
    const Tensor& coff,
    Tensor& y) {
  AT_DISPATCH_MONT_TYPES(x.scalar_type(), "poly_reduce_cuda", [&] {
    auto x_ptr = reinterpret_cast<scalar_t::compute_type*>(
        x.mutable_data_ptr<scalar_t>());
    auto coff_ptr = reinterpret_cast<scalar_t::compute_type*>(
        coff.mutable_data_ptr<scalar_t>());
    auto y_ptr = reinterpret_cast<scalar_t::compute_type*>(
        y.mutable_data_ptr<scalar_t>());
    int64_t N = x.numel() / num_uint64(x.scalar_type());
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (N > (BLOCK_SIZE * MAX_NUM_BLOCKS)) {
      grid = MAX_NUM_BLOCKS;
    }
    auto temp = at::empty(
        {grid, num_uint64(x.scalar_type())},
        x.scalar_type(),
        x.layout(),
        x.device(),
        c10::nullopt,
        c10::nullopt);
    auto temp_ptr = reinterpret_cast<scalar_t::compute_type*>(
        temp.mutable_data_ptr<scalar_t>());
    auto stream = at::cuda::getCurrentCUDAStream();
    poly_reduce_kernel_first<<<grid, BLOCK_SIZE, 0, stream>>>(
        N, x_ptr, coff_ptr, temp_ptr);
    poly_reduce_kernel_second<<<1, grid, 0, stream>>>(grid, temp_ptr, y_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

BIN_OP_TEMPLATE(add);
BIN_OP_TEMPLATE(sub);
BIN_OP_TEMPLATE(mul);
BIN_OP_TEMPLATE(div);
SCALAR_OP_TEMPLATE(add);
SCALAR_OP_TEMPLATE(sub);
SCALAR_OP_TEMPLATE(mul);
SCALAR_OP_TEMPLATE(div);

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

Tensor inv_mod_cuda(const Tensor& input) {
  Tensor output = input.clone();
  inv_mod_cuda_template(output);
  return output;
}
Tensor& inv_mod_cuda_(Tensor& self) {
  inv_mod_cuda_template(self);
  return self;
}
Tensor& inv_mod_out_cuda(const Tensor& input, Tensor& output) {
  copy(output, input);
  inv_mod_cuda_template(output);
  return output;
}

Tensor exp_mod_cuda(const Tensor& input, int64_t exp) {
  Tensor output = input.clone();
  exp_mod_cuda_template(output, exp);
  return output;
}
Tensor& exp_mod_cuda_(Tensor& self, int64_t exp) {
  exp_mod_cuda_template(self, exp);
  return self;
}
Tensor& exp_mod_out_cuda(const Tensor& input, int64_t exp, Tensor& output) {
  copy(output, input);
  exp_mod_cuda_template(output, exp);
  return output;
}

Tensor poly_eval_cuda(const Tensor& x, int64_t N) {
  auto poly = at::empty(
      {N, x.numel()},
      x.scalar_type(),
      x.layout(),
      x.device(),
      c10::nullopt,
      c10::nullopt);
  poly_eval_cuda_template(x, poly);
  return poly;
}

Tensor poly_reduce_cuda(const Tensor& x, const Tensor& coff) {
  auto y = at::empty(
      {num_uint64(x.scalar_type())},
      x.scalar_type(),
      x.layout(),
      x.device(),
      c10::nullopt,
      c10::nullopt);
  poly_reduce_cuda_template(x, coff, y);
  return y;
}

BIN_OP(add);
BIN_OP(sub);
BIN_OP(mul);
BIN_OP(div);
SCALAR_OP(add);
SCALAR_OP(sub);
SCALAR_OP(mul);
SCALAR_OP(div);

} // namespace native
} // namespace at
