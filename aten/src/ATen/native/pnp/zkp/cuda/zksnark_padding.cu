#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/native/pnp/mont/cuda/curve_def.cuh>

namespace at {
namespace native {

template <typename T>
__global__ void padding(T* input, T* output, int64_t N, int64_t pad_len) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        output[tid] = input[tid];
    }
    if(tid >= N - pad_len)
    {
        output[tid].zero();
    }
}

static void padding_template(const Tensor& input, Tensor& output, int64_t N, int64_t pad_len) {
  AT_DISPATCH_MONT_TYPES(input.scalar_type(), "padding_cuda", [&] {
    auto input_ptr = reinterpret_cast<scalar_t::compute_type*>(
        input.mutable_data_ptr<scalar_t>());
    auto output_ptr = reinterpret_cast<scalar_t::compute_type*>(
        output.mutable_data_ptr<scalar_t>());    
    TORCH_INTERNAL_ASSERT((N + pad_len) > 0 && (N + pad_len) <= std::numeric_limits<int32_t>::max());
    int64_t grid = ((N + pad_len) + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    padding<<<grid, block_work_size(), 0, stream>>>(input_ptr, output_ptr, N, pad_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

Tensor padding_cuda(const Tensor& input, int64_t pad_len){  
    int64_t N = input.numel()/num_uint64(input.scalar_type());
    Tensor output = at::empty(
      {N + pad_len, num_uint64(input.scalar_type())},
      input.options());          
    padding_template(input, output, N, pad_len);
    return output;
}

} // namespace native
} // namespace at