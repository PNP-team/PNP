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
#include <iostream>
namespace at {
namespace native {

template <typename T>
__global__ void compute_query_table_poly(T* q_lookup, T* w_l, T* w_r, T* w_o, T* w_4,
                                         T* t, T* f_0, T* f_1, T* f_2, T* f_3, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        if(q_lookup[tid].is_zero()){
                f_0[tid] = *t;
                f_1[tid].zero();
                f_2[tid].zero();
                f_3[tid].zero();
            }
        else{
            f_0[tid] = w_l[tid];
            f_1[tid] = w_r[tid];
            f_2[tid] = w_o[tid];
            f_3[tid] = w_4[tid];
        }
    }
}


template <typename T>
__global__ void compress_poly_cuda(T* f, T* f_0, T* challenge, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        f[tid] = f[tid] * challenge[0] + f_0[tid];
    }
}

static void compute_query_table_poly_template(const Tensor& padded_q_lookup, const Tensor& w_l_scalar,
                                    const Tensor& w_r_scalar, const Tensor& w_o_scalar, const Tensor& w_4_scalar, 
                                    const Tensor& t_poly, Tensor& f) {
  AT_DISPATCH_MONT_TYPES(padded_q_lookup.scalar_type(), "compute_query_table_cuda", [&] {
    int64_t N = padded_q_lookup.numel() / num_uint64(padded_q_lookup.scalar_type());
    auto q_lookup_ptr = reinterpret_cast<scalar_t::compute_type*>(
        padded_q_lookup.mutable_data_ptr<scalar_t>());
    auto w_l_ptr = reinterpret_cast<scalar_t::compute_type*>(
        w_l_scalar.mutable_data_ptr<scalar_t>());
    auto w_r_ptr = reinterpret_cast<scalar_t::compute_type*>(
        w_r_scalar.mutable_data_ptr<scalar_t>());
    auto w_o_ptr = reinterpret_cast<scalar_t::compute_type*>(
        w_o_scalar.mutable_data_ptr<scalar_t>());
    auto w_4_ptr = reinterpret_cast<scalar_t::compute_type*>(
        w_4_scalar.mutable_data_ptr<scalar_t>());
    auto t_ptr = reinterpret_cast<scalar_t::compute_type*>(
        t_poly.mutable_data_ptr<scalar_t>());
    auto f_0_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f.mutable_data_ptr<scalar_t>());
    auto f_1_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f.mutable_data_ptr<scalar_t>())+ N;
    auto f_2_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f.mutable_data_ptr<scalar_t>())+ 2*N;
    auto f_3_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f.mutable_data_ptr<scalar_t>())+ 3*N;

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_query_table_poly<<<grid, block_work_size(), 0, stream>>>(q_lookup_ptr, w_l_ptr, w_r_ptr, w_o_ptr, w_4_ptr,
                                                                    t_ptr, f_0_ptr, f_1_ptr, f_2_ptr, f_3_ptr, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

static void compress_poly_template(Tensor& f,
                                   const Tensor& f_0, const Tensor& f_1, const Tensor& f_2,
                                   const Tensor& challenge) {
  AT_DISPATCH_MONT_TYPES(f.scalar_type(), "compress_poly_cuda", [&] {
    int64_t N = f.numel() / num_uint64(f.scalar_type());
    auto f_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f.mutable_data_ptr<scalar_t>());
    auto f_0_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f_0.mutable_data_ptr<scalar_t>());
    auto f_1_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f_1.mutable_data_ptr<scalar_t>());
    auto f_2_ptr = reinterpret_cast<scalar_t::compute_type*>(
        f_2.mutable_data_ptr<scalar_t>());
    auto challenge_ptr = reinterpret_cast<scalar_t::compute_type*>(
        challenge.mutable_data_ptr<scalar_t>());

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f_ptr, f_2_ptr, challenge_ptr, N);
    compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f_ptr, f_1_ptr, challenge_ptr, N);
    compress_poly_cuda<<<grid, block_work_size(), 0, stream>>>(f_ptr, f_0_ptr, challenge_ptr, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

Tensor compute_query_table_cuda(const Tensor& padded_q_lookup, 
                                const Tensor& w_l_scalar, const Tensor& w_r_scalar, const Tensor& w_o_scalar, const Tensor& w_4_scalar,
                                const Tensor& t_poly){
   
    int64_t N = padded_q_lookup.numel() * 4/num_uint64(padded_q_lookup.scalar_type());
    Tensor output = at::empty(
      {N, num_uint64(padded_q_lookup.scalar_type())},
      padded_q_lookup.options());   

    compute_query_table_poly_template(padded_q_lookup, w_l_scalar,
                                      w_r_scalar, w_o_scalar, w_4_scalar, 
                                      t_poly, output);
    
    return output;
}

Tensor compress_cuda(const Tensor& f_0, const Tensor& f_1, const Tensor& f_2, const Tensor& f_3,
                     const Tensor& challenge){

    Tensor output = f_3.clone();
    compress_poly_template(output, f_0, f_1, f_2, challenge);
    return output;
}
} // namespace native
} // namespace at