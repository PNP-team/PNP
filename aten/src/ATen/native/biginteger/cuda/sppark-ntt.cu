#include <stddef.h>
#include <stdint.h>

#include "sppark-ntt/nttkernels/ntt.cuh"
#include "sppark-ntt/parameters/parameters.cuh"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>

#include <math.h>

//temporarily set device_id to 0, set InputOutputOrder to NN
//TODO: optimize memory copy for inout
namespace at {
namespace native {
    

static void params_zkp_template(Tensor& self, int gpu_id, bool is_intt){
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "load_ntt_params_cuda", [&] {     
        auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
        NTTParameters(is_intt, gpu_id, self_ptr);
    });
}

Tensor params_zkp_cuda(int64_t domain_size, int64_t gpu_id, bool is_intt, 
                        c10::optional<ScalarType> dtype,
                        c10::optional<Layout> layout,
                        c10::optional<Device> device,
                        c10::optional<bool> pin_memory) {

    auto partial_sz = WINDOW_NUM * WINDOW_SIZE;
    auto S1 = 2 * partial_sz;
    auto S2 = 32+64+128+256+512;
    auto S3 = 64*64 + 4096*64 + 128*128 + 256*256 + 512*512;
    auto S4 = domain_size + 1;

    auto params = at::empty({S1 + S2 + S3 + S4, num_uint64(*dtype)}, dtype, layout, device, pin_memory, c10::nullopt); 
    params_zkp_template(params, gpu_id, is_intt);
    return params;
}


static void ntt_zkp(Tensor& inout, const Tensor& params) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE;
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512);
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512);

        auto pt_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L1;
        auto rp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L2;
        auto rpm_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L3;
        auto size_inverse_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L4;
        auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(inout.mutable_data_ptr<scalar_t>());
        
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
            Direction::forward,
            Type::standard
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_zkp(Tensor& inout, const Tensor& params) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "intt_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE;
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512);
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512);

        auto pt_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L1;
        auto rp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L2;
        auto rpm_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L3;
        auto size_inverse_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L4;
        auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(inout.mutable_data_ptr<scalar_t>());
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
            Direction::inverse,
            Type::standard
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void ntt_coset_zkp(Tensor& inout, const Tensor& params) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_coset_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE;
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512);
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512);

        auto pt_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L1;
        auto rp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L2;
        auto rpm_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L3;
        auto size_inverse_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L4;
        auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(inout.mutable_data_ptr<scalar_t>());
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
            Direction::forward,
            Type::coset
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_coset_zkp(Tensor& inout, const Tensor& params) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "intt_coset_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE;
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512);
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512);

        auto pt_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L1;
        auto rp_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L2;
        auto rpm_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L3;
        auto size_inverse_ptr = reinterpret_cast<scalar_t::compute_type*>(params.mutable_data_ptr<scalar_t>()) + L4;
        auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(inout.mutable_data_ptr<scalar_t>());
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
            Direction::inverse,
            Type::coset
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

Tensor ntt_zkp_cuda(const Tensor& in, const Tensor& params) {
    Tensor input = in.clone();
    ntt_zkp(input, params);
    return input;
}

Tensor& ntt_zkp_cuda_(Tensor& in, const Tensor& params) {
    ntt_zkp(in, params);
    return in;
}

Tensor& ntt_zkp_out_cuda(const Tensor& inout, const Tensor& params, Tensor& output) {
    copy(output, inout);   
    ntt_zkp(output, params);
    return output;
}

Tensor intt_zkp_cuda(const Tensor& in, const Tensor& params) {
    Tensor input = in.clone();
    intt_zkp(input, params);
    return input;
}

Tensor& intt_zkp_cuda_(Tensor& in, const Tensor& params) {                      
    intt_zkp(in, params);
    return in;
}

Tensor& intt_zkp_out_cuda(const Tensor& in, const Tensor& params, Tensor& output) {
    copy(output, in);
    intt_zkp(output, params);
    return output;
}

Tensor ntt_coset_zkp_cuda(const Tensor& in, const Tensor& params) {
    Tensor input = in.clone();
    ntt_coset_zkp(input, params);
    return input;
}

Tensor& ntt_coset_zkp_cuda_(Tensor& in, const Tensor& params) {
    ntt_coset_zkp(in, params);
    return in;
}

Tensor& ntt_coset_zkp_out_cuda(const Tensor& in, const Tensor& params, Tensor& output) {
    copy(output, in);
    ntt_coset_zkp(output, params);
    return output;
}

Tensor intt_coset_zkp_cuda(const Tensor& in, const Tensor& params) {
    Tensor input = in.clone();
    intt_coset_zkp(input, params);
    return input;
}

Tensor& intt_coset_zkp_cuda_(Tensor& in, const Tensor& params) {   
    intt_coset_zkp(in, params);
    return in;
}

Tensor& intt_coset_zkp_out_cuda(const Tensor& in, const Tensor& params, Tensor& output) {
    copy(output, in);
    intt_coset_zkp(output,params);
    return output;
}

}//namespace native
}//namespace at