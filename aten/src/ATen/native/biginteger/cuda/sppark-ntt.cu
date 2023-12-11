#include <stddef.h>
#include <stdint.h>

#include <ATen/native/biginteger/cuda/sppark-ntt/ntt.cuh>
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
namespace at {
namespace native {


static void params_zkp_template(Tensor& self, int gpu_id, bool is_intt){
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "load_ntt_params_cuda", [&] {     
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(self.mutable_data_ptr<scalar_t>());
        NTTParameters ntt_parameters(is_intt, gpu_id, self_ptr);
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
    auto params = at::empty({S1 + S2 + S3 + S4, 4}, kBLS12_381_Fr_G1_Mont, layout, device, pin_memory, c10::nullopt); 
    params_zkp_template(params, gpu_id, is_intt);
    return params;
}


static void ntt_zkp(Tensor& self, const Tensor& params) {
    auto len = self.numel() / num_uint64(self.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "ntt_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE * num_uint64(params.scalar_type());
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512) * num_uint64(params.scalar_type());
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512) * num_uint64(params.scalar_type());

        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L1);
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L2);
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L3);
        auto size_inverse_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L4);
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(self.mutable_data_ptr<scalar_t>());
        
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

static void intt_zkp(Tensor& self, const Tensor& params) {
    auto len = self.numel() / num_uint64(self.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "intt_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE;
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512);
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512);

        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>()) + L1;
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>()) + L2;
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>()) + L3;
        auto size_inverse_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>()) + L4;
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(self.mutable_data_ptr<scalar_t>());
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

static void ntt_coset_zkp(Tensor& self, const Tensor& params) {
    auto len = self.numel() / num_uint64(self.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "ntt_coset_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE * num_uint64(params.scalar_type());
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512) * num_uint64(params.scalar_type());
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512) * num_uint64(params.scalar_type());

        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L1);
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L2);
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L3);
        auto size_inverse_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L4);
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(self.mutable_data_ptr<scalar_t>());
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

static void intt_coset_zkp(Tensor& self, const Tensor& params) {
    auto len = self.numel() / num_uint64(self.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "intt_coset_cuda", [&] {
        auto L1 = WINDOW_NUM * WINDOW_SIZE * num_uint64(params.scalar_type());
        auto L2 = 2 * L1;
        auto L3 = L2 + (32+64+128+256+512) * num_uint64(params.scalar_type());
        auto L4 = L3 + (64*64 + 4096*64 + 128*128 + 256*256 + 512*512) * num_uint64(params.scalar_type());

        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>());
        auto pggp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L1);
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L2);
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L3);
        auto size_inverse_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(params.mutable_data_ptr<scalar_t>() + L4);
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(self.mutable_data_ptr<scalar_t>());
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

Tensor ntt_zkp_cuda(const Tensor& inout, const Tensor& params) {
    Tensor output = inout.clone();
    ntt_zkp(output, params);
    return output;
}

Tensor& ntt_zkp_cuda_(Tensor& inout, const Tensor& params) {
    ntt_zkp(inout, params);
    return inout;
}

Tensor& ntt_zkp_out_cuda(const Tensor& inout, const Tensor& params, Tensor& output) {
    copy(output, inout);   
    ntt_zkp(output, params);
    return output;
}

Tensor intt_zkp_cuda(const Tensor& inout, const Tensor& params) {
    Tensor output = inout.clone();
    intt_zkp(output, params);
    return output;
}

Tensor& intt_zkp_cuda_(Tensor& inout, const Tensor& params) {                          
    intt_zkp(inout, params);
    return inout;
}

Tensor& intt_zkp_out_cuda(const Tensor& inout, const Tensor& params, Tensor& output) {
    copy(output, inout);
    intt_zkp(output, params);
    return output;
}

Tensor ntt_coset_zkp_cuda(const Tensor& inout, const Tensor& params) {
    Tensor output = inout.clone();
    ntt_coset_zkp(output, params);
    return output;
}

Tensor& ntt_coset_zkp_cuda_(Tensor& inout, const Tensor& params) {
    ntt_coset_zkp(inout, params);
    return inout;
}

Tensor& ntt_coset_zkp_out_cuda(const Tensor& inout, const Tensor& params, Tensor& output) {
    copy(output, inout);
    ntt_coset_zkp(output, params);
    return output;
}

Tensor intt_coset_zkp_cuda(const Tensor& inout, const Tensor& params) {
    Tensor output = inout.clone();
    intt_coset_zkp(output, params);
    return output;
}

Tensor& intt_coset_zkp_cuda_(Tensor& inout, const Tensor& params) {   
    intt_coset_zkp(inout, params);
    return inout;
}

Tensor& intt_coset_zkp_out_cuda(const Tensor& inout, const Tensor& params, Tensor& output) {
    copy(output, inout);
    intt_coset_zkp(output, params);
    return output;
}

}//namespace native
}//namespace at