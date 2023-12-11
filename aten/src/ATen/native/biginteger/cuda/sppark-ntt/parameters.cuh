#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/sppark-util/gpu_t.cuh"
#include "parameters/bls12_381.h"
#include "gen_twiddles.cuh"



// Maximum domain size supported. Can be adjusted at will, but with the
// target field in mind. Most fields handle up to 2^32 elements, BLS12-377
// can handle up to 2^47, alt_bn128 - 2^28...
namespace at { 
namespace native {



__device__ __constant__ BLS12_381_Fr_G1 forward_radix6_twiddles[32];
__device__ __constant__ BLS12_381_Fr_G1 inverse_radix6_twiddles[32];


class NTTParameters {
private:
    stream_t& gpu;
    bool inverse;

public:
    BLS12_381_Fr_G1* partial_twiddles;

    BLS12_381_Fr_G1* radix6_twiddles, * radix7_twiddles, * radix8_twiddles,
        * radix9_twiddles, * radix10_twiddles;

    BLS12_381_Fr_G1* radix6_twiddles_6, * radix6_twiddles_12, * radix7_twiddles_7,
        * radix8_twiddles_8, * radix9_twiddles_9;

    BLS12_381_Fr_G1* partial_group_gen_powers; // for LDE

    BLS12_381_Fr_G1* Domain_size_inverse;
    
private:
    BLS12_381_Fr_G1* twiddles_X(int num_blocks, int block_size, const BLS12_381_Fr_G1* root)
    {
        BLS12_381_Fr_G1* ret = (BLS12_381_Fr_G1*)gpu.Dmalloc(num_blocks * block_size * sizeof(BLS12_381_Fr_G1));
        generate_radixX_twiddles_X<<<16, block_size, 0, gpu>>>(ret, num_blocks, root);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return ret;
    }

public:
    NTTParameters(const bool _inverse, int id)
        : gpu(select_gpu(id)), inverse(_inverse)
    {
        const BLS12_381_Fr_G1* roots = NTTParameters::inverse ? (const BLS12_381_Fr_G1*)inverse_roots_of_unity
                                    : (const BLS12_381_Fr_G1*)forward_roots_of_unity;

        const size_t blob_sz = 64 + 128 + 256 + 512 + 32;
        
        cudaGetSymbolAddress((void**)&radix6_twiddles,
                                     inverse ? inverse_radix6_twiddles
                                             : forward_radix6_twiddles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        radix7_twiddles = (BLS12_381_Fr_G1*)gpu.Dmalloc(blob_sz * sizeof(BLS12_381_Fr_G1));

        radix8_twiddles = radix7_twiddles + 64;
        radix9_twiddles = radix8_twiddles + 128;
        radix10_twiddles = radix9_twiddles + 256;

        generate_all_twiddles<<<blob_sz/32, 32, 0, gpu>>>(radix7_twiddles,
                                                          roots+6,
                                                          roots+7,
                                                          roots+8,
                                                          roots+9,
                                                          roots+10);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        cudaMemcpyAsync(radix6_twiddles, radix10_twiddles + 512,
                            32 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
                            gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        radix6_twiddles_6 = twiddles_X(64, 64, roots+12);
        radix6_twiddles_12 = twiddles_X(4096, 64, roots+18);
        radix7_twiddles_7 = twiddles_X(128, 128, roots+14);
        radix8_twiddles_8 = twiddles_X(256, 256, roots+16);
        radix9_twiddles_9 = twiddles_X(512, 512, roots+18);

        const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;

        partial_twiddles = reinterpret_cast<decltype(partial_twiddles)>
                           (gpu.Dmalloc(2 * partial_sz * sizeof(BLS12_381_Fr_G1)));
        partial_group_gen_powers = &partial_twiddles[WINDOW_NUM];

        const size_t inverse_sz = S + 1;
        //Domain_size_inverse = (uint32_t*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
        Domain_size_inverse = (BLS12_381_Fr_G1*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Domain_size_inverse, domain_size_inverse, 
                                inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                                gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (partial_twiddles, roots + MAX_LG_DOMAIN_SIZE);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        BLS12_381_Fr_G1* Group_gen_inverse = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Group_gen_inverse, group_gen_inverse, 
                            sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                            gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        BLS12_381_Fr_G1* Group_gen = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Group_gen, group_gen, 
                            sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                            gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (partial_group_gen_powers, inverse ? Group_gen_inverse
                                               : Group_gen);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }



    NTTParameters(const bool _inverse, int id, BLS12_381_Fr_G1* data_ptr)
        : gpu(select_gpu(id)), inverse(_inverse)
    {
        const size_t blob_sz = 64 + 128 + 256 + 512 + 32;
        
        const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;

        const size_t inverse_sz = S + 1;
        
        BLS12_381_Fr_G1* Inverse_roots_of_unity = (BLS12_381_Fr_G1*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Inverse_roots_of_unity, inverse_roots_of_unity, 
                        inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                        gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        BLS12_381_Fr_G1* Forward_roots_of_unity = (BLS12_381_Fr_G1*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Forward_roots_of_unity, forward_roots_of_unity, 
                        inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                        gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const BLS12_381_Fr_G1* roots = NTTParameters::inverse ? (const BLS12_381_Fr_G1*)(Inverse_roots_of_unity)
                                    : (const BLS12_381_Fr_G1*)Forward_roots_of_unity;
        
        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (data_ptr, roots + MAX_LG_DOMAIN_SIZE);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        BLS12_381_Fr_G1* Group_gen_inverse = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Group_gen_inverse, group_gen_inverse, 
                            sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                            gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        BLS12_381_Fr_G1* Group_gen = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
        
        cudaMemcpyAsync(Group_gen, group_gen, 
                            sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                            gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (data_ptr + partial_sz, inverse ? Group_gen_inverse
                                               : Group_gen);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // radix7_twiddles = data_ptr + 2 * partial_sz;
        // radix8_twiddles = data_ptr + 2 * partial_sz + 64;
        // radix9_twiddles = data_ptr + 2 * partial_sz + 128;
        // radix10_twiddles = data_ptr + 2 * partial_sz + 256;

        generate_all_twiddles<<<blob_sz/32, 32, 0, gpu>>>(data_ptr + 2 * partial_sz,
                                                          roots+6,
                                                          roots+7,
                                                          roots+8,
                                                          roots+9,
                                                          roots+10);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        
        generate_radixX_twiddles_X<<<16, 64, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz, 64, roots+12);
        generate_radixX_twiddles_X<<<16, 64, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64, 4096, roots+18);
        generate_radixX_twiddles_X<<<16, 128, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64, 128, roots+14);
        generate_radixX_twiddles_X<<<16, 256, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128, 256, roots+16);
        generate_radixX_twiddles_X<<<16, 512, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128 + 256*256, 512, roots+18);

        // cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz, radix6_twiddles_6,
        //                         64*64 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
        //                         gpu);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64, radix6_twiddles_12,
        //                         4096*64 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
        //                         gpu);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64, radix7_twiddles_7,
        //                         128*128 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
        //                         gpu);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128, radix8_twiddles_8,
        //                         256*256 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
        //                         gpu);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128 + 256*256, radix6_twiddles_6,
        //                         512*512 * sizeof(BLS12_381_Fr_G1), cudaMemcpyDeviceToDevice,
        //                         gpu);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        
        cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128 + 256*256 + 512*512, domain_size_inverse, 
                                inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                                gpu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    //NTTParameters(const NTTParameters&) = delete;

    // ~NTTParameters()
    // {
    //     gpu.Dfree(partial_twiddles);

    //     gpu.Dfree(radix9_twiddles_9);
    //     gpu.Dfree(radix8_twiddles_8);
    //     gpu.Dfree(radix7_twiddles_7);
    //     gpu.Dfree(radix6_twiddles_12);
    //     gpu.Dfree(radix6_twiddles_6);

    //     gpu.Dfree(radix7_twiddles);
    // }

    inline void sync() const    { gpu.sync(); }

private:
    class all_params { friend class NTTParameters;
        std::vector<const NTTParameters*> forward;
        std::vector<const NTTParameters*> inverse;

        all_params()
        {
            int current_id;
            cudaGetDevice(&current_id);

            size_t nids = ngpus();
            for (size_t id = 0; id < nids; id++)
                forward.push_back(new NTTParameters(false, id));
            for (size_t id = 0; id < nids; id++)
                inverse.push_back(new NTTParameters(true, id));
            for (size_t id = 0; id < nids; id++)
                inverse[id]->sync();

            cudaSetDevice(current_id);
        }
        ~all_params()
        {
            for (auto* ptr: forward) delete ptr;
            for (auto* ptr: inverse) delete ptr;
        }
    };

public:
    static const auto& all(bool inverse = false)
    {
        static all_params params;
        return inverse ? params.inverse : params.forward;
    }
};

}
}
