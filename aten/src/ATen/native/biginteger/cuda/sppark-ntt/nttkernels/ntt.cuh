#pragma once

#include <cassert>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/sppark-util/gpu_t.cuh"
#include "ATen/native/biginteger/cuda/CurveDef.cuh"
#include "kernels.cuh"
#include "algorithm.cuh"
#include <cuda.h>

namespace at { 
namespace native {

enum class InputOutputOrder { NN, NR, RN, RR };
enum class Direction { forward, inverse };
enum class Type { standard, coset };
enum class Algorithm { GS, CT };

template <typename fr_t>
void bit_rev(fr_t* d_out, const fr_t* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
{
    TORCH_CHECK(lg_domain_size <= MAX_LG_DOMAIN_SIZE, "NTT length cannot exceed MAX_LG_DOMAIN_SIZE!");
    size_t domain_size = (size_t)1 << lg_domain_size;

    if (domain_size <= WARP_SZ)
        bit_rev_permutation
            <<<1, domain_size, 0, stream>>>
            (d_out, d_inp, lg_domain_size);
    else if (d_out == d_inp)
        bit_rev_permutation
            <<<domain_size/WARP_SZ, WARP_SZ, 0, stream>>>
            (d_out, d_inp, lg_domain_size);
    else if (domain_size < 1024)
        bit_rev_permutation_aux
            <<<1, domain_size / 8, domain_size * sizeof(fr_t), stream>>>
            (d_out, d_inp, lg_domain_size);
    else
        bit_rev_permutation_aux
            <<<domain_size / 1024, 1024 / 8, 1024 * sizeof(fr_t), stream>>>
            (d_out, d_inp, lg_domain_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename fr_t>
void LDE_powers(fr_t* inout, fr_t* pggp, 
                        bool bitrev,
                        uint32_t lg_domain_size, uint32_t lg_blowup,
                        stream_t& stream, bool ext_pow = false)
{
    size_t domain_size = (size_t)1 << lg_domain_size;

    if (domain_size < WARP_SZ)
        LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);
    else if (domain_size < 512)
        LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);
    else
        LDE_distribute_powers<<<domain_size / 512, 512, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename fr_t>
void NTT_internal(fr_t* d_inout,
                    fr_t* partial_twiddles,
                    fr_t* radix_twiddles,
                    fr_t* radix_middles,
                    fr_t* partial_group_gen_powers,
                    fr_t* Domain_size_inverse,
                    uint32_t lg_domain_size,
                    InputOutputOrder order, Direction direction,
                    Type type, stream_t& stream,
                    bool coset_ext_pow = false)
{
    // Pick an NTT algorithm based on the input order and the desired output
    // order of the data. In certain cases, bit reversal can be avoided which
    // results in a considerable performance gain.

    const bool intt = direction == Direction::inverse;
    bool bitrev;
    Algorithm algorithm;

    switch (order) {
        case InputOutputOrder::NN:
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
            bitrev = true;
            algorithm = Algorithm::CT;
            break;
        case InputOutputOrder::NR:
            bitrev = false;
            algorithm = Algorithm::GS;
            break;
        case InputOutputOrder::RN:
            bitrev = true;
            algorithm = Algorithm::CT;
            break;
        case InputOutputOrder::RR:
            bitrev = true;
            algorithm = Algorithm::GS;
            break;
        default:
            assert(false);
    }

    if (!intt && type == Type::coset)
        LDE_powers(d_inout, partial_group_gen_powers, bitrev, lg_domain_size, 0, stream,
                    coset_ext_pow);

    switch (algorithm) {
        case Algorithm::GS:
            GS_NTT(d_inout, lg_domain_size, intt, stream,
                   partial_twiddles, 
                   radix_twiddles, radix_middles, 
                   partial_group_gen_powers, Domain_size_inverse);
            break;
        case Algorithm::CT:
            CT_NTT(d_inout, lg_domain_size, intt, stream,
                   partial_twiddles, 
                   radix_twiddles, radix_middles, 
                   partial_group_gen_powers, Domain_size_inverse);
            break;
    }

    if (intt && type == Type::coset)
        LDE_powers(d_inout, partial_group_gen_powers, !bitrev, lg_domain_size, 0, stream,
                    coset_ext_pow);

    if (order == InputOutputOrder::RR)
        bit_rev(d_inout, d_inout, lg_domain_size, stream);
}

template <typename fr_t>
void compute_ntt(size_t device_id,
                 fr_t* in,
                 fr_t* partial_twiddles,
                 fr_t* radix_twiddles,
                 fr_t* radix_middles,
                 fr_t* partial_group_gen_powers,
                 fr_t* Domain_size_inverse,
                 uint32_t lg_domain_size,
                 InputOutputOrder ntt_order,
                 Direction ntt_direction,
                 Type ntt_type)
{
    TORCH_CHECK(lg_domain_size != 0, "NTT Length cannot be 0!");

    auto& gpu = select_gpu(device_id);
    gpu.select();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t domain_size = (size_t)1 << lg_domain_size;

    NTT_internal(in, 
                 partial_twiddles, radix_twiddles, radix_middles,
                 partial_group_gen_powers, Domain_size_inverse,
                 lg_domain_size, ntt_order, ntt_direction, ntt_type, gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gpu.sync();
}

}//namespace native
}//namespace at