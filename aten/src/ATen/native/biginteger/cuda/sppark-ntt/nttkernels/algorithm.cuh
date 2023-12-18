#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/CurveDef.cuh"
#include "ATen/native/biginteger/cuda/sppark-ntt/ntt_config.h"
#include "ct_mixed_radix_wide.cuh"
#include "gs_mixed_radix_wide.cuh"
#include "kernels.cuh"
namespace at { 
namespace native {

template <typename fr_t>
void CT_NTT(fr_t* d_inout, const int lg_domain_size, const bool is_intt,
            const cudaStream_t& stream,
            fr_t* partial_twiddles,
            fr_t* radix_twiddles,
            fr_t* radix_middles,
            fr_t* partial_group_gen_powers,
            fr_t* Domain_size_inverse)
{
    TORCH_CHECK(lg_domain_size <= 40, "CT_NTT length cannot exceed 40");
    int stage = 0;
    if (lg_domain_size <= 10) {
        CTkernel(lg_domain_size, d_inout, 
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 17) {
        CTkernel(lg_domain_size / 2 + lg_domain_size % 2, d_inout, 
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(lg_domain_size / 2, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 30) {
        int step = lg_domain_size / 3;
        int rem = lg_domain_size % 3;
        CTkernel(step, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(step + (lg_domain_size == 29 ? 1 : 0), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(step + (lg_domain_size == 29 ? 1 : rem), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 40) {
        int step = lg_domain_size / 4;
        int rem = lg_domain_size % 4;
        CTkernel(step, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(step + (rem > 2), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(step + (rem > 1), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        CTkernel(step + (rem > 0), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } 
}

template <typename fr_t>
void GS_NTT(fr_t* d_inout, const int lg_domain_size, const bool is_intt,
    const cudaStream_t& stream,
    fr_t* partial_twiddles,
    fr_t* radix_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse)
{
    TORCH_CHECK(lg_domain_size <= 40, "GS_NTT length cannot exceed 40!");
    int stage = lg_domain_size;

    if (lg_domain_size <= 10) {
        GSkernel(lg_domain_size, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 12) {
        GSkernel(lg_domain_size - 6, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(6, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 18) {
        GSkernel(lg_domain_size / 2 + lg_domain_size % 2, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(lg_domain_size / 2, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 30) {
        int step = lg_domain_size / 3;
        int rem = lg_domain_size % 3;
        GSkernel(step + (rem > 0), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(step + (rem > 1), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(step, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } else if (lg_domain_size <= 40) {
        int step = lg_domain_size / 4;
        int rem = lg_domain_size % 4;
        GSkernel(step + (rem > 0), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(step + (rem > 1), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(step + (rem > 2), d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
        GSkernel(step, d_inout,
                 partial_twiddles,
                 radix_twiddles, radix_middles,
                 partial_group_gen_powers,
                 Domain_size_inverse,
                 lg_domain_size, is_intt, stream, &stage);
    } 
}

}//namespace native
}//namespace at