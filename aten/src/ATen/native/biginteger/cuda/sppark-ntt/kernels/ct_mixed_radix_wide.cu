#include "algorithm.cuh"
#include "kernels.cuh"

namespace at { 
namespace native {

template <int intermediate_mul>
__launch_bounds__(768, 1) __global__
void _CT_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             BLS12_381_Fr_G1* d_inout, const BLS12_381_Fr_G1 (*d_partial_twiddles)[WINDOW_SIZE],
             const BLS12_381_Fr_G1* d_radix6_twiddles, const BLS12_381_Fr_G1* d_radixX_twiddles,
             const BLS12_381_Fr_G1* d_intermediate_twiddles,
             const unsigned int intermediate_twiddle_shift,
             const bool is_intt, const uint32_t* d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
#endif

    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t inp_ntt_size = (index_t)1 << stage;
    const index_t out_ntt_size = (index_t)1 << (stage + iterations - 1);
#if 1
    const index_t thread_ntt_pos = (tid & (out_ntt_size - 1)) >> (iterations - 1);
#else
    const index_t thread_ntt_pos = (tid >> (iterations - 1)) & (inp_ntt_size - 1);
#endif

    // rearrange |tid|'s bits
    index_t idx0 = tid & ~(out_ntt_size - 1);
    idx0 += (tid << stage) & (out_ntt_size - 1);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + inp_ntt_size;

    BLS12_381_Fr_G1 r0 = d_inout[idx0];
    BLS12_381_Fr_G1 r1 = d_inout[idx1];

    if (intermediate_mul == 1) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits) * thread_ntt_pos;
        index_t root_idx1 = thread_ntt_pos << (nbits - 1);

        BLS12_381_Fr_G1 first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
        second_root *= first_root;

        r0 *= first_root;
        r1 *= second_root;
    } else if (intermediate_mul == 2) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = intermediate_twiddle_shift + iterations;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx1 = bit_rev(thread_ntt_idx + 1, nbits);

        BLS12_381_Fr_G1 t0 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx0];
        BLS12_381_Fr_G1 t1 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx1];

        r0 *= t0;
        r1 *= t1;
    }

    {
        BLS12_381_Fr_G1 t = r1;
        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (int s = 1; s < min(iterations, 6); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

#ifdef __CUDA_ARCH__
        BLS12_381_Fr_G1 x = BLS12_381_Fr_G1::csel(r1, r0, pos);
        shfl_bfly(x, laneMask);
        r0 = BLS12_381_Fr_G1::csel(x, r0, !pos);
        r1 = BLS12_381_Fr_G1::csel(x, r1, pos);
#endif
        BLS12_381_Fr_G1 t = d_radix6_twiddles[rank << (6 - (s + 1))];
        t *= r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        BLS12_381_Fr_G1 t = d_radixX_twiddles[rank << (radix - (s + 1))];

        // shfl_bfly through the shared memory
        extern __shared__ BLS12_381_Fr_G1 shared_exchange[];

#ifdef __CUDA_ARCH__
        BLS12_381_Fr_G1 x = BLS12_381_Fr_G1::csel(r1, r0, pos);
        __syncthreads();
        shared_exchange[threadIdx.x] = x;
        __syncthreads();
        x = shared_exchange[threadIdx.x ^ laneMask];
        r0 = BLS12_381_Fr_G1::csel(x, r0, !pos);
        r1 = BLS12_381_Fr_G1::csel(x, r1, pos);
#endif
        t *= r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }
    

    if (is_intt && (stage + iterations) == lg_domain_size) {
        r0 *= *(reinterpret_cast<const BLS12_381_Fr_G1*>(d_domain_size_inverse));
        r1 *= *(reinterpret_cast<const BLS12_381_Fr_G1*>(d_domain_size_inverse));
    }

    // rotate "iterations" bits in indices
    index_t mask = ((index_t)1 << (stage + iterations)) - ((index_t)1 << stage);
    index_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    d_inout[idx0] = r0;
    d_inout[idx1] = r1;
}

#define NTT_ARGUMENTS \
        unsigned int, unsigned int, unsigned int, unsigned int, BLS12_381_Fr_G1*, \
        const BLS12_381_Fr_G1 (*)[WINDOW_SIZE], const BLS12_381_Fr_G1*, const BLS12_381_Fr_G1*, const BLS12_381_Fr_G1*, \
        unsigned int, bool, const uint32_t*

template __global__ void _CT_NTT<0>(NTT_ARGUMENTS);
template __global__ void _CT_NTT<1>(NTT_ARGUMENTS);
template __global__ void _CT_NTT<2>(NTT_ARGUMENTS);

#undef NTT_ARGUMENTS

void CTkernel(int iterations, BLS12_381_Fr_G1* d_inout, int lg_domain_size, bool is_intt,
        const NTTParameters& ntt_parameters, const cudaStream_t& stream, int* stage)
{
    //assert(iterations <= 10);
    TORCH_CHECK(iterations <= 10, "NTT iterations check!");
    const int radix = iterations < 6 ? 6 : iterations;


    index_t num_threads = (index_t)1 << (lg_domain_size - 1);
    index_t block_size = 1 << (radix - 1);
    index_t num_blocks;

    block_size = (num_threads <= block_size) ? num_threads : block_size;
    num_blocks = (num_threads + block_size - 1) / block_size;

    //assert(num_blocks == (unsigned int)num_blocks);
    TORCH_CHECK(num_blocks == (unsigned int)num_blocks, "NTT blocks check!");
    
    BLS12_381_Fr_G1* d_radixX_twiddles = nullptr;
    BLS12_381_Fr_G1* d_intermediate_twiddles = nullptr;
    
    unsigned int intermediate_twiddle_shift = 0;

    #define NTT_CONFIGURATION \
            num_blocks, block_size, sizeof(BLS12_381_Fr_G1) * block_size, stream

    #define NTT_ARGUMENTS radix, lg_domain_size, *stage, iterations, \
            d_inout, ntt_parameters.partial_twiddles, \
            ntt_parameters.radix6_twiddles, d_radixX_twiddles, \
            d_intermediate_twiddles, intermediate_twiddle_shift, \
            is_intt, ntt_parameters.Domain_size_inverse+lg_domain_size*8

    switch (radix) {
    case 6:
        switch (*stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 6:
            intermediate_twiddle_shift = std::max(12 - lg_domain_size, 0);
            d_intermediate_twiddles = ntt_parameters.radix6_twiddles_6;
            _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 12:
            intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
            d_intermediate_twiddles = ntt_parameters.radix6_twiddles_12;
            _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }
        break;
    case 7:
        d_radixX_twiddles = ntt_parameters.radix7_twiddles;
        switch (*stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 7:
            intermediate_twiddle_shift = std::max(14 - lg_domain_size, 0);
            d_intermediate_twiddles = ntt_parameters.radix7_twiddles_7;
            _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }
        break;
    case 8:
        d_radixX_twiddles = ntt_parameters.radix8_twiddles;
        switch (*stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 8:
            intermediate_twiddle_shift = std::max(16 - lg_domain_size, 0);
            d_intermediate_twiddles = ntt_parameters.radix8_twiddles_8;
            _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }
        break;
    case 9:
        d_radixX_twiddles = ntt_parameters.radix9_twiddles;
        switch (*stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 9:
            intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
            d_intermediate_twiddles = ntt_parameters.radix9_twiddles_9;
            _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }
        break;
    case 10:
        d_radixX_twiddles = ntt_parameters.radix10_twiddles;
        switch (*stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }
        break;
    default:
        assert(false);
    }

    *stage += radix;
    #undef NTT_CONFIGURATION
    #undef NTT_ARGUMENTS
    
    //CUDA_OK(cudaGetLastError());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
}

void CT_NTT(BLS12_381_Fr_G1* d_inout, const int lg_domain_size, const bool is_intt,
            const NTTParameters& ntt_parameters, const cudaStream_t& stream)
{
    TORCH_CHECK(lg_domain_size <= 40, "NTT length check!");
    int stage = 0;
    if (lg_domain_size <= 10) {
        CTkernel(lg_domain_size, d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
    } else if (lg_domain_size <= 17) {
        CTkernel(lg_domain_size / 2 + lg_domain_size % 2, d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(lg_domain_size / 2, d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
    } else if (lg_domain_size <= 30) {
        int step = lg_domain_size / 3;
        int rem = lg_domain_size % 3;
        CTkernel(step, d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(step + (lg_domain_size == 29 ? 1 : 0), d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(step + (lg_domain_size == 29 ? 1 : rem), d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
    } else if (lg_domain_size <= 40) {
        int step = lg_domain_size / 4;
        int rem = lg_domain_size % 4;
        CTkernel(step, d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(step + (rem > 2), d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(step + (rem > 1), d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
        CTkernel(step + (rem > 0), d_inout, lg_domain_size, is_intt, ntt_parameters, stream, &stage);
    } 
}

}//namespace native
}//namespace at
