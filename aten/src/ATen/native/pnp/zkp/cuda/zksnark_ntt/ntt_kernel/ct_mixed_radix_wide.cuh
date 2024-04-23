#pragma once

#include "kernels.cuh"

namespace at {
namespace native {

template <int intermediate_mul, typename fr_t>
__launch_bounds__(768, 1) __global__ void _CT_NTT(
    const unsigned int radix,
    const unsigned int lg_domain_size,
    const unsigned int stage,
    const unsigned int iterations,
    fr_t* d_inout,
    const fr_t* d_partial_twiddles,
    const fr_t* d_radix6_twiddles,
    const fr_t* d_radixX_twiddles,
    const fr_t* d_intermediate_twiddles,
    const unsigned int intermediate_twiddle_shift,
    const bool is_intt,
    const fr_t* d_domain_size_inverse) {
#if (__CUDACC_VER_MAJOR__ - 0) >= 11
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

  fr_t r0 = d_inout[idx0];
  fr_t r1 = d_inout[idx1];

  if (intermediate_mul == 1) {
    unsigned int diff_mask = (1 << (iterations - 1)) - 1;
    unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
    unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;

    index_t root_idx0 = bit_rev(thread_ntt_idx, nbits) * thread_ntt_pos;
    index_t root_idx1 = thread_ntt_pos << (nbits - 1);

    fr_t first_root, second_root;
    get_intermediate_roots(
        first_root, second_root, root_idx0, root_idx1, d_partial_twiddles);
    second_root *= first_root;

    r0 *= first_root;
    r1 *= second_root;
  } else if (intermediate_mul == 2) {
    unsigned int diff_mask = (1 << (iterations - 1)) - 1;
    unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
    unsigned int nbits = intermediate_twiddle_shift + iterations;

    index_t root_idx0 = bit_rev(thread_ntt_idx, nbits);
    index_t root_idx1 = bit_rev(thread_ntt_idx + 1, nbits);

    fr_t t0 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx0];
    fr_t t1 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx1];

    r0 *= t0;
    r1 *= t1;
  }

  {
    fr_t t = r1;
    r1 = r0 - t;
    r0 = r0 + t;
  }

  for (int s = 1; s < min(iterations, 6); s++) {
    unsigned int laneMask = 1 << (s - 1);
    unsigned int thrdMask = (1 << s) - 1;
    unsigned int rank = threadIdx.x & thrdMask;
    bool pos = rank < laneMask;

#ifdef __CUDA_ARCH__
    fr_t x = fr_t::csel(r1, r0, pos);
    shfl_bfly(x, laneMask);
    r0 = fr_t::csel(x, r0, !pos);
    r1 = fr_t::csel(x, r1, pos);
#endif
    fr_t t = d_radix6_twiddles[rank << (6 - (s + 1))];
    t *= r1;

    r1 = r0 - t;
    r0 = r0 + t;
  }

  for (int s = 6; s < iterations; s++) {
    unsigned int laneMask = 1 << (s - 1);
    unsigned int thrdMask = (1 << s) - 1;
    unsigned int rank = threadIdx.x & thrdMask;
    bool pos = rank < laneMask;

    fr_t t = d_radixX_twiddles[rank << (radix - (s + 1))];

    // shfl_bfly through the shared memory
    extern __shared__ char shmem[];
    auto shared_exchange = reinterpret_cast<fr_t*>(shmem);

    // extern __shared__ fr_t shared_exchange[];

#ifdef __CUDA_ARCH__
    fr_t x = fr_t::csel(r1, r0, pos);
    __syncthreads();
    shared_exchange[threadIdx.x] = x;
    __syncthreads();
    x = shared_exchange[threadIdx.x ^ laneMask];
    r0 = fr_t::csel(x, r0, !pos);
    r1 = fr_t::csel(x, r1, pos);
#endif
    t *= r1;

    r1 = r0 - t;
    r0 = r0 + t;
  }

  if (is_intt && (stage + iterations) == lg_domain_size) {
    r0 *= *(reinterpret_cast<const fr_t*>(d_domain_size_inverse));
    r1 *= *(reinterpret_cast<const fr_t*>(d_domain_size_inverse));
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

template <typename fr_t>
void CTkernel(
    int iterations,
    fr_t* d_inout,
    fr_t* partial_twiddles,
    fr_t* radix7_twiddles,
    fr_t* radix_middles,
    fr_t* partial_group_gen_powers,
    fr_t* Domain_size_inverse,
    int lg_domain_size,
    bool is_intt,
    int* stage) {
  TORCH_CHECK(iterations <= 10, "CT_NTT iterations cannot exceed 10!");
  const int radix = iterations < 6 ? 6 : iterations;

  index_t num_threads = (index_t)1 << (lg_domain_size - 1);
  index_t block_size = 1 << (radix - 1);
  index_t num_blocks;

  block_size = (num_threads <= block_size) ? num_threads : block_size;
  num_blocks = (num_threads + block_size - 1) / block_size;

  TORCH_CHECK(num_blocks == (unsigned int)num_blocks, "NTT blocks check!");

  fr_t* d_radixX_twiddles = nullptr;
  fr_t* d_intermediate_twiddles = nullptr;

  unsigned int intermediate_twiddle_shift = 0;

#define NTT_CONFIGURATION num_blocks, block_size, sizeof(fr_t) * block_size

#define NTT_ARGUMENTS                                                   \
  radix, lg_domain_size, *stage, iterations, d_inout, partial_twiddles, \
      radix7_twiddles + 64 + 128 + 256 + 512, d_radixX_twiddles,        \
      d_intermediate_twiddles, intermediate_twiddle_shift, is_intt,     \
      Domain_size_inverse + lg_domain_size

  switch (radix) {
    case 6:
      switch (*stage) {
        case 0:
          _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        case 6:
          intermediate_twiddle_shift = std::max(12 - lg_domain_size, 0);
          d_intermediate_twiddles = radix_middles;
          _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        case 12:
          intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
          d_intermediate_twiddles = radix_middles + 64 * 64;
          _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        default:
          _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
      }
      break;
    case 7:
      d_radixX_twiddles = radix7_twiddles;
      switch (*stage) {
        case 0:
          _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        case 7:
          intermediate_twiddle_shift = std::max(14 - lg_domain_size, 0);
          d_intermediate_twiddles = radix_middles + 64 * 64 + 4096 * 64;
          _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        default:
          _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
      }
      break;
    case 8:
      d_radixX_twiddles = radix7_twiddles + 64; // radix8_twiddles
      switch (*stage) {
        case 0:
          _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        case 8:
          intermediate_twiddle_shift = std::max(16 - lg_domain_size, 0);
          d_intermediate_twiddles = radix_middles + 64 * 64 + 4096 * 64 +
              128 * 128; // radix8_twiddles_8
          _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        default:
          _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
      }
      break;
    case 9:
      d_radixX_twiddles = radix7_twiddles + 64 + 128; // radix9_twiddles
      switch (*stage) {
        case 0:
          _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        case 9:
          intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
          d_intermediate_twiddles = radix_middles + 64 * 64 + 4096 * 64 +
              128 * 128 + 256 * 256; // radix9_twiddles_9
          _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
        default:
          _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
          break;
      }
      break;
    case 10:
      d_radixX_twiddles = radix7_twiddles + 64 + 128 + 256; // radix10_twiddles
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

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace native
} // namespace at
