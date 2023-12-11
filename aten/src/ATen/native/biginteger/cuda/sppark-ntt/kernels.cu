#include "ATen/native/biginteger/cuda/sppark-ntt/kernels.cuh"
#include "kernels/kernels.cuh"
namespace at { 
namespace native {
// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
__launch_bounds__(1024) __global__
void bit_rev_permutation(BLS12_381_Fr_G1* d_out, const BLS12_381_Fr_G1 *d_in, uint32_t lg_domain_size)
{
    index_t i = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    index_t r = bit_rev(i, lg_domain_size);

    if (i < r || (d_out != d_in && i == r)) {
        BLS12_381_Fr_G1 t0 = d_in[i];
        BLS12_381_Fr_G1 t1 = d_in[r];
        d_out[r] = t0;
        d_out[i] = t1;
    }
}

__launch_bounds__(1024) __global__
void bit_rev_permutation_aux(BLS12_381_Fr_G1* out, const BLS12_381_Fr_G1* in, uint32_t lg_domain_size)
{
    extern __shared__ BLS12_381_Fr_G1 exchange[];
    BLS12_381_Fr_G1 (*xchg)[8][8] = reinterpret_cast<decltype(xchg)>(exchange);

    index_t step = (index_t)1 << (lg_domain_size - 3);
    index_t group_idx = (threadIdx.x + blockDim.x * (index_t)blockIdx.x) >> 3;
    uint32_t brev_limit = lg_domain_size - 6;
    index_t brev_mask = ((index_t)1 << brev_limit) - 1;
    index_t group_idx_brev =
        (group_idx & ~brev_mask) | bit_rev(group_idx & brev_mask, brev_limit);
    uint32_t group_thread = threadIdx.x & 0x7;
    uint32_t group_thread_rev = bit_rev(group_thread, 3);
    uint32_t group_in_block_idx = threadIdx.x >> 3;

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        xchg[group_in_block_idx][i][group_thread_rev] =
            in[group_idx * 8 + i * step + group_thread];
    }

    __syncwarp();

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        out[group_idx_brev * 8 + i * step + group_thread] =
            xchg[group_in_block_idx][group_thread_rev][i];
    }
}

__device__ __forceinline__
BLS12_381_Fr_G1 get_intermediate_root(index_t pow, const BLS12_381_Fr_G1 *roots,
                           unsigned int nbits)
{
    unsigned int off = 0;

    BLS12_381_Fr_G1 root = roots[off * WINDOW_SIZE + pow % WINDOW_SIZE];
    #pragma unroll 1
    while (pow >>= LG_WINDOW_SIZE){
        off += 1;
        root *= roots[off * WINDOW_SIZE + pow % WINDOW_SIZE];
    }

    return root;
}

__launch_bounds__(1024) __global__
void LDE_distribute_powers(BLS12_381_Fr_G1* d_inout, uint32_t lg_blowup, bool bitrev,
                           const BLS12_381_Fr_G1 *gen_powers,
                           bool ext_pow)
{
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    index_t pow = idx;
    BLS12_381_Fr_G1 r = d_inout[idx];

    if (bitrev) {
        size_t domain_size = gridDim.x * (size_t)blockDim.x;
        assert((domain_size & (domain_size-1)) == 0);
        uint32_t lg_domain_size = 63 - __clzll(domain_size);

        pow = bit_rev(idx, lg_domain_size);
    }

    if (ext_pow)
        pow <<= lg_blowup;
        
    r = r * get_intermediate_root(pow << lg_blowup, gen_powers);

    d_inout[idx] = r;
}

__launch_bounds__(1024) __global__
void LDE_spread_distribute_powers(BLS12_381_Fr_G1* out, BLS12_381_Fr_G1* in,
                                  const BLS12_381_Fr_G1* gen_powers,
                                  uint32_t lg_domain_size, uint32_t lg_blowup)
{
    extern __shared__ BLS12_381_Fr_G1 exchange[]; // block size

    assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE);

    size_t domain_size = (size_t)1 << lg_domain_size;
    uint32_t blowup = 1u << lg_blowup;
    uint32_t stride = gridDim.x * blockDim.x;

    assert(&out[domain_size * (blowup - 1)] == &in[0] &&
           (stride & (stride-1)) == 0);

    index_t idx0 = blockDim.x * blockIdx.x;
    uint32_t thread_pos = threadIdx.x & (blowup - 1);

#if 0
    index_t iters = domain_size / stride;
#else
    index_t iters = domain_size >> (31 - __clz(stride));
#endif
    index_t iterx = (blowup - 1) * (iters >> lg_blowup);

    for (index_t iter = 0; iter < iters; iter++) {
        index_t idx = idx0 + threadIdx.x;

        BLS12_381_Fr_G1 r = in[idx];

        // TODO: winterfell does not shift by lg_blowup - need to resolve
        // discrepency with Polygon
#ifdef HERMEZ
        index_t pow = bit_rev(idx, lg_domain_size + lg_blowup);
#else
        index_t pow = bit_rev(idx, lg_domain_size);
#endif

        r = r * get_intermediate_root(pow, gen_powers);

        __syncthreads();

        exchange[threadIdx.x] = r;

        if (iter >= iterx)
            cooperative_groups::this_grid().sync();
        else
            __syncthreads();

        r.zero();
        for (uint32_t i = 0; i < blowup; i++) {
            uint32_t offset = i * blockDim.x + threadIdx.x;

            if (thread_pos == 0)
                r = exchange[offset >> lg_blowup];

            out[(idx0 << lg_blowup) + offset] = r;
        }

        idx0 += stride;
    }
}

}//namespace native
}//namespace at
