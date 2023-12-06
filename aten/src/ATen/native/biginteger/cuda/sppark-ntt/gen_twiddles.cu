#include "gen_twiddles.cuh"

namespace at { 
namespace native {
__global__ 
void generate_partial_twiddles( BLS12_381_Fr_G1(*roots)[WINDOW_SIZE],
                               const BLS12_381_Fr_G1 root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);
    BLS12_381_Fr_G1 root;

    if (tid == 0)
        root = ONE;
    else if (tid == 1)
        root = root_of_unity;
    else
        root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
            #if defined(__CUDA_ARCH__)
                root.sqr();
            #else
                root *= root;
            #endif
            roots[off][tid] = root;
        }
}

__global__
void generate_all_twiddles(BLS12_381_Fr_G1* d_radixX_twiddles, const BLS12_381_Fr_G1 root6,
                                                    const BLS12_381_Fr_G1 root7,
                                                    const BLS12_381_Fr_G1 root8,
                                                    const BLS12_381_Fr_G1 root9,
                                                    const BLS12_381_Fr_G1 root10)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int pow;
    BLS12_381_Fr_G1 root_of_unity;

    if (tid < 64) {
        pow = tid;
        root_of_unity = root7;
    } 
    else if (tid < 64 + 128) {
        pow = tid - 64;
        root_of_unity = root8;
    } 
    else if (tid < 64 + 128 + 256) {
        pow = tid - 64 - 128;
        root_of_unity = root9;
    } 
    else if (tid < 64 + 128 + 256 + 512) {
        pow = tid - 64 - 128 - 256;
        root_of_unity = root10;
    } 
    else if (tid < 64 + 128 + 256 + 512 + 32) {
        pow = tid - 64 - 128 - 256 - 512;
        root_of_unity = root6;
    } 
    else {
        assert(false);
    }

    if (pow == 0)
        d_radixX_twiddles[tid] = ONE;
    else if (pow == 1)
        d_radixX_twiddles[tid] = root_of_unity;
    else
        d_radixX_twiddles[tid] = root_of_unity^pow;
}

__launch_bounds__(512) __global__
void generate_radixX_twiddles_X(BLS12_381_Fr_G1* d_radixX_twiddles_X, int n,
                                const BLS12_381_Fr_G1 root_of_unity)
{
    if (gridDim.x == 1) {
        BLS12_381_Fr_G1 root0;

        d_radixX_twiddles_X[threadIdx.x] = ONE;
        d_radixX_twiddles_X += blockDim.x;

        if (threadIdx.x == 0)
            root0 = ONE;
        else if (threadIdx.x == 1)
            root0 = root_of_unity;
        else
            root0 = root_of_unity^threadIdx.x;

        d_radixX_twiddles_X[threadIdx.x] = root0;
        d_radixX_twiddles_X += blockDim.x;

        BLS12_381_Fr_G1 root1 = root0;

        for (int i = 2; i < n; i++) {
            root1 *= root0;
            d_radixX_twiddles_X[threadIdx.x] = root1;
            d_radixX_twiddles_X += blockDim.x;
        }
    } else {
        BLS12_381_Fr_G1 root0;

        if (threadIdx.x == 0)
            root0 = ONE;
        else
            root0 = root_of_unity ^ (threadIdx.x * gridDim.x);

        unsigned int pow = blockIdx.x * threadIdx.x;
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        BLS12_381_Fr_G1 root1;

        if (pow == 0)
            root1 = ONE;
        else if (pow == 1)
            root1 = root_of_unity;
        else
            root1 = root_of_unity^pow;

        d_radixX_twiddles_X[tid] = root1;
        d_radixX_twiddles_X += gridDim.x * blockDim.x;

        for (int i = gridDim.x; i < n; i += gridDim.x) {
            root1 *= root0;
            d_radixX_twiddles_X[tid] = root1;
            d_radixX_twiddles_X += gridDim.x * blockDim.x;
        }
    }
}

}//namespace native
}//namespace at