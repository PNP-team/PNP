#include "ATen/native/biginteger/cuda/sppark-ntt/ntt.cuh"

namespace at { 
namespace native {
    
void bit_rev(BLS12_381_Fr_G1* d_out, const BLS12_381_Fr_G1* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
{
    //assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
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
            <<<1, domain_size / 8, domain_size * sizeof(BLS12_381_Fr_G1), stream>>>
            (d_out, d_inp, lg_domain_size);
    else
        bit_rev_permutation_aux
            <<<domain_size / 1024, 1024 / 8, 1024 * sizeof(BLS12_381_Fr_G1), stream>>>
            (d_out, d_inp, lg_domain_size);

    //CUDA_OK(cudaGetLastError());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LDE_powers(BLS12_381_Fr_G1* inout, BLS12_381_Fr_G1* pggp, 
                        bool bitrev,
                        uint32_t lg_domain_size, uint32_t lg_blowup,
                        stream_t& stream, bool ext_pow)
{
    size_t domain_size = (size_t)1 << lg_domain_size;
    // const auto gen_powers =
    //     NTTParameters::all(intt)[stream]->partial_group_gen_powers;

    if (domain_size < WARP_SZ)
        LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);
    else if (domain_size < 512)
        LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);
    else
        LDE_distribute_powers<<<domain_size / 512, 512, 0, stream>>>
                                (inout, lg_blowup, bitrev, pggp, ext_pow);

    //CUDA_OK(cudaGetLastError());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void NTT_internal(BLS12_381_Fr_G1* d_inout,
                    BLS12_381_Fr_G1* partial_twiddles,
                    BLS12_381_Fr_G1* radix_twiddles,
                    BLS12_381_Fr_G1* radix_middles,
                    BLS12_381_Fr_G1* partial_group_gen_powers,
                    BLS12_381_Fr_G1* Domain_size_inverse,
                    uint32_t lg_domain_size,
                    InputOutputOrder order, Direction direction,
                    Type type, stream_t& stream,
                    bool coset_ext_pow)
{
    // Pick an NTT algorithm based on the input order and the desired output
    // order of the data. In certain cases, bit reversal can be avoided which
    // results in a considerable performance gain.

    const bool intt = direction == Direction::inverse;
    //const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
    //NTTParameters ntt_parameters(intt,stream);

    //const auto& ntt_parameters = parameters.all()[stream];
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

void Base(const gpu_t& gpu, BLS12_381_Fr_G1* inout,
          BLS12_381_Fr_G1 *partial_twiddles,
          BLS12_381_Fr_G1* radix_twiddles,
          BLS12_381_Fr_G1* radix_middles,
          BLS12_381_Fr_G1 *partial_group_gen_powers,
          BLS12_381_Fr_G1* Domain_size_inverse,
          uint32_t lg_domain_size,
          InputOutputOrder order, Direction direction, Type type, bool coset_ext_pow)
{
    TORCH_CHECK(lg_domain_size != 0, "NTT Length cannot be 0!");
    // if (lg_domain_size == 0)
    //     return RustError{cudaSuccess};

    //try {

    gpu.select();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t domain_size = (size_t)1 << lg_domain_size;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    
    NTT_internal(inout, 
                 partial_twiddles, radix_twiddles, radix_middles,
                 partial_group_gen_powers, Domain_size_inverse,
                 lg_domain_size, order, direction, type, gpu,
                 coset_ext_pow);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gpu.sync();

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);

    // float elapsed;
    // cudaEventElapsedTime(&elapsed, start, stop);

    // std::cout << "NTT_internal: " << elapsed << " ms" << std::endl;


    //} 
//  catch (const cuda_error& e) {
//         gpu.sync();
// #ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
//         return RustError{e.code(), e.what()};
// #else
//         return RustError{e.code()};
// #endif
//     }

//     return RustError{cudaSuccess};
}

void compute_ntt(size_t device_id, BLS12_381_Fr_G1* inout, 
                 BLS12_381_Fr_G1* partial_twiddles,
                 BLS12_381_Fr_G1* radix_twiddles,
                 BLS12_381_Fr_G1* radix_middles,
                 BLS12_381_Fr_G1* partial_group_gen_powers,
                 BLS12_381_Fr_G1* Domain_size_inverse,
                 uint32_t lg_domain_size,
                 InputOutputOrder ntt_order,
                 Direction ntt_direction,
                 Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    Base(gpu, inout, 
         partial_twiddles, radix_twiddles, radix_middles,
         partial_group_gen_powers, Domain_size_inverse,
         lg_domain_size,
         ntt_order, ntt_direction, ntt_type);
}

}//namespace native
}//namespace at