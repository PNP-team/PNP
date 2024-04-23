#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cassert>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "batch_addition.cuh"
#include "sort.cuh"

namespace at {
namespace native {

#ifndef WARP_SZ
#define WARP_SZ 32
#endif
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif
#ifndef SHARED_MEM_PER_BLOCK
#define SHARED_MEM_PER_BLOCK (48 * 1024)
#endif
/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

// Transposed scalar_t
template <class scalar_t>
class scalar_T {
  uint32_t val[sizeof(scalar_t) / sizeof(uint32_t)][WARP_SZ];

 public:
  __device__ const uint32_t& operator[](size_t i) const {
    return val[i][0];
  }
  __device__ scalar_T& operator()(uint32_t laneid) {
    return *reinterpret_cast<scalar_T*>(&val[0][laneid]);
  }
  __device__ scalar_T& operator=(const scalar_t& rhs) {
    for (size_t i = 0; i < sizeof(scalar_t) / sizeof(uint32_t); i++)
      val[i][0] = rhs[i];
    return *this;
  }
};

template <class scalar_t>
__device__ __forceinline__ static uint32_t get_wval(
    const scalar_T<scalar_t>& scalar,
    uint32_t off,
    uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1) {
  uint32_t i = off / 32;
  uint64_t ret = scalar[i];

  if (i < top_i)
    ret |= (uint64_t)scalar[i + 1] << 32;

  return ret >> (off % 32);
}

__device__ __forceinline__ static uint32_t booth_encode(
    uint32_t wval,
    uint32_t wmask,
    uint32_t wbits) {
  uint32_t sign = (wval >> wbits) & 1;
  wval = ((wval + 1) & wmask) >> 1;
  return sign ? 0 - wval : wval;
}

template <class scalar_t>
__launch_bounds__(1024) __global__ void breakdown(
    uint32_t* digits,
    const scalar_t scalars[],
    size_t len,
    const uint32_t digit_stride,
    uint32_t nwins,
    uint32_t wbits,
    bool mont = true) {
  assert(len <= (1U << 31) && wbits < 32);

  extern __shared__ char shmem[];
  auto xchange = reinterpret_cast<scalar_T<scalar_t>*>(shmem);
  // extern __shared__ scalar_T<scalar_t> xchange[];
  const uint32_t tid = threadIdx.x;
  const uint32_t tix = threadIdx.x + blockIdx.x * blockDim.x;

  const uint32_t top_i =
      (scalar_t::nbits + 31) / 32 - 1; // find the first 32bits index
  const uint32_t wmask = 0xffffffffU >> (31 - wbits); // (1U << (wbits+1)) - 1;

  auto& scalar = xchange[tid / WARP_SZ](tid % WARP_SZ);

#pragma unroll 1
  for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x * blockDim.x) {
    auto s = scalars[i];

    if (mont)
      s.from();

    // clear the most significant bit
    uint32_t msb = s[top_i] >> ((scalar_t::nbits - 1) % 32);
    s.cneg(msb);
    msb <<= 31;

    scalar = s;

#pragma unroll 1
    for (uint32_t bit0 = nwins * wbits - 1, win = nwins; --win;) {
      bit0 -= wbits;
      uint32_t wval = get_wval(scalar, bit0, top_i);
      wval = booth_encode(wval, wmask, wbits);
      if (wval)
        wval ^= msb;
      *(digits + win * digit_stride + i) = wval;
    }

    uint32_t wval = s[0] << 1;
    wval = booth_encode(wval, wmask, wbits);
    if (wval)
      wval ^= msb;
    *(digits + i) = wval;
  }
}

#ifndef LARGE_L1_CODE_CACHE
#if __CUDA_ARCH__ - 0 >= 800
#define LARGE_L1_CODE_CACHE 1
#define ACCUMULATE_NTHREADS 384
#else
#define LARGE_L1_CODE_CACHE 0
#define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
#endif
#endif

#ifndef MSM_NTHREADS
#define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS - 1)) != 0
#error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
#define MSM_NSTREAMS 8
#elif MSM_NSTREAMS < 2
#error "invalid MSM_NSTREAMS"
#endif

template <
    class bucket_t,
    class affine_h,
    class bucket_h = class bucket_t::mem_t,
    class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__ void accumulate(
    bucket_h buckets[],
    uint32_t nwins,
    uint32_t wbits,
    const uint32_t digit_stride,
    const uint32_t hist_stride,
    affine_h points_[],
    uint32_t* digits,
    uint32_t* histogram,
    uint32_t sid = 0) {
  const uint32_t bucket_stride = 1U << --wbits;
  const affine_h* points = points_;

  static __device__ uint32_t streams[MSM_NSTREAMS];
  uint32_t& current = streams[sid % MSM_NSTREAMS];
  uint32_t laneid;
  asm("mov.u32 %0, %laneid;" : "=r"(laneid));
  const uint32_t degree = bucket_t::degree;
  const uint32_t warp_sz = WARP_SZ / degree;
  const uint32_t lane_id = laneid / degree;

  uint32_t x, y;

  __shared__ uint32_t xchg;

  if (threadIdx.x == 0)
    xchg = atomicAdd(&current, blockDim.x / degree);
  __syncthreads();
  x = xchg + threadIdx.x / degree;

  while (x < (nwins << wbits)) {
    y = x >> wbits;
    x &= (1U << wbits) - 1;
    uint32_t* h = histogram + y * hist_stride + x;
    uint32_t idx, len = h[0];

    asm("{ .reg.pred %did;"
        "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
        "  @!%did mov.b32 %0, 0;"
        "}"
        : "=r"(idx)
        : "r"(len), "r"(degree));

    if (lane_id == 0 && x != 0)
      idx = h[-1];

    if ((len -= idx) && !(x == 0 && y == 0)) {
      const uint32_t* digs_ptr = digits + y * digit_stride + idx;
      uint32_t digit = *digs_ptr++;

      affine_t p = points[digit & 0x7fffffff];
      bucket_t bucket = p;
      bucket.cneg(digit >> 31);

      while (--len) {
        digit = *digs_ptr++;
        p = points[digit & 0x7fffffff];
        if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
          bucket.add(p, digit >> 31);
        else
          bucket.uadd(p, digit >> 31);
      }

      buckets[y * bucket_stride + x] = bucket;
    } else {
      buckets[y * bucket_stride + x].inf();
    }

    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
  }

  cooperative_groups::this_grid().sync();

  if (threadIdx.x + blockIdx.x == 0)
    current = 0;
}

template <class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__ void integrate(
    bucket_h buckets[],
    uint32_t nwins,
    uint32_t wbits,
    uint32_t nbits) {
  const uint32_t degree = bucket_t::degree;
  uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

  assert((blockDim.x & (blockDim.x - 1)) == 0 && wbits - 1 > Nthrbits);

  const uint32_t bucket_stride = 1U << (wbits - 1);
  extern __shared__ uint4 scratch_[];
  auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
  const uint32_t tid = threadIdx.x / degree;
  const uint32_t bid = blockIdx.x;

  auto* row = buckets + bid * bucket_stride;
  uint32_t i = 1U << (wbits - 1 - Nthrbits);
  row += tid * i;

  uint32_t mask = 0;
  if ((bid + 1) * wbits > nbits) {
    uint32_t lsbits = nbits - bid * wbits;
    mask = (1U << (wbits - lsbits)) - 1;
  }

  bucket_t res, acc = row[--i];

  if (i & mask) {
    if (sizeof(res) <= 128)
      res.inf();
    else
      scratch[tid].inf();
  } else {
    if (sizeof(res) <= 128)
      res = acc;
    else
      scratch[tid] = acc;
  }

  bucket_t p;

#pragma unroll 1
  while (i--) {
    p = row[i];

    uint32_t pc = i & mask ? 2 : 0;
#pragma unroll 1
    do {
      if (sizeof(bucket_t) <= 128) {
        p.add(acc);
        if (pc == 1) {
          res = p;
        } else {
          acc = p;
          if (pc == 0)
            p = res;
        }
      } else {
        if (LARGE_L1_CODE_CACHE && degree == 1)
          p.add(acc);
        else
          p.uadd(acc);
        if (pc == 1) {
          scratch[tid] = p;
        } else {
          acc = p;
          if (pc == 0)
            p = scratch[tid];
        }
      }
    } while (++pc < 2);
  }

  __syncthreads();

  buckets[bid * bucket_stride + 2 * tid] = p;
  buckets[bid * bucket_stride + 2 * tid + 1] = acc;
}
#undef asm
} // namespace native
} // namespace at

template <typename... Types>
void launch_coop(
    void (*f)(Types...),
    dim3 gridDim,
    dim3 blockDim,
    size_t shared_sz,
    Types... args) {
  if (SHARED_MEM_PER_BLOCK < shared_sz) {
    cudaFuncSetAttribute(
        f, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_sz);
  }

  if (gridDim.x == 0 || blockDim.x == 0) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, f);
    if (blockDim.x == 0)
      blockDim.x = blockSize;
    if (gridDim.x == 0)
      gridDim.x = minGridSize;
  }
  void* va_args[sizeof...(args)] = {&args...};
  cudaLaunchCooperativeKernel(
      (const void*)f, gridDim, blockDim, va_args, shared_sz);
}

#include <vector>

namespace at {
namespace native {

template <
    class bucket_t,
    class point_t,
    class affine_t,
    class scalar_t,
    typename affine_h = typename affine_t::mem_t,
    typename bucket_h = typename bucket_t::mem_t>
class msm_t {
  size_t npoints, smcount;
  uint32_t wbits, nwins;
  bucket_h* d_buckets;
  uint32_t* d_hist;

  class result_t {
    bucket_t ret[MSM_NTHREADS / bucket_t::degree][2];

   public:
    result_t() {}
    inline operator decltype(ret)&() {
      return ret;
    }
    inline const bucket_t* operator[](size_t i) const {
      return ret[i];
    }
  };

  constexpr static int lg2(size_t n) {
    int ret = 0;
    while (n >>= 1)
      ret++;
    return ret;
  }

 public:
  msm_t(
      const affine_t points[],
      bucket_h* buckets,
      size_t np,
      size_t sm,
      size_t ffi_affine_sz = sizeof(affine_t),
      int device_id = -1)
      : smcount(sm), d_buckets(buckets) {
    npoints = (np + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);
    // Ensure that npoints are multiples of WARP_SZ and are as close as possible
    // to the original np value.

    wbits = 17;
    if (npoints > 192) {
      wbits = std::min(lg2(npoints + npoints / 2) - 8, 18);
      if (wbits < 10)
        wbits = 10;
    } else if (npoints > 0) {
      wbits = 10;
    }
    nwins = (scalar_t::bit_length() - 1) / wbits + 1;

    uint32_t row_sz = 1U << (wbits - 1);

    size_t d_buckets_sz =
        (nwins * row_sz) + (smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

    d_hist = (uint32_t*)(d_buckets + d_buckets_sz);
  }

 private:
  void digits(
      const scalar_t d_scalars[],
      size_t len,
      uint32_t* d_digits,
      uint2* d_temps,
      const uint32_t temp_stride,
      const uint32_t digit_stride,
      const uint32_t hist_stride,
      bool mont) {
    // Using larger grid size doesn't make 'sort' run faster, actually
    // quite contrary. Arguably because global memory bus gets
    // thrashed... Stepping far outside the sweet spot has significant
    // impact, 30-40% degradation was observed. It's assumed that all
    // GPUs are "balanced" in an approximately the same manner. The
    // coefficient was observed to deliver optimal performance on
    // Turing and Ampere...
    uint32_t grid_size = smcount / 3;
    while (grid_size & (grid_size - 1))
      grid_size -= (grid_size & (0 - grid_size));

    breakdown<<<2 * grid_size, 1024, sizeof(scalar_t) * 1024>>>(
        d_digits, d_scalars, len, digit_stride, nwins, wbits, mont);
    C10_CUDA_CHECK(cudaGetLastError());

    const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;

    // On the other hand a pair of kernels launched in parallel run
    // ~50% slower but sort twice as much data...
    uint32_t top = scalar_t::bit_length() - wbits * (nwins - 1);
    uint32_t win;
    for (win = 0; win < nwins - 1; win += 2) {
      launch_coop(
          sort,
          dim3{grid_size, 2},
          dim3{SORT_BLOCKDIM},
          shared_sz,
          d_digits,
          len,
          win,
          temp_stride,
          digit_stride,
          hist_stride,
          d_temps,
          d_hist,
          wbits - 1,
          wbits - 1,
          win == nwins - 2 ? top - 1 : wbits - 1);
    }
    if (win < nwins) {
      launch_coop(
          sort,
          dim3{grid_size, 1},
          dim3{SORT_BLOCKDIM},
          shared_sz,
          d_digits,
          len,
          win,
          temp_stride,
          digit_stride,
          hist_stride,
          d_temps,
          d_hist,
          wbits - 1,
          top - 1,
          0u);
    }
  }

 public:
  void invoke(
      bucket_t* out,
      affine_t* _points,
      size_t npoints,
      scalar_t* scalars,
      uint8_t* temp,
      bool mont = true,
      size_t ffi_affine_sz = sizeof(affine_t)) {
    assert(this->npoints == 0 || npoints <= this->npoints);

    uint32_t lg_npoints = lg2(npoints + npoints / 2);
    size_t batch = 1;
    uint32_t stride = npoints;
    // Round up to the nearest multiple greater than or equal to WARP_SZ
    stride = (stride + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);
    point_t p;

    // |scalars| being nullptr means the scalars are pre-loaded to
    // |d_scalars|, otherwise allocate stride.
    size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
    temp_sz = stride * std::max(2 * sizeof(uint2), temp_sz);

    // |points| being nullptr means the points are pre-loaded to
    // |d_points|, otherwise allocate double-stride.
    affine_h* points = reinterpret_cast<affine_h*>(_points);

    uint2* d_temps = (uint2*)temp;
    uint32_t* d_digits = (uint32_t*)(temp + temp_sz);

    size_t d_off = 0; // device offset
    size_t h_off = 0; // host offset
    size_t num = stride > npoints ? npoints : stride;
    const uint32_t hist_stride = 1U << (wbits - 1);
    const uint32_t temp_stride = stride;
    const uint32_t digit_stride = stride;

    // it seems that we do not need to copy `scalars` to `d_temps`
    digits(
        scalars,
        num,
        d_digits,
        d_temps,
        temp_stride,
        digit_stride,
        hist_stride,
        mont);

    batch_addition<bucket_t><<<smcount, BATCH_ADD_BLOCK_SIZE, 0>>>(
        d_buckets + (nwins << (wbits - 1)),
        points + d_off,
        num,
        &d_digits[0],
        d_hist[0]);
    C10_CUDA_CHECK(cudaGetLastError());

    launch_coop(
        accumulate<bucket_t, affine_h>,
        dim3{smcount},
        dim3{0},
        (size_t)0,
        d_buckets,
        nwins,
        wbits,
        digit_stride,
        hist_stride,
        points + d_off,
        d_digits,
        d_hist,
        (uint32_t)0);
    C10_CUDA_CHECK(cudaGetLastError());
    integrate<bucket_t>
        <<<nwins,
           MSM_NTHREADS,
           sizeof(bucket_t) * MSM_NTHREADS / bucket_t::degree>>>(
            d_buckets, nwins, wbits, scalar_t::bit_length());
    C10_CUDA_CHECK(cudaGetLastError());

    C10_CUDA_CHECK(cudaMemcpy(
        out + nwins * MSM_NTHREADS / bucket_t::degree * 2,
        d_buckets + (nwins << (wbits - 1)),
        sizeof(bucket_t) * smcount * BATCH_ADD_BLOCK_SIZE / WARP_SZ,
        cudaMemcpyDeviceToDevice));

    C10_CUDA_CHECK(cudaMemcpy(
        out, d_buckets, sizeof(result_t) * nwins, cudaMemcpyDeviceToDevice));
  }
};

template <
    class point_t,
    class bucket_t,
    class affine_t,
    class scalar_t,
    class bucket_h = class bucket_t::mem_t>
static void mult_pippenger(
    bucket_t* out,
    affine_t points[],
    size_t npoints,
    scalar_t scalars[],
    bucket_h* bucket,
    uint8_t* temp,
    int64_t sm,
    bool mont = true,
    size_t ffi_affine_sz = sizeof(affine_t)) {
  msm_t<bucket_t, point_t, affine_t, scalar_t> msm{
      nullptr, bucket, npoints, sm};
  msm.invoke(out, points, npoints, scalars, temp, mont, ffi_affine_sz);
}
} // namespace native
} // namespace at
