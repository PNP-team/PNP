#include <iostream>
#include <vector>
#include <variant>

#include "primes.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

static __device__ __constant__ __align__(16) const uint32_t MOD[2] = {1, 2};
static __device__ __constant__ const uint32_t M0 = 0xefffffff;


__global__ void fhe_mul_mod_kernel(uint32_t *c, const uint32_t *a, const uint32_t *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  fhe_mul_mod(c + i*2, a + i*2, b + i*2, MOD, M0);
}

void test() {
  std::vector<uint32_t> a = {0x30644e72, 0xe131a029};
  std::vector<uint32_t> b = {0x30644e72, 0xe131a029};
  std::vector<uint32_t> c = {0x30644e72, 0xe131a029};

  uint32_t *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, a.size() * sizeof(uint32_t));
  cudaMalloc(&d_b, b.size() * sizeof(uint32_t));
  cudaMalloc(&d_c, c.size() * sizeof(uint32_t));

  cudaMemcpy(d_a, a.data(), a.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), b.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

  fhe_mul_mod_kernel<<<1, 1>>>(d_c, d_a, d_b);

  cudaMemcpy(c.data(), d_c, c.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  std::cout << "c = {" << std::hex << c[0] << ", " << c[1] << "}" << std::endl;
}

} // namespace native
} // namespace at
