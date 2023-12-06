#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <endian.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

struct ProjectPoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
};
struct AffinePoint {
    uint64_t x[4];
    uint64_t y[4];
};

void NTT(uint64_t* vector, bool forward, uint32_t N_times_Limbs);

void iNTT(uint64_t* vector, bool forward, uint32_t N_times_Limbs);

void MSM(uint64_t coeffs[][4], struct AffinePoint* base, struct ProjectPoint acc,struct ProjectPoint* result, uint64_t len);

