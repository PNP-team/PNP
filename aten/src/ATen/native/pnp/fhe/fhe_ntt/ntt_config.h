#pragma once

namespace at {
namespace native {

#ifndef WARP_SZ
#define WARP_SZ 32
#endif

#define MAX_LG_DOMAIN_SIZE 28 // tested only up to 2^31 for now
typedef unsigned int
    index_t; // for MAX_LG_DOMAIN_SIZE <= 32, otherwise use size_t

#if MAX_LG_DOMAIN_SIZE <= 28
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2)
#else
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 2) / 3)
#endif
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

} // namespace native
} // namespace at