#include <vector>
#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <cstdio>
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"
#include "thread_pool_t.hpp"

namespace at {
namespace native {

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 256
#endif

#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0
# error "bad MSM_NTHREADS value"
#endif

#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8
#elif MSM_NSTREAMS<2
# error "invalid MSM_NSTREAMS"
#endif

constexpr static int lg2(size_t n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

template<class bucket_t>
bucket_t sum_up(const bucket_t inp[], size_t n)
{
    bucket_t sum = inp[0];
    for (size_t i = 1; i < n; i++)
        sum.add(inp[i]);
    return sum;
}

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = class affine_t::mem_t,
         class bucket_h = class bucket_t::mem_t>
class collect_t {
public:
    size_t npoints;
    uint32_t wbits, nwins;

    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

public:
    collect_t(size_t np){
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ); 
        //Ensure that npoints are multiples of WARP_SZ and are as close as possible to the original np value.
        
        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;
    }
public:
    void collect(point_t* out, bucket_t* res, const bucket_t* ones, uint32_t lenofone)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);
        bucket_t sum_res = sum_up(res, nwins * MSM_NTHREADS/1 * 2);
        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = scalar_t::bit_length() - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        thread_pool_t pool{"SPPARK_GPU_T_AFFINITY"};
        auto ncpus = pool.size(); 
        auto n_workers = (uint32_t)ncpus;
        if(n_workers > total)
            {n_workers = total;}
        while (n_workers--) {
            pool.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res + y * MSM_NTHREADS/bucket_t::degree * 2, item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones, lenofone);
        out->inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out->add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out->dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }

        out->add(one);
    }

public:
    point_t integrate_row(bucket_t* row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);

        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_NTHREADS/bucket_t::degree - 1;
        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = *(row + i * 2 + 1);

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(*(row + i * 2 + 1));
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res; 
        }

        point_t  res = *(row + i * 2); 
        bucket_t acc = *(row + i * 2 + 1);

        while (i--) {
            point_t raise = acc; 
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(*(row + i * 2));
            if (i)
            {
                acc.add(*(row + i * 2 + 1));
            }
                
        }

        return res;
    }
};

}}//namespace at::native
