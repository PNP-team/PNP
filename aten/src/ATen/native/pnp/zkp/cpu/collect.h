
#if __cplusplus < 201103L && !(defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
#error C++11 or later is required.
#endif

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#ifdef _GNU_SOURCE
#include <sched.h>
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"

namespace at {
namespace native {

#ifndef WARP_SZ
#define WARP_SZ 32
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

class semaphore_t {
 private:
  size_t counter;
  std::mutex mtx;
  std::condition_variable cvar;

 public:
  semaphore_t() : counter(0) {}

  void notify() {
    std::unique_lock<std::mutex> lock(mtx);
    counter++;
    cvar.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx);
    cvar.wait(lock, [&] { return counter != 0; });
    counter--;
  }
};

class thread_pool_t {
 private:
  std::vector<std::thread> threads;

  std::mutex mtx; // Inter-thread synchronization
  std::condition_variable cvar;
  std::atomic<bool> done;

  typedef std::function<void()> job_t;
  std::deque<job_t> fifo;

  void init(unsigned int num_threads) {
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
#ifdef _GNU_SOURCE
      cpu_set_t set;
      if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        size_t i, n;
        for (n = 0, i = num_threads; i--;)
          n += CPU_ISSET(i, &set);
        if (n != 0)
          num_threads = n;
      }
#endif
    }

    threads.reserve(num_threads);

    for (unsigned int i = 0; i < num_threads; i++)
      threads.push_back(std::thread([this]() {
        while (execute())
          ;
      }));
  }

 public:
  thread_pool_t(unsigned int num_threads = 0) : done(false) {
    init(num_threads);
  }

  thread_pool_t(const char* affinity_env) : done(false) {
#ifdef _GNU_SOURCE
    while ((affinity_env = getenv(affinity_env))) {
      if (affinity_env[0] != '0')
        break;
      char base = affinity_env[1];
      if (base == 'x')
        base = 16;
      else if (base == 'b')
        base = 2;
      else if (base >= '0' && base < '8')
        base = 8;
      else
        break;

      size_t len = strlen(affinity_env += 1 + (base != 8));
      if (len == 0)
        break;

      cpu_set_t oset, nset;
      CPU_ZERO(&oset);
      CPU_ZERO(&nset);

      if (sched_getaffinity(0, sizeof(oset), &oset) != 0)
        break;

      unsigned int num_threads = 0;
      for (int cpu = 0; cpu < CPU_SETSIZE && len--;) {
        char nibble = nibble_from_hex(affinity_env[len]);
        for (char mask = 1; mask < base; mask <<= 1, cpu++) {
          if (nibble & mask) {
            CPU_SET(cpu, &nset);
            num_threads++;
          }
        }
      }

      if (sched_setaffinity(0, sizeof(nset), &nset) != 0)
        break;
      init(num_threads);
      sched_setaffinity(0, sizeof(oset), &oset);

      return;
    }
#endif
    init(0);
  }

  virtual ~thread_pool_t() {
    done = true;
    cvar.notify_all();
    for (auto& tid : threads)
      tid.join();
  }

  size_t size() const {
    return threads.size();
  }

  template <class Workable>
  void spawn(Workable work) {
    std::unique_lock<std::mutex> lock(mtx);
    fifo.emplace_back(work);
    cvar.notify_one(); // wake up a worker thread
  }

 private:
  bool execute() {
    job_t work;
    {
      std::unique_lock<std::mutex> lock(mtx);

      while (!done && fifo.empty())
        cvar.wait(lock);

      if (done && fifo.empty())
        return false;

      work = fifo.front();
      fifo.pop_front();
    }
    work();

    return true;
  }

 public:
  // call work(size_t idx) with idx=[0..num_items) in parallel, e.g.
  // pool.par_map(20, [&](size_t i) { std::cout << i << std::endl; });
  template <class Workable>
  void par_map(
      size_t num_items,
      size_t stride,
      Workable work,
      size_t max_workers = 0) {
    size_t num_steps = (num_items + stride - 1) / stride;
    size_t num_workers = std::min(size(), num_steps);
    if (max_workers > 0)
      num_workers = std::min(num_workers, max_workers);

    if (num_steps == num_workers)
      stride = (num_items + num_steps - 1) / num_steps;

    std::atomic<size_t> counter(0);
    std::atomic<size_t> done(num_workers);
    semaphore_t barrier;

    while (num_workers--) {
      spawn([&, num_items, stride, num_steps]() {
        size_t idx;
        while ((idx = counter.fetch_add(1, std::memory_order_relaxed)) <
               num_steps) {
          size_t off = idx * stride, n = stride;
          if (off + n > num_items)
            n = num_items - off;
          while (n--)
            work(off++);
        }
        if (--done == 0)
          barrier.notify();
      });
    }

    barrier.wait();
  }
  template <class Workable>
  void par_map(size_t num_items, Workable work, size_t max_workers = 0) {
    par_map(num_items, 1, work, max_workers);
  }

#ifdef _GNU_SOURCE
  static char nibble_from_hex(char c) {
    int mask, ret;

    mask = (('a' - c - 1) & (c - 1 - 'f')) >> 31;
    ret = (10 + c - 'a') & mask;
    mask = (('A' - c - 1) & (c - 1 - 'F')) >> 31;
    ret |= (10 + c - 'A') & mask;
    mask = (('0' - c - 1) & (c - 1 - '9')) >> 31;
    ret |= (c - '0') & mask;
    mask = ((ret - 1) & ~mask) >> 31;
    ret |= 16 & mask;

    return (char)ret;
  }
#endif
};

template <class T>
class channel_t {
 private:
  std::deque<T> fifo;
  std::mutex mtx;
  std::condition_variable cvar;

 public:
  void send(const T& msg) {
    std::unique_lock<std::mutex> lock(mtx);
    fifo.push_back(msg);
    cvar.notify_one();
  }

  T recv() {
    std::unique_lock<std::mutex> lock(mtx);
    cvar.wait(lock, [&] { return !fifo.empty(); });
    auto msg = fifo.front();
    fifo.pop_front();
    return msg;
  }
};

template <typename T>
class counter_t {
  struct inner {
    std::atomic<T> val;
    std::atomic<size_t> ref_cnt;
    inline inner(T v) {
      val = v, ref_cnt = 1;
    };
  };
  inner* ptr;

 public:
  counter_t(T v = 0) {
    ptr = new inner(v);
  }
  counter_t(const counter_t& r) {
    (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);
  }
  ~counter_t() {
    if (ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1)
      delete ptr;
  }
  counter_t& operator=(const counter_t& r) = delete;
  size_t ref_cnt() const {
    return ptr->ref_cnt;
  }
  T operator++(int) const {
    return ptr->val.fetch_add(1, std::memory_order_relaxed);
  }
  T operator++() const {
    return ptr->val++ + 1;
  }
  T operator--(int) const {
    return ptr->val.fetch_sub(1, std::memory_order_relaxed);
  }
  T operator--() const {
    return ptr->val-- - 1;
  }
};

constexpr static int lg2(size_t n) {
  int ret = 0;
  while (n >>= 1)
    ret++;
  return ret;
}

template <class bucket_t>
bucket_t sum_up(const bucket_t inp[], size_t n) {
  bucket_t sum = inp[0];
  for (size_t i = 1; i < n; i++) {
    sum.add(inp[i]);
  }
  return sum;
}

template <
    class bucket_t,
    class point_t,
    class affine_t,
    class scalar_t,
    typename affine_h = typename affine_t::mem_t,
    typename bucket_h = typename bucket_t::mem_t>
class collect_t {
 public:
  size_t npoints;
  uint32_t wbits, nwins;

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

 public:
  collect_t(size_t np) {
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
  }

 public:
  void collect(
      point_t* out,
      bucket_t* res,
      const bucket_t* ones,
      uint32_t lenofone) {
    struct tile_t {
      uint32_t x, y, dy;
      point_t p;
      tile_t() {}
    };
    std::vector<tile_t> grid(nwins);
    uint32_t y = nwins - 1, total = 0;

    grid[0].x = 0;
    grid[0].y = y;
    grid[0].dy = scalar_t::bit_length() - y * wbits;
    total++;

    while (y--) {
      grid[total].x = grid[0].x;
      grid[total].y = y;
      grid[total].dy = wbits;
      total++;
    }

    std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
    counter_t<size_t> counter(0);
    channel_t<size_t> ch;

    thread_pool_t pool{"SPPARK_GPU_T_AFFINITY"};
    auto ncpus = pool.size();
    auto n_workers = (uint32_t)ncpus;
    if (n_workers > total) {
      n_workers = total;
    }
    while (n_workers--) {
      pool.spawn([&, this, total, counter]() {
        for (size_t work; (work = counter++) < total;) {
          auto item = &grid[work];
          auto y = item->y;
          item->p = integrate_row(
              res + y * MSM_NTHREADS / bucket_t::degree * 2, item->dy);
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
  point_t integrate_row(bucket_t* row, uint32_t lsbits) {
    const int NTHRBITS = lg2(MSM_NTHREADS / bucket_t::degree);

    assert(wbits - 1 > NTHRBITS);

    size_t i = MSM_NTHREADS / bucket_t::degree - 1;
    if (lsbits - 1 <= NTHRBITS) {
      size_t mask = (1U << (NTHRBITS - (lsbits - 1))) - 1;
      bucket_t res, acc = *(row + i * 2 + 1);

      if (mask)
        res.inf();
      else
        res = acc;

      while (i--) {
        acc.add(*(row + i * 2 + 1));
        if ((i & mask) == 0)
          res.add(acc);
      }

      return res;
    }

    point_t res = *(row + i * 2);
    bucket_t acc = *(row + i * 2 + 1);

    while (i--) {
      point_t raise = acc;
      for (size_t j = 0; j < lsbits - 1 - NTHRBITS; j++)
        raise.dbl();
      res.add(raise);
      res.add(*(row + i * 2));
      if (i) {
        acc.add(*(row + i * 2 + 1));
      }
    }

    return res;
  }
};

} // namespace native
} // namespace at
