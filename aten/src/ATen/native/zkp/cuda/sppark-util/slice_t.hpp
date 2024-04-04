#pragma once

#include <vector>

#ifdef __CUDACC__
# ifdef inline
#  define slice_t_saved_inline inline
#  undef inline
# endif
# define inline inline __host__ __device__
#endif
namespace at { 
namespace native {
// A simple way to pack a constant pointer and array's size length,
// and to "borrow" std::vector<T>&...
template<typename T> class slice_t {
    T* ptr;
    size_t nelems;
public:
    slice_t() : ptr(nullptr), nelems(0)                                 {}
    slice_t(void* p, size_t n) : ptr(reinterpret_cast<T*>(p)), nelems(n){}
    slice_t(T* p, size_t n) : ptr(p), nelems(n)                   {}
    slice_t(const std::vector<T>& v) : ptr(v.data()), nelems(v.size())  {}

    inline operator void*() const               { return (void*)ptr; }
    inline operator decltype(ptr)() const       { return ptr; }
    inline T* data() const                { return ptr; }
    inline size_t size() const                  { return nelems; }
    inline T& operator[](size_t i) const  { return ptr[i]; }
};
}//namespace native
}//namespace at
#ifdef __CUDACC__
# undef inline
# ifdef slice_t_saved_inline
#  define inline slice_t_saved_inline
# endif
#endif

