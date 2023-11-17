#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include<iostream>
#include "CurveDef.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

#define CONVERT_ELEM(name) \
else if (type == ScalarType::name##_Base) { return caffe2::TypeMeta::Make<name##_Mont>();} \
else if (type == ScalarType::name##_Mont) { return caffe2::TypeMeta::Make<name##_Base>();}

caffe2::TypeMeta get_corresponding_type(const ScalarType type) {
  if (false) { ; }
  APPLY_ALL_CURVE(CONVERT_ELEM)
  else {
    throw std::runtime_error("Unsupported curve type");
  }
}
#undef CONVERT_ELEM

static void to_mont_cpu_template(Tensor& self) {

  AT_DISPATCH_FR_BASE_TYPES(self.scalar_type(), "to_mont_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].to();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

/////////////////////////////////////////////////////////////////////
static void Arry_add_template(Tensor &a,const Tensor &b) {
  if (a.scalar_type() != b.scalar_type()) {
    throw std::runtime_error("Data types of input tensors must be the same.");
  }
    AT_DISPATCH_FR_MONT_TYPES(a.scalar_type(), "arry_add_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(b.mutable_data_ptr<scalar_t>());
  
    int64_t a_num_ = a.numel() / num_uint64(a.scalar_type());
    int64_t b_num_ = b.numel() / num_uint64(b.scalar_type());
    if(a_num_<b_num_)
    {
      throw std::runtime_error("length check!");
    }
    for(auto i = 0; i < a_num_; i++) {
      a_ptr[i]+=b_ptr[i];
    }
  });
  //a.set_dtype(get_corresponding_type(a.scalar_type()));
}


static void Arry_sub_template(Tensor &a,const Tensor &b) {
  if (a.scalar_type() != b.scalar_type()) {
    throw std::runtime_error("Data types of input tensors must be the same.");
  }
  AT_DISPATCH_FR_BASE_TYPES(a.scalar_type(), "Subtraction_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(b.mutable_data_ptr<scalar_t>());

    int64_t a_num_ = a.numel() / num_uint64(a.scalar_type());
    int64_t b_num_ = b.numel() / num_uint64(b.scalar_type());
    if(a_num_<b_num_)
    {
      throw std::runtime_error("length check!");
    }
    for(auto i = 0; i < a_num_; i++) {
      a_ptr[i]-=b_ptr[i];
    }
  });
}


static void Arry_mul_template(Tensor &a,const Tensor &b) {
    if (a.scalar_type() != b.scalar_type()) {
    throw std::runtime_error("Data types of input tensors must be the same.");
  }
  AT_DISPATCH_FR_BASE_TYPES(a.scalar_type(), "Multiplication_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(b.mutable_data_ptr<scalar_t>());

    int64_t a_num_ = a.numel() / num_uint64(a.scalar_type());
    int64_t b_num_ = b.numel() / num_uint64(b.scalar_type());
    if(a_num_<b_num_)
    {
      throw std::runtime_error("length check!");
    }
    for(auto i = 0; i < a_num_; i++) {
      a_ptr[i]*=b_ptr[i];
    }
  });
}


static void Arry_div_template(Tensor &a,const Tensor &b) {
    if (a.scalar_type() != b.scalar_type()) {
    throw std::runtime_error("Data types of input tensors must be the same.");
  }
  AT_DISPATCH_FR_BASE_TYPES(a.scalar_type(), "Division_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(b.mutable_data_ptr<scalar_t>());

    int64_t a_num_ = a.numel() / num_uint64(a.scalar_type());
    int64_t b_num_ = b.numel() / num_uint64(b.scalar_type());
    if(a_num_<b_num_)
    {
      throw std::runtime_error("length check!");
    }
    for(auto i = 0; i < a_num_; i++) {
      a_ptr[i]/=b_ptr[i];
    }
  });
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
static void to_base_cpu_template(Tensor& self) {
  AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "to_base_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].from();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

} // namespace

Tensor to_mont_cpu(const Tensor& input) {
  Tensor output = input.clone();
  to_mont_cpu_template(output);
  return output;
}

Tensor& to_mont_cpu_(Tensor& self) {
  to_mont_cpu_template(self);
  return self;
}

Tensor& to_mont_out_cpu(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_mont_cpu_template(output);
  return output;
}

Tensor to_base_cpu(const Tensor& input) {
  Tensor output = input.clone();
  to_base_cpu_template(output);
  return output;
}

Tensor& to_base_cpu_(Tensor& self) {
  to_base_cpu_template(self);
  return self;
}

Tensor& to_base_out_cpu(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_base_cpu_template(output);
  return output;
}

Tensor arry_add_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  std::cout<<"\n \n sadsa";
  Arry_add_template(c, b);
  return c;
}

Tensor& arry_add_cpu_(Tensor& self,const Tensor&b) {
  Arry_add_template(self,b);
  return self;
}

Tensor& arry_add_cpu_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  Arry_add_template(c, b);
  return c;
}


Tensor Subtraction_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  Arry_sub_template(c, b);
  return c;
}


Tensor& Subtraction_cpu_(Tensor& self,const Tensor&b) {
  Arry_sub_template(self,b);
  return self;
}

Tensor Multiplication_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  Arry_mul_template(c, b);
  return c;
}

Tensor Division_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = a.clone();
  Arry_div_template(c, b);
  return c;
}


} // namespace native
} // namespace at
