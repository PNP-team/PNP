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

  AT_DISPATCH_BASE_TYPES(self.scalar_type(), "to_mont_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].to();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

/////////////////////////////////////////////////////////////////////

static bool equal_mod_template(const Tensor &in_a,const Tensor &in_b) {
    bool result=true;

    AT_DISPATCH_FR_MONT_TYPES(in_a.scalar_type(), "equal_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
  
    int64_t a_num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    int64_t b_num_ = in_b.numel() / num_uint64(in_b.scalar_type());
    TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

    for(auto i = 0; i < a_num_; i++) {
      if(a_ptr[i]!=b_ptr[i])
      {
        result=false;
      }
    }
   
  });
  //a.set_dtype(get_corresponding_type(a.scalar_type()));
   return result;
}
static bool equal_base_template(const Tensor &in_a, const Tensor &in_b)
{

    bool result=true;

    AT_DISPATCH_FR_BASE_TYPES(in_a.scalar_type(), "equal_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
  
    int64_t a_num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    int64_t b_num_ = in_b.numel() / num_uint64(in_b.scalar_type());
    TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

    for(auto i = 0; i < a_num_; i++) {
      if(a_ptr[i]!=b_ptr[i])
      {
        result=false;
      }
    }
    });
    return result;
}

static void add_template(const Tensor &in_a,const Tensor &in_b, Tensor &out_c) {

    TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

    AT_DISPATCH_FR_MONT_TYPES(in_a.scalar_type(), "add_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(out_c.mutable_data_ptr<scalar_t>());

    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());

    for(auto i = 0; i < num_; i++) {
      c_ptr[i]=a_ptr[i]+b_ptr[i];
    }
  });
  //a.set_dtype(get_corresponding_type(a.scalar_type()));
}


static void sub_template(const Tensor &in_a, const Tensor &in_b, Tensor& out_c) {

  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

  AT_DISPATCH_FR_MONT_TYPES(in_a.scalar_type(), "subtraction_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(out_c.mutable_data_ptr<scalar_t>());

    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    for(auto i = 0; i < num_; i++) {
      c_ptr[i] = a_ptr[i]-b_ptr[i];
    }

  });
}


static void mul_template(const Tensor &in_a,const Tensor &in_b,Tensor &out_c) {

  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

  AT_DISPATCH_FR_MONT_TYPES(in_a.scalar_type(), "multiplication_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(out_c.mutable_data_ptr<scalar_t>());

    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());

    for(auto i = 0; i < num_; i++) {
      c_ptr[i]=a_ptr[i]*b_ptr[i];
    }
  });
}

static void div_template(const Tensor &in_a,const Tensor &in_b,Tensor &out_c) {
    TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");

    AT_DISPATCH_FR_MONT_TYPES(in_a.scalar_type(), "division_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(out_c.mutable_data_ptr<scalar_t>());

    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());

    for(auto i = 0; i < num_; i++) {
     c_ptr[i]=a_ptr[i]/b_ptr[i];
    }
  });
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
static void to_base_cpu_template(Tensor& self) {
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "to_base_cpu", [&] {
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
  Tensor output =input.clone();
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
  //Tensor output=at::empty_like(input);
  Tensor output=input.clone();
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
//****************add**************
Tensor add_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  add_template(a, b,c);
  return c;
}

Tensor& add_cpu_(Tensor& self,const Tensor&b) {
  add_template(self,b,self);
  return self;
}

Tensor& add_cpu_out(const Tensor& a, const Tensor& b, Tensor& c) {
  copy(c, a);
  add_template(a, b,c);
  return c;
}
//****************sub**************
Tensor subtraction_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  sub_template(a, b, c);
  return c;
}

Tensor& subtraction_cpu_(Tensor& self,const Tensor&b) {
  sub_template(self,b,self);
  return self;
}


Tensor& subtraction_cpu_out(const Tensor& a, const Tensor& b, Tensor& c) {
  sub_template(a, b, c);
  return c;
}
//****************mul************************
Tensor multiplication_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  mul_template(a, b,c);
  return c;
}

Tensor multiplication_cpu_(Tensor& self, const Tensor& b) {
  mul_template(self, b,self);
  return self;
}

Tensor& multiplication_cpu_out(const Tensor& a, const Tensor& b, Tensor& c) {
  mul_template(a, b,c);
  return c;
}
//****************div**********************
Tensor division_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  div_template(a, b,c);
  return c;
}

Tensor division_cpu_(Tensor& self, const Tensor& b) {
  div_template(self, b,self);
  return self;
}

Tensor& division_cpu_out(const Tensor& a, const Tensor& b, Tensor& c) {
  div_template(a, b,c);
  return c;
}
//********************equal*******************
bool equal_cpu_mod(const Tensor &a,const Tensor &b)
{
  return equal_mod_template(a,b);
}


bool equal_cpu_base(const Tensor &a,const Tensor &b)
{
  return equal_base_template(a,b);
}



} // namespace native
} // namespace at
