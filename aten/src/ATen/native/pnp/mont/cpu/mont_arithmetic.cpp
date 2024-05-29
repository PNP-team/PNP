#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/native/pnp/mont/cpu/curve_def.h>
#include <ATen/ops/copy.h>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

#define CONVERT_ELEM(name)                        \
  else if (type == ScalarType::name##_Base) {     \
    return caffe2::TypeMeta::Make<name##_Mont>(); \
  }                                               \
  else if (type == ScalarType::name##_Mont) {     \
    return caffe2::TypeMeta::Make<name##_Base>(); \
  }

caffe2::TypeMeta get_corresponding_type(const ScalarType type) {
  if (false) {
    ;
  }
  APPLY_ALL_CURVE(CONVERT_ELEM)
  else {
    throw std::runtime_error("Unsupported curve type");
  }
}
#undef CONVERT_ELEM

static void to_mont_cpu_template(Tensor& self) {
  AT_DISPATCH_BASE_TYPES(self.scalar_type(), "to_mont_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for (auto i = 0; i < num_; i++) {
      self_ptr[i].to();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

static void to_base_cpu_template(Tensor& self) {
  AT_DISPATCH_MONT_TYPES(self.scalar_type(), "to_base_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(
        self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for (auto i = 0; i < num_; i++) {
      self_ptr[i].from();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

static void add_template(
    const Tensor& in_a,
    const Tensor& in_b,
    Tensor& out_c) {
  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(in_a.scalar_type(), "add_mod_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(
        out_c.mutable_data_ptr<scalar_t>());
    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    for (auto i = 0; i < num_; i++) {
      c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  });
}

static void sub_template(
    const Tensor& in_a,
    const Tensor& in_b,
    Tensor& out_c) {
  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(in_a.scalar_type(), "sub_mod_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(
        out_c.mutable_data_ptr<scalar_t>());
    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    for (auto i = 0; i < num_; i++) {
      c_ptr[i] = a_ptr[i] - b_ptr[i];
    }
  });
}

static void mul_template(
    const Tensor& in_a,
    const Tensor& in_b,
    Tensor& out_c) {
  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(in_a.scalar_type(), "mul_mod_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(
        out_c.mutable_data_ptr<scalar_t>());
    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    for (auto i = 0; i < num_; i++) {
      c_ptr[i] = a_ptr[i] * b_ptr[i];
    }
  });
}

static void div_template(
    const Tensor& in_a,
    const Tensor& in_b,
    Tensor& out_c) {
  TORCH_CHECK(in_a.numel() == in_b.numel(), "Length check!");
  AT_DISPATCH_MONT_TYPES(in_a.scalar_type(), "div_mod_cpu", [&] {
    auto a_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_a.mutable_data_ptr<scalar_t>());
    auto b_ptr = reinterpret_cast<scalar_t::compute_type*>(
        in_b.mutable_data_ptr<scalar_t>());
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(
        out_c.mutable_data_ptr<scalar_t>());
    int64_t num_ = in_a.numel() / num_uint64(in_a.scalar_type());
    for (auto i = 0; i < num_; i++) {
      c_ptr[i] = a_ptr[i] / b_ptr[i];
    }
  });
}

static void exp_template(Tensor& out, int exp) {
  if (exp == 1) {
    return;
  }
  AT_DISPATCH_MONT_TYPES(out.scalar_type(), "exp_mod_cpu", [&] {
    auto c_ptr = reinterpret_cast<scalar_t::compute_type*>(
        out.mutable_data_ptr<scalar_t>());
    int64_t num_ = out.numel() / num_uint64(out.scalar_type());
    if (exp == 0) {
      for (auto i = 0; i < num_; i++) {
        c_ptr[i] = c_ptr[i].one();
      }
    } else {
      for (auto i = 0; i < num_; i++) {
        c_ptr[i] = c_ptr[i] ^ exp;
      }
    }
  });
}

} // anonymous namespace

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

Tensor add_mod_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  add_template(a, b, c);
  return c;
}

Tensor& add_mod_cpu_(Tensor& self, const Tensor& b) {
  add_template(self, b, self);
  return self;
}

Tensor& add_mod_out_cpu(const Tensor& a, const Tensor& b, Tensor& c) {
  add_template(a, b, c);
  return c;
}

Tensor sub_mod_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  sub_template(a, b, c);
  return c;
}

Tensor& sub_mod_cpu_(Tensor& self, const Tensor& b) {
  sub_template(self, b, self);
  return self;
}

Tensor& sub_mod_out_cpu(const Tensor& a, const Tensor& b, Tensor& c) {
  sub_template(a, b, c);
  return c;
}

Tensor mul_mod_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  mul_template(a, b, c);
  return c;
}

Tensor& mul_mod_cpu_(Tensor& self, const Tensor& b) {
  mul_template(self, b, self);
  return self;
}

Tensor& mul_mod_out_cpu(const Tensor& a, const Tensor& b, Tensor& c) {
  mul_template(a, b, c);
  return c;
}

Tensor div_mod_cpu(const Tensor& a, const Tensor& b) {
  Tensor c = at::empty_like(a);
  div_template(a, b, c);
  return c;
}

Tensor& div_mod_cpu_(Tensor& self, const Tensor& b) {
  div_template(self, b, self);
  return self;
}

Tensor& div_mod_out_cpu(const Tensor& a, const Tensor& b, Tensor& c) {
  div_template(a, b, c);
  return c;
}

Tensor exp_mod_cpu(const Tensor& input, long exp) {
  Tensor output = input.clone();
  exp_template(output, exp);
  return output;
}
Tensor& exp_mod_cpu_(Tensor& self, long exp) {
  exp_template(self, exp);
  return self;
}
Tensor& exp_mod_out_cpu(const Tensor& input, long exp, Tensor& output) {
  copy(output, input);
  exp_template(output, exp);
  return output;
}

} // namespace native
} // namespace at
