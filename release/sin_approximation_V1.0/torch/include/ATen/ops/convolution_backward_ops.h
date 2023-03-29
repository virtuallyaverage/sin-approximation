#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API convolution_backward {
  using schema = ::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::OptionalSymIntArrayRef, at::IntArrayRef, c10::SymIntArrayRef, at::IntArrayRef, bool, c10::SymIntArrayRef, int64_t, ::std::array<bool,3>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::convolution_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor> call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, at::IntArrayRef stride, c10::SymIntArrayRef padding, at::IntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask);
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, at::IntArrayRef stride, c10::SymIntArrayRef padding, at::IntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask);
};

struct TORCH_API convolution_backward_out {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::OptionalSymIntArrayRef, at::IntArrayRef, c10::SymIntArrayRef, at::IntArrayRef, bool, c10::SymIntArrayRef, int64_t, ::std::array<bool,3>, at::Tensor &, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::convolution_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))")
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, at::IntArrayRef stride, c10::SymIntArrayRef padding, at::IntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2);
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, at::IntArrayRef stride, c10::SymIntArrayRef padding, at::IntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2);
};

}} // namespace at::_ops
