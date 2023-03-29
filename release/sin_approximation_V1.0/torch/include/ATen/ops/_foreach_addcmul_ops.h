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


struct TORCH_API _foreach_addcmul__Scalar {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, const at::Scalar &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul_")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Scalar")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value);
};

struct TORCH_API _foreach_addcmul__ScalarList {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, at::ArrayRef<at::Scalar>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul_")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "ScalarList")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars);
};

struct TORCH_API _foreach_addcmul__Tensor {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul_")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars);
};

struct TORCH_API _foreach_addcmul_Scalar {
  using schema = ::std::vector<at::Tensor> (at::TensorList, at::TensorList, at::TensorList, const at::Scalar &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Scalar")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]")
  static ::std::vector<at::Tensor> call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value);
};

struct TORCH_API _foreach_addcmul_ScalarList {
  using schema = ::std::vector<at::Tensor> (at::TensorList, at::TensorList, at::TensorList, at::ArrayRef<at::Scalar>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "ScalarList")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]")
  static ::std::vector<at::Tensor> call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars);
};

struct TORCH_API _foreach_addcmul_Tensor {
  using schema = ::std::vector<at::Tensor> (at::TensorList, at::TensorList, at::TensorList, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]")
  static ::std::vector<at::Tensor> call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars);
};

struct TORCH_API _foreach_addcmul_Scalar_out {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, const at::Scalar &, at::TensorList);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Scalar_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value, at::TensorList out);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value, at::TensorList out);
};

struct TORCH_API _foreach_addcmul_ScalarList_out {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, at::ArrayRef<at::Scalar>, at::TensorList);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "ScalarList_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars, at::TensorList out);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars, at::TensorList out);
};

struct TORCH_API _foreach_addcmul_Tensor_out {
  using schema = void (at::TensorList, at::TensorList, at::TensorList, const at::Tensor &, at::TensorList);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_foreach_addcmul")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_foreach_addcmul.Tensor_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars, *, Tensor(a!)[] out) -> ()")
  static void call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars, at::TensorList out);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars, at::TensorList out);
};

}} // namespace at::_ops
