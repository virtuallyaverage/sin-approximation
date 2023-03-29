#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/remainder_meta.h>

namespace at {
namespace native {
TORCH_API at::Tensor remainder(const at::Tensor & self, const at::Scalar & other);
TORCH_API at::Tensor & remainder_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
TORCH_API at::Tensor & remainder_(at::Tensor & self, const at::Scalar & other);
struct TORCH_API structured_remainder_out : public at::meta::structured_remainder_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
};
TORCH_API at::Tensor & remainder_Scalar_Tensor_out(const at::Scalar & self, const at::Tensor & other, at::Tensor & out);
TORCH_API at::Tensor remainder(const at::Scalar & self, const at::Tensor & other);
} // namespace native
} // namespace at
