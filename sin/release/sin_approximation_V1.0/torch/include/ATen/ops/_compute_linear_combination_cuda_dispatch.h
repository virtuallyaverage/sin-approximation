#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace cuda {

TORCH_API at::Tensor _compute_linear_combination(const at::Tensor & input, const at::Tensor & coefficients);
TORCH_API at::Tensor & _compute_linear_combination_out(at::Tensor & out, const at::Tensor & input, const at::Tensor & coefficients);
TORCH_API at::Tensor & _compute_linear_combination_outf(const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out);

} // namespace cuda
} // namespace at
