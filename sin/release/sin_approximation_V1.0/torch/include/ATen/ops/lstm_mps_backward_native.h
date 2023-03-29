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


namespace at {
namespace native {
TORCH_API void lstm_mps_backward_out(const at::Tensor & grad_y, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & z_state, const at::Tensor & cell_state_fwd, const at::Tensor & input, const at::Tensor & layersOutputs, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, at::Tensor & out0, at::TensorList out1, at::TensorList out2);
} // namespace native
} // namespace at
