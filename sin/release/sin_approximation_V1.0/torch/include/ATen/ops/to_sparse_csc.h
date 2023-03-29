#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/to_sparse_csc_ops.h>

namespace at {


// aten::to_sparse_csc.out(Tensor self, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & to_sparse_csc_out(at::Tensor & out, const at::Tensor & self, c10::optional<int64_t> dense_dim=c10::nullopt) {
    return at::_ops::to_sparse_csc_out::call(self, dense_dim, out);
}
// aten::to_sparse_csc.out(Tensor self, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & to_sparse_csc_outf(const at::Tensor & self, c10::optional<int64_t> dense_dim, at::Tensor & out) {
    return at::_ops::to_sparse_csc_out::call(self, dense_dim, out);
}

}
