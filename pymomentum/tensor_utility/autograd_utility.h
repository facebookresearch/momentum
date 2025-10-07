/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <pymomentum/tensor_utility/tensor_utility.h>
#include <torch/torch.h>

namespace pymomentum {

template <typename T>
inline bool hasFloat64(T&& val) {
  if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
    return val.scalar_type() == toScalarType<double>();
  } else {
    return false;
  }
}

template <typename T, typename... Rest>
inline bool hasFloat64(T&& val, Rest&&... rest) {
  return hasFloat64(std::forward<T>(val)) || hasFloat64(std::forward<Rest>(rest)...);
}

// Takes an autograd function that is templated on float type and apply the
// version that we think is most appropriate, using the heuristic "if any
// of the inputs are double precision Tensors then promote everything to
// double."
template <template <class> class Fn, class... Args>
inline torch::autograd::variable_list applyTemplatedAutogradFunction(Args&&... args) {
  if (hasFloat64(std::forward<Args>(args)...)) {
    return Fn<double>::apply(std::forward<Args>(args)...);
  } else {
    return Fn<float>::apply(std::forward<Args>(args)...);
  }
}

} // namespace pymomentum
