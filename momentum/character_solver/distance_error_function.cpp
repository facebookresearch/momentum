/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/distance_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void DistanceErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    FuncType& f,
    std::array<VType, 1>& v,
    std::array<DfdvType, 1>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const DistanceConstraintDataT<T>& constr = this->constraints_[constrIndex];

  // Transform local offset to world space
  v[0] = state.transform * constr.offset;

  // Compute difference and distance
  const Vector3<T> diff = v[0] - constr.origin;
  const T distance = diff.norm();

  // Handle degenerate case (point exactly at origin)
  if (distance == T(0)) {
    f[0] = T(0);
    dfdv[0].setZero();
    return;
  }

  // Compute residual: f = distance - target
  f[0] = distance - constr.target;

  // df/dv = diff / distance (normalized direction)
  // This is a 1x3 matrix (row vector) since FuncDim=1
  dfdv[0] = (diff / distance).transpose();
}

template <typename T>
DistanceConstraintDataT<T> DistanceConstraintDataT<T>::createFromLocator(
    const momentum::Locator& locator) {
  DistanceConstraintDataT<T> result;
  result.parent = locator.parent;
  result.offset = locator.offset.template cast<T>();
  result.weight = 1;
  result.target = 0;
  return result;
}

template class DistanceErrorFunctionT<float>;
template class DistanceErrorFunctionT<double>;

template struct DistanceConstraintDataT<float>;
template struct DistanceConstraintDataT<double>;

} // namespace momentum
