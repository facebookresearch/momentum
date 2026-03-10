/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/position_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void PositionErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    std::array<Vector3<T>, 1>& v,
    std::array<Matrix3<T>, 1>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const PositionDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.transform * constr.offset;
  f = v[0] - constr.target;
  dfdv[0].setIdentity();
}

template class PositionErrorFunctionT<float>;
template class PositionErrorFunctionT<double>;

} // namespace momentum
