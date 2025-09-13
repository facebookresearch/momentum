/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void PositionErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    optional_ref<std::array<Vector3<T>, 1>> v,
    optional_ref<std::array<Matrix3<T>, 1>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const PositionDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> vec = state.transformation * constr.offset;
  f = vec - constr.target;
  if (v) {
    v->get().at(0) = std::move(vec);
  }
  if (dfdv) {
    dfdv->get().at(0).setIdentity();
  }
}

template class PositionErrorFunctionT<float>;
template class PositionErrorFunctionT<double>;

} // namespace momentum
