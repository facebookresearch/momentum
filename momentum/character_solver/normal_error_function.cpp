/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/normal_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void NormalErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    optional_ref<std::array<Vector3<T>, 2>> v,
    optional_ref<std::array<Eigen::Matrix<T, 1, 3>, 2>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const NormalDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> point = state.transformation * constr.localPoint;
  Vector3<T> normal = state.rotation() * constr.localNormal;
  const Vector3<T> dist = point - constr.globalPoint;

  f[0] = normal.dot(dist);
  if (v) {
    v->get().at(0) = std::move(point);
    v->get().at(1) = std::move(normal);
  }
  if (dfdv) {
    if (v) {
      dfdv->get().at(0) = v->get().at(1); // normal
    } else {
      dfdv->get().at(0) = normal;
    }
    dfdv->get().at(1) = dist;
  }
}

template class NormalErrorFunctionT<float>;
template class NormalErrorFunctionT<double>;

} // namespace momentum
