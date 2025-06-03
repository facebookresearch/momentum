/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/orientation_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void OrientationErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 9>& f,
    optional_ref<std::array<Vector3<T>, 3>> v,
    optional_ref<std::array<Eigen::Matrix<T, 9, 3>, 3>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const OrientationDataT<T>& constr = this->constraints_[constrIndex];
  Matrix3<T> vec = state.rotation() * constr.offset.toRotationMatrix();
  Matrix3<T> val = vec - constr.target.toRotationMatrix();
  f = Eigen::Map<Vector<T, 9>>(val.data(), val.size());

  if (v) {
    v->get().at(0) = std::move(vec.col(0));
    v->get().at(1) = std::move(vec.col(1));
    v->get().at(2) = std::move(vec.col(2));
  }
  if (dfdv) {
    for (size_t iVec = 0; iVec < 3; ++iVec) {
      dfdv->get()[iVec].setZero();
      dfdv->get()[iVec].template middleRows<3>(iVec * 3).setIdentity();
    }
  }
}

template class OrientationErrorFunctionT<float>;
template class OrientationErrorFunctionT<double>;

} // namespace momentum
