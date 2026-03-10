/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/orientation_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void OrientationErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 9>& f,
    std::array<Vector3<T>, 3>& v,
    std::array<Eigen::Matrix<T, 9, 3>, 3>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const OrientationDataT<T>& constr = this->constraints_[constrIndex];
  const Matrix3<T> rotMat = constr.offset.toRotationMatrix();
  v[0] = state.rotation() * rotMat.col(0);
  v[1] = state.rotation() * rotMat.col(1);
  v[2] = state.rotation() * rotMat.col(2);

  Matrix3<T> vec;
  vec.col(0) = v[0];
  vec.col(1) = v[1];
  vec.col(2) = v[2];
  Matrix3<T> val = vec - constr.target.toRotationMatrix();
  f = Eigen::Map<Vector<T, 9>>(val.data(), val.size());

  for (size_t iVec = 0; iVec < 3; ++iVec) {
    dfdv[iVec].setZero();
    dfdv[iVec].template middleRows<3>(iVec * 3).setIdentity();
  }
}

template class OrientationErrorFunctionT<float>;
template class OrientationErrorFunctionT<double>;

} // namespace momentum
