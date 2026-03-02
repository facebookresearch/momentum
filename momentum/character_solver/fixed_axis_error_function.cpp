/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/fixed_axis_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void FixedAxisDiffErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    std::array<Vector3<T>, 1>& v,
    std::array<Matrix3<T>, 1>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.rotation() * constr.localAxis;
  f = v[0] - constr.globalAxis;
  dfdv[0].setIdentity();
}

template <typename T>
void FixedAxisCosErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    std::array<Vector3<T>, 1>& v,
    std::array<Eigen::Matrix<T, 1, 3>, 1>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.rotation() * constr.localAxis;
  f[0] = 1 - v[0].dot(constr.globalAxis);
  dfdv[0] = -constr.globalAxis.transpose();
}

template <typename T>
void FixedAxisAngleErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    std::array<Vector3<T>, 1>& v,
    std::array<Eigen::Matrix<T, 1, 3>, 1>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.rotation() * constr.localAxis;
  const T dot = v[0].dot(constr.globalAxis);
  f[0] = std::acos(std::clamp(dot, -T(1), T(1)));
  // The derivative of d[acos(x)]/dx  = -1/sqrt(1-x^2), where x is the cosine of the angle.
  // When the angle is 0 or 180, x=+/-1.0, and d[acos(x)] is infinity. But because dx=sine(angle)
  // is also zero, the final derivative will be zero as well.
  // Comparing to the Cos version, this Acos version scales up the jacobian by the inverse of
  // sine.
  const T sine = std::sqrt(1 - dot * dot);
  if (sine > 1e-9) {
    dfdv[0] = -constr.globalAxis.transpose() / sine;
  } else {
    dfdv[0].setZero();
  }
}

template class FixedAxisDiffErrorFunctionT<float>;
template class FixedAxisDiffErrorFunctionT<double>;

template class FixedAxisCosErrorFunctionT<float>;
template class FixedAxisCosErrorFunctionT<double>;

template class FixedAxisAngleErrorFunctionT<float>;
template class FixedAxisAngleErrorFunctionT<double>;

} // namespace momentum
