/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/aim_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void AimDistErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    std::array<Vector3<T>, 2>& v,
    std::array<Eigen::Matrix3<T>, 2>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const AimDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.transform * constr.localPoint;
  v[1] = state.rotation() * constr.localDir;
  const Vector3<T> tgtVec = constr.globalTarget - v[0];
  const T projLength = v[1].dot(tgtVec);

  // f = (globalTarget - point).dot(srcDir) * srcDir - (globalTarget - point)
  f = projLength * v[1] - tgtVec;
  // df/d(point) = I - outterProd(srcDir, srcDir)
  dfdv[0].setIdentity();
  dfdv[0].noalias() -= v[1] * v[1].transpose();
  // df/d(dir) = projLength * I + outterProd(srcDir, tgtVec)
  dfdv[1].noalias() = v[1] * tgtVec.transpose();
  dfdv[1].diagonal().array() += projLength;
}

template <typename T>
void AimDirErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    std::array<Vector3<T>, 2>& v,
    std::array<Eigen::Matrix3<T>, 2>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const AimDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.transform * constr.localPoint;
  v[1] = state.rotation() * constr.localDir;
  const Vector3<T> tgtVec = constr.globalTarget - v[0];
  const T tgtNorm = tgtVec.norm();
  Vector3<T> tgtDir = Vector3<T>::Zero();
  if (tgtNorm > 1e-16) {
    tgtDir = tgtVec / tgtNorm;
  }

  // f = srcDir - (globalTarget - point).normalize()
  f = v[1] - tgtDir;
  // df/d(point) = (I - outterProd(tgtDir, tgtDir)) / tgtNorm
  dfdv[0].setZero();
  if (tgtNorm > 1e-16) {
    dfdv[0].noalias() -= (tgtDir * tgtDir.transpose()) / tgtNorm;
    dfdv[0].diagonal().array() += T(1) / tgtNorm;
  }
  dfdv[1].setIdentity();
}

template class AimDistErrorFunctionT<float>;
template class AimDistErrorFunctionT<double>;

template class AimDirErrorFunctionT<float>;
template class AimDirErrorFunctionT<double>;

} // namespace momentum
