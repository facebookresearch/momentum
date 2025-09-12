/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/aim_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void AimDistErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    optional_ref<std::array<Vector3<T>, 2>> v,
    optional_ref<std::array<Eigen::Matrix3<T>, 2>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const AimDataT<T>& constr = this->constraints_[constrIndex];
  const Vector3<T> point = state.transform * constr.localPoint;
  const Vector3<T> srcDir = state.rotation() * constr.localDir;
  const Vector3<T> tgtVec = constr.globalTarget - point;
  const T projLength = srcDir.dot(tgtVec);

  // f = (globalTarget - point).dot(srcDir) * srcDir - (globalTarget - point)
  f = projLength * srcDir - tgtVec;
  if (v) {
    v->get().at(0) = point;
    v->get().at(1) = srcDir;
  }
  if (dfdv) {
    // df/d(point) = I - outterProd(srcDir, srcDir)
    dfdv->get().at(0).setIdentity();
    dfdv->get().at(0).noalias() -= srcDir * srcDir.transpose();
    // df/d(dir) = projLength * I + outterProd(srcDir, tgtVec)
    dfdv->get().at(1).noalias() = srcDir * tgtVec.transpose();
    dfdv->get().at(1).diagonal().array() += projLength;
  }
}

template <typename T>
void AimDirErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    optional_ref<std::array<Vector3<T>, 2>> v,
    optional_ref<std::array<Eigen::Matrix3<T>, 2>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const AimDataT<T>& constr = this->constraints_[constrIndex];
  const Vector3<T> point = state.transform * constr.localPoint;
  const Vector3<T> srcDir = state.rotation() * constr.localDir;
  const Vector3<T> tgtVec = constr.globalTarget - point;
  const T tgtNorm = tgtVec.norm();
  Vector3<T> tgtDir = Vector3<T>::Zero();
  if (tgtNorm > 1e-16) {
    tgtDir = tgtVec / tgtNorm;
  }

  // f = srcDir - (globalTarget - point).normalize()
  f = srcDir - tgtDir;
  if (v) {
    v->get().at(0) = point;
    v->get().at(1) = srcDir;
  }
  if (dfdv) {
    // df/d(point) = (I - outterProd(tgtDir, tgtDir)) / tgtNorm
    dfdv->get().at(0).setZero();
    if (tgtNorm > 1e-16) {
      dfdv->get().at(0).noalias() -= (tgtDir * tgtDir.transpose()) / tgtNorm;
      dfdv->get().at(0).diagonal().array() += T(1) / tgtNorm;
    }
    dfdv->get().at(1).setIdentity();
  }
}

template class AimDistErrorFunctionT<float>;
template class AimDistErrorFunctionT<double>;

template class AimDirErrorFunctionT<float>;
template class AimDirErrorFunctionT<double>;

} // namespace momentum
