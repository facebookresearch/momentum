/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/normal_error_function.h"

#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void NormalErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    std::array<Vector3<T>, 2>& v,
    std::array<Eigen::Matrix<T, 1, 3>, 2>& dfdv) const {
  MT_PROFILE_FUNCTION();

  const NormalDataT<T>& constr = this->constraints_[constrIndex];
  v[0] = state.transform * constr.localPoint;
  v[1] = state.rotation() * constr.localNormal;
  const Vector3<T> dist = v[0] - constr.globalPoint;

  f[0] = v[1].dot(dist);
  dfdv[0] = v[1].transpose();
  dfdv[1] = dist.transpose();
}

template class NormalErrorFunctionT<float>;
template class NormalErrorFunctionT<double>;

} // namespace momentum
