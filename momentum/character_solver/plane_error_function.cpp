/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/plane_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
std::vector<PlaneDataT<T>> createFloorConstraints(
    const std::string& prefix,
    const LocatorList& locators,
    const Vector3<T>& floorNormal,
    const T& floorOffset,
    const float weight) {
  std::vector<PlaneDataT<T>> result;
  for (const auto& loc : locators) {
    if (loc.name.find(prefix) == 0) {
      result.emplace_back(PlaneDataT<T>(
          loc.offset.cast<T>(),
          floorNormal,
          floorOffset,
          loc.parent,
          loc.weight * weight,
          loc.name));
    }
  }
  return result;
}

template std::vector<PlaneDataT<float>> createFloorConstraints<float>(
    const std::string& prefix,
    const LocatorList& locators,
    const Vector3f& floorNormal,
    const float& floorOffset,
    const float weight);
template std::vector<PlaneDataT<double>> createFloorConstraints<double>(
    const std::string& prefix,
    const LocatorList& locators,
    const Vector3d& floorNormal,
    const double& floorOffset,
    const float weight);

template <typename T>
void PlaneErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    optional_ref<std::array<Vector3<T>, 1>> v,
    optional_ref<std::array<Eigen::Matrix<T, 1, 3>, 1>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const PlaneDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> vec = state.transform * constr.offset;
  T val = vec.dot(constr.normal) - constr.d;
  if (halfPlane_ && val > T(0)) {
    val = T(0);
  }
  f[0] = val;

  if (v) {
    v->get().at(0) = std::move(vec);
  }
  if (dfdv) {
    dfdv->get().at(0).setZero();
    if (!halfPlane_ || val < T(0)) {
      dfdv->get().at(0) = constr.normal;
    }
  }
}

template class PlaneErrorFunctionT<float>;
template class PlaneErrorFunctionT<double>;

} // namespace momentum
