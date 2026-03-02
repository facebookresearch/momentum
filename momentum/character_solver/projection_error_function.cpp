/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/projection_error_function.h"

#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
ProjectionErrorFunctionT<T>::ProjectionErrorFunctionT(
    const momentum::Skeleton& skel,
    const momentum::ParameterTransform& pt,
    T nearClip)
    : Base(skel, pt), nearClip_(nearClip) {}

template <typename T>
void ProjectionErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const JointStateT<T>& state,
    FuncType& f,
    std::array<VType, 1>& v,
    std::array<DfdvType, 1>& dfdv) const {
  const auto& cons = this->constraints_[constrIndex];

  // Transform local offset to world space
  v[0] = state.transform * cons.offset;

  // Project using the 3x4 projection matrix: (x, y, z) = P * [v; 1]
  const Eigen::Vector3<T> p_projected = cons.projection * v[0].homogeneous();

  // Check near-clip plane - if behind camera, return zero residual to skip this constraint
  if (p_projected.z() < nearClip_) {
    f.setZero();
    dfdv[0].setZero();
    return;
  }

  const T z = p_projected.z();

  // Compute 2D residual: f = (x/z, y/z) - target
  f = p_projected.template head<2>() / z - cons.target;

  // The chain rule: df/dv = df/dp_projected * dp_projected/dv
  // where p_projected = projection * [v; 1]
  //
  // Let P = projection (3x4), then p_projected = P * [v; 1] = P_33 * v + p_3
  // where P_33 is the top-left 3x3 block and p_3 is the last column
  //
  // f = (x/z, y/z) - target, so:
  // df/dp_projected = [(1/z, 0, -x/z^2), (0, 1/z, -y/z^2)]
  //
  // Then df/dv = df/dp_projected * P_33

  const T z_sqr = sqr(z);
  const T x_zz = p_projected.x() / z_sqr;
  const T y_zz = p_projected.y() / z_sqr;
  const T inv_z = T(1) / z;

  // df/dp_projected is 2x3
  Eigen::Matrix<T, 2, 3> df_dp;
  df_dp << inv_z, T(0), -x_zz, T(0), inv_z, -y_zz;

  // Multiply by P_33 (top-left 3x3 of projection matrix)
  dfdv[0] = df_dp * cons.projection.template topLeftCorner<3, 3>();
}

template class ProjectionErrorFunctionT<float>;
template class ProjectionErrorFunctionT<double>;

} // namespace momentum
