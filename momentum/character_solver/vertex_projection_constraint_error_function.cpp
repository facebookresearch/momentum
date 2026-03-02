/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_projection_constraint_error_function.h"

#include "momentum/character/character.h"
#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
VertexProjectionConstraintErrorFunctionT<T>::VertexProjectionConstraintErrorFunctionT(
    const Character& character,
    const ParameterTransform& parameterTransform,
    const T& lossAlpha,
    const T& lossC)
    : VertexConstraintErrorFunctionT<T, VertexProjectionConstraintDataT<T>, 2>(
          character,
          parameterTransform,
          lossAlpha,
          lossC) {}

template <typename T>
void VertexProjectionConstraintErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < this->constraints_.size(), "Constraint index out of range");
  MT_CHECK(worldVecs.size() == 1, "Expected exactly one world vector for vertex projection");

  const auto& constraint = this->constraints_[constrIndex];
  const Eigen::Vector3<T>& v = worldVecs[0];

  // Project 3D point to 2D using the projection matrix
  // p_hom = P * [v; 1]
  Eigen::Vector4<T> vHom;
  vHom << v, T(1);
  const Eigen::Vector3<T> pHom = constraint.projectionMatrix * vHom;

  // Near-clip: skip vertices behind the camera (matching legacy _nearClip = 1.0)
  if (pHom(2) < T(1)) {
    f.setZero();
    return;
  }

  // Perspective divide
  const T invZ = T(1) / pHom(2);
  const Eigen::Vector2<T> projected(pHom(0) * invZ, pHom(1) * invZ);

  // Residual: f = projected - target
  f = projected - constraint.target;

  // Derivative: df/dv = d(p/w)/dv where p = Proj * v_hom
  if (!dfdv.empty()) {
    MT_CHECK(dfdv.size() == 1, "Expected exactly one dfdv for vertex projection");

    // df/dv = [d(x/z)/dv; d(y/z)/dv]
    // Using quotient rule: d(a/b)/dx = (da/dx * b - a * db/dx) / b^2
    const T invZ2 = invZ * invZ;

    // P[:3, :3] is the 3x3 submatrix of projection matrix
    const Eigen::Matrix<T, 3, 3> P33 = constraint.projectionMatrix.template leftCols<3>();

    // df/dv row 0: d(x/z)/dv = (P[0,:3] * z - x * P[2,:3]) / z^2
    // df/dv row 1: d(y/z)/dv = (P[1,:3] * z - y * P[2,:3]) / z^2
    dfdv[0].row(0) = (P33.row(0) * pHom(2) - pHom(0) * P33.row(2)) * invZ2;
    dfdv[0].row(1) = (P33.row(1) * pHom(2) - pHom(1) * P33.row(2)) * invZ2;
  }
}

// Explicit template instantiations
template class VertexProjectionConstraintErrorFunctionT<float>;
template class VertexProjectionConstraintErrorFunctionT<double>;

} // namespace momentum
