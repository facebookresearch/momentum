/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_plane_constraint_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

namespace momentum {

template <typename T>
VertexPlaneConstraintErrorFunctionT<T>::VertexPlaneConstraintErrorFunctionT(
    const Character& character,
    const ParameterTransform& parameterTransform,
    const T& lossAlpha,
    const T& lossC)
    : VertexConstraintErrorFunctionT<T, VertexPlaneConstraintDataT<T>, 1>(
          character,
          parameterTransform,
          lossAlpha,
          lossC) {
  this->legacyWeight_ = this->kLegacyWeight;
}

template <typename T>
void VertexPlaneConstraintErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& meshState,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < this->constraints_.size(), "Constraint index out of range");
  MT_CHECK(worldVecs.size() == 1, "Expected exactly one world vector for vertex plane");
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  const auto& constraint = this->constraints_[constrIndex];

  // Match legacy sign convention: flip the plane normal if it faces away from
  // the posed mesh normal at this vertex.  This ensures that the signed distance
  // is consistent regardless of whether the caller provided a normal pointing
  // toward or away from the mesh surface.
  const Eigen::Vector3<T> sourceNormal =
      meshState.posedMesh_->normals[constraint.vertexIndex].template cast<T>();
  Eigen::Vector3<T> normal = constraint.normal;
  if (sourceNormal.dot(normal) < T(0)) {
    normal *= T(-1);
  }

  // Signed distance from vertex to plane: d = (v - point) · normal
  const T distance = (worldVecs[0] - constraint.point).dot(normal);

  // Half-space constraint: clamp to zero when above the plane
  if (constraint.above && distance > T(0)) {
    f(0) = T(0);
    if (!dfdv.empty()) {
      dfdv[0].setZero();
    }
  } else {
    f(0) = distance;
    if (!dfdv.empty()) {
      MT_CHECK(dfdv.size() == 1, "Expected exactly one dfdv for vertex plane");
      dfdv[0] = normal.transpose();
    }
  }
}

// Explicit template instantiations
template class VertexPlaneConstraintErrorFunctionT<float>;
template class VertexPlaneConstraintErrorFunctionT<double>;

} // namespace momentum
