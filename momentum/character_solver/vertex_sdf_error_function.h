/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/fwd.h>

#include <vector>

namespace momentum {

/// Constraint specifying that a vertex should be at a target signed distance from the SDF.
template <typename T>
struct VertexSDFConstraintT {
  int vertexIndex = -1; ///< Mesh vertex index
  T targetDistance = T(0); ///< Target signed distance value
  T weight = T(1); ///< Per-constraint weight
};

/// Error function that penalizes deviation of mesh vertex SDF distance from a target value.
///
/// Takes a single SDFCollider (with optional parent joint and offset transform) and allows
/// adding constraints that specify target signed distance values for individual vertices.
/// Error per constraint: weight * kVertexSDFWeight * (distance - targetDistance)^2
template <typename T>
class VertexSDFErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Weight constant for vertex SDF penalties.
  static constexpr T kVertexSDFWeight = T(5e-3);

  /// Constructor.
  ///
  /// @param character The character containing skeleton, mesh, and skin weights
  /// @param sdfCollider The SDF collider to test against (with optional parent joint and offset)
  explicit VertexSDFErrorFunctionT(
      const Character& character,
      const SDFColliderT<float>& sdfCollider);

  /// Destructor
  ~VertexSDFErrorFunctionT() override = default;

  /// Add a constraint specifying a target distance for a vertex.
  void addConstraint(const VertexSDFConstraintT<T>& constraint);

  /// Clear all constraints.
  void clearConstraints();

  /// Returns the current constraints.
  [[nodiscard]] const std::vector<VertexSDFConstraintT<T>>& getConstraints() const;

  /// Returns the number of constraints.
  [[nodiscard]] size_t numConstraints() const;

  /// Computes the error for the given model parameters and skeleton state.
  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) override;

  /// Computes the error and its gradient.
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Computes the Jacobian and residual.
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  /// Returns the maximum number of Jacobian rows needed.
  [[nodiscard]] size_t getJacobianSize() const override;

  /// Returns true because this error function requires mesh state.
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

 private:
  /// Computes the world-to-SDF transform.
  [[nodiscard]] TransformT<T> computeWorldToSDFTransform(const SkeletonStateT<T>& state) const;

  /// Samples the SDF distance using a pre-computed world-to-SDF transform.
  [[nodiscard]] T sampleSDFDistanceWithTransform(
      const Eigen::Vector3<T>& worldPos,
      const TransformT<T>& worldToSdfTransform) const;

  /// Computes both SDF distance and gradient using pre-computed transforms.
  [[nodiscard]] std::pair<T, Eigen::Vector3<T>> sampleSDFDistanceAndGradientWithTransform(
      const Eigen::Vector3<T>& worldPos,
      const TransformT<T>& worldToSdfTransform,
      const TransformT<T>& sdfToWorldTransform) const;

  /// Accumulates gradient contributions for a single joint's 7 parameters.
  void accumulateJointGradient(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulates jacobian contributions for a single joint's 7 parameters.
  void accumulateJointJacobian(
      const JointStateT<T>& jointState,
      size_t paramIndex,
      const Eigen::Vector3<T>& posd,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  /// Walks up the collider hierarchy accumulating gradient contributions.
  void accumulateColliderHierarchyGradient(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Walks up the collider hierarchy accumulating jacobian contributions.
  void accumulateColliderHierarchyJacobian(
      const SkeletonStateT<T>& state,
      size_t colliderParent,
      size_t commonAncestor,
      const Eigen::Vector3<T>& sdfSurfacePoint,
      const Eigen::Vector3<T>& direction,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const;

  const Character& character_; ///< Reference to the character
  SDFColliderT<float> sdfCollider_; ///< SDF collider (stored by value; cheap due to shared_ptr)
  std::vector<VertexSDFConstraintT<T>> constraints_; ///< Active constraints
};

using VertexSDFErrorFunction = VertexSDFErrorFunctionT<float>;
using VertexSDFErrorFunctiond = VertexSDFErrorFunctionT<double>;

} // namespace momentum
