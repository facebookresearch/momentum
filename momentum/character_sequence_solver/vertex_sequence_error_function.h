/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/sequence_error_function.h>
#include <momentum/character_solver/vertex_error_function.h>
#include <momentum/math/mesh.h>

#include <span>

namespace momentum {

/// Constraint structure for vertex velocity.
template <typename T>
struct VertexVelocityConstraintT {
  int vertexIndex = -1;
  T weight = 1;
  Eigen::Vector3<T> targetVelocity;

  template <typename T2>
  VertexVelocityConstraintT<T2> cast() const {
    return {
        this->vertexIndex, static_cast<T2>(this->weight), this->targetVelocity.template cast<T2>()};
  }
};

/// Error function that penalizes differences in vertex velocity between source and target motion.
/// This function computes vertex velocities by taking the difference between consecutive frames
/// and penalizes deviations from target vertex velocities. It combines the sequence aspect of
/// StateSequenceErrorFunction with vertex constraints from VertexErrorFunction.
template <typename T>
class VertexSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  explicit VertexSequenceErrorFunctionT(const Character& character);

  [[nodiscard]] size_t numFrames() const final {
    return 2;
  }

  double getError(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates) const final;
  double getGradient(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const final;

  // modelParameters: [numFrames() * parameterTransform] parameter vector
  // skelStates: [numFrames()] array of skeleton states
  // jacobian: [getJacobianSize()] x [numFrames() * parameterTransform] Jacobian matrix
  // residual: [getJacobianSize()] residual vector.
  double getJacobian(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const final;

  bool needsMesh() const final {
    return true;
  }

  [[nodiscard]] size_t getJacobianSize() const final;

  /// Add a vertex velocity constraint.
  /// @param vertexIndex Index of the vertex to constrain.
  /// @param weight Weight for this constraint.
  /// @param targetVelocity Target velocity for the vertex.
  void addConstraint(int vertexIndex, T weight, const Eigen::Vector3<T>& targetVelocity);

  /// Clear all vertex velocity constraints.
  void clearConstraints();

  /// Get all vertex velocity constraints.
  [[nodiscard]] const std::vector<VertexVelocityConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Get the number of constraints.
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  /// Get the character reference.
  [[nodiscard]] const Character& getCharacter() const {
    return character_;
  }

  static constexpr T kVelocityWeight = 1e-3f;

 private:
  /// Calculate gradient for a single vertex velocity constraint.
  double calculateVelocityGradient(
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      const VertexVelocityConstraintT<T>& constraint,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Calculate Jacobian for a single vertex velocity constraint.
  double calculateVelocityJacobian(
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      const VertexVelocityConstraintT<T>& constraint,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      Eigen::Index startRow) const;

  const Character& character_;
  std::vector<VertexVelocityConstraintT<T>> constraints_;
};

} // namespace momentum
