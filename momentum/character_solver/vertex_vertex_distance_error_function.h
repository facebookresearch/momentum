/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/fwd.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace momentum {

/// Constraint for vertex-to-vertex distance errors
template <typename T>
struct VertexVertexDistanceConstraintT {
  int vertexIndex1 = -1; ///< First vertex index
  int vertexIndex2 = -1; ///< Second vertex index
  T weight = 1; ///< Constraint weight
  T targetDistance = 0; ///< Desired distance between the two vertices

  template <typename T2>
  VertexVertexDistanceConstraintT<T2> cast() const {
    return {
        this->vertexIndex1,
        this->vertexIndex2,
        static_cast<T2>(this->weight),
        static_cast<T2>(this->targetDistance)};
  }
};

/// Error function for vertex-to-vertex distance constraints
template <typename T>
class VertexVertexDistanceErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit VertexVertexDistanceErrorFunctionT(const Character& character);
  ~VertexVertexDistanceErrorFunctionT() override;

  VertexVertexDistanceErrorFunctionT(const VertexVertexDistanceErrorFunctionT& other) = delete;
  VertexVertexDistanceErrorFunctionT(VertexVertexDistanceErrorFunctionT&& other) noexcept = delete;
  VertexVertexDistanceErrorFunctionT& operator=(const VertexVertexDistanceErrorFunctionT& other) =
      delete;
  VertexVertexDistanceErrorFunctionT& operator=(VertexVertexDistanceErrorFunctionT&& other) =
      delete;

  [[nodiscard]] double getError(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  /// Add a vertex-to-vertex distance constraint
  void addConstraint(int vertexIndex1, int vertexIndex2, T weight, T targetDistance);

  /// Clear all constraints
  void clearConstraints();

  /// Get all constraints
  [[nodiscard]] const std::vector<VertexVertexDistanceConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Get the number of constraints
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  /// Get the character
  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  /// Override to indicate this function requires mesh state
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

 private:
  /// Calculate jacobian for a distance constraint
  double calculateJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexVertexDistanceConstraintT<T>& constraint,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      T& residual) const;

  /// Calculate gradient for a distance constraint
  double calculateGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexVertexDistanceConstraintT<T>& constraint,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Calculate gradient contribution from a single vertex
  void calculateVertexGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      int vertexIndex,
      const Eigen::Vector3<T>& gradientDirection,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Calculate jacobian contribution from a single vertex
  void calculateVertexJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      int vertexIndex,
      const Eigen::Vector3<T>& jacobianDirection,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian) const;

  /// Calculate world space position derivative for blend shape parameters
  void calculateDWorldPos(
      const SkeletonStateT<T>& state,
      int vertexIndex,
      const Eigen::Vector3<T>& d_restPos,
      Eigen::Vector3<T>& d_worldPos) const;

  const Character& character_;

  std::vector<VertexVertexDistanceConstraintT<T>> constraints_;
};

} // namespace momentum
