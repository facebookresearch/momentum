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

#include <vector>

namespace momentum {

/// Error function that constrains the minimum vertex height to a target value.
///
/// Given a set of candidate vertex indices (e.g. foot vertices), this error
/// function finds the k lowest vertices along a specified "up" direction and
/// penalizes the difference between their average projection and a target
/// height. This is useful for ground contact constraints where feet should
/// touch the floor (target_height = 0) while allowing individual feet to
/// lift off the ground.
///
/// Unlike HeightErrorFunction, this function:
/// - Uses the solver's posed mesh (needsMesh() = true), so it responds to
///   all model parameters including pose
/// - Only constrains the minimum projection (not the height = max - min)
/// - Operates on a specified subset of vertices rather than all vertices
template <typename T>
class FloorErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Construct a floor error function.
  ///
  /// @param character The character (must have a mesh)
  /// @param vertexIndices Indices of candidate vertices to consider (e.g. foot
  ///   vertices). The k lowest among these will be constrained.
  /// @param targetHeight Target height for the minimum vertices (default: 0)
  /// @param upDirection Direction to measure height along (default: Y-axis)
  /// @param k Number of lowest vertices to average (default: 5)
  explicit FloorErrorFunctionT(
      const Character& character,
      std::vector<size_t> vertexIndices,
      T targetHeight = T(0),
      const Eigen::Vector3<T>& upDirection = Eigen::Vector3<T>::UnitY(),
      size_t k = 5);
  ~FloorErrorFunctionT() override;

  FloorErrorFunctionT(const FloorErrorFunctionT& other) = delete;
  FloorErrorFunctionT(FloorErrorFunctionT&& other) noexcept = delete;
  FloorErrorFunctionT& operator=(const FloorErrorFunctionT& other) = delete;
  FloorErrorFunctionT& operator=(FloorErrorFunctionT&& other) = delete;

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

  /// Set the target height for the minimum vertices.
  void setTargetHeight(T height);

  /// Get the target height.
  [[nodiscard]] T getTargetHeight() const {
    return targetHeight_;
  }

  /// Set the up direction for height measurement.
  /// @param upDirection The direction to measure height along (will be
  ///   normalized)
  void setUpDirection(const Eigen::Vector3<T>& upDirection);

  /// Get the up direction.
  [[nodiscard]] const Eigen::Vector3<T>& getUpDirection() const {
    return upDirection_;
  }

  /// Set the candidate vertex indices.
  void setVertexIndices(std::vector<size_t> indices);

  /// Get the candidate vertex indices.
  [[nodiscard]] const std::vector<size_t>& getVertexIndices() const {
    return vertexIndices_;
  }

  /// Get the character.
  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  /// This function requires the solver to provide a posed mesh.
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

 private:
  /// Result of finding the k lowest vertices.
  struct MinHeightResult {
    T avgProjection{};
    std::vector<size_t> vertexIndices;
    std::vector<T> vertexWeights;
  };

  /// Find the k lowest vertices among the candidates.
  [[nodiscard]] MinHeightResult calculateMinHeight(const MeshStateT<T>& meshState) const;

  /// Calculate jacobian contribution from a vertex.
  template <typename Derived>
  void calculateVertexJacobian(
      size_t vertexIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const Eigen::Vector3<T>& jacobianDirection,
      Eigen::MatrixBase<Derived>& jacobian) const;

  /// Calculate gradient contribution from a vertex.
  void calculateVertexGradient(
      size_t vertexIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const Eigen::Vector3<T>& gradientDirection,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  const Character& character_;

  std::vector<size_t> vertexIndices_;
  T targetHeight_;
  Eigen::Vector3<T> upDirection_;
  size_t k_;
};

} // namespace momentum
