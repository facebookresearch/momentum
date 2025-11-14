/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/mesh_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/fwd.h>

#include <Eigen/Dense>

#include <queue>
#include <vector>

namespace momentum {

/// Error function for character height constraints
///
/// This error function measures the height of a character mesh by projecting
/// all vertices onto a specified "up" direction and computing the difference
/// between the maximum and minimum projections.
///
/// IMPORTANT: Unlike most error functions, this one only depends on specific
/// active parameters that are automatically determined in the constructor.
/// Specifically, it uses blend shape, face expression, and scale parameters only
/// (not pose parameters). The error function achieves this by zeroing out all
/// inactive parameters before computing the skeleton and mesh state. This allows
/// us to constrain height independent of pose, e.g. the solver doesn't try to fix
/// height by having the character bend over.
template <typename T>
class HeightErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Construct a height error function
  /// @param character The character to measure height for
  /// @param targetHeight The target height for the character (required)
  /// @param upDirection The direction to measure height along (defaults to Y-axis)
  /// @param k Number of vertices to average for min/max height calculation (defaults to 1)
  ///
  /// This error function automatically uses blend shape, face expression, and
  /// scale parameters only. All other parameters (pose, etc.) are kept inactive.
  explicit HeightErrorFunctionT(
      const Character& character,
      T targetHeight,
      const Eigen::Vector3<T>& upDirection = Eigen::Vector3<T>::UnitY(),
      size_t k = 1);
  ~HeightErrorFunctionT() override;

  HeightErrorFunctionT(const HeightErrorFunctionT& other) = delete;
  HeightErrorFunctionT(HeightErrorFunctionT&& other) noexcept = delete;
  HeightErrorFunctionT& operator=(const HeightErrorFunctionT& other) = delete;
  HeightErrorFunctionT& operator=(HeightErrorFunctionT&& other) = delete;

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

  /// Set the target height
  /// @param height The desired height
  void setTargetHeight(T height);

  /// Get the target height
  [[nodiscard]] T getTargetHeight() const {
    return targetHeight_;
  }

  /// Set the up direction for height measurement
  /// @param upDirection The direction to measure height along (will be normalized)
  void setUpDirection(const Eigen::Vector3<T>& upDirection);

  /// Get the up direction
  [[nodiscard]] const Eigen::Vector3<T>& getUpDirection() const {
    return upDirection_;
  }

  /// Get the character
  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  /// Override to indicate this function does NOT require mesh state
  /// (we maintain our own meshes)
  [[nodiscard]] bool needsMesh() const override {
    return false;
  }

 private:
  /// Create a copy of model parameters with inactive parameters zeroed out
  /// @param modelParameters The input parameters
  /// @return Modified parameters with only active parameters
  [[nodiscard]] ModelParametersT<T> applyActiveParameters(
      const ModelParametersT<T>& modelParameters) const;

  /// Result of height calculation
  struct HeightResult {
    T height{};
    std::vector<size_t> minVertexIndices;
    std::vector<T> minVertexWeights;
    std::vector<size_t> maxVertexIndices;
    std::vector<T> maxVertexWeights;
  };

  /// Calculate the current height of the mesh
  /// @return Height result containing the height and the min/max vertex indices
  [[nodiscard]] HeightResult calculateHeight() const;

  /// Calculate jacobian contribution from a vertex
  template <typename Derived>
  void calculateVertexJacobian(
      size_t vertexIndex,
      const Eigen::Vector3<T>& jacobianDirection,
      Eigen::MatrixBase<Derived>& jacobian) const;

  /// Calculate gradient contribution from a vertex
  void calculateVertexGradient(
      size_t vertexIndex,
      const Eigen::Vector3<T>& gradientDirection,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Calculate world space position derivative for blend shape parameters
  void calculateDWorldPos(
      size_t vertexIndex,
      const Eigen::Vector3<T>& d_restPos,
      Eigen::Vector3<T>& d_worldPos) const;

  const Character& character_;

  T targetHeight_;
  Eigen::Vector3<T> upDirection_;
  size_t k_;

  /// The set of active model parameters (blend shapes, face expressions, and scale)
  ///
  /// This is automatically set up in the constructor and determines which parameters
  /// affect the height measurement. Inactive parameters are zeroed out before
  /// computing the skeleton and mesh state.
  ParameterSet activeModelParams_;

  /// Internal skeleton state for skinning (computed from active parameters only)
  ///
  /// This is updated in getError/getGradient/getJacobian by zeroing out all
  /// inactive parameters from the input model parameters. This ensures derivatives
  /// w.r.t. inactive parameters are automatically zero.
  mutable SkeletonStateT<T> skeletonState_;

  /// Internal mesh state (updated using skeletonState_ and active parameters only)
  ///
  /// We maintain our own mesh state because we use skeletonState_ (computed from
  /// active parameters only) rather than the state parameter passed to
  /// getError/getGradient/getJacobian. This is marked mutable so we can update
  /// it in const methods like getError.
  mutable MeshStateT<T> meshState_;
};

} // namespace momentum
