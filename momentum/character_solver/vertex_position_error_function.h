/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/vertex_constraint_error_function.h>

namespace momentum {

/// Constraint data for a vertex position target.
///
/// Constrains a mesh vertex to a target 3D position.
template <typename T>
struct VertexPositionDataT : public VertexConstraintData {
  /// Target position in world space
  Eigen::Vector3<T> target = Eigen::Vector3<T>::Zero();

  VertexPositionDataT() = default;
  VertexPositionDataT(
      size_t vIndex,
      const Eigen::Vector3<T>& targetPos,
      float w,
      const std::string& n = "")
      : VertexConstraintData(vIndex, w, n), target(targetPos) {}
};

/// Error function that constrains mesh vertices to target positions.
///
/// The residual is the 3D vector from the vertex position to the target:
///   f = vertexPosition - target
///
/// @tparam T Scalar type (float or double)
template <typename T>
class VertexPositionErrorFunctionT
    : public VertexConstraintErrorFunctionT<T, VertexPositionDataT<T>, 3> {
 public:
  using Base = VertexConstraintErrorFunctionT<T, VertexPositionDataT<T>, 3>;
  using typename Base::DfdvType;
  using typename Base::FuncType;

  explicit VertexPositionErrorFunctionT(
      const Character& character,
      const ParameterTransform& parameterTransform,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~VertexPositionErrorFunctionT() override = default;

  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const final;

  /// Specialized getGradient that avoids the dfdv*derivative matrix multiply.
  /// Since dfdv=Identity for position constraints, we use
  /// accumulateVertexGradientIdentityDfdv which computes
  /// weightedResidual.dot(derivative) directly instead of
  /// weightedResidual.dot(I * derivative).
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Specialized getJacobian with the same dfdv=Identity optimization.
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;
};

} // namespace momentum
