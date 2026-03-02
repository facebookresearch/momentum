/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/vertex_constraint_error_function.h>

namespace momentum {

/// Constraint data for a vertex 2D projection target.
///
/// Constrains a mesh vertex projected to a 2D target position.
template <typename T>
struct VertexProjectionConstraintDataT : public VertexConstraintData {
  /// Target position in 2D screen space
  Eigen::Vector2<T> target = Eigen::Vector2<T>::Zero();

  /// Camera projection matrix (3x4)
  Eigen::Matrix<T, 3, 4> projectionMatrix = Eigen::Matrix<T, 3, 4>::Zero();

  VertexProjectionConstraintDataT() = default;
  VertexProjectionConstraintDataT(
      size_t vIndex,
      const Eigen::Vector2<T>& targetPos,
      const Eigen::Matrix<T, 3, 4>& proj,
      float w,
      const std::string& nm = "")
      : VertexConstraintData(vIndex, w, nm), target(targetPos), projectionMatrix(proj) {}
};

/// Error function that constrains mesh vertices to 2D projected positions.
///
/// The residual is the 2D vector from the projected vertex to the target:
///   p = project(vertexPosition)
///   f = p - target
///
/// @tparam T Scalar type (float or double)
template <typename T>
class VertexProjectionConstraintErrorFunctionT
    : public VertexConstraintErrorFunctionT<T, VertexProjectionConstraintDataT<T>, 2> {
 public:
  using Base = VertexConstraintErrorFunctionT<T, VertexProjectionConstraintDataT<T>, 2>;
  using typename Base::DfdvType;
  using typename Base::FuncType;

  explicit VertexProjectionConstraintErrorFunctionT(
      const Character& character,
      const ParameterTransform& parameterTransform,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~VertexProjectionConstraintErrorFunctionT() override = default;

  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const final;
};

} // namespace momentum
