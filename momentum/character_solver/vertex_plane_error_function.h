/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/vertex_constraint_error_function.h>

namespace momentum {

/// Constraint data for a vertex plane target.
///
/// Constrains a mesh vertex to lie on or above a plane.
template <typename T>
struct VertexPlaneDataT : public VertexConstraintData {
  /// Plane normal (must be normalized)
  Eigen::Vector3<T> normal = Eigen::Vector3<T>::UnitY();

  /// A point on the plane
  Eigen::Vector3<T> point = Eigen::Vector3<T>::Zero();

  /// If true, constrain vertex to be above the plane (half-space)
  bool above = false;

  VertexPlaneDataT() = default;
  VertexPlaneDataT(
      size_t vIndex,
      const Eigen::Vector3<T>& n,
      const Eigen::Vector3<T>& p,
      bool abv,
      float w,
      const std::string& nm = "")
      : VertexConstraintData(vIndex, w, nm), normal(n), point(p), above(abv) {}
};

/// Error function that constrains mesh vertices to a plane.
///
/// The residual is the signed distance from the vertex to the plane:
///   f = dot(vertexPosition - point, normal)
///
/// If above=true, the constraint is a half-space constraint where
/// f is clamped to zero when the vertex is above the plane.
///
/// @tparam T Scalar type (float or double)
template <typename T>
class VertexPlaneErrorFunctionT : public VertexConstraintErrorFunctionT<T, VertexPlaneDataT<T>, 1> {
 public:
  using Base = VertexConstraintErrorFunctionT<T, VertexPlaneDataT<T>, 1>;
  using typename Base::DfdvType;
  using typename Base::FuncType;

  explicit VertexPlaneErrorFunctionT(
      const Character& character,
      const ParameterTransform& parameterTransform,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~VertexPlaneErrorFunctionT() override = default;

  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const final;
};

} // namespace momentum
