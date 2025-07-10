/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/plane_error_function.h>
#include <momentum/character_solver/position_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/vertex_error_function.h>
#include <momentum/math/fwd.h>

namespace momentum {

template <typename T>
struct PointTriangleVertexConstraintT {
  int srcVertexIndex = -1;
  Eigen::Vector3i tgtTriangleIndices;
  Eigen::Vector3<T> tgtTriangleBaryCoords;
  T depth = 0;
  T weight = 1;

  template <typename T2>
  VertexConstraintT<T2> cast() const {
    return {
        this->srcVertexIndex,
        this->tgtTriangleIndices,
        this->tgtTriangleBaryCoords.template cast<T2>(),
        static_cast<T2>(this->depth),
        static_cast<T2>(this->weight)};
  }
};

/// Support constraining different parts of the mesh together by specifying that a source vertex
/// should target a location on the mesh defined by a target triangle and its normal.  The target
/// point is specified as sum_i (bary_i * target_triangle_vertex_i) + depth *
/// target_triangle_normal. Note that this constraint applies "forces" both to the source vertex and
/// every vertex in the target triangle, so it will actually try to pull both toward each other.
template <typename T>
class PointTriangleVertexErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit PointTriangleVertexErrorFunctionT(
      const Character& character,
      VertexConstraintType type = VertexConstraintType::Position);
  virtual ~PointTriangleVertexErrorFunctionT() override;

  [[nodiscard]] double getError(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state) final;

  double getGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void addConstraint(
      int vertexIndex,
      const Eigen::Vector3i& triangleIndices,
      const Eigen::Vector3<T>& triangleBaryCoords,
      float depth = 0,
      T weight = 1);
  void clearConstraints();

  [[nodiscard]] const std::vector<PointTriangleVertexConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  static constexpr T kPositionWeight = PositionErrorFunctionT<T>::kLegacyWeight;
  static constexpr T kPlaneWeight = PlaneErrorFunctionT<T>::kLegacyWeight;

  size_t getNumVertices() const;
  [[nodiscard]] const Character& getCharacter() const {
    return character_;
  }

 private:
  double calculatePositionJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const PointTriangleVertexConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  double calculateNormalJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const PointTriangleVertexConstraintT<T>& constr,
      T sourceNormalWeight,
      T targetNormalWeight,
      Ref<Eigen::MatrixX<T>> jac,
      T& res) const;

  double calculatePositionGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const PointTriangleVertexConstraintT<T>& constr,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  double calculateNormalGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const PointTriangleVertexConstraintT<T>& constr,
      T sourceNormalWeight,
      T targetNormalWeight,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  std::pair<T, T> computeNormalWeights() const;

  const Character& character_;

  std::vector<PointTriangleVertexConstraintT<T>> constraints_;

  std::unique_ptr<MeshT<T>>
      neutralMesh_; // Rest mesh without facial expression basis,
                    // used to restore the neutral shape after facial expressions are applied.
                    // Not used with there is a shape basis.
  std::unique_ptr<MeshT<T>> restMesh_; // The rest positions of the mesh after shape basis
                                       // (and potentially facial expression) has been applied
  std::unique_ptr<MeshT<T>>
      posedMesh_; // The posed mesh after the skeleton transforms have been applied.

  const VertexConstraintType constraintType_;

  void updateMeshes(const ModelParametersT<T>& modelParameters, const SkeletonStateT<T>& state);
};

} // namespace momentum
