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
#include <momentum/math/fwd.h>

namespace momentum {

template <typename T>
struct VertexConstraintT {
  int vertexIndex = -1;
  T weight = 1;
  Eigen::Vector3<T> targetPosition;
  Eigen::Vector3<T> targetNormal;

  template <typename T2>
  VertexConstraintT<T2> cast() const {
    return {
        this->vertexIndex,
        (T)this->weight,
        this->targetPosition.template cast<T2>(),
        this->targetNormal.template cast<T2>()};
  }
};

enum class VertexConstraintType {
  Position, // Target the vertex position
  Plane, // point-to-plane distance using the target normal
  Normal, // point-to-plane distance using the source (body) normal
  SymmetricNormal, // Point-to-plane using a 50/50 mix of source and target normal
};

[[nodiscard]] std::string_view toString(VertexConstraintType type);

template <typename T>
class VertexErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit VertexErrorFunctionT(
      const Character& character,
      VertexConstraintType type = VertexConstraintType::Position,
      size_t maxThreads = 0);
  virtual ~VertexErrorFunctionT() override;

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
      T weight,
      const Eigen::Vector3<T>& targetPosition,
      const Eigen::Vector3<T>& targetNormal);
  void clearConstraints();

  [[nodiscard]] const std::vector<VertexConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  static constexpr T kPositionWeight = PositionErrorFunctionT<T>::kLegacyWeight;
  static constexpr T kPlaneWeight = PlaneErrorFunctionT<T>::kLegacyWeight;

 private:
  double calculatePositionJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const VertexConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  double calculateNormalJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const VertexConstraintT<T>& constr,
      T sourceNormalWeight,
      T targetNormalWeight,
      Ref<Eigen::MatrixX<T>> jac,
      T& res) const;

  double calculatePositionGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const VertexConstraintT<T>& constr,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  double calculateNormalGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const VertexConstraintT<T>& constr,
      T sourceNormalWeight,
      T targetNormalWeight,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  // Utility function used now in calculateNormalJacobian and calculatePositionGradient
  // to calculate derivatives with respect to position in world space (considering skinning)
  void calculateDWorldPos(
      const SkeletonStateT<T>& state,
      const VertexConstraintT<T>& constr,
      const Eigen::Vector3<T>& d_restPos,
      Eigen::Vector3<T>& d_worldPos) const;

  std::pair<T, T> computeNormalWeights() const;

  const Character& character_;

  std::vector<VertexConstraintT<T>> constraints_;

  std::unique_ptr<MeshT<T>>
      neutralMesh_; // Rest mesh without facial expression basis,
                    // used to restore the neutral shape after facial expressions are applied.
                    // Not used with there is a shape basis.
  std::unique_ptr<MeshT<T>> restMesh_; // The rest positions of the mesh after shape basis
                                       // (and potentially facial expression) has been applied
  std::unique_ptr<MeshT<T>>
      posedMesh_; // The posed mesh after the skeleton transforms have been applied.

  const VertexConstraintType constraintType_;

  size_t maxThreads_;

  void updateMeshes(const ModelParametersT<T>& modelParameters, const SkeletonStateT<T>& state);
};

} // namespace momentum
