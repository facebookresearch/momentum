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
        static_cast<T2>(this->weight),
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
      uint32_t maxThreads = 0);
  virtual ~VertexErrorFunctionT() override;

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

  void addConstraint(
      int vertexIndex,
      T weight,
      const Eigen::Vector3<T>& targetPosition,
      const Eigen::Vector3<T>& targetNormal);
  void clearConstraints();

  [[nodiscard]] const std::vector<VertexConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  static constexpr T kPositionWeight = PositionErrorFunctionT<T>::kLegacyWeight;
  static constexpr T kPlaneWeight = PlaneErrorFunctionT<T>::kLegacyWeight;

  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  /// Override to indicate this function requires mesh state
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

 private:
  double calculatePositionJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  double calculateNormalJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexConstraintT<T>& constr,
      T sourceNormalWeight,
      T targetNormalWeight,
      Ref<Eigen::MatrixX<T>> jac,
      T& res) const;

  double calculatePositionGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexConstraintT<T>& constr,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  double calculateNormalGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
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

  const VertexConstraintType constraintType_;

  uint32_t maxThreads_;
};

} // namespace momentum
