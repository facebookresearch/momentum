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

namespace momentum {

template <typename T>
struct VertexProjectionConstraintT {
  int vertexIndex = -1;
  T weight = 1;
  Eigen::Vector2<T> targetPosition;
  Eigen::Matrix<T, 3, 4> projection; // Projection matrix

  template <typename T2>
  VertexProjectionConstraintT<T2> cast() const {
    return {
        this->vertexIndex,
        (T)this->weight,
        this->targetPosition.template cast<T2>(),
    };
  }
};

template <typename T>
class VertexProjectionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit VertexProjectionErrorFunctionT(const Character& character, uint32_t maxThreads = 0);
  ~VertexProjectionErrorFunctionT() override;

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
      const Eigen::Vector2<T>& targetPosition,
      const Eigen::Matrix<T, 3, 4>& projection);
  void clearConstraints();

  [[nodiscard]] const std::vector<VertexProjectionConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  /// Override to indicate this function requires mesh state
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }

 private:
  double calculateJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexProjectionConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  double calculateGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      const VertexProjectionConstraintT<T>& constr,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  // Utility function used now in calculateJacobian and calculateGradient
  // to calculate derivatives with respect to position in world space (considering skinning)
  void calculateDWorldPos(
      const SkeletonStateT<T>& state,
      const VertexProjectionConstraintT<T>& constr,
      const Eigen::Vector3<T>& d_restPos,
      Eigen::Vector3<T>& d_worldPos) const;

  const Character& character_;

  std::vector<VertexProjectionConstraintT<T>> constraints_;

  uint32_t maxThreads_;

  T _nearClip = 1.0f;
};

} // namespace momentum
