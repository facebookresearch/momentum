/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>

namespace momentum {

template <typename T>
class ModelParametersErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  ModelParametersErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit ModelParametersErrorFunctionT(const Character& character);

  // Create a ModelParametersError that only targets the specified parameters:
  ModelParametersErrorFunctionT(const Character& character, const ParameterSet& active);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;
  double getGradient(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<Eigen::VectorX<T>> gradient) final;
  double getJacobian(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<Eigen::MatrixX<T>> jacobian,
      Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;
  [[nodiscard]] size_t getJacobianSize() const final;

  void setTargetParameters(const ModelParametersT<T>& params, const Eigen::VectorX<T>& weights) {
    this->targetParameters_ = params;
    this->targetWeights_ = weights;
  }

  [[nodiscard]] const ModelParametersT<T>& getTargetParameters() const {
    return this->targetParameters_;
  }

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return this->targetWeights_;
  }

 private:
  ModelParametersT<T> targetParameters_;
  Eigen::VectorX<T> targetWeights_;

 public:
  static constexpr T kMotionWeight = 1e-1;
};

} // namespace momentum
