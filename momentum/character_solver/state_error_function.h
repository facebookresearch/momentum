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
class StateErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  StateErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit StateErrorFunctionT(const Character& character);

  [[nodiscard]] double getError(const ModelParametersT<T>& params, const SkeletonStateT<T>& state)
      final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void reset();
  void setTargetState(const SkeletonStateT<T>* target);
  void setTargetState(const SkeletonStateT<T>& target);
  void setTargetState(TransformListT<T> target);
  void setTargetWeight(const Eigen::VectorX<T>& weights);
  void setTargetWeights(const Eigen::VectorX<T>& posWeight, const Eigen::VectorX<T>& rotWeight);
  void setWeights(const float posWeight, const float rotationWeight) {
    posWgt_ = posWeight;
    rotWgt_ = rotationWeight;
  }

  [[nodiscard]] const TransformListT<T>& getTargetState() const {
    return this->targetState_;
  }

  [[nodiscard]] const Eigen::VectorX<T>& getPositionWeights() const {
    return targetPositionWeights_;
  }
  [[nodiscard]] const Eigen::VectorX<T>& getRotationWeights() const {
    return targetRotationWeights_;
  }
  [[nodiscard]] const T& getPositionWeight() const {
    return posWgt_;
  }
  [[nodiscard]] const T& getRotationWeight() const {
    return rotWgt_;
  }

 private:
  TransformListT<T> targetState_;
  Eigen::VectorX<T> targetPositionWeights_;
  Eigen::VectorX<T> targetRotationWeights_;

  T posWgt_;
  T rotWgt_;

 public:
  // weights for the error functions
  static constexpr T kPositionWeight = 1e-3f;
  static constexpr T kOrientationWeight = 1e+0f;
};

} // namespace momentum
