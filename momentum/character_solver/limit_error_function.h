/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_limits.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>

namespace momentum {

template <typename T>
class LimitErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  LimitErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const ParameterLimits& pl = ParameterLimits());
  explicit LimitErrorFunctionT(const Character& character);
  LimitErrorFunctionT(const Character& character, const ParameterLimits& pl);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<Eigen::MatrixX<T>> jacobian,
      Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void setLimits(const ParameterLimits& lm);
  void setLimits(const Character& character);

 private:
  // weights for the error functions
  static constexpr float kLimitWeight = 1e+1;
  ParameterLimits limits_;
};

} // namespace momentum
