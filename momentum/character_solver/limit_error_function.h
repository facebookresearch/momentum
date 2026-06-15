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
#include <momentum/math/generalized_loss.h>

namespace momentum {

/// Penalizes model/joint parameters that violate their configured limits.
///
/// Each limit contributes a squared residual (e.g., how far a parameter is past a min/max bound)
/// that is transformed by the generalized loss function (see math/generalized_loss.h) before being
/// weighted and accumulated. The default loss is L2 (alpha = 2, c = 1), which reproduces the
/// classic quadratic penalty; other alpha values make the penalty robust to outliers.
template <typename T>
class LimitErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Constructor
  ///
  /// @param[in] skel: character skeleton
  /// @param[in] pt: character parameter transform
  /// @param[in] pl: parameter limits to enforce
  /// @param[in] lossAlpha: alpha (shape) parameter for the generalized loss function
  /// @param[in] lossC: c (scale) parameter for the generalized loss function
  LimitErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const ParameterLimits& pl = ParameterLimits(),
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  /// A convenience constructor where character contains info of the skeleton, parameter transform,
  /// and parameter limits.
  ///
  /// @param[in] character: character definition
  /// @param[in] lossAlpha: alpha (shape) parameter for the generalized loss function
  /// @param[in] lossC: c (scale) parameter for the generalized loss function
  explicit LimitErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  /// A convenience constructor where character supplies the skeleton and parameter transform, with
  /// an explicit set of parameter limits.
  ///
  /// @param[in] character: character definition
  /// @param[in] pl: parameter limits to enforce
  /// @param[in] lossAlpha: alpha (shape) parameter for the generalized loss function
  /// @param[in] lossC: c (scale) parameter for the generalized loss function
  LimitErrorFunctionT(
      const Character& character,
      const ParameterLimits& pl,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

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
  /// The generalized loss function applied to each squared limit residual.
  const GeneralizedLossT<T> loss_;

  // Implementations of the three public entry points, templated on whether the loss is a plain L2
  // loss. The L2 path (the default) is compiled with no per-constraint loss-function calls so it
  // matches the original hand-optimized quadratic penalty with no performance regression; the
  // non-L2 path applies the generalized loss. getError/getGradient/getJacobian dispatch to these
  // based on loss_.isL2().
  template <bool IsL2>
  [[nodiscard]] double getErrorImpl(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state) const;

  template <bool IsL2>
  [[nodiscard]] double getGradientImpl(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Ref<Eigen::VectorX<T>> gradient) const;

  template <bool IsL2>
  double getJacobianImpl(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Ref<Eigen::MatrixX<T>> jacobian,
      Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const;
};

} // namespace momentum
