/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/math/random.h>
#include <momentum/math/transform.h>
#include <momentum/math/types.h>
#include <momentum/solver/fwd.h>

namespace momentum {

/// Function which computes a new set of model parameters such that the character pose is a rigidly
/// transformed version of the original pose.  While there is technically a closed form solution
/// for any given skeleton, this is complicated in momentum because different
/// characters attach the rigid parameters to different joints, so a fully general solution uses IK.
/// However, getting it right requires dealing with local minima issues in Euler angles and other
/// challenges.  So centralizing this functionality in a single place is useful.
///
/// @param[in] ensureContinuousOutput If true, the output model parameters will be solved
/// sequentially so that the resulting output is continuous where possible.  This helps to avoid
/// issues with Euler flipping.
template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const std::vector<TransformT<T>>& transforms,
    bool ensureContinuousOutput = true);

template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const TransformT<T>& transforms);

/// Solver which can be reused to repeatedly apply the transformPose function listed above.
template <typename T>
class PoseTransformSolverT {
 public:
  explicit PoseTransformSolverT(const Character& character);
  ~PoseTransformSolverT();

  /// Applies a rigid transform to the given model parameters.
  /// @param[in,out] modelParameters The model parameters to be transformed.
  /// @param[in] transform The transform to apply.
  /// @param[in] modelParametersPrev The previous pose, which is used to ensure continuity (the
  /// solver will initialize using the previous frame's rigid parameters).
  void transformPose(
      ModelParametersT<T>& modelParameters,
      const TransformT<T>& transform,
      const ModelParametersT<T>& modelParametersPrev = ModelParametersT<T>{});

 private:
  Eigen::Index numModelParametersFull_;

  momentum::Character characterSimplified_;
  momentum::ParameterTransformT<T> parameterTransformSimplified_;
  size_t rootJointSimplified_;
  momentum::ParameterSet rigidParametersSimplified_;
  std::vector<size_t> simplifiedParamToFullParamIdx_;

  momentum::ModelParametersT<T> solvedParametersSimplified_;
  momentum::SkeletonStateT<T> skelStateSimplified_;

  std::shared_ptr<momentum::PositionErrorFunctionT<T>> positionError_;
  std::shared_ptr<momentum::OrientationErrorFunctionT<T>> orientationError_;

  std::unique_ptr<momentum::SkeletonSolverFunctionT<T>> solverFunction_;
  std::unique_ptr<momentum::SolverT<T>> solver_;

  momentum::Random<> rng_;
};

} // namespace momentum
