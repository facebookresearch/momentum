/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>

#include <vector>

namespace momentum {

/// Multi-threaded scalar-per-joint state-error function (RotationMatrixDifference variant).
///
/// Equivalent to `StateErrorFunctionT<float>` in `character_solver/` with the
/// `RotationMatrixDifference` rotation error. Each joint contributes a 12-element residual
/// (3 translation + 9 rotation matrix entries) parallelized across joints with
/// `dispenso::parallel_for`.
///
/// Lives in `character_solver_simd/` for API consistency. The per-joint math is not SIMD-packet
/// vectorized because state error has exactly one residual per joint (not many constraints
/// per joint as in the position/normal/plane variants), so the existing per-joint constraint-bucket
/// SIMD pattern doesn't apply. SIMD-izing across joints would require a different storage layout
/// and is left as future work.
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdStateErrorFunction : public SkeletonErrorFunction {
 public:
  static constexpr size_t kConstraintDim = 12;

  explicit SimdStateErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdStateErrorFunction(
      const Character& character,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  [[nodiscard]] double getError(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */) override;

  double getGradient(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */,
      Ref<VectorXf> gradient) override;

  double getJacobian(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */,
      Ref<MatrixXf> jacobian,
      Ref<VectorXf> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const override;

  void setTargetState(const SkeletonState& target);

  void setTargetWeights(const Eigen::VectorXf& posWeight, const Eigen::VectorXf& rotWeight);

  void setWeights(float posWeight, float rotWeight) {
    posWgt_ = posWeight;
    rotWgt_ = rotWeight;
  }

 protected:
  /// Match the legacy `StateErrorFunctionT::kPositionWeight` and `kOrientationWeight`.
  static constexpr float kPositionWeight = 1e-3f;
  static constexpr float kOrientationWeight = 1.0f;

  TransformListT<float> targetState_;
  Eigen::VectorXf targetPositionWeights_;
  Eigen::VectorXf targetRotationWeights_;
  float posWgt_{1.0f};
  float rotWgt_{1.0f};

  uint32_t maxThreads_;
};

} // namespace momentum
