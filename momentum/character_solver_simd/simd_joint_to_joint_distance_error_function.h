/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/common/aligned.h>

#include <cstdint>
#include <vector>

namespace momentum {

/// Storage for joint-to-joint distance constraints.
///
/// Unlike single-joint SIMD constraints (which stripe by parent joint), each constraint here
/// involves two joints, so storage is a flat per-constraint list. SIMD vectorization across
/// constraints is not practical because each lane would require a different (joint1, joint2)
/// transform pair (gather pattern unfit for the existing SIMD math). This implementation instead
/// uses Dispenso to parallelize *across* constraints, leaving the per-constraint math scalar.
struct SimdJointToJointDistanceConstraints final {
 public:
  void clearConstraints();

  /// Add a constraint between two joints with local offsets.
  ///
  /// @param joint1 Index of the first joint.
  /// @param offset1 Offset from joint1 in joint1's local frame.
  /// @param joint2 Index of the second joint.
  /// @param offset2 Offset from joint2 in joint2's local frame.
  /// @param targetDistance Desired distance between the two world points.
  /// @param weight Per-constraint weight.
  void addConstraint(
      size_t j1,
      const Vector3f& off1,
      size_t j2,
      const Vector3f& off2,
      float targetDistance,
      float weight);

  [[nodiscard]] size_t getNumConstraints() const noexcept {
    return joint1.size();
  }

  /// Flat per-constraint storage; arrays are kept in sync size-wise.
  std::vector<uint32_t> joint1;
  std::vector<uint32_t> joint2;
  std::vector<Vector3f> offset1;
  std::vector<Vector3f> offset2;
  std::vector<float> targetDistance;
  std::vector<float> weight;
};

/// Multi-threaded scalar-per-constraint error function for joint-to-joint distance.
///
/// Functionally equivalent to `JointToJointDistanceErrorFunctionT<float>` in `character_solver/`
/// but parallelized across constraints with `dispenso::parallel_for`. Lives in
/// `character_solver_simd/` for API consistency, even though the inner math is not SIMD-packet
/// vectorized — see the constraint storage doc-comment for the architectural reason.
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdJointToJointDistanceErrorFunction : public SkeletonErrorFunction {
 public:
  static constexpr size_t kConstraintDim = 1;

  explicit SimdJointToJointDistanceErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdJointToJointDistanceErrorFunction(
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

  void setConstraints(const SimdJointToJointDistanceConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Matches `JointToJointDistanceErrorFunctionT::kDistanceWeight = 1e-2f`.
  static constexpr float kDistanceWeight = 1e-2f;

  /// Skip gradient/Jacobian contribution for distances below this threshold (singular Jacobian).
  static constexpr float kDistanceEps = 1e-7f;

  const SimdJointToJointDistanceConstraints* constraints_;

  uint32_t maxThreads_;
};

} // namespace momentum
