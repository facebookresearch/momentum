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

#include <array>
#include <atomic>
#include <cstdint>

namespace momentum {

/// Striped storage for fixed-axis constraints.
///
/// A fixed-axis constraint penalizes the deviation between a local-frame axis (rotated by the
/// joint into world space) and a global target axis: `f = R(joint) * localAxis - globalAxis`. The
/// residual is purely rotational (no translation/scale dependency), so only the rotation chain
/// contributes to the gradient.
struct SimdFixedAxisConstraints final {
 public:
  explicit SimdFixedAxisConstraints(const Skeleton* skel);

  ~SimdFixedAxisConstraints();

  SimdFixedAxisConstraints(const SimdFixedAxisConstraints&) = delete;
  SimdFixedAxisConstraints& operator=(const SimdFixedAxisConstraints&) = delete;
  SimdFixedAxisConstraints(SimdFixedAxisConstraints&&) = delete;
  SimdFixedAxisConstraints& operator=(SimdFixedAxisConstraints&&) = delete;

  void clearConstraints();

  /// Append a new constraint for `jointIndex`. Thread-safe via lock-free CAS.
  ///
  /// @param jointIndex Joint that owns the local-axis source.
  /// @param localAxis Source axis in the joint's local frame (no need to be normalized at storage).
  /// @param globalAxis Target axis in the global frame.
  /// @param targetWeight Per-constraint weight.
  void addConstraint(
      size_t jointIndex,
      const Vector3f& localAxis,
      const Vector3f& globalAxis,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  static constexpr size_t kMaxConstraints = 4096;
  static constexpr size_t kMaxJoints = 512;

  // 7 striped float blocks: localAxis (xyz), globalAxis (xyz), weight.
  std::unique_ptr<float, AlignedDeleter> data;
  float* localAxisX;
  float* localAxisY;
  float* localAxisZ;
  float* globalAxisX;
  float* globalAxisY;
  float* globalAxisZ;
  float* weights;

  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// SIMD-accelerated error function for fixed-axis (axis-alignment) constraints.
///
/// Equivalent to the per-constraint `FixedAxisDiffErrorFunctionT<float>` in `character_solver/`.
/// The residual is the element-wise difference between the rotated local axis and the target
/// global axis (3D vector). For Cos and Angle variants, see the Diff variant first; this SIMD
/// version implements the Diff form which is the most common.
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdFixedAxisErrorFunction : public SkeletonErrorFunction {
 public:
  static constexpr size_t kConstraintDim = 3;

  explicit SimdFixedAxisErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdFixedAxisErrorFunction(
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

  void setConstraints(const SimdFixedAxisConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Matches the legacy `FixedAxisDiffErrorFunctionT::kLegacyWeight = 1e-1f` convention used by
  /// the marker tracker.
  static constexpr float kFixedAxisWeight = 1e-1f;

  mutable std::vector<size_t> jacobianOffset_;

  const SimdFixedAxisConstraints* constraints_;

  uint32_t maxThreads_;
};

} // namespace momentum
