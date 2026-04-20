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

/// Striped storage for aim constraints.
///
/// Each constraint defines a ray (origin and direction) attached to a parent joint, and a global
/// target point. The "Distance" residual is the perpendicular component of the target relative to
/// the ray: `f = (dir·tgtVec)·dir - tgtVec`, where `tgtVec = globalTarget - rayOrigin_world`.
struct SimdAimConstraints final {
 public:
  explicit SimdAimConstraints(const Skeleton* skel);

  ~SimdAimConstraints();

  SimdAimConstraints(const SimdAimConstraints&) = delete;
  SimdAimConstraints& operator=(const SimdAimConstraints&) = delete;
  SimdAimConstraints(SimdAimConstraints&&) = delete;
  SimdAimConstraints& operator=(SimdAimConstraints&&) = delete;

  void clearConstraints();

  /// Append a new constraint for `jointIndex`. Thread-safe via lock-free CAS.
  ///
  /// @param jointIndex Joint that owns the local ray.
  /// @param localPoint Ray origin in the joint's local frame.
  /// @param localDir Ray direction in the joint's local frame (will be normalized).
  /// @param globalTarget Target point in world coordinates.
  /// @param targetWeight Per-constraint weight.
  void addConstraint(
      size_t jointIndex,
      const Vector3f& localPoint,
      const Vector3f& localDir,
      const Vector3f& globalTarget,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  static constexpr size_t kMaxConstraints = 4096;
  static constexpr size_t kMaxJoints = 512;

  // 10 striped float blocks: localPoint (xyz), localDir (xyz), globalTarget (xyz), weight.
  std::unique_ptr<float, AlignedDeleter> data;
  float* localPointX;
  float* localPointY;
  float* localPointZ;
  float* localDirX;
  float* localDirY;
  float* localDirZ;
  float* targetX;
  float* targetY;
  float* targetZ;
  float* weights;

  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// SIMD-accelerated error function for aim-distance (point-to-ray) constraints.
///
/// Equivalent to the per-constraint `AimDistErrorFunctionT<float>` in `character_solver/`. The
/// residual is the perpendicular component of `(globalTarget - rayOrigin_world)` relative to
/// `dir_world` — minimized when the target lies on the ray.
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdAimDistErrorFunction : public SkeletonErrorFunction {
 public:
  static constexpr size_t kConstraintDim = 3;

  explicit SimdAimDistErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdAimDistErrorFunction(
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

  void setConstraints(const SimdAimConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Matches the legacy `AimDistErrorFunctionT::kLegacyWeight = 1e-1f` convention.
  static constexpr float kAimWeight = 1e-1f;

  mutable std::vector<size_t> jacobianOffset_;

  const SimdAimConstraints* constraints_;

  uint32_t maxThreads_;
};

} // namespace momentum
