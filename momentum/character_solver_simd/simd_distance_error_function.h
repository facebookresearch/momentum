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

/// Striped storage for distance constraints, with one packed block of slots per joint.
///
/// A distance constraint penalizes the deviation between the world-space distance
/// `||transform(joint, offset) - origin||` and a scalar `target`. The storage layout (one block
/// per scalar field, all blocks SIMD-aligned and sized to the per-joint capacity) mirrors the
/// other SIMD constraint structs so that the same load/store patterns can be used.
struct SimdDistanceConstraints final {
 public:
  explicit SimdDistanceConstraints(const Skeleton* skel);

  ~SimdDistanceConstraints();

  SimdDistanceConstraints(const SimdDistanceConstraints&) = delete;
  SimdDistanceConstraints& operator=(const SimdDistanceConstraints&) = delete;
  SimdDistanceConstraints(SimdDistanceConstraints&&) = delete;
  SimdDistanceConstraints& operator=(SimdDistanceConstraints&&) = delete;

  /// Reset all per-joint constraint counts and weights to zero.
  void clearConstraints();

  /// Append a new constraint for `jointIndex`. Thread-safe via a lock-free CAS loop.
  ///
  /// @param jointIndex Parent joint that owns the constraint.
  /// @param offset Local-space offset on the joint that defines the constrained world point.
  /// @param origin World-space anchor point against which the distance is measured.
  /// @param target Desired scalar distance from `origin` to the constrained world point.
  /// @param targetWeight Per-constraint weight.
  void addConstraint(
      size_t jointIndex,
      const Vector3f& offset,
      const Vector3f& origin,
      float target,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  /// Maximum constraints stored per joint; constraints beyond this are dropped by addConstraint.
  static constexpr size_t kMaxConstraints = 4096;

  /// Maximum joints supported.
  static constexpr size_t kMaxJoints = 512;

  // Single aligned float buffer striped into 8 blocks (offset xyz, origin xyz, target, weight).
  std::unique_ptr<float, AlignedDeleter> data;
  float* offsetX;
  float* offsetY;
  float* offsetZ;
  float* originX;
  float* originY;
  float* originZ;
  float* targets;
  float* weights;

  // Per-joint constraint counters (atomic so addConstraint is safe from concurrent producers).
  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// SIMD-accelerated error function for scalar point-to-point distance constraints.
///
/// Conceptually equivalent to the per-constraint `DistanceErrorFunctionT<float>` in
/// `character_solver/`, but parallelized across joints with `dispenso::parallel_for` and vectorized
/// across constraints within each joint with DrJit packets. Use this when the number of distance
/// constraints is large enough to outweigh the per-call dispatch overhead; otherwise prefer the
/// generic `DistanceErrorFunctionT`.
///
/// @warning Multi-threaded reductions make `getError` / `getGradient` / `getJacobian`
/// non-deterministic; results may differ slightly across calls with identical inputs.
class SimdDistanceErrorFunction : public SkeletonErrorFunction {
 public:
  /// Each constraint produces one residual row.
  static constexpr size_t kConstraintDim = 1;

  /// @param maxThreads Maximum threads dispenso may use; zero forces serial execution. Defaults
  /// to dispenso's own default.
  explicit SimdDistanceErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  /// @param maxThreads Same semantics as the other constructor.
  explicit SimdDistanceErrorFunction(
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

  void setConstraints(const SimdDistanceConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Per-call multiplier applied on top of the per-constraint weight. Matches the legacy
  /// non-SIMD `DistanceErrorFunctionT::kDistanceWeight`.
  static constexpr float kDistanceWeight = 1.0f;

  /// Distances below this threshold skip gradient/Jacobian contribution: the gradient of the
  /// scalar distance w.r.t. the world point is undefined at zero magnitude (singular Jacobian).
  static constexpr float kDistanceEps = 1e-7f;

  mutable std::vector<size_t> jacobianOffset_;

  const SimdDistanceConstraints* constraints_;

  uint32_t maxThreads_;
};

} // namespace momentum
