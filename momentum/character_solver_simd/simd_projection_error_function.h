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

#include <Eigen/Core>

#include <array>
#include <atomic>
#include <cstdint>

namespace momentum {

/// Striped storage for projection (3x4 camera matrix) constraints.
///
/// Each constraint defines a 3D point on a parent joint plus a precomputed 3x4 projection matrix
/// and a target 2D pixel. The residual is the 2D pixel-space difference between the projected
/// point and the target. Points behind the near clip plane are silently dropped.
struct SimdProjectionConstraints final {
 public:
  explicit SimdProjectionConstraints(const Skeleton* skel);

  ~SimdProjectionConstraints();

  SimdProjectionConstraints(const SimdProjectionConstraints&) = delete;
  SimdProjectionConstraints& operator=(const SimdProjectionConstraints&) = delete;
  SimdProjectionConstraints(SimdProjectionConstraints&&) = delete;
  SimdProjectionConstraints& operator=(SimdProjectionConstraints&&) = delete;

  void clearConstraints();

  /// Append a new constraint for `jointIndex`. Thread-safe via lock-free CAS.
  ///
  /// @param jointIndex Joint that owns the source 3D point.
  /// @param offset Source 3D point in the joint's local frame.
  /// @param projection 3x4 projection matrix mapping homogeneous world points to (rxz, ryz, z).
  /// @param target Target 2D pixel.
  /// @param targetWeight Per-constraint weight.
  void addConstraint(
      size_t jointIndex,
      const Vector3f& offset,
      const Eigen::Matrix<float, 3, 4>& projection,
      const Eigen::Vector2f& target,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  static constexpr size_t kMaxConstraints = 4096;
  static constexpr size_t kMaxJoints = 512;

  // 18 striped float blocks: offset (3), projection (12), target (2), weight (1).
  std::unique_ptr<float, AlignedDeleter> data;
  float* offsetX;
  float* offsetY;
  float* offsetZ;
  // Projection matrix entries (row-major: r0c0, r0c1, ..., r2c3).
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  float* projection[12]{};
  float* targetU;
  float* targetV;
  float* weights;

  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// SIMD-accelerated error function for 2D projection constraints.
///
/// Equivalent to the per-constraint `ProjectionErrorFunctionT<float>` in `character_solver/`. The
/// near-clip behavior is preserved via per-lane masking (lanes whose projected z is below the near
/// clip contribute nothing to error/gradient/Jacobian).
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdProjectionErrorFunction : public SkeletonErrorFunction {
 public:
  static constexpr size_t kConstraintDim = 2;

  explicit SimdProjectionErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      float nearClip = 1.0f,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdProjectionErrorFunction(
      const Character& character,
      float nearClip = 1.0f,
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

  void setConstraints(const SimdProjectionConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Matches the legacy `ProjectionErrorFunctionT::kProjectionWeight = 1.0f` constant.
  static constexpr float kProjectionWeight = 1.0f;

  mutable std::vector<size_t> jacobianOffset_;

  const SimdProjectionConstraints* constraints_;

  float nearClip_;
  uint32_t maxThreads_;
};

} // namespace momentum
