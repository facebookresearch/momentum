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

/// Striped storage for orientation constraints (Frobenius / RotationMatrixDifference variant).
///
/// The residual is `||R_joint * R_offset - R_target||_F` (Frobenius norm of the rotation matrix
/// difference). Both `R_offset` and `R_target` are precomputed once at `addConstraint` time and
/// stored column-by-column in striped float blocks (9 floats each). This avoids per-evaluation
/// quaternion-to-matrix conversion in the SIMD inner loop.
struct SimdOrientationConstraints final {
 public:
  explicit SimdOrientationConstraints(const Skeleton* skel);

  ~SimdOrientationConstraints();

  SimdOrientationConstraints(const SimdOrientationConstraints&) = delete;
  SimdOrientationConstraints& operator=(const SimdOrientationConstraints&) = delete;
  SimdOrientationConstraints(SimdOrientationConstraints&&) = delete;
  SimdOrientationConstraints& operator=(SimdOrientationConstraints&&) = delete;

  void clearConstraints();

  /// Append a new constraint for `jointIndex`. Thread-safe via lock-free CAS.
  ///
  /// Quaternion inputs are normalized and converted to rotation matrices on insertion; the SIMD
  /// inner loop only ever touches the matrix form.
  ///
  /// @param jointIndex Joint that owns the source orientation.
  /// @param offset Local-frame offset rotation.
  /// @param target Global-frame target rotation.
  /// @param targetWeight Per-constraint weight.
  void addConstraint(
      size_t jointIndex,
      const Eigen::Quaternionf& offset,
      const Eigen::Quaternionf& target,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  static constexpr size_t kMaxConstraints = 4096;
  static constexpr size_t kMaxJoints = 512;

  // 19 striped float blocks: offset rot matrix (9), target rot matrix (9), weight (1).
  // Layout is column-major: offsetCol0XYZ, offsetCol1XYZ, offsetCol2XYZ, then target the same.
  std::unique_ptr<float, AlignedDeleter> data;
  // Offset rotation matrix columns
  float* offsetC0X;
  float* offsetC0Y;
  float* offsetC0Z;
  float* offsetC1X;
  float* offsetC1Y;
  float* offsetC1Z;
  float* offsetC2X;
  float* offsetC2Y;
  float* offsetC2Z;
  // Target rotation matrix columns
  float* targetC0X;
  float* targetC0Y;
  float* targetC0Z;
  float* targetC1X;
  float* targetC1Y;
  float* targetC1Z;
  float* targetC2X;
  float* targetC2Y;
  float* targetC2Z;
  float* weights;

  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// SIMD-accelerated error function for orientation constraints (Frobenius rotation difference).
///
/// Equivalent to the per-constraint `OrientationErrorFunctionT<float>` in `character_solver/`
/// (RotationMatrixDifference form). Each constraint contributes 3 axis-style residuals (one per
/// rotation matrix column), which SIMD batches across the constraints attached to a joint.
///
/// @warning Multi-threaded reductions make `getError`/`getGradient`/`getJacobian`
/// non-deterministic.
class SimdOrientationErrorFunction : public SkeletonErrorFunction {
 public:
  /// Each constraint produces a 9-element residual (3 axis columns, 3 components each).
  static constexpr size_t kConstraintDim = 9;

  explicit SimdOrientationErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdOrientationErrorFunction(
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

  void setConstraints(const SimdOrientationConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  /// Matches the legacy `OrientationErrorFunctionT::kLegacyWeight = 1e-1f` convention.
  static constexpr float kOrientationWeight = 1e-1f;

  mutable std::vector<size_t> jacobianOffset_;

  const SimdOrientationConstraints* constraints_;

  uint32_t maxThreads_;
};

} // namespace momentum
