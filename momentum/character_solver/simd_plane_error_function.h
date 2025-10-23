/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/common/aligned.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace momentum {

struct SimdPlaneConstraints final {
 public:
  explicit SimdPlaneConstraints(const Skeleton* skel);

  ~SimdPlaneConstraints();

  void clearConstraints();

  void addConstraint(
      size_t jointIndex,
      const Vector3f& offset,
      const Vector3f& targetNormal,
      float targetOffset,
      float targetWeight);

  [[nodiscard]] VectorXi getNumConstraints() const;

 public:
  // max number of constraints per segment
  static constexpr size_t kMaxConstraints = 4096;
  static constexpr size_t kMaxJoints = 512;

  // striped arrays for storing the offset
  std::unique_ptr<float, AlignedDeleter> data;
  float* offsetX;
  float* offsetY;
  float* offsetZ;
  // striped arrays for storing the normal + target
  float* normalX;
  float* normalY;
  float* normalZ;
  float* targets;
  // array for constraint weight
  float* weights;

  // store the number of constraints per joint
  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// A highly optimized error function for "point-on-plane" or "point-in-half-plane" errors.
///
/// This function is recommended for use only when dealing with a large number of constraints. For a
/// smaller number of constraints, consider using the generic PlaneErrorFunction.
///
/// @warning Due to the multi-threaded evaluation of the error/gradient, the functions are
/// non-deterministic. This might lead to numerical inconsistencies, resulting in slightly different
/// outcomes on multiple calls with the same data.
class SimdPlaneErrorFunction : public SkeletonErrorFunction {
 public:
  /// @param maxThreads An optional parameter that specifies the maximum number of threads to be
  /// used with dispenso::parallel_for. If this parameter is set to zero, the function will run in
  /// serial mode, i.e., it will not use any additional threads. By default, the value is set to the
  /// maximum allowable size of a uint32_t, which is also the default for dispenso.
  explicit SimdPlaneErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  /// @param maxThreads An optional parameter that specifies the maximum number of threads to be
  /// used with dispenso::parallel_for. If this parameter is set to zero, the function will run in
  /// serial mode, i.e., it will not use any additional threads. By default, the value is set to the
  /// maximum allowable size of a uint32_t, which is also the default for dispenso.
  explicit SimdPlaneErrorFunction(
      const Character& character,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  [[nodiscard]] double getError(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */) override;

  double getGradient(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* meshState */,
      Ref<VectorXf> gradient) override;

  double getJacobian(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* meshState */,
      Ref<MatrixXf> jacobian,
      Ref<VectorXf> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const override;

  void setConstraints(const SimdPlaneConstraints* cstrs);

 protected:
  // weights for the error functions
  static constexpr float kPlaneWeight = 1e-4f;

  mutable std::vector<size_t> jacobianOffset_;

  // constraints to use
  const SimdPlaneConstraints* constraints_;

  uint32_t maxThreads_;
};

#ifdef MOMENTUM_ENABLE_AVX

// A version of SimdPlaneErrorFunction where the Jacobian has been hand-unrolled using
// AVX instructions.  On some platforms this performs better than the generic SIMD version
// but it only works on Intel platforms that support AVX.
class SimdPlaneErrorFunctionAVX : public SimdPlaneErrorFunction {
 public:
  explicit SimdPlaneErrorFunctionAVX(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  explicit SimdPlaneErrorFunctionAVX(
      const Character& character,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  [[nodiscard]] double getError(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */) final;

  double getGradient(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */,
      Ref<VectorXf> gradient) final;

  double getJacobian(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& /* unused */,
      Ref<MatrixXf> jacobian,
      Ref<VectorXf> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;
};

#endif // MOMENTUM_ENABLE_AVX

} // namespace momentum
