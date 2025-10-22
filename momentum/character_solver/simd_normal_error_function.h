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

namespace momentum {

struct SimdNormalConstraints {
 public:
  explicit SimdNormalConstraints(const Skeleton* skel);
  ~SimdNormalConstraints();

  void clearConstraints();

  bool addConstraint(
      size_t jointIndex,
      const Vector3f& offset,
      const Vector3f& normal,
      const Vector3f& target,
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
  float* targetX;
  float* targetY;
  float* targetZ;
  // array for constraint weight
  float* weights;

  // store the number of constraints per joint
  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount;
  int numJoints;
};

/// A highly optimized error function for "point-to-plane" (signed) distance errors. See
/// NormalErrorFunction for detailed explanation.
///
/// This function should primarily be used when dealing with a large number of constraints. For
/// smaller numbers of constraints, consider using the generic NormalErrorFunction.
///
/// @warning Due to the multi-threaded evaluation of the error/gradient, the functions are
/// non-deterministic, which may cause numerical inconsistencies. As a result, slightly different
/// results may occur on multiple calls with the same data.
class SimdNormalErrorFunction : public SkeletonErrorFunction {
 public:
  /// @param maxThreads An optional parameter that specifies the maximum number of threads to be
  /// used with dispenso::parallel_for. If this parameter is set to zero, the function will run in
  /// serial mode, i.e., it will not use any additional threads. By default, the value is set to the
  /// maximum allowable size of a uint32_t, which is also the default for dispenso.
  explicit SimdNormalErrorFunction(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  /// @param maxThreads An optional parameter that specifies the maximum number of threads to be
  /// used with dispenso::parallel_for. If this parameter is set to zero, the function will run in
  /// serial mode, i.e., it will not use any additional threads. By default, the value is set to the
  /// maximum allowable size of a uint32_t, which is also the default for dispenso.
  explicit SimdNormalErrorFunction(
      const Character& character,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max());

  [[nodiscard]] double getError(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& meshState) final;

  double getGradient(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& meshState,
      Ref<VectorXf> gradient) final;

  double getJacobian(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& meshState,
      Ref<MatrixXf> jacobian,
      Ref<VectorXf> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const override;

  void setConstraints(const SimdNormalConstraints* cstrs) {
    constraints_ = cstrs;
  }

 protected:
  // weights for the error functions
  static constexpr float kPlaneWeight = 1e-4f;

  mutable std::vector<size_t> jacobianOffset_;

  // constraints to use
  const SimdNormalConstraints* constraints_;

  uint32_t maxThreads_;
};

#ifdef MOMENTUM_ENABLE_AVX

// A version of SimdNormalErrorFunction where the Jacobian has been hand-unrolled using
// AVX instructions.  On some platforms this performs better than the generic SIMD version
// but it only works on Intel platforms that support AVX.
class SimdNormalErrorFunctionAVX : public SimdNormalErrorFunction {
 public:
  explicit SimdNormalErrorFunctionAVX(
      const Skeleton& skel,
      const ParameterTransform& pt,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max())
      : SimdNormalErrorFunction(skel, pt, maxThreads) {}
  explicit SimdNormalErrorFunctionAVX(
      const Character& character,
      uint32_t maxThreads = std::numeric_limits<uint32_t>::max())
      : SimdNormalErrorFunction(character, maxThreads) {}

  double getJacobian(
      const ModelParameters& params,
      const SkeletonState& state,
      const MeshState& meshState,
      Ref<MatrixXf> jacobian,
      Ref<VectorXf> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;
};

#endif // MOMENTUM_ENABLE_AVX

} // namespace momentum
