/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_sequence_solver/fwd.h>

namespace momentum {

template <typename T>
class SequenceErrorFunctionT {
 public:
  SequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt)
      : skeleton_(skel), parameterTransform_(pt), activeJointParams_(pt.activeJointParams) {
    enabledParameters_.flip(); // all parameters enabled by default
  }
  virtual ~SequenceErrorFunctionT() = default;

  // Number of contiguous frames that this error function affects.
  [[nodiscard]] virtual size_t numFrames() const = 0;

  void setWeight(T w) {
    weight_ = w;
  }
  [[nodiscard]] float getWeight() const {
    return weight_;
  }

  void setActiveJoints(const VectorX<bool>& aj) {
    activeJointParams_ = aj;
  }

  void setEnabledParameters(const ParameterSet& ps) {
    enabledParameters_ = ps;
  }

  [[nodiscard]] virtual bool needsMesh() const {
    return false;
  }

  [[nodiscard]] virtual double getError(
      std::span<const ModelParametersT<T>> /* modelParameters */,
      std::span<const SkeletonStateT<T>> /* skelStates */,
      std::span<const MeshStateT<T>> /* meshStates */) const {
    return 0.0f;
  }

  // Get the gradient of the error.
  // modelParameters: numFrames() array of parameter vectors
  // skelStates: [numFrames()] array of skeleton states
  // meshStates: [numFrames()] array of mesh states
  // gradient: [numFrames() * parameterTransform] gradient vector
  virtual double getGradient(
      std::span<const ModelParametersT<T>> /* modelParameters */,
      std::span<const SkeletonStateT<T>> /* skelStates */,
      std::span<const MeshStateT<T>> /* meshStates */,
      Eigen::Ref<Eigen::VectorX<T>> /* gradient */) const {
    return 0.0f;
  }

  // modelParameters: [numFrames() * parameterTransform] parameter vector
  // skelStates: [numFrames()] array of skeleton states
  // meshStates: [numFrames()] array of mesh states
  // jacobian: [getJacobianSize()] x [numFrames() * parameterTransform] Jacobian matrix
  // residual: [getJacobianSize()] residual vector.
  virtual double getJacobian(
      std::span<const ModelParametersT<T>> /* modelParameters */,
      std::span<const SkeletonStateT<T>> /* skelStates */,
      std::span<const MeshStateT<T>> /* meshStates */,
      Eigen::Ref<Eigen::MatrixX<T>> /* jacobian */,
      Eigen::Ref<Eigen::VectorX<T>> /* residual */,
      int& usedRows) const {
    usedRows = 0;
    return 0.0f;
  }

  [[nodiscard]] virtual size_t getJacobianSize() const {
    return 0;
  }

  [[nodiscard]] const Skeleton& getSkeleton() const {
    return this->skeleton_;
  }

  [[nodiscard]] const ParameterTransform& getParameterTransform() const {
    return this->parameterTransform_;
  }

 protected:
  const Skeleton& skeleton_;
  const ParameterTransform& parameterTransform_;
  T weight_ = 1.0f;
  VectorX<bool> activeJointParams_;
  ParameterSet enabledParameters_; // set to zero by default constr
};

} // namespace momentum
