/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skeleton_only_solver_function.h"

#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"

namespace momentum {

namespace {

// The mesh state passed to the error functions, which read none. Never destroyed: the mesh state's
// destructor is defined with its update, which links the skinning code.
template <typename T>
const MeshStateT<T>& noMeshState() {
  static const auto* const kNoMesh = new MeshStateT<T>();
  return *kNoMesh;
}

} // namespace

template <typename T>
SkeletonOnlySolverFunctionT<T>::SkeletonOnlySolverFunctionT(
    const Skeleton& skeleton,
    const ParameterTransformT<T>& parameterTransform)
    : skeleton_(skeleton),
      parameterTransform_(parameterTransform),
      state_(std::make_unique<SkeletonStateT<T>>(parameterTransform_.zero(), skeleton_)) {
  this->numParameters_ = parameterTransform_.numAllModelParameters();
  this->actualParameters_ = this->numParameters_;
}

template <typename T>
SkeletonOnlySolverFunctionT<T>::~SkeletonOnlySolverFunctionT() = default;

template <typename T>
void SkeletonOnlySolverFunctionT<T>::setEnabledParameters(const ParameterSet& ps) {
  this->actualParameters_ = 0;
  for (size_t i = 0; i < this->numParameters_; i++) {
    if (ps.test(i)) {
      this->actualParameters_ = i + 1;
    }
  }
  const auto activeJointParams = parameterTransform_.computeActiveJointParams(ps);
  for (auto&& errorFunction : errorFunctions_) {
    errorFunction->setActiveJoints(activeJointParams);
    errorFunction->setEnabledParameters(ps);
  }
}

template <typename T>
double SkeletonOnlySolverFunctionT<T>::getError(const Eigen::VectorX<T>& parameters) {
  state_->set(parameterTransform_.apply(parameters), skeleton_, false);
  double error = 0.0;
  for (auto&& errorFunction : errorFunctions_) {
    if (errorFunction->getWeight() > 0.0f) {
      error += errorFunction->getError(parameters, *state_, noMeshState<T>());
    }
  }
  return error;
}

template <typename T>
double SkeletonOnlySolverFunctionT<T>::getGradient(
    const Eigen::VectorX<T>& parameters,
    Eigen::VectorX<T>& gradient) {
  state_->set(parameterTransform_.apply(parameters), skeleton_);
  gradient.setZero(parameters.size());
  double error = 0.0;
  for (auto&& errorFunction : errorFunctions_) {
    if (errorFunction->getWeight() > 0.0f) {
      error += errorFunction->getGradient(parameters, *state_, noMeshState<T>(), gradient);
    }
  }
  return error;
}

template <typename T>
void SkeletonOnlySolverFunctionT<T>::initializeJacobianComputation(
    const Eigen::VectorX<T>& parameters) {
  state_->set(parameterTransform_.apply(parameters), skeleton_);
}

template <typename T>
size_t SkeletonOnlySolverFunctionT<T>::getJacobianBlockCount() const {
  return errorFunctions_.size();
}

template <typename T>
size_t SkeletonOnlySolverFunctionT<T>::getJacobianBlockSize(size_t blockIndex) const {
  if (blockIndex >= errorFunctions_.size() || errorFunctions_[blockIndex]->getWeight() <= 0.0f) {
    return 0;
  }
  return errorFunctions_[blockIndex]->getJacobianSize();
}

template <typename T>
double SkeletonOnlySolverFunctionT<T>::computeJacobianBlock(
    const Eigen::VectorX<T>& parameters,
    size_t blockIndex,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
    Eigen::Ref<Eigen::VectorX<T>> residualBlock,
    size_t& actualRows) {
  actualRows = 0;
  if (blockIndex >= errorFunctions_.size() || errorFunctions_[blockIndex]->getWeight() <= 0.0f) {
    return 0.0;
  }
  int rows = 0;
  const double error = errorFunctions_[blockIndex]->getJacobian(
      parameters, *state_, noMeshState<T>(), jacobianBlock, residualBlock, rows);
  actualRows = static_cast<size_t>(rows);
  return error;
}

template <typename T>
void SkeletonOnlySolverFunctionT<T>::updateParameters(
    Eigen::VectorX<T>& parameters,
    const Eigen::VectorX<T>& delta) {
  MT_CHECK(parameters.size() == delta.size());
  parameters -= delta;
  if (lowerBounds_.size() == parameters.size()) {
    parameters = parameters.cwiseMax(lowerBounds_).cwiseMin(upperBounds_);
  }
}

template <typename T>
void SkeletonOnlySolverFunctionT<T>::addErrorFunction(
    std::shared_ptr<SkeletonErrorFunctionT<T>> errorFunction) {
  MT_THROW_IF(errorFunction->needsMesh(), "SkeletonOnlySolverFunction has no mesh");
  errorFunctions_.push_back(std::move(errorFunction));
}

template <typename T>
void SkeletonOnlySolverFunctionT<T>::setParameterBounds(
    Eigen::VectorX<T> lower,
    Eigen::VectorX<T> upper) {
  MT_THROW_IF(
      lower.size() != static_cast<Eigen::Index>(this->numParameters_) ||
          upper.size() != lower.size(),
      "Parameter bounds need one entry per model parameter");
  for (Eigen::Index i = 0; i < lower.size(); ++i) {
    MT_THROW_IF(
        !(lower(i) <= upper(i)), "Invalid bounds [{}, {}] for parameter {}", lower(i), upper(i), i);
  }
  lowerBounds_ = std::move(lower);
  upperBounds_ = std::move(upper);
}

template class SkeletonOnlySolverFunctionT<float>;
template class SkeletonOnlySolverFunctionT<double>;

} // namespace momentum
