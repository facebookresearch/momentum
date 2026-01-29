/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skeleton_solver_function.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_base.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
SkeletonSolverFunctionT<T>::SkeletonSolverFunctionT(
    const Character& character,
    const ParameterTransformT<T>& parameterTransform,
    std::span<const std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions)
    : character_(character), parameterTransform_(parameterTransform), needsMeshState_(false) {
  this->numParameters_ = parameterTransform_.numAllModelParameters();
  this->actualParameters_ = this->numParameters_;
  state_ = std::make_unique<SkeletonStateT<T>>(parameterTransform_.zero(), character_.skeleton);
  activeJointParams_ = parameterTransform_.activeJointParams;
  meshState_ = std::make_unique<MeshStateT<T>>();

  for (auto& errf : errorFunctions) {
    addErrorFunction(std::move(errf));
  }
}

template <typename T>
SkeletonSolverFunctionT<T>::~SkeletonSolverFunctionT() = default;

template <typename T>
void SkeletonSolverFunctionT<T>::setEnabledParameters(const ParameterSet& ps) {
  // find the last enabled parameter
  this->actualParameters_ = 0;
  for (size_t i = 0; i < this->numParameters_; i++) {
    if (ps.test(i)) {
      this->actualParameters_ = gsl::narrow_cast<int>(i + 1);
    }
  }
  // set the enabled joints based on the parameter set
  activeJointParams_ = parameterTransform_.computeActiveJointParams(ps);

  // give data to helper functions
  for (auto&& solvable : errorFunctions_) {
    solvable->setActiveJoints(activeJointParams_);
    solvable->setEnabledParameters(ps);
  }
}

template <typename T>
double SkeletonSolverFunctionT<T>::getError(const Eigen::VectorX<T>& parameters) {
  // update the state according to the transformed parameters
  state_->set(parameterTransform_.apply(parameters), character_.skeleton, false);

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }

  double error = 0.0;

  // sum up error for all solvables
  for (auto&& solvable : errorFunctions_) {
    if (solvable->getWeight() > 0.0f) {
      error += (double)solvable->getError(parameters, *state_, *meshState_);
    }
  }

  return (float)error;
}

template <typename T>
double SkeletonSolverFunctionT<T>::getGradient(
    const Eigen::VectorX<T>& parameters,
    Eigen::VectorX<T>& gradient) {
  // update the state according to the transformed parameters
  state_->set(parameterTransform_.apply(parameters), character_.skeleton);

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }

  double error = 0.0;

  // sum up error and gradients for all solvables
  if (gradient.size() != parameters.size()) {
    gradient.resize(parameters.size());
  }
  gradient.setZero();
  for (auto&& solvable : errorFunctions_) {
    if (solvable->getWeight() > 0.0f) {
      error += solvable->getGradient(parameters, *state_, *meshState_, gradient);
    }
  }

  return error;
}

/* override */
template <typename T>
double SkeletonSolverFunctionT<T>::getSolverDerivatives(
    const Eigen::VectorX<T>& parameters,
    Eigen::MatrixX<T>& hess,
    Eigen::VectorX<T>& grad) {
  MT_PROFILE_FUNCTION();

  // update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("UpdateState");
    state_->set(parameterTransform_.apply(parameters), character_.skeleton);
  }

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }

  if (hess.cols() != gsl::narrow_cast<Eigen::Index>(this->actualParameters_)) {
    hess.resize(this->actualParameters_, this->actualParameters_);
    grad.resize(this->actualParameters_);
  }

  hess.setZero();
  grad.setZero();

  double error = 0.0;
  for (size_t i = 0; i < errorFunctions_.size(); i++) {
    const auto& solvable = errorFunctions_[i];
    if (solvable->getWeight() > 0.0f) {
      error += solvable->getSolverDerivatives(
          parameters, *state_, *meshState_, this->actualParameters_, hess, grad);
    }
  }

  return error;
}

template <typename T>
void SkeletonSolverFunctionT<T>::updateParameters(
    Eigen::VectorX<T>& parameters,
    const Eigen::VectorX<T>& delta) {
  // check for sizes
  MT_CHECK(parameters.size() == delta.size());
  parameters -= delta;
}

template <typename T>
void SkeletonSolverFunctionT<T>::addErrorFunction(
    std::shared_ptr<SkeletonErrorFunctionT<T>> solvable) {
  // Update mesh state requirement if this error function needs mesh
  if (solvable->needsMesh()) {
    needsMeshState_ = true;
  }
  errorFunctions_.push_back(std::move(solvable));
}

template <typename T>
void SkeletonSolverFunctionT<T>::clearErrorFunctions() {
  errorFunctions_.clear();
}

template <typename T>
const std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>&
SkeletonSolverFunctionT<T>::getErrorFunctions() const {
  return errorFunctions_;
}

template <typename T>
bool SkeletonSolverFunctionT<T>::needsMeshState() const {
  return needsMeshState_;
}

template <typename T>
void SkeletonSolverFunctionT<T>::updateMeshState(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& state) {
  if (!needsMeshState()) {
    return; // Skip if no error functions need mesh
  }

  meshState_->update(parameters, state, character_);
}

// Block-wise Jacobian interface implementation
template <typename T>
void SkeletonSolverFunctionT<T>::initializeJacobianComputation(
    const Eigen::VectorX<T>& parameters) {
  MT_PROFILE_FUNCTION();

  // Update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("Initialize - update state");
    state_->set(parameterTransform_.apply(parameters), character_.skeleton);
  }

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }
}

template <typename T>
size_t SkeletonSolverFunctionT<T>::getJacobianBlockCount() const {
  return errorFunctions_.size();
}

template <typename T>
size_t SkeletonSolverFunctionT<T>::getJacobianBlockSize(size_t blockIndex) const {
  if (blockIndex >= errorFunctions_.size()) {
    return 0;
  }
  const auto& solvable = errorFunctions_[blockIndex];
  // Return 0 for disabled error functions so we don't need to process it:
  if (solvable->getWeight() <= 0.0f) {
    return 0;
  }
  return solvable->getJacobianSize();
}

template <typename T>
double SkeletonSolverFunctionT<T>::computeJacobianBlock(
    const Eigen::VectorX<T>& parameters,
    size_t blockIndex,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
    Eigen::Ref<Eigen::VectorX<T>> residualBlock,
    size_t& actualRows) {
  MT_PROFILE_FUNCTION();

  if (blockIndex >= errorFunctions_.size()) {
    actualRows = 0;
    return 0.0;
  }

  const auto& solvable = errorFunctions_[blockIndex];

  if (solvable->getWeight() <= 0.0f) {
    actualRows = 0;
    return 0.0;
  }

  int rows = 0;
  const double error =
      solvable->getJacobian(parameters, *state_, *meshState_, jacobianBlock, residualBlock, rows);

  actualRows = static_cast<size_t>(rows);
  return error;
}

template <typename T>
void SkeletonSolverFunctionT<T>::finalizeJacobianComputation() {
  // No cleanup needed
}

template class SkeletonSolverFunctionT<float>;
template class SkeletonSolverFunctionT<double>;

} // namespace momentum
