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

template <typename T>
std::pair<size_t, std::vector<size_t>> getDimensions(
    const std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>& error_func) {
  MT_PROFILE_FUNCTION();
  std::vector<size_t> offset(error_func.size());
  size_t jacobianSize = 0;
  for (size_t i = 0; i < error_func.size(); i++) {
    const auto& solvable = error_func[i];

    // ! offset values are only defined when the cost function is active
    if (solvable->getWeight() > 0.0f) {
      const auto jSize = solvable->getJacobianSize();
      offset[i] = jSize;
      jacobianSize += jSize;
    }
  }
  jacobianSize += 8 - (jacobianSize % 8);
  return std::make_pair(jacobianSize, std::move(offset));
}

template <typename T>
double SkeletonSolverFunctionT<T>::getJacobian(
    const Eigen::VectorX<T>& parameters,
    Eigen::MatrixX<T>& jacobian,
    Eigen::VectorX<T>& residual,
    size_t& actualRows) {
  MT_PROFILE_FUNCTION();

  // update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("Set state");
    state_->set(parameterTransform_.apply(parameters), character_.skeleton);
  }

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }

  double error = 0.0;

  // calculate the jacobian size
  const auto dimensions = getDimensions(errorFunctions_);
  const auto& jacobianSize = dimensions.first;
  const auto& offset = dimensions.second;

  if (jacobianSize > static_cast<size_t>(jacobian.rows()) || parameters.size() != jacobian.cols()) {
    MT_PROFILE_EVENT("ResizeJacobian");
    jacobian.resize(jacobianSize, parameters.size());
    residual.resize(jacobianSize);
  }
  {
    MT_PROFILE_EVENT("InitializeJacobian");
    jacobian.setZero();
    residual.setZero();
  }
  actualRows = jacobianSize;

  // add values to the jacobian
  size_t position = 0;

  MT_PROFILE_EVENT("Collect all jacobians");

  for (size_t i = 0; i < errorFunctions_.size(); i++) {
    int rows = 0;
    const auto& solvable = errorFunctions_[i];
    if (solvable->getWeight() > 0.0f) {
      const auto& n = offset[i];
      error += solvable->getJacobian(
          parameters,
          *state_,
          *meshState_,
          jacobian.block(position, 0, n, parameters.size()),
          residual.middleRows(position, n),
          rows);
      position += n;
    }
  }

  return error;
}

/* override */
template <typename T>
double SkeletonSolverFunctionT<T>::getJtJR(
    const Eigen::VectorX<T>& parameters,
    Eigen::MatrixX<T>& JtJ,
    Eigen::VectorX<T>& JtR) {
  MT_PROFILE_FUNCTION();

  // update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("JtJR - update state");
    state_->set(parameterTransform_.apply(parameters), character_.skeleton);
  }

  // Update mesh state if needed
  if (needsMeshState()) {
    updateMeshState(parameters, *state_);
  }

  // calculate the jacobian size
  const auto dimensions = getDimensions(errorFunctions_);
  const auto& jacobianSize = dimensions.first;
  const auto n_parameters = parameters.size();

  if (dimensions.first > static_cast<size_t>(tJacobian_.rows()) ||
      parameters.size() != tJacobian_.cols()) {
    MT_PROFILE_EVENT("ResizetJacobian");
    tJacobian_.resize(jacobianSize, n_parameters);
    tResidual_.resize(jacobianSize);
  }

  if (JtJ.cols() != gsl::narrow_cast<Eigen::Index>(this->actualParameters_)) {
    JtJ.resize(this->actualParameters_, this->actualParameters_);
    JtR.resize(this->actualParameters_);
  }

  {
    MT_PROFILE_EVENT("JtJR - set to Zero");

    tJacobian_.topRows(jacobianSize).setZero();
    tResidual_.head(jacobianSize).setZero();
    JtJ.setZero();
    JtR.setZero();
  }

  // block compute JtJ and JtR on the fly
  size_t position = 0;
  double error = 0.0;
  for (size_t i = 0; i < errorFunctions_.size(); i++) {
    const auto& solvable = errorFunctions_[i];

    if (solvable->getWeight() > 0.0f) {
      int rows = 0;
      const auto& n = dimensions.second[i];

      // This is timed through the corresponding solvable call
      error += solvable->getJacobian(
          parameters,
          *state_,
          *meshState_,
          tJacobian_.block(position, 0, n, n_parameters),
          tResidual_.segment(position, n),
          rows);
      MT_CHECK(rows <= gsl::narrow_cast<int>(n));

      // Update JtJ
      if (rows > 0) {
        MT_PROFILE_EVENT("Partial JtJ JtR");

        // ! In truth, on the the "this->actualParameters_" leftmost block will be used
        // We take advantage of this here and skip the other computations
        const auto JtBlock =
            tJacobian_.block(position, 0, rows, this->actualParameters_).transpose();

        // Efficiently update JtJ (JtJ = J^T * J) using selfadjointView with rankUpdate,
        // replacing triangularView to leverage symmetry and improve performance
        JtJ.template selfadjointView<Eigen::Lower>().rankUpdate(JtBlock);

        // Update JtR
        JtR.noalias() += JtBlock * tResidual_.segment(position, rows);
      }
      position += n;
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

template class SkeletonSolverFunctionT<float>;
template class SkeletonSolverFunctionT<double>;

} // namespace momentum
