/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/multipose_solver_function.h"

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
MultiposeSolverFunctionT<T>::MultiposeSolverFunctionT(
    const Skeleton* skel,
    const ParameterTransformT<T>* parameterTransform,
    gsl::span<const int> universal,
    const size_t frames)
    : skeleton_(skel), parameterTransform_(parameterTransform) {
  MT_CHECK(universal.size() == static_cast<size_t>(parameterTransform_->numAllModelParameters()));

  states_.reserve(frames);
  for (size_t i = 0; i < frames; i++) {
    states_.push_back(std::make_unique<SkeletonStateT<T>>());
    states_[states_.size() - 1]->set(parameterTransform_->zero(), *skeleton_);
  }
  errorFunctions_.resize(frames);
  frameParameters_.resize(
      frames, ModelParametersT<T>::Zero(parameterTransform_->numAllModelParameters()));
  activeJointParams_ = parameterTransform_->activeJointParams;

  // calculate needed offsets and indices
  universal_.setZero(universal.size());
  parameterIndexMap_.resize(static_cast<size_t>(parameterTransform_->numAllModelParameters()), 0);
  for (size_t i = 0; i < universal.size(); i++) {
    if (universal[i] != 0) {
      universal_[i] = 1.0f;
      parameterIndexMap_[i] = universalParameters_.size();
      universalParameters_.push_back(i);
    } else {
      parameterIndexMap_[i] = genericParameters_.size();
      genericParameters_.push_back(i);
    }
  }

  this->numParameters_ = genericParameters_.size() * getNumFrames() + universalParameters_.size();
  this->actualParameters_ = this->numParameters_;
}

template <typename T>
void MultiposeSolverFunctionT<T>::setEnabledParameters(const ParameterSet& parameterSet) {
  // find the last enabled parameter
  this->actualParameters_ = 0;
  for (Eigen::Index i = 0; i < parameterTransform_->numAllModelParameters(); i++) {
    if (parameterSet.test(i)) {
      this->actualParameters_ = gsl::narrow_cast<int>(i);
    }
  }
  if (this->actualParameters_ > 0) {
    if (universal_[this->actualParameters_] > 0) {
      this->actualParameters_ = genericParameters_.size() * getNumFrames() +
          parameterIndexMap_[this->actualParameters_] + 1;
    } else {
      this->actualParameters_ = genericParameters_.size() * (getNumFrames() - 1) +
          parameterIndexMap_[this->actualParameters_] + 1;
    }
  }

  // set the enabled joints based on the parameter set
  activeJointParams_ = parameterTransform_->computeActiveJointParams(parameterSet);

  // give data to helper functions
  for (size_t f = 0; f < getNumFrames(); f++) {
    for (auto&& solvable : errorFunctions_[f]) {
      solvable->setActiveJoints(activeJointParams_);
    }
  }
}

template <typename T>
void MultiposeSolverFunctionT<T>::setFrameParameters(
    const size_t frame,
    const ModelParametersT<T>& parameters) {
  MT_CHECK(frame < states_.size());
  MT_CHECK(
      parameters.size() == gsl::narrow<Eigen::Index>(parameterTransform_->numAllModelParameters()));
  frameParameters_[frame] = parameters;
}

template <typename T>
Eigen::VectorX<T> MultiposeSolverFunctionT<T>::getUniversalParameters() const {
  return frameParameters_[0].v.cwiseProduct(universal_);
}

template <typename T>
Eigen::VectorX<T> MultiposeSolverFunctionT<T>::getJoinedParameterVector() const {
  Eigen::VectorX<T> res(this->numParameters_);
  for (size_t f = 0; f < frameParameters_.size(); f++) {
    for (Eigen::Index i = 0; i < parameterTransform_->numAllModelParameters(); i++) {
      // grab parameter from the right position in the index map, either universal at the end or
      // generic for each frame
      if (universal_[i] > 0.0f) {
        res[genericParameters_.size() * getNumFrames() + parameterIndexMap_[i]] =
            frameParameters_[f][i];
      } else {
        res[genericParameters_.size() * f + parameterIndexMap_[i]] = frameParameters_[f][i];
      }
    }
  }

  return res;
}

template <typename T>
void MultiposeSolverFunctionT<T>::setJoinedParameterVector(
    const Eigen::VectorX<T>& joinedParameters) {
  setFrameParametersFromJoinedParameterVector(joinedParameters);
}

template <typename T>
void MultiposeSolverFunctionT<T>::setFrameParametersFromJoinedParameterVector(
    const Eigen::VectorX<T>& parameters) {
  for (size_t f = 0; f < frameParameters_.size(); f++) {
    for (Eigen::Index i = 0; i < parameterTransform_->numAllModelParameters(); i++) {
      // grab parameter from the right position in the index map, either universal at the end or
      // generic for each frame
      if (universal_[i] > 0.0f) {
        frameParameters_[f][i] =
            parameters[genericParameters_.size() * getNumFrames() + parameterIndexMap_[i]];
      } else {
        frameParameters_[f][i] = parameters[genericParameters_.size() * f + parameterIndexMap_[i]];
      }
    }
  }
}

template <typename T>
ParameterSet MultiposeSolverFunctionT<T>::getUniversalParameterSet() const {
  ParameterSet res;

  for (size_t i = 0; i < universalParameters_.size(); i++) {
    res.set(genericParameters_.size() * getNumFrames() + i);
  }

  return res;
}

template <typename T>
ParameterSet MultiposeSolverFunctionT<T>::getUniversalLocatorParameterSet() const {
  ParameterSet res = ParameterSet();

  size_t count = 0;
  for (Eigen::Index i = 0; i < parameterTransform_->numAllModelParameters(); i++) {
    if (universal_[i] > 0.0f) {
      if (parameterTransform_->name[i][0] == '_') {
        res.set(genericParameters_.size() * getNumFrames() + count);
      }
      count++;
    }
  }

  return res;
}

template <typename T>
double MultiposeSolverFunctionT<T>::getError(const Eigen::VectorX<T>& parameters) {
  // update states
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  for (size_t f = 0; f < getNumFrames(); f++) {
    states_[f]->set(parameterTransform_->apply(frameParameters_[f]), *skeleton_);
  }

  double error = 0.0;

  // sum up error for all solvables
  for (size_t f = 0; f < getNumFrames(); f++) {
    for (auto&& solvable : errorFunctions_[f]) {
      if (solvable->getWeight() > 0.0f) {
        error += solvable->getError(frameParameters_[f], *states_[f]);
      }
    }
  }

  return error;
}

template <typename T>
double MultiposeSolverFunctionT<T>::getGradient(
    const Eigen::VectorX<T>& parameters,
    Eigen::VectorX<T>& gradient) {
  // update states
  MT_CHECK(
      universal_.size() ==
      gsl::narrow_cast<Eigen::Index>(parameterTransform_->numAllModelParameters()));
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  for (size_t f = 0; f < getNumFrames(); f++) {
    states_[f]->set(parameterTransform_->apply(frameParameters_[f]), *skeleton_);
  }

  double error = 0.0;
  gradient.setZero(parameters.size());

  // get error and gradients, gradients each into their own slot
  for (size_t f = 0; f < getNumFrames(); f++) {
    Eigen::VectorX<T> subGradient =
        Eigen::VectorX<T>::Zero(parameterTransform_->numAllModelParameters());
    for (auto&& solvable : errorFunctions_[f]) {
      if (solvable->getWeight() > 0.0f) {
        error += solvable->getGradient(frameParameters_[f], *states_[f], subGradient);
      }
    }
    // put the subGradient at the right positions
    for (Eigen::Index i = 0; i < parameterTransform_->numAllModelParameters(); i++) {
      if (universal_[i] > 0.0f) {
        gradient[genericParameters_.size() * getNumFrames() + parameterIndexMap_[i]] +=
            subGradient[i];
      } else {
        gradient[genericParameters_.size() * f + parameterIndexMap_[i]] += subGradient[i];
      }
    }
  }

  return error;
}

template <typename T>
double MultiposeSolverFunctionT<T>::getJacobian(
    const Eigen::VectorX<T>& parameters,
    Eigen::MatrixX<T>& jacobian,
    Eigen::VectorX<T>& residual,
    size_t& actualRows) {
  MT_PROFILE_FUNCTION();
  // update states
  MT_CHECK(
      universal_.size() ==
      gsl::narrow_cast<Eigen::Index>(parameterTransform_->numAllModelParameters()));
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("UpdateState");
    for (size_t f = 0; f < getNumFrames(); f++) {
      states_[f]->set(parameterTransform_->apply(frameParameters_[f]), *skeleton_);
    }
  }

  double error = 0.0;

  // calculate the jacobian size
  size_t jacobianSize = 0;
  size_t maxFrameJacobianSize = 0;
  {
    MT_PROFILE_EVENT("GetJacobianSize");
    for (size_t f = 0; f < getNumFrames(); f++) {
      size_t frameJacobianSize = 0;
      for (auto&& solvable : errorFunctions_[f]) {
        if (solvable->getWeight() > 0.0f) {
          const auto js = solvable->getJacobianSize();
          frameJacobianSize += js;
          jacobianSize += js;
        }
      }

      frameJacobianSize += 8 - (frameJacobianSize % 8);
      maxFrameJacobianSize = std::max(maxFrameJacobianSize, frameJacobianSize);
    }
    jacobianSize += 8 - (jacobianSize % 8);
  }

  if (jacobianSize > static_cast<size_t>(jacobian.rows()) ||
      gsl::narrow_cast<Eigen::Index>(this->numParameters_) != jacobian.cols()) {
    MT_PROFILE_EVENT("ResizeAndInitializeJacobian");
    jacobian.setZero(jacobianSize, this->numParameters_);
    residual.setZero(jacobianSize);
  }
  actualRows = jacobianSize;

  const auto& parameterSize = parameterTransform_->numAllModelParameters();
  Eigen::MatrixX<T> tempJac(maxFrameJacobianSize, parameterSize);

  // add values to the jacobian
  size_t position = 0;
  int rows = 0;
  for (size_t f = 0; f < getNumFrames(); f++) {
    // fill in temporary jacobian
    size_t localPosition = 0;
    const size_t startPosition = position;
    {
      MT_PROFILE_EVENT("GetFrameJacobian");
      {
        MT_PROFILE_EVENT("InitializeTempJacobian");
        tempJac.setZero();
      }
      for (auto&& solvable : errorFunctions_[f]) {
        if (solvable->getWeight() > 0.0f) {
          const size_t n = solvable->getJacobianSize();
          error += solvable->getJacobian(
              frameParameters_[f],
              *states_[f],
              tempJac.block(localPosition, 0, n, parameterSize),
              residual.middleRows(position, n),
              rows);
          localPosition += n;
          position += n;
        }
      }
    }

    // move values to correct columns in actual jacobian according to the structure
    {
      MT_PROFILE_EVENT("MoveJacobianBlocks");
      for (Eigen::Index i = 0; i < parameterSize; i++) {
        if (universal_[i] > 0.0f) {
          jacobian.block(
              startPosition,
              genericParameters_.size() * getNumFrames() + parameterIndexMap_[i],
              localPosition,
              1) = tempJac.block(0, i, localPosition, 1);
        } else {
          jacobian.block(
              startPosition,
              genericParameters_.size() * f + parameterIndexMap_[i],
              localPosition,
              1) = tempJac.block(0, i, localPosition, 1);
        }
      }
    }
  }

  return error;
}

template <typename T>
void MultiposeSolverFunctionT<T>::updateParameters(
    Eigen::VectorX<T>& parameters,
    const Eigen::VectorX<T>& gradient) {
  // check for sizes
  MT_CHECK(parameters.size() == gradient.size());

  // take care of the remaining gradients
  parameters -= gradient;

  // update frame parameters
  setFrameParametersFromJoinedParameterVector(parameters);
}

template <typename T>
void MultiposeSolverFunctionT<T>::addErrorFunction(
    const size_t frame,
    SkeletonErrorFunctionT<T>* errorFunction) {
  MT_CHECK(frame < states_.size());
  errorFunctions_[frame].push_back(errorFunction);
}

template class MultiposeSolverFunctionT<float>;
template class MultiposeSolverFunctionT<double>;

} // namespace momentum
