/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/sequence_solver_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/sequence_error_function.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/online_householder_qr.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

template <typename T>
SequenceSolverFunctionT<T>::SequenceSolverFunctionT(
    const Character& character,
    const ParameterTransformT<T>& parameterTransform,
    const ParameterSet& universal,
    const size_t nFrames)
    : character_(character),
      parameterTransform_(parameterTransform),
      universalParameters_(universal) {
  states_.resize(nFrames);
  meshStates_.resize(nFrames);
  perFrameErrorFunctions_.resize(nFrames);
  sequenceErrorFunctions_.resize(nFrames);
  frameParameters_.resize(
      nFrames, ModelParametersT<T>::Zero(parameterTransform_.numAllModelParameters()));
  activeJointParams_ = parameterTransform_.activeJointParams;

  updateParameterSets(allParams());
}

template <typename T>
SequenceSolverFunctionT<T>::~SequenceSolverFunctionT() = default;

template <typename T>
void SequenceSolverFunctionT<T>::setEnabledParameters(const ParameterSet& parameterSet) {
  updateParameterSets(parameterSet);

  // set the enabled joints based on the parameter set
  activeJointParams_ = parameterTransform_.computeActiveJointParams(parameterSet);

  // give data to helper functions
  for (size_t f = 0; f < getNumFrames(); f++) {
    for (auto solvable : perFrameErrorFunctions_[f]) {
      solvable->setActiveJoints(activeJointParams_);
    }
  }

  for (size_t f = 0; f < getNumFrames(); f++) {
    for (const auto& solvable : sequenceErrorFunctions_[f]) {
      solvable->setActiveJoints(activeJointParams_);
    }
  }
}

template <typename T>
void SequenceSolverFunctionT<T>::updateParameterSets(const ParameterSet& activeParams) {
  universalParameterIndices_.clear();
  perFrameParameterIndices_.clear();
  this->numParameters_ = 0;
  this->actualParameters_ = 0;

  const auto np = parameterTransform_.numAllModelParameters();

  // calculate needed offsets and indices
  for (Eigen::Index i = 0; i < np; i++) {
    if (!activeParams.test(i)) {
      continue;
    }

    if (universalParameters_.test(i)) {
      universalParameterIndices_.push_back(i);
    } else {
      perFrameParameterIndices_.push_back(i);
    }
  }

  this->numParameters_ =
      perFrameParameterIndices_.size() * getNumFrames() + universalParameterIndices_.size();
  this->actualParameters_ = this->numParameters_;
}

template <typename T>
void SequenceSolverFunctionT<T>::setFrameParameters(
    const size_t frame,
    const ModelParametersT<T>& parameters) {
  MT_CHECK(frame < states_.size());
  MT_CHECK(
      parameters.size() == gsl::narrow<Eigen::Index>(parameterTransform_.numAllModelParameters()));
  frameParameters_[frame] = parameters;
}

template <typename T>
ModelParametersT<T> SequenceSolverFunctionT<T>::getUniversalParameters() const {
  const auto np = this->parameterTransform_.numAllModelParameters();
  ModelParametersT<T> result = ModelParametersT<T>::Zero(np);
  for (Eigen::Index i = 0; i < np; ++i) {
    if (this->universalParameters_.test(i)) {
      result(i) = frameParameters_[0](i);
    }
  }
  return result;
}

template <typename T>
Eigen::VectorX<T> SequenceSolverFunctionT<T>::getJoinedParameterVectorFromFrameParameters(
    gsl::span<const ModelParametersT<T>> frameParameters) const {
  MT_CHECK(frameParameters.size() == frameParameters_.size());
  const auto nFrames = getNumFrames();

  Eigen::VectorX<T> res = Eigen::VectorX<T>::Zero(this->numParameters_);
  for (size_t f = 0; f < frameParameters.size(); f++) {
    MT_CHECK(frameParameters[f].size() == parameterTransform_.numAllModelParameters());

    // Fill in all the per-frame parameters:
    for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
      res(perFrameParameterIndices_.size() * f + k) =
          frameParameters[f](perFrameParameterIndices_[k]);
    }
  }

  // Then take the universal parameters from the first frame:
  for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
    res(perFrameParameterIndices_.size() * nFrames + k) =
        frameParameters[0](universalParameterIndices_[k]);
  }

  return res;
}

template <typename T>
Eigen::VectorX<T> SequenceSolverFunctionT<T>::getJoinedParameterVector() const {
  return this->getJoinedParameterVectorFromFrameParameters(frameParameters_);
}

template <typename T>
const Skeleton* SequenceSolverFunctionT<T>::getSkeleton() const {
  return &character_.skeleton;
}

template <typename T>
const Character& SequenceSolverFunctionT<T>::getCharacter() const {
  return character_;
}

template <typename T>
void SequenceSolverFunctionT<T>::setJoinedParameterVector(
    const Eigen::VectorX<T>& joinedParameters) {
  setFrameParametersFromJoinedParameterVector(joinedParameters);
}

template <typename T>
void SequenceSolverFunctionT<T>::setFrameParametersFromJoinedParameterVector(
    const Eigen::VectorX<T>& parameters) {
  const auto nFrames = getNumFrames();
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  dispenso::parallel_for(0, nFrames, [&](size_t f) {
    for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
      frameParameters_[f](perFrameParameterIndices_[k]) =
          parameters(perFrameParameterIndices_.size() * f + k);
    }

    for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
      frameParameters_[f](universalParameterIndices_[k]) =
          parameters(perFrameParameterIndices_.size() * nFrames + k);
    }
  });
}

template <typename T>
double SequenceSolverFunctionT<T>::getError(const Eigen::VectorX<T>& parameters) {
  const auto nFrames = getNumFrames();

  // update states
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  dispenso::parallel_for(0, nFrames, [&](size_t f) {
    states_[f].set(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
  });

  // sum up error for all per-frame error functions
  std::vector<double> perFrameErrors(nFrames);
  dispenso::parallel_for(0, nFrames, [&](size_t f) {
    for (const auto& errf : perFrameErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      perFrameErrors[f] += errf->getError(frameParameters_[f], states_[f], meshStates_[f]);
    }
  });

  // Now sum up the sequence error functions:
  dispenso::parallel_for(0, nFrames, [&](size_t f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      const auto nFramesSubset = errf->numFrames();
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      perFrameErrors[f] += errf->getError(
          gsl::make_span(frameParameters_).subspan(f, nFramesSubset),
          gsl::make_span(states_).subspan(f, nFramesSubset),
          gsl::make_span(meshStates_).subspan(f, nFramesSubset));
    }
  });

  return std::accumulate(perFrameErrors.begin(), perFrameErrors.end(), 0.0);
}

template <typename T>
double SequenceSolverFunctionT<T>::getGradient(
    const Eigen::VectorX<T>& parameters,
    Eigen::VectorX<T>& gradient) {
  const auto nFrames = getNumFrames();
  const auto np = this->parameterTransform_.numAllModelParameters();

  // update states
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  dispenso::parallel_for(0, nFrames, [&](size_t f) {
    states_[f].set(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
    if (needsMesh_) {
      meshStates_[f].update(frameParameters_[f], states_[f], character_);
    }
  });

  double error = 0.0;

  // Now accumulate all sequenceErrorFunctions_ which operate on multiple frames.
  Eigen::VectorX<T> fullGradient = Eigen::VectorX<T>::Zero(nFrames * np);
  for (size_t f = 0; f < nFrames; ++f) {
    for (const auto& errf : perFrameErrorFunctions_[f]) {
      if (errf->getWeight() <= 0) {
        continue;
      }
      error += errf->getGradient(
          frameParameters_[f], states_[f], meshStates_[f], fullGradient.segment(f * np, np));
    }
  }

  for (size_t f = 0; f < nFrames; ++f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      const auto nFramesSubset = errf->numFrames();
      error += errf->getGradient(
          gsl::make_span(frameParameters_).subspan(f, nFramesSubset),
          gsl::make_span(states_).subspan(f, nFramesSubset),
          gsl::make_span(meshStates_).subspan(f, nFramesSubset),
          fullGradient.segment(f * np, nFramesSubset * np));
    }
  }

  gradient.setZero(parameters.size());
  for (size_t f = 0; f < frameParameters_.size(); f++) {
    Eigen::Ref<const Eigen::VectorX<T>> subGradient = fullGradient.segment(f * np, np);

    // Fill in all the per-frame parameters:
    for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
      gradient(perFrameParameterIndices_.size() * f + k) +=
          subGradient(perFrameParameterIndices_[k]);
    }

    for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
      gradient(perFrameParameterIndices_.size() * nFrames + k) +=
          subGradient(universalParameterIndices_[k]);
    }
  }

  return error;
}

template <typename T>
double SequenceSolverFunctionT<T>::getJacobian(
    const Eigen::VectorX<T>& parameters,
    Eigen::MatrixX<T>& jacobian,
    Eigen::VectorX<T>& residual,
    size_t& actualRows) {
  const auto np = this->parameterTransform_.numAllModelParameters();
  const auto nFrames = this->getNumFrames();

  MT_PROFILE_EVENT("GetMultiposeJacobian");
  // update states
  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // update the state according to the transformed parameters
  {
    MT_PROFILE_EVENT("UpdateState");
    dispenso::parallel_for(0, nFrames, [&](size_t f) {
      states_[f].set(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
      if (needsMesh_) {
        meshStates_[f].update(frameParameters_[f], states_[f], character_);
      }
    });
  }

  double error = 0.0;

  // calculate the jacobian size
  size_t jacobianSize = 0;
  {
    MT_PROFILE_EVENT("GetJacobianSize");
    for (size_t f = 0; f < getNumFrames(); f++) {
      for (const auto& errf : perFrameErrorFunctions_[f]) {
        if (errf->getWeight() <= 0.0f) {
          continue;
        }

        jacobianSize += errf->getJacobianSize();
      }

      for (const auto& errf : sequenceErrorFunctions_[f]) {
        if (errf->getWeight() <= 0.0f) {
          continue;
        }

        jacobianSize += errf->getJacobianSize();
      }
    }
  }

  if (jacobianSize > static_cast<size_t>(jacobian.rows()) ||
      gsl::narrow_cast<Eigen::Index>(this->numParameters_) != jacobian.cols()) {
    MT_PROFILE_EVENT("ResizeAndInitializeJacobian");
    jacobian.resize(jacobianSize, this->numParameters_);
    residual.resize(jacobianSize);
    jacobian.setZero();
    residual.setZero();
  }
  actualRows = jacobianSize;

  ResizeableMatrix<T> tempJac;

  // add values to the jacobian
  size_t position = 0;
  for (size_t f = 0; f < getNumFrames(); f++) {
    // fill in temporary jacobian
    MT_PROFILE_EVENT("GetFrameJacobian");
    for (const auto& errf : perFrameErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      // TODO pad jacobian
      const auto n = errf->getJacobianSize();
      tempJac.resizeAndSetZero(n, np);

      int rows = 0;
      error += errf->getJacobian(
          frameParameters_[f],
          states_[f],
          meshStates_[f],
          tempJac.mat(),
          residual.middleRows(position, n),
          rows);

      // move values to correct columns in actual jacobian according to the structure
      {
        MT_PROFILE_EVENT("MoveJacobianBlocks");
        for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
          jacobian.block(position, perFrameParameterIndices_.size() * f + k, rows, 1) =
              tempJac.mat().block(0, perFrameParameterIndices_[k], rows, 1);
        }

        for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
          jacobian.block(position, perFrameParameterIndices_.size() * nFrames + k, rows, 1) =
              tempJac.mat().block(0, universalParameterIndices_[k], rows, 1);
        }
      }

      position += rows;
    }
  }

  for (size_t f = 0; f < getNumFrames(); ++f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      const auto nFramesSubset = errf->numFrames();
      const auto js = errf->getJacobianSize();

      tempJac.resizeAndSetZero(js, nFramesSubset * np);
      int rows = 0;
      error += errf->getJacobian(
          gsl::make_span(frameParameters_).subspan(f, nFramesSubset),
          gsl::make_span(states_).subspan(f, nFramesSubset),
          gsl::make_span(meshStates_).subspan(f, nFramesSubset),
          tempJac.mat(),
          residual.middleRows(position, js),
          rows);

      for (size_t iSubframe = 0; iSubframe < nFramesSubset; ++iSubframe) {
        for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
          jacobian.block(
              position, perFrameParameterIndices_.size() * (f + iSubframe) + k, rows, 1) =
              tempJac.mat().block(0, iSubframe * np + perFrameParameterIndices_[k], rows, 1);
        }

        for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
          jacobian.block(position, perFrameParameterIndices_.size() * nFrames + k, rows, 1) +=
              tempJac.mat().block(0, iSubframe * np + universalParameterIndices_[k], rows, 1);
        }
      }

      position += rows;
    }
  }

  return error;
}

template <typename T>
void SequenceSolverFunctionT<T>::updateParameters(
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
void SequenceSolverFunctionT<T>::addErrorFunction(
    const size_t frame,
    std::shared_ptr<SkeletonErrorFunctionT<T>> errorFunction) {
  if (errorFunction->needsMesh()) {
    needsMesh_.store(true, std::memory_order_relaxed);
  }
  if (frame == kAllFrames) {
    dispenso::parallel_for(0, getNumFrames(), [this, errorFunction](size_t iFrame) {
      addErrorFunction(iFrame, errorFunction);
    });
  } else {
    MT_CHECK(frame < states_.size());
    perFrameErrorFunctions_[frame].push_back(errorFunction);
    ++numTotalPerFrameErrorFunctions_;
  }
}

template <typename T>
void SequenceSolverFunctionT<T>::addSequenceErrorFunction(
    const size_t frame,
    std::shared_ptr<SequenceErrorFunctionT<T>> errorFunction) {
  if (errorFunction->needsMesh()) {
    needsMesh_.store(true, std::memory_order_relaxed);
  }
  if (frame == kAllFrames) {
    if (getNumFrames() >= errorFunction->numFrames()) {
      dispenso::parallel_for(
          0, getNumFrames() - errorFunction->numFrames() + 1, [this, errorFunction](size_t iFrame) {
            addSequenceErrorFunction(iFrame, errorFunction);
          });
    }
  } else {
    MT_CHECK(frame < states_.size());
    MT_CHECK(frame + errorFunction->numFrames() <= states_.size());
    sequenceErrorFunctions_[frame].push_back(errorFunction);
    ++numTotalSequenceErrorFunctions_;
  }
}

template class SequenceSolverFunctionT<float>;
template class SequenceSolverFunctionT<double>;

} // namespace momentum
