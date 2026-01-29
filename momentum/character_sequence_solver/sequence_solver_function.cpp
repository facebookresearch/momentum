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
      nFrames_(nFrames),
      universalParameters_(universal) {
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
  MT_CHECK(frame < getNumFrames());
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
    std::span<const ModelParametersT<T>> frameParameters) const {
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

  // sum up error for all per-frame error functions using thread-local states
  std::vector<double> perFrameErrors(nFrames);
  std::vector<PerFrameStateT<T>> states;
  dispenso::parallel_for(
      states,
      []() -> PerFrameStateT<T> { return PerFrameStateT<T>(); },
      dispenso::makeChunkedRange(0, nFrames, 1),
      [&](PerFrameStateT<T>& state, size_t rangeStart, size_t rangeEnd) {
        for (size_t f = rangeStart; f < rangeEnd; ++f) {
          state.skeletonState.set(
              parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
          if (needsMeshPerFrame_) {
            state.meshState.update(frameParameters_[f], state.skeletonState, character_);
          }

          for (const auto& errf : perFrameErrorFunctions_[f]) {
            if (errf->getWeight() <= 0.0f) {
              continue;
            }

            perFrameErrors[f] +=
                errf->getError(frameParameters_[f], state.skeletonState, state.meshState);
          }
        }
      });

  // Now sum up the sequence error functions (single-threaded)
  std::vector<SkeletonStateT<T>> skelStates;
  std::vector<MeshStateT<T>> meshStates;
  for (size_t f = 0; f < nFrames; ++f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      const auto nFramesSubset = errf->numFrames();
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      // Compute skeleton/mesh states for this sequence
      skelStates.resize(nFramesSubset);
      if (needsMeshSequence_) {
        meshStates.resize(nFramesSubset);
      }

      for (size_t kSubFrame = 0; kSubFrame < nFramesSubset; ++kSubFrame) {
        skelStates[kSubFrame].set(
            parameterTransform_.apply(frameParameters_[f + kSubFrame]), character_.skeleton);
        if (needsMeshSequence_) {
          meshStates[kSubFrame].update(
              frameParameters_[f + kSubFrame], skelStates[kSubFrame], character_);
        }
      }

      perFrameErrors[f] += errf->getError(
          std::span(frameParameters_).subspan(f, nFramesSubset),
          std::span(skelStates),
          std::span(meshStates));
    }
  }

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

  double error = 0.0;

  // Accumulate per-frame error functions (single-threaded)
  Eigen::VectorX<T> fullGradient = Eigen::VectorX<T>::Zero(nFrames * np);
  SkeletonStateT<T> skelState;
  MeshStateT<T> meshState;

  for (size_t f = 0; f < nFrames; ++f) {
    skelState.set(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
    if (needsMeshPerFrame_) {
      meshState.update(frameParameters_[f], skelState, character_);
    }

    for (const auto& errf : perFrameErrorFunctions_[f]) {
      if (errf->getWeight() <= 0) {
        continue;
      }
      error += errf->getGradient(
          frameParameters_[f], skelState, meshState, fullGradient.segment(f * np, np));
    }
  }

  // Now accumulate sequence error functions (single-threaded)
  std::vector<SkeletonStateT<T>> skelStates;
  std::vector<MeshStateT<T>> meshStates;
  for (size_t f = 0; f < nFrames; ++f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      const auto nFramesSubset = errf->numFrames();

      // Compute skeleton/mesh states for this sequence
      skelStates.resize(nFramesSubset);
      if (needsMeshSequence_) {
        meshStates.resize(nFramesSubset);
      }

      for (size_t kSubFrame = 0; kSubFrame < nFramesSubset; ++kSubFrame) {
        skelStates[kSubFrame].set(
            parameterTransform_.apply(frameParameters_[f + kSubFrame]), character_.skeleton);
        if (needsMeshSequence_) {
          meshStates[kSubFrame].update(
              frameParameters_[f + kSubFrame], skelStates[kSubFrame], character_);
        }
      }

      error += errf->getGradient(
          std::span(frameParameters_).subspan(f, nFramesSubset),
          std::span(skelStates),
          std::span(meshStates),
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
double SequenceSolverFunctionT<T>::getPerFrameJacobian(
    Eigen::MatrixX<T>& jacobian,
    Eigen::VectorX<T>& residual,
    size_t& position) {
  const auto np = this->parameterTransform_.numAllModelParameters();
  const auto nFrames = this->getNumFrames();

  double error = 0.0;

  ResizeableMatrix<T> tempJac;
  SkeletonStateT<T> skelState;
  MeshStateT<T> meshState;

  for (size_t f = 0; f < nFrames; ++f) {
    skelState.set(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);
    if (needsMeshPerFrame_) {
      meshState.update(frameParameters_[f], skelState, character_);
    }

    for (const auto& errf : perFrameErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      const auto n = errf->getJacobianSize();
      tempJac.resizeAndSetZero(n, np);

      int rows = 0;
      error += errf->getJacobian(
          frameParameters_[f],
          skelState,
          meshState,
          tempJac.mat(),
          residual.middleRows(position, n),
          rows);

      // move values to correct columns in actual jacobian according to the structure
      for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
        jacobian.block(position, perFrameParameterIndices_.size() * f + k, rows, 1) =
            tempJac.mat().block(0, perFrameParameterIndices_[k], rows, 1);
      }

      for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
        jacobian.block(position, perFrameParameterIndices_.size() * nFrames + k, rows, 1) =
            tempJac.mat().block(0, universalParameterIndices_[k], rows, 1);
      }

      position += rows;
    }
  }

  return error;
}

template <typename T>
double SequenceSolverFunctionT<T>::getSequenceErrorFunctionsJacobian(
    Eigen::MatrixX<T>& jacobian,
    Eigen::VectorX<T>& residual,
    size_t& position) {
  const auto np = this->parameterTransform_.numAllModelParameters();
  const auto nFrames = this->getNumFrames();

  double error = 0.0;

  ResizeableMatrix<T> tempJac;
  std::vector<SkeletonStateT<T>> skelStates;
  std::vector<MeshStateT<T>> meshStates;

  for (size_t f = 0; f < getNumFrames(); ++f) {
    for (const auto& errf : sequenceErrorFunctions_[f]) {
      if (errf->getWeight() <= 0.0f) {
        continue;
      }

      const auto nFramesSubset = errf->numFrames();
      const auto js = errf->getJacobianSize();

      // Compute skeleton/mesh states for this sequence
      skelStates.resize(nFramesSubset);
      if (needsMeshSequence_) {
        meshStates.resize(nFramesSubset);
      }

      for (size_t kSubFrame = 0; kSubFrame < nFramesSubset; ++kSubFrame) {
        skelStates[kSubFrame].set(
            parameterTransform_.apply(frameParameters_[f + kSubFrame]), character_.skeleton);
        if (needsMeshSequence_) {
          meshStates[kSubFrame].update(
              frameParameters_[f + kSubFrame], skelStates[kSubFrame], character_);
        }
      }

      tempJac.resizeAndSetZero(js, nFramesSubset * np);
      int rows = 0;
      error += errf->getJacobian(
          std::span(frameParameters_).subspan(f, nFramesSubset),
          std::span(skelStates),
          std::span(meshStates),
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

// Block-wise Jacobian interface implementation
// Each block corresponds to a single frame, containing all per-frame and sequence error functions
template <typename T>
void SequenceSolverFunctionT<T>::initializeJacobianComputation(
    const Eigen::VectorX<T>& parameters) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(parameters.size() == gsl::narrow_cast<Eigen::Index>(this->numParameters_));
  setFrameParametersFromJoinedParameterVector(parameters);

  // Pre-allocate temporary Jacobian storage
  size_t maxBlockSize = 0;
  for (size_t f = 0; f < getNumFrames(); f++) {
    maxBlockSize = std::max(maxBlockSize, getJacobianBlockSize(f));
  }
  const auto np = parameterTransform_.numAllModelParameters();
  // For sequence error functions, we may need nFrames * np columns
  const size_t maxCols = getNumFrames() * np;
  tempJac_.resizeAndSetZero(maxBlockSize, maxCols);
}

template <typename T>
size_t SequenceSolverFunctionT<T>::getJacobianBlockCount() const {
  return getNumFrames();
}

template <typename T>
size_t SequenceSolverFunctionT<T>::getJacobianBlockSize(size_t blockIndex) const {
  if (blockIndex >= getNumFrames()) {
    return 0;
  }

  size_t blockSize = 0;

  // Add per-frame error functions for this frame
  for (const auto& errf : perFrameErrorFunctions_[blockIndex]) {
    if (errf->getWeight() > 0.0f) {
      blockSize += errf->getJacobianSize();
    }
  }

  // Add sequence error functions starting at this frame
  for (const auto& errf : sequenceErrorFunctions_[blockIndex]) {
    if (errf->getWeight() > 0.0f) {
      blockSize += errf->getJacobianSize();
    }
  }

  return blockSize;
}

template <typename T>
double SequenceSolverFunctionT<T>::computeJacobianBlock(
    const Eigen::VectorX<T>& /* parameters */,
    size_t blockIndex,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
    Eigen::Ref<Eigen::VectorX<T>> residualBlock,
    size_t& actualRows) {
  MT_PROFILE_FUNCTION();

  if (blockIndex >= getNumFrames()) {
    actualRows = 0;
    return 0.0;
  }

  const size_t f = blockIndex;
  const size_t nFrames = getNumFrames();
  const size_t np = parameterTransform_.numAllModelParameters();
  double error = 0.0;
  size_t position = 0;

  // Compute skeleton state for this frame
  SkeletonStateT<T> skelState(parameterTransform_.apply(frameParameters_[f]), character_.skeleton);

  // Compute mesh state if needed
  MeshStateT<T> meshState;
  if (needsMeshPerFrame_) {
    meshState.update(frameParameters_[f], skelState, character_);
  }

  // Compute per-frame error functions
  for (const auto& errf : perFrameErrorFunctions_[f]) {
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    const auto n = errf->getJacobianSize();
    tempJac_.resizeAndSetZero(n, np);

    int rows = 0;
    error += errf->getJacobian(
        frameParameters_[f],
        skelState,
        meshState,
        tempJac_.mat(),
        residualBlock.segment(position, n),
        rows);

    // Copy Jacobian into the block using the joined parameter layout
    // Per-frame parameters for frame f start at column: perFrameParameterIndices_.size() * f
    for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
      jacobianBlock.block(position, perFrameParameterIndices_.size() * f + k, rows, 1) =
          tempJac_.mat().block(0, perFrameParameterIndices_[k], rows, 1);
    }

    // Universal parameters are at the end: perFrameParameterIndices_.size() * nFrames + k
    for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
      jacobianBlock.block(position, perFrameParameterIndices_.size() * nFrames + k, rows, 1) =
          tempJac_.mat().block(0, universalParameterIndices_[k], rows, 1);
    }

    position += rows;
  }

  // Compute sequence error functions starting at this frame
  std::vector<SkeletonStateT<T>> skelStates;
  std::vector<MeshStateT<T>> meshStates;

  for (const auto& errf : sequenceErrorFunctions_[f]) {
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    // Cap nFramesSubset to avoid writing past the end of the Jacobian
    const auto nFramesSubset = std::min(errf->numFrames(), nFrames - f);
    const auto js = errf->getJacobianSize();

    // Compute skeleton/mesh states for this sequence
    skelStates.resize(nFramesSubset);
    if (needsMeshSequence_) {
      meshStates.resize(nFramesSubset);
    }
    for (size_t iSubframe = 0; iSubframe < nFramesSubset; ++iSubframe) {
      const size_t iFrame = f + iSubframe;
      skelStates[iSubframe] = SkeletonStateT<T>(
          parameterTransform_.apply(frameParameters_[iFrame]), character_.skeleton);
      if (needsMeshSequence_) {
        meshStates[iSubframe].update(frameParameters_[iFrame], skelStates[iSubframe], character_);
      }
    }

    const size_t seqCols = nFramesSubset * np;
    tempJac_.resizeAndSetZero(js, seqCols);

    int rows = 0;
    error += errf->getJacobian(
        std::span(frameParameters_).subspan(f, nFramesSubset),
        std::span(skelStates),
        std::span(meshStates),
        tempJac_.mat(),
        residualBlock.segment(position, js),
        rows);

    // Scatter the Jacobian into the output block using joined parameter layout
    for (size_t iSubframe = 0; iSubframe < nFramesSubset; ++iSubframe) {
      // Per-frame parameters for this subframe
      for (size_t k = 0; k < perFrameParameterIndices_.size(); ++k) {
        jacobianBlock.block(
            position, perFrameParameterIndices_.size() * (f + iSubframe) + k, rows, 1) =
            tempJac_.mat().block(0, iSubframe * np + perFrameParameterIndices_[k], rows, 1);
      }

      // Universal parameters (accumulate contributions from all subframes)
      for (size_t k = 0; k < universalParameterIndices_.size(); ++k) {
        jacobianBlock.block(position, perFrameParameterIndices_.size() * nFrames + k, rows, 1) +=
            tempJac_.mat().block(0, iSubframe * np + universalParameterIndices_[k], rows, 1);
      }
    }

    position += rows;
  }

  actualRows = position;
  return error;
}

template <typename T>
void SequenceSolverFunctionT<T>::finalizeJacobianComputation() {
  // Release temporary storage
  tempJac_.resizeAndSetZero(0, 0);
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
    needsMeshPerFrame_.store(true, std::memory_order_relaxed);
  }
  if (frame == kAllFrames) {
    dispenso::parallel_for(0, getNumFrames(), [this, errorFunction](size_t iFrame) {
      addErrorFunction(iFrame, errorFunction);
    });
  } else {
    MT_CHECK(frame < getNumFrames());
    perFrameErrorFunctions_[frame].push_back(errorFunction);
    ++numTotalPerFrameErrorFunctions_;
  }
}

template <typename T>
void SequenceSolverFunctionT<T>::addSequenceErrorFunction(
    const size_t frame,
    std::shared_ptr<SequenceErrorFunctionT<T>> errorFunction) {
  if (errorFunction->needsMesh()) {
    needsMeshSequence_.store(true, std::memory_order_relaxed);
  }
  if (frame == kAllFrames) {
    if (getNumFrames() >= errorFunction->numFrames()) {
      dispenso::parallel_for(
          0, getNumFrames() - errorFunction->numFrames() + 1, [this, errorFunction](size_t iFrame) {
            addSequenceErrorFunction(iFrame, errorFunction);
          });
    }
  } else {
    MT_CHECK(frame < getNumFrames());
    MT_CHECK(frame + errorFunction->numFrames() <= getNumFrames());
    sequenceErrorFunctions_[frame].push_back(errorFunction);
    ++numTotalSequenceErrorFunctions_;
  }
}

template class SequenceSolverFunctionT<float>;
template class SequenceSolverFunctionT<double>;

} // namespace momentum
