/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/sequence_solver.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/common/progress_bar.h"
#include "momentum/math/online_householder_qr.h"

#include <dispenso/parallel_for.h>

#include <mutex>

namespace momentum {

template <typename T>
SequenceSolverT<T>::SequenceSolverT(
    const SolverOptions& options,
    SequenceSolverFunctionT<T>* function)
    : SolverT<T>(options, function) {
  // Set default values from SequenceSolverOptions
  regularization_ = SequenceSolverOptions().regularization;
  doLineSearch_ = SequenceSolverOptions().doLineSearch;
  multithreaded_ = SequenceSolverOptions().multithreaded;
  progressBar_ = SequenceSolverOptions().progressBar;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view SequenceSolverT<T>::getName() const {
  return "SequenceSolver";
}

template <typename T>
void SequenceSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const SequenceSolverOptions*>(&options)) {
    this->regularization_ = derivedOptions->regularization;
    this->doLineSearch_ = derivedOptions->doLineSearch;
    this->multithreaded_ = derivedOptions->multithreaded;
    this->progressBar_ = derivedOptions->progressBar;
  }
}

template <typename T>
void SequenceSolverT<T>::initializeSolver() {
  MT_PROFILE_FUNCTION();

  auto* fn = dynamic_cast<SequenceSolverFunctionT<T>*>(this->solverFunction_);
  MT_CHECK(fn != nullptr);

  this->bandwidth_ = 1;
  for (const auto& frameErrf : fn->sequenceErrorFunctions_) {
    for (const auto& errf : frameErrf) {
      this->bandwidth_ = std::max(this->bandwidth_, errf->numFrames());
    }
  }
}

namespace {

template <typename IntType>
IntType padForSSE(IntType value) {
  return value + 8 - (value % 8);
}

template <typename ErrorFunctionType>
size_t computeJacobianSize(std::vector<std::shared_ptr<ErrorFunctionType>>& functions) {
  size_t result = 0;
  for (const auto& f : functions) {
    if (f->getWeight() <= 0) {
      continue;
    }
    result += f->getJacobianSize();
  }
  return result;
}

} // namespace

// Compute the full per-frame Jacobian and update the skeleton state:
template <typename T>
typename SequenceSolverT<T>::JacobianResidual SequenceSolverT<T>::computePerFrameJacobian(
    SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    SkeletonStateT<T>& skelState,
    MeshStateT<T>& meshState) {
  const auto& frameParameters = fn->frameParameters_[iFrame];

  const auto& character = fn->getCharacter();

  skelState.set(fn->parameterTransform_.apply(frameParameters), character.skeleton);

  if (fn->needsMeshPerFrame()) {
    meshState.update(frameParameters, skelState, character);
  }

  const auto nFullParameters = fn->parameterTransform_.numAllModelParameters();
  const auto jacobianSize = padForSSE(computeJacobianSize(fn->perFrameErrorFunctions_[iFrame]));
  Eigen::MatrixX<T> jacobian = Eigen::MatrixX<T>::Zero(jacobianSize, nFullParameters);
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);

  Eigen::Index offset = 0;
  double errorCur = 0;
  for (const auto& errf : fn->perFrameErrorFunctions_[iFrame]) {
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    const auto n = errf->getJacobianSize();

    int rows = 0;
    errorCur += errf->getJacobian(
        frameParameters,
        skelState,
        meshState,
        jacobian.block(offset, 0, n, nFullParameters),
        residual.middleRows(offset, n),
        rows);

    MT_CHECK(rows >= 0 && rows <= static_cast<Eigen::Index>(n));

    offset += rows;
  }

  return {
      .frameIndex = iFrame,
      .jacobian = std::move(jacobian),
      .residual = std::move(residual),
      .error = errorCur,
      .nFunctions = fn->perFrameErrorFunctions_[iFrame].size(),
      .usedRows = offset};
}

template <typename T>
typename SequenceSolverT<T>::JacobianResidual SequenceSolverT<T>::computeSequenceJacobian(
    SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    size_t bandwidth,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> meshStates) {
  const size_t bandwidth_cur = std::min(fn->getNumFrames() - iFrame, bandwidth);
  MT_CHECK(skelStates.size() >= bandwidth_cur);
  MT_CHECK(meshStates.size() >= bandwidth_cur);

  const auto nFullParameters = fn->parameterTransform_.numAllModelParameters();
  const size_t jacobianSize = padForSSE(computeJacobianSize(fn->sequenceErrorFunctions_[iFrame]));
  Eigen::MatrixX<T> jacobian =
      Eigen::MatrixX<T>::Zero(jacobianSize, bandwidth_cur * nFullParameters);
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);

  Eigen::Index offset = 0;
  double errorCur = 0;
  for (const auto& errf : fn->sequenceErrorFunctions_[iFrame]) {
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    const auto nFrames = errf->numFrames();
    MT_CHECK(nFrames <= bandwidth_cur);
    const size_t n = errf->getJacobianSize();
    int rows = 0;
    errorCur += errf->getJacobian(
        std::span(fn->frameParameters_).subspan(iFrame, nFrames),
        skelStates.subspan(0, nFrames),
        meshStates.subspan(0, nFrames),
        jacobian.block(offset, 0, n, nFrames * nFullParameters),
        residual.middleRows(offset, n),
        rows);

    MT_CHECK(rows >= 0 && rows <= static_cast<Eigen::Index>(n));
    offset += rows;
  }

  // Because we have only one copy of the universal parameters, we need to merge all the
  // per-frame Jacobians into the first block.
  for (size_t iUnivParam = 0; iUnivParam < fn->universalParameterIndices_.size(); ++iUnivParam) {
    for (size_t kSubFrame = 1; kSubFrame < bandwidth_cur; ++kSubFrame) {
      jacobian.col(fn->universalParameterIndices_[iUnivParam]) +=
          jacobian.col(kSubFrame * nFullParameters + fn->universalParameterIndices_[iUnivParam]);
    }
  }

  return {
      .frameIndex = iFrame,
      .jacobian = std::move(jacobian),
      .residual = std::move(residual),
      .error = errorCur,
      .nFunctions = fn->sequenceErrorFunctions_[iFrame].size(),
      .usedRows = offset};
}

// Indices into the multi-frame, all-parameters Jacobian that correspond to the submatrix which is
// just the per-frame parameters across multiple frames:
template <typename T>
std::vector<Eigen::Index> SequenceSolverT<T>::buildSequenceColumnIndices(
    const SequenceSolverFunctionT<T>* fn,
    size_t bandwidth) {
  std::vector<Eigen::Index> result;
  for (size_t iSubFrame = 0; iSubFrame < bandwidth; ++iSubFrame) {
    std::transform(
        fn->perFrameParameterIndices_.begin(),
        fn->perFrameParameterIndices_.end(),
        std::back_inserter(result),
        [offset = iSubFrame * fn->parameterTransform_.numAllModelParameters()](
            Eigen::Index kParam) -> Eigen::Index { return kParam + offset; });
  }

  return result;
}

template <typename T>
double SequenceSolverT<T>::processPerFrameErrors_serial(
    SequenceSolverFunctionT<T>* fn,
    OnlineBandedHouseholderQR<T>& qrSolver,
    ProgressBar* progress) {
  SkeletonStateT<T> skelState;
  MeshStateT<T> meshState;
  double errorSum = 0;
  for (size_t iFrame = 0; iFrame < fn->getNumFrames(); ++iFrame) {
    auto jacRes = computePerFrameJacobian(fn, iFrame, skelState, meshState);
    errorSum += jacRes.error;

    if (jacRes.usedRows > 0) {
      qrSolver.addMutating(
          iFrame * fn->perFrameParameterIndices_.size(),
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(
              jacRes.jacobian.topRows(jacRes.usedRows), fn->perFrameParameterIndices_),
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(
              jacRes.jacobian.topRows(jacRes.usedRows), fn->universalParameterIndices_),
          jacRes.residual.topRows(jacRes.usedRows));
    }

    if (progress) {
      progress->increment(jacRes.nFunctions);
    }
  }
  return errorSum;
}

namespace {

template <typename T, typename Comparator>
class PriorityQueue {
 public:
  void push(T value) {
    queue.push_back(std::move(value));
    std::push_heap(queue.begin(), queue.end(), Comparator());
  }

  void pop() {
    std::pop_heap(queue.begin(), queue.end(), Comparator());
    queue.pop_back();
  }

  [[nodiscard]] T& top() {
    return queue.front();
  }

  [[nodiscard]] bool empty() const {
    return queue.empty();
  }

 private:
  std::vector<T> queue;
};

} // namespace

template <typename T>
double SequenceSolverT<T>::processErrorFunctions_parallel(
    const std::function<JacobianResidual(
        size_t,
        SequenceSolverFunctionT<T>*,
        OnlineBandedHouseholderQR<T>&,
        SkeletonStateT<T>,
        MeshStateT<T>&)>& processJac,
    SequenceSolverFunctionT<T>* fn,
    OnlineBandedHouseholderQR<T>& qrSolver,
    ProgressBar* progress) {
  const size_t end = fn->getNumFrames();

  // We will buffer at the end of the pipeline to ensure the universal parts of the matrices are
  // always processed in the same order; this will make it deterministic and simplify debugging.
  PriorityQueue<JacobianResidual, std::greater<>> reorderBuffer;
  std::mutex reorderBufferMutex;

  std::deque<JacobianResidual> readyJacobians;
  std::mutex readyJacobiansMutex;
  std::size_t nextToProcess{0};

  // Mutex to lock access to the universal part of the matrix.
  double errorSum = 0;
  std::mutex qrSolverMutex;

  // The way this pipeline works is:
  //   1. We compute the full Jacobian/residual for each from each frame.
  //   2. We zero out the non-shared parts of the Jacobian in parallel
  //   3. We zero out the shared parts of the Jacobian serially in order in the last stage.
  std::vector<PerFrameStateT<T>> states;
  dispenso::parallel_for(
      states,
      []() -> PerFrameStateT<T> { return PerFrameStateT<T>(); },
      dispenso::makeChunkedRange(0, end, 1),
      [&](PerFrameStateT<T>& state, size_t rangeStart, size_t rangeEnd) {
        for (size_t iFrame = rangeStart; iFrame < rangeEnd; ++iFrame) {
          if (iFrame >= end) {
            // nothing to do:
            return;
          }

          JacobianResidual universalJacRes =
              processJac(iFrame, fn, qrSolver, state.skeletonState, state.meshState);

          // Need to push into the queue:
          std::unique_lock<std::mutex> reorderBufferLock(reorderBufferMutex);
          reorderBuffer.push(std::move(universalJacRes));

          // Move any Jacobians that are "ready" out of the reorder buffer and into the
          // "up-next" queue.
          while (!reorderBuffer.empty() && reorderBuffer.top().frameIndex == nextToProcess) {
            {
              std::unique_lock<std::mutex> readyJacobiansLock(readyJacobiansMutex);
              readyJacobians.push_back(std::move(reorderBuffer.top()));
            }
            nextToProcess++;
            reorderBuffer.pop();
          }
        }

        // Now maybe drain the queue:
        std::unique_lock<std::mutex> qrSolverLock(qrSolverMutex, std::try_to_lock);
        if (qrSolverLock.owns_lock()) {
          while (true) {
            JacobianResidual toProcess;
            {
              std::unique_lock<std::mutex> lock(readyJacobiansMutex);
              if (!readyJacobians.empty()) {
                toProcess = std::move(readyJacobians.front());
                readyJacobians.pop_front();
              }
            }

            MT_CHECK(toProcess.usedRows <= toProcess.residual.rows());

            if (toProcess.frameIndex == SIZE_MAX) {
              break;
            } else {
              if (toProcess.residual.rows() > 0) {
                qrSolver.addMutating(
                    toProcess.jacobian.topRows(toProcess.usedRows),
                    toProcess.residual.topRows(toProcess.usedRows));
              }
              errorSum += toProcess.error;
              if (progress) {
                progress->increment(toProcess.nFunctions);
              }
            }
          }
        }
      });

  // Drain the rest of the queue:
  {
    std::unique_lock<std::mutex> solverLock(qrSolverMutex);

    {
      std::unique_lock<std::mutex> reorderLock(readyJacobiansMutex);
      while (!readyJacobians.empty()) {
        if (readyJacobians.front().usedRows > 0) {
          qrSolver.addMutating(
              readyJacobians.front().jacobian.topRows(readyJacobians.front().usedRows),
              readyJacobians.front().residual.topRows(readyJacobians.front().usedRows));
        }
        errorSum += readyJacobians.front().error;
        if (progress) {
          progress->increment(readyJacobians.front().nFunctions);
        }
        readyJacobians.pop_front();
      }
    }

    {
      std::unique_lock<std::mutex> reorderLock(reorderBufferMutex);
      while (!reorderBuffer.empty()) {
        if (reorderBuffer.top().usedRows > 0) {
          qrSolver.addMutating(
              reorderBuffer.top().jacobian.topRows(reorderBuffer.top().usedRows),
              reorderBuffer.top().residual.topRows(reorderBuffer.top().usedRows));
        }
        errorSum += reorderBuffer.top().error;
        if (progress) {
          progress->increment(reorderBuffer.top().nFunctions);
        }
        reorderBuffer.pop();
      }
    }
  }

  return errorSum;
}

// Process the per-frame error functions that lie along the diagonal:
template <typename T>
double SequenceSolverT<T>::processPerFrameErrors_parallel(
    SequenceSolverFunctionT<T>* fn,
    OnlineBandedHouseholderQR<T>& qrSolver,
    ProgressBar* progress) {
  return processErrorFunctions_parallel(
      [](size_t iFrame,
         SequenceSolverFunctionT<T>* fn,
         OnlineBandedHouseholderQR<T>& qrSolver,
         SkeletonStateT<T> skelState,
         MeshStateT<T>& meshState) -> JacobianResidual {
        // Construct the Jacobian/residual for a single frame and zero out the parts
        // of the Jacobian that only affect that frame; this is safe to do because
        // the non-shared parameters for each frame don't overlap.
        auto jacRes = computePerFrameJacobian(fn, iFrame, skelState, meshState);

        if (jacRes.usedRows > 0) {
          qrSolver.zeroBandedPart(
              iFrame * fn->perFrameParameterIndices_.size(),
              ColumnIndexedMatrix<Eigen::MatrixX<T>>(
                  jacRes.jacobian.topRows(jacRes.usedRows), fn->perFrameParameterIndices_),
              ColumnIndexedMatrix<Eigen::MatrixX<T>>(
                  jacRes.jacobian.topRows(jacRes.usedRows), fn->universalParameterIndices_),
              jacRes.residual.topRows(jacRes.usedRows));
        }

        if (fn->universalParameterIndices_.empty()) {
          // Optimization: don't hang onto the residual if we aren't going to need it later.
          return {
              .frameIndex = iFrame,
              .jacobian = Eigen::MatrixX<T>(),
              .residual = Eigen::VectorX<T>(),
              .error = jacRes.error,
              .nFunctions = jacRes.nFunctions,
              .usedRows = 0};
        } else {
          // Now, pass the Jacobian/residual pair for the universal parameters to be handled by the
          // last stage.  These Jacobians must be handled one at a time because the universal
          // parameters are shared between all frames.
          return {
              .frameIndex = iFrame,
              .jacobian = jacRes.jacobian.topRows(jacRes.usedRows)(
                  Eigen::all, fn->universalParameterIndices_),
              .residual = jacRes.residual.topRows(jacRes.usedRows),
              .error = jacRes.error,
              .nFunctions = jacRes.nFunctions,
              .usedRows = jacRes.usedRows};
        }
      },
      fn,
      qrSolver,
      progress);
}

// Process the "sequence" error functions that span multiple frames:
template <typename T>
double SequenceSolverT<T>::processSequenceErrors_serial(
    SequenceSolverFunctionT<T>* fn,
    OnlineBandedHouseholderQR<T>& qrSolver,
    ProgressBar* progress) {
  if (fn->numTotalSequenceErrorFunctions_ == 0) {
    return 0;
  }

  const size_t bandwidth = qrSolver.bandwidth() / fn->perFrameParameterIndices_.size();
  MT_CHECK(qrSolver.bandwidth() % fn->perFrameParameterIndices_.size() == 0);

  // Our per-frame matrices our now (bandwidth) frames wide, so we need (bandwidth)
  // copies of the perFrameParameterIndices_:
  const auto sequenceColumnIndices = buildSequenceColumnIndices(fn, bandwidth);

  std::vector<SkeletonStateT<T>> skelStates(bandwidth);
  std::vector<MeshStateT<T>> meshStates(bandwidth);

  double errorSum = 0;
  for (size_t iFrame = 0; iFrame < fn->getNumFrames(); ++iFrame) {
    const size_t bandwidth_cur = std::min<size_t>(fn->getNumFrames() - iFrame, bandwidth);

    // Determine how many frames are already valid from the previous iteration
    // First frame: 0 (compute all), subsequent frames: bandwidth_cur - 1 (reuse all but last)
    const size_t numValidFrames = (iFrame == 0) ? 0 : (bandwidth_cur - 1);

    // Shift valid frames left by 1 to reuse already-computed states
    for (size_t kSubFrame = 0; kSubFrame < numValidFrames; ++kSubFrame) {
      skelStates[kSubFrame] = std::move(skelStates[kSubFrame + 1]);
      if (fn->needsMeshSequence()) {
        meshStates[kSubFrame] = std::move(meshStates[kSubFrame + 1]);
      }
    }

    // Compute new frames (all frames if iFrame == 0, or just the last frame if iFrame > 0)
    for (size_t kSubFrame = numValidFrames; kSubFrame < bandwidth_cur; ++kSubFrame) {
      skelStates[kSubFrame].set(
          fn->parameterTransform_.apply(fn->frameParameters_[iFrame + kSubFrame]),
          fn->getCharacter().skeleton);

      if (fn->needsMeshSequence()) {
        meshStates[kSubFrame].update(
            fn->frameParameters_[iFrame + kSubFrame], skelStates[kSubFrame], fn->getCharacter());
      }
    }

    auto jacRes = computeSequenceJacobian(fn, iFrame, bandwidth, skelStates, meshStates);
    errorSum += jacRes.error;

    if (jacRes.usedRows > 0) {
      qrSolver.addMutating(
          iFrame * fn->perFrameParameterIndices_.size(),
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(
              jacRes.jacobian.topRows(jacRes.usedRows),
              std::span(sequenceColumnIndices)
                  .subspan(0, bandwidth_cur * fn->perFrameParameterIndices_.size())),
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(
              jacRes.jacobian.topRows(jacRes.usedRows), fn->universalParameterIndices_),
          jacRes.residual.topRows(jacRes.usedRows));
    }

    if (progress) {
      progress->increment(jacRes.nFunctions);
    }
  }
  return errorSum;
}

template <typename T>
void SequenceSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  auto* fn = dynamic_cast<SequenceSolverFunctionT<T>*>(this->solverFunction_);
  MT_CHECK(fn != nullptr);

  OnlineBandedHouseholderQR<T> qrSolver(
      fn->getNumFrames() * fn->perFrameParameterIndices_.size(),
      fn->universalParameterIndices_.size(),
      this->bandwidth_ * fn->perFrameParameterIndices_.size(),
      std::sqrt(regularization_));

  fn->setFrameParametersFromJoinedParameterVector(this->parameters_);

  std::unique_ptr<ProgressBar> progress;
  if (progressBar_) {
    progress = std::make_unique<ProgressBar>(
        "Solving sequence",
        fn->numTotalPerFrameErrorFunctions_ + fn->numTotalSequenceErrorFunctions_);
  }

  this->error_ = 0;
  if (this->multithreaded_) {
    this->error_ += processPerFrameErrors_parallel(fn, qrSolver, progress.get());

    // Sequence errors still have to be be processed serially, at least for now.
    this->error_ += processSequenceErrors_serial(fn, qrSolver, progress.get());
  } else {
    this->error_ += processPerFrameErrors_serial(fn, qrSolver, progress.get());
    this->error_ += processSequenceErrors_serial(fn, qrSolver, progress.get());
  }

  // Now, extract the delta:
  const Eigen::VectorX<T> searchDir = qrSolver.x_dense();

  const double error_orig = this->error_;
  if (doLineSearch_) {
    const double innerProd = -qrSolver.At_times_b().dot(searchDir);

    // Line search:
    const float c_1 = 1e-4f;
    const float tau = 0.5f;
    float alpha = 1.0f;

    const Eigen::VectorX<T> parameters_orig = this->parameters_;
    for (size_t kStep = 0; kStep < 10 && std::fpclassify(alpha) == FP_NORMAL; ++kStep) {
      // update the this->parameters_
      this->parameters_ = parameters_orig;
      this->solverFunction_->updateParameters(this->parameters_, alpha * searchDir);

      const double error_new = this->solverFunction_->getError(this->parameters_);

      if ((error_orig - error_new) >= c_1 * alpha * -innerProd) {
        break;
      }

      // Reduce step size:
      alpha = alpha * tau;
    }
  } else {
    this->solverFunction_->updateParameters(this->parameters_, searchDir);
  }
}

template class SequenceSolverT<float>;
template class SequenceSolverT<double>;

} // namespace momentum
