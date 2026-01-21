/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/gauss_newton_solver.h"

#include "momentum/common/profile.h"
#include "momentum/solver/solver_function.h"

#include <gsl/narrow>

namespace momentum {

template <typename T>
GaussNewtonSolverT<T>::GaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver)
    : SolverT<T>(options, solver) {
  // Set default values from GaussNewtonSolverOptions
  doLineSearch_ = GaussNewtonSolverOptions().doLineSearch;
  regularization_ = GaussNewtonSolverOptions().regularization;
  useBlockJtJ_ = GaussNewtonSolverOptions().useBlockJtJ;
  targetRowsPerChunk_ = GaussNewtonSolverOptions().targetRowsPerChunk;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view GaussNewtonSolverT<T>::getName() const {
  return "GaussNewton";
}

template <typename T>
void GaussNewtonSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const GaussNewtonSolverOptions*>(&options)) {
    regularization_ = derivedOptions->regularization;
    doLineSearch_ = derivedOptions->doLineSearch;
    useBlockJtJ_ = derivedOptions->useBlockJtJ;
    targetRowsPerChunk_ = derivedOptions->targetRowsPerChunk;
  }
}

template <typename T>
void GaussNewtonSolverT<T>::initializeSolver() {
  // This is called from the solver base class .solve()

  // Build enabled parameters mapping from activeParameters_ bitset
  updateEnabledParameters();
}

template <typename T>
void GaussNewtonSolverT<T>::updateEnabledParameters() {
  enabledParameters_.clear();
  enabledParameters_.reserve(this->numParameters_);

  for (int i = 0; i < gsl::narrow_cast<int>(this->numParameters_); ++i) {
    if (this->activeParameters_.test(i)) {
      enabledParameters_.push_back(i);
    }
  }
}

template <typename T>
void GaussNewtonSolverT<T>::computeJtJFromBlockJtJ() {
  MT_PROFILE_EVENT("Get JtJ and JtR");

  // getJtJR() returns JtJ sized up to the highest enabled parameter index (actualParameters_).
  // It doesn't compact the parameters - gaps between enabled parameters still have entries.
  // We do in-place row/column shifting to avoid allocating a temporary matrix.
  this->error_ = this->solverFunction_->getJtJR(this->parameters_, hessianApprox_, JtR_);

  // Compact JtR in-place: shift enabled parameter entries to the front
  const int nSubsetParams = gsl::narrow_cast<int>(enabledParameters_.size());
  for (int iSubset = 0; iSubset < nSubsetParams; ++iSubset) {
    const int iFull = enabledParameters_[iSubset];
    if (iFull > iSubset) {
      JtR_(iSubset) = JtR_(iFull);
    }
  }

  // Compact JtJ in-place: shift rows and columns for enabled parameters to the top-left
  // Process in order so we don't overwrite data we still need (since iFull >= iSubset always)
  for (int iSubset = 0; iSubset < nSubsetParams; ++iSubset) {
    const int iFull = enabledParameters_[iSubset];

    // First, copy the column from iFull to iSubset (for rows 0 to iSubset that are already shifted)
    if (iFull > iSubset) {
      for (int jSubset = 0; jSubset <= iSubset; ++jSubset) {
        const int jFull = enabledParameters_[jSubset];
        // Copy from (jFull, iFull) to (jSubset, iSubset) - but jFull row is already at jSubset
        hessianApprox_(jSubset, iSubset) = hessianApprox_(jFull, iFull);
      }

      // Copy the row from iFull to iSubset (for columns 0 to iSubset-1 that are already shifted)
      for (int jSubset = 0; jSubset < iSubset; ++jSubset) {
        const int jFull = enabledParameters_[jSubset];
        // Copy from (iFull, jFull) to (iSubset, jSubset) - but jFull col is already at jSubset
        hessianApprox_(iSubset, jSubset) = hessianApprox_(iFull, jFull);
      }
    }
  }
}

template <typename T>
void GaussNewtonSolverT<T>::computeJtJFromJacobianBlocks() {
  const int nSubsetParams = gsl::narrow_cast<int>(enabledParameters_.size());

  // Allocate and zero JtJ and JtR for subset parameters
  hessianApprox_.setZero(nSubsetParams, nSubsetParams);
  JtR_.setZero(nSubsetParams);

  this->solverFunction_->initializeJacobianComputation(this->parameters_);

  const size_t numBlocks = this->solverFunction_->getJacobianBlockCount();

  // Calculate block sizes and find the maximum block size
  std::vector<size_t> blockSizes(numBlocks);
  size_t maxBlockSize = 0;
  size_t totalRows = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    blockSizes[i] = this->solverFunction_->getJacobianBlockSize(i);
    maxBlockSize = std::max(maxBlockSize, blockSizes[i]);
    totalRows += blockSizes[i];
  }

  // Effective target: at least the largest block, or all rows if targetRowsPerChunk_ is SIZE_MAX
  const size_t effectiveTarget =
      (targetRowsPerChunk_ == SIZE_MAX) ? totalRows : std::max(targetRowsPerChunk_, maxBlockSize);

  double error = 0.0;

  // Process blocks in chunks using greedy bin-packing
  size_t blockIdx = 0;
  while (blockIdx < numBlocks) {
    // Determine which blocks go into this chunk
    size_t chunkRows = 0;
    size_t chunkStartBlock = blockIdx;

    // Greedily add blocks until we exceed the target
    while (blockIdx < numBlocks) {
      const size_t blockSize = blockSizes[blockIdx];
      if (blockSize == 0) {
        ++blockIdx;
        continue;
      }

      // Add this block if: chunk is empty OR adding it doesn't exceed target
      if (chunkRows == 0 || chunkRows + blockSize <= effectiveTarget) {
        chunkRows += blockSize;
        ++blockIdx;
      } else {
        break;
      }
    }

    if (chunkRows == 0) {
      continue;
    }

    // Resize Jacobian storage for this chunk (ResizeableMatrix avoids reallocations)
    jacobian_.resizeAndSetZero(chunkRows, this->numParameters_);
    residual_.resizeAndSetZero(chunkRows);

    // Fill the chunk with block data
    size_t rowPos = 0;
    for (size_t i = chunkStartBlock; i < blockIdx; ++i) {
      const size_t blockSize = blockSizes[i];
      if (blockSize == 0) {
        continue;
      }

      auto jacBlock = jacobian_.mat().block(rowPos, 0, blockSize, this->numParameters_);
      auto resBlock = residual_.vec().segment(rowPos, blockSize);

      size_t actualRows = 0;
      error += this->solverFunction_->computeJacobianBlock(
          this->parameters_, i, jacBlock, resBlock, actualRows);

      rowPos += blockSize;
    }

    // Remap columns to subset parameter space (in-place, similar to SubsetGaussNewtonSolver)
    // Move the enabled columns of the Jacobian into place.
    // Doing this in-place minimizes total work.
    // For the common case where all "dropped" parameters are at the end,
    // this will effectively be a no-op since enabledParameters_[iSubsetParam] == iSubsetParam.
    for (int iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
      if (enabledParameters_[iSubsetParam] > iSubsetParam) {
        jacobian_.mat().col(iSubsetParam) = jacobian_.mat().col(enabledParameters_[iSubsetParam]);
      }
    }

    // Accumulate JtJ and JtR for this chunk
    const auto J_chunk = jacobian_.mat().leftCols(nSubsetParams);
    const auto r_chunk = residual_.vec();

    hessianApprox_.template triangularView<Eigen::Lower>() += J_chunk.transpose() * J_chunk;
    JtR_.noalias() += J_chunk.transpose() * r_chunk;
  }

  this->solverFunction_->finalizeJacobianComputation();
  this->error_ = error;
}

template <typename T>
void GaussNewtonSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  const int nSubsetParams = gsl::narrow_cast<int>(enabledParameters_.size());

  // Compute JtJ and JtR using either pre-computed JtJ or Jacobian blocks
  // Each code path handles its own matrix sizing
  if (useBlockJtJ_) {
    computeJtJFromBlockJtJ();
  } else {
    computeJtJFromJacobianBlocks();
  }

  // Solve for delta in subset parameter space
  // For BlockJtJ, hessianApprox_ may be larger than nSubsetParams but the compacted
  // data is in the top-left corner. For Jacobian blocks, it's exactly nSubsetParams.
  {
    MT_PROFILE_EVENT("Dense gauss newton step");

    // Get the relevant portion of the matrices
    auto jtjBlock = hessianApprox_.topLeftCorner(nSubsetParams, nSubsetParams);
    auto jtrBlock = JtR_.head(nSubsetParams);

    // Add regularization
    jtjBlock.diagonal().array() += regularization_;

    // Solve
    Eigen::VectorX<T> subsetDelta = llt_.compute(jtjBlock).solve(jtrBlock);

    // Map subset delta back to full parameter space
    Eigen::VectorX<T> delta = Eigen::VectorX<T>::Zero(this->numParameters_);
    for (int iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
      delta(enabledParameters_[iSubsetParam]) = subsetDelta(iSubsetParam);
    }

    updateParameters(delta);
  }

  // Store history (if enabled)
  {
    MT_PROFILE_EVENT("Store history");
    if (this->storeHistory) {
      auto& jtjHist = this->iterationHistory_["jtj"];
      if (jtjHist.rows() !=
              gsl::narrow_cast<Eigen::Index>(this->actualParameters_ * this->getMaxIterations()) ||
          jtjHist.cols() != gsl::narrow_cast<Eigen::Index>(this->actualParameters_)) {
        jtjHist.resize(this->actualParameters_ * this->getMaxIterations(), this->actualParameters_);
      }
      jtjHist.block(
          this->iteration_ * this->actualParameters_,
          0,
          this->actualParameters_,
          this->actualParameters_) =
          hessianApprox_.topLeftCorner(this->actualParameters_, this->actualParameters_);
    }
  }
}

template <typename T>
void GaussNewtonSolverT<T>::updateParameters(Eigen::VectorX<T>& delta) {
  if (!doLineSearch_) {
    MT_PROFILE_EVENT("Update params");
    this->solverFunction_->updateParameters(this->parameters_, delta);
    return;
  }

  MT_PROFILE_EVENT("Line search");

  static constexpr T kC1 = 1e-3;
  static constexpr T kTau = 0.5;
  static constexpr size_t kMaxLineSearchSteps = 10;

  const T scaledError = kC1 * this->error_;
  const Eigen::VectorX<T> parametersOrig = this->parameters_;
  T lineSearchScale = 1.0;

  for (size_t i = 0; i < kMaxLineSearchSteps && std::isnormal(lineSearchScale); ++i) {
    this->parameters_ = parametersOrig;

    this->solverFunction_->updateParameters(this->parameters_, lineSearchScale * delta);

    const double errorNew = this->solverFunction_->getError(this->parameters_);
    if ((this->error_ - errorNew) >= lineSearchScale * scaledError) {
      break;
    }

    // Reduce step size:
    lineSearchScale *= kTau;
  }
}

template class GaussNewtonSolverT<float>;
template class GaussNewtonSolverT<double>;

} // namespace momentum
