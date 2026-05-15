/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/sequence_cholesky_solver.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/common/progress_bar.h"

#include <dispenso/parallel_for.h>

#include <Eigen/Cholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace momentum {

namespace {

template <typename T>
struct JacobianResidual {
  Eigen::MatrixX<T> jacobian;
  Eigen::VectorX<T> residual;
  double error = 0.0;
  size_t nFunctions = 0;
  Eigen::Index usedRows = 0;
  Eigen::Index nBandColumns = 0;
  size_t nFrameColumns = 0;
};

enum class SequenceJacobianSelection { All, InternalToChunk, CrossesChunk };

template <typename T>
bool useSequenceErrorFunction(
    const SequenceErrorFunctionT<T>& errf,
    size_t iFrame,
    size_t chunkEnd,
    SequenceJacobianSelection selection) {
  switch (selection) {
    case SequenceJacobianSelection::All:
      return true;
    case SequenceJacobianSelection::InternalToChunk:
      return iFrame + errf.numFrames() <= chunkEnd;
    case SequenceJacobianSelection::CrossesChunk:
      return iFrame + errf.numFrames() > chunkEnd;
  }
  MT_THROW("Unhandled sequence Jacobian selection");
}

template <typename T>
class NormalEquationBlock {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  NormalEquationBlock(
      Eigen::Index iColOffset,
      Eigen::Index nBand,
      Eigen::Index nCommon,
      Eigen::Index bandwidth,
      T regularization,
      Eigen::Index blockSize = 1)
      : iColOffset_(iColOffset),
        nBand_(nBand),
        bandwidth_(bandwidth),
        blockSize_(blockSize),
        common_(MatrixType::Zero(nBand + nCommon, nCommon)),
        rhs_(VectorType::Zero(nBand + nCommon)) {
    MT_CHECK(nBand == 0 || bandwidth > 0);
    if (blockSize_ > 1 && nBand > 0 && nBand % blockSize_ == 0 && bandwidth % blockSize_ == 0) {
      blockBandwidth_ = (bandwidth - 1) / blockSize_ + 1;
      blockBand_ = MatrixType::Zero(nBand, blockBandwidth_ * blockSize_);
    } else {
      band_ = MatrixType::Zero(bandwidth, nBand);
    }

    if (hasBlockBand()) {
      for (Eigen::Index iBlock = 0; iBlock < nBand / blockSize_; ++iBlock) {
        blockBandBlock(iBlock, iBlock).diagonal().array() += regularization;
      }
    } else {
      for (Eigen::Index iCol = 0; iCol < nBand; ++iCol) {
        bandEntry(iCol, iCol) = regularization;
      }
    }
    if (nCommon > 0) {
      common_.bottomRightCorner(nCommon, nCommon).diagonal().setConstant(regularization);
    }
  }

  [[nodiscard]] Eigen::Index iColOffset() const {
    return iColOffset_;
  }

  [[nodiscard]] Eigen::Index nBand() const {
    return nBand_;
  }

  [[nodiscard]] Eigen::Index nCommon() const {
    return common_.cols();
  }

  [[nodiscard]] Eigen::Index bandwidth() const {
    return bandwidth_;
  }

  [[nodiscard]] bool hasBlockBand() const {
    return blockBandwidth_ > 0;
  }

  [[nodiscard]] Eigen::Index blockSize() const {
    return blockSize_;
  }

  [[nodiscard]] Eigen::Index blockBandwidth() const {
    return blockBandwidth_;
  }

  [[nodiscard]] const MatrixType& band() const {
    return band_;
  }

  [[nodiscard]] const MatrixType& common() const {
    return common_;
  }

  [[nodiscard]] const VectorType& rhs() const {
    return rhs_;
  }

  auto blockBandBlock(Eigen::Index iBlock, Eigen::Index jBlock) {
    MT_CHECK(hasBlockBand());
    MT_CHECK(iBlock >= jBlock);
    const Eigen::Index offset = iBlock - jBlock;
    MT_CHECK(offset < blockBandwidth_);
    return blockBand_.block(iBlock * blockSize_, offset * blockSize_, blockSize_, blockSize_);
  }

  [[nodiscard]] auto blockBandBlock(Eigen::Index iBlock, Eigen::Index jBlock) const {
    MT_CHECK(hasBlockBand());
    MT_CHECK(iBlock >= jBlock);
    const Eigen::Index offset = iBlock - jBlock;
    MT_CHECK(offset < blockBandwidth_);
    return blockBand_.block(iBlock * blockSize_, offset * blockSize_, blockSize_, blockSize_);
  }

  [[nodiscard]] T bandEntry(Eigen::Index iRowLocal, Eigen::Index jColLocal) const {
    MT_CHECK(iRowLocal <= jColLocal);
    MT_CHECK(jColLocal - iRowLocal < bandwidth());
    return band_(bandwidth() + iRowLocal - jColLocal - 1, jColLocal);
  }

  [[nodiscard]] T& bandEntry(Eigen::Index iRowLocal, Eigen::Index jColLocal) {
    MT_CHECK(iRowLocal <= jColLocal);
    MT_CHECK(jColLocal - iRowLocal < bandwidth());
    return band_(bandwidth() + iRowLocal - jColLocal - 1, jColLocal);
  }

  template <typename JacobianDerived, typename ResidualDerived>
  void add(
      Eigen::Index iColOffset,
      const Eigen::MatrixBase<JacobianDerived>& jacobian,
      Eigen::Index nBandColumns,
      const Eigen::MatrixBase<ResidualDerived>& residual,
      size_t targetRowsPerJtJChunk) {
    MT_CHECK(jacobian.rows() == residual.rows());
    MT_CHECK(iColOffset >= iColOffset_);
    MT_CHECK(iColOffset + nBandColumns <= iColOffset_ + nBand());
    MT_CHECK(jacobian.cols() == nBandColumns + nCommon());

    const Eigen::Index nCommonColumns = nCommon();
    const Eigen::Index nRows = jacobian.rows();
    const Eigen::Index rowChunkSize = static_cast<Eigen::Index>(
        std::max<size_t>(1, std::min<size_t>(targetRowsPerJtJChunk, nRows)));
    const Eigen::Index localBandOffset = iColOffset - iColOffset_;

    for (Eigen::Index rowStart = 0; rowStart < nRows; rowStart += rowChunkSize) {
      const Eigen::Index rowsCur = std::min(rowChunkSize, nRows - rowStart);
      const MatrixType jacobianChunk =
          jacobian.derived().middleRows(rowStart, rowsCur).template cast<T>();
      const VectorType residualChunk =
          residual.derived().segment(rowStart, rowsCur).template cast<T>();

      const VectorType localRhs = jacobianChunk.transpose() * residualChunk;
      rhs_.segment(localBandOffset, nBandColumns).noalias() += localRhs.head(nBandColumns);
      if (nCommonColumns > 0) {
        rhs_.tail(nCommonColumns).noalias() += localRhs.tail(nCommonColumns);
      }

      const MatrixType localHessian = jacobianChunk.transpose() * jacobianChunk;
      if (nCommonColumns > 0) {
        common_.middleRows(localBandOffset, nBandColumns).noalias() +=
            localHessian.topRightCorner(nBandColumns, nCommonColumns);
        common_.bottomRows(nCommonColumns).noalias() +=
            localHessian.bottomRightCorner(nCommonColumns, nCommonColumns);
      }

      if (hasBlockBand() && localBandOffset % blockSize_ == 0 && nBandColumns % blockSize_ == 0) {
        const Eigen::Index blockOffset = localBandOffset / blockSize_;
        const Eigen::Index nBlocks = nBandColumns / blockSize_;
        for (Eigen::Index iBlock = 0; iBlock < nBlocks; ++iBlock) {
          for (Eigen::Index jBlock = 0; jBlock <= iBlock; ++jBlock) {
            const Eigen::Index iGlobalBlock = blockOffset + iBlock;
            const Eigen::Index jGlobalBlock = blockOffset + jBlock;
            if (iGlobalBlock - jGlobalBlock >= blockBandwidth_) {
              continue;
            }
            auto block = blockBandBlock(iGlobalBlock, jGlobalBlock);
            for (Eigen::Index i = 0; i < blockSize_; ++i) {
              for (Eigen::Index j = 0; j < blockSize_; ++j) {
                const Eigen::Index iGlobal = iGlobalBlock * blockSize_ + i;
                const Eigen::Index jGlobal = jGlobalBlock * blockSize_ + j;
                if (std::abs(iGlobal - jGlobal) < bandwidth_) {
                  block(i, j) += localHessian(iBlock * blockSize_ + i, jBlock * blockSize_ + j);
                }
              }
            }
          }
        }
      } else {
        for (Eigen::Index iBand = 0; iBand < nBandColumns; ++iBand) {
          const Eigen::Index iLocal = localBandOffset + iBand;
          for (Eigen::Index jBand = iBand; jBand < nBandColumns; ++jBand) {
            const Eigen::Index jLocal = localBandOffset + jBand;
            if (jLocal - iLocal >= bandwidth()) {
              continue;
            }
            bandEntry(iLocal, jLocal) += localHessian(iBand, jBand);
          }
        }
      }
    }
  }

  void mergeBlock(const NormalEquationBlock& rhs) {
    MT_CHECK(nCommon() == rhs.nCommon());
    MT_CHECK(bandwidth() >= rhs.bandwidth());
    MT_CHECK(rhs.iColOffset() >= iColOffset());
    MT_CHECK(rhs.iColOffset() + rhs.nBand() <= iColOffset() + nBand());

    const Eigen::Index localOffset = rhs.iColOffset() - iColOffset();
    if (hasBlockBand() && rhs.hasBlockBand() && blockSize() == rhs.blockSize() &&
        localOffset % blockSize_ == 0 && blockBandwidth() == rhs.blockBandwidth()) {
      blockBand_.middleRows(localOffset, rhs.nBand()).noalias() += rhs.blockBand_;
    } else {
      for (Eigen::Index iCol = 0; iCol < rhs.nBand(); ++iCol) {
        for (Eigen::Index iRow = std::max<Eigen::Index>(0, iCol - rhs.bandwidth() + 1);
             iRow <= iCol;
             ++iRow) {
          bandEntry(localOffset + iRow, localOffset + iCol) += rhs.bandEntry(iRow, iCol);
        }
      }
    }

    if (rhs.nBand() > 0) {
      if (nCommon() > 0) {
        common_.middleRows(localOffset, rhs.nBand()).noalias() += rhs.common().topRows(rhs.nBand());
      }
      rhs_.segment(localOffset, rhs.nBand()).noalias() += rhs.rhs().head(rhs.nBand());
    }
    if (nCommon() > 0) {
      common_.bottomRows(nCommon()).noalias() += rhs.common().bottomRows(rhs.nCommon());
      rhs_.tail(nCommon()).noalias() += rhs.rhs().tail(rhs.nCommon());
    }
  }

 private:
  Eigen::Index iColOffset_ = 0;
  Eigen::Index nBand_ = 0;
  Eigen::Index bandwidth_ = 0;
  Eigen::Index blockSize_ = 1;
  Eigen::Index blockBandwidth_ = 0;
  MatrixType band_;
  MatrixType blockBand_;
  MatrixType common_;
  VectorType rhs_;
};

template <typename T>
void computeState(
    const SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    bool needsMesh,
    SkeletonStateT<T>& skelState,
    MeshStateT<T>& meshState) {
  skelState.set(
      (*fn->getParameterTransform()).apply(fn->getFrameParameters()[iFrame]),
      fn->getCharacter().skeleton);
  if (needsMesh) {
    meshState.update(fn->getFrameParameters()[iFrame], skelState, fn->getCharacter());
  }
}

template <typename T>
Eigen::MatrixX<T> compactJacobianColumns(
    Eigen::Ref<const Eigen::MatrixX<T>> jacobian,
    std::span<const Eigen::Index> bandColumnIndices,
    std::span<const Eigen::Index> commonColumnIndices) {
  Eigen::MatrixX<T> result(
      jacobian.rows(),
      static_cast<Eigen::Index>(bandColumnIndices.size() + commonColumnIndices.size()));
  Eigen::Index iOut = 0;
  for (const Eigen::Index iCol : bandColumnIndices) {
    MT_CHECK(iCol >= 0 && iCol < jacobian.cols());
    result.col(iOut++) = jacobian.col(iCol);
  }
  for (const Eigen::Index iCol : commonColumnIndices) {
    MT_CHECK(iCol >= 0 && iCol < jacobian.cols());
    result.col(iOut++) = jacobian.col(iCol);
  }
  return result;
}

template <typename T>
JacobianResidual<T> computePerFrameJacobian(
    SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    SkeletonStateT<T>& skelState,
    MeshStateT<T>& meshState) {
  computeState(fn, iFrame, fn->needsMeshPerFrame(), skelState, meshState);

  const auto nFullParameters = (*fn->getParameterTransform()).numAllModelParameters();
  const auto jacobianSize =
      padToSimdAlignment(detail::computeJacobianSize(fn->getErrorFunctions()[iFrame]));
  Eigen::MatrixX<T> jacobian = Eigen::MatrixX<T>::Zero(jacobianSize, nFullParameters);
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);

  Eigen::Index offset = 0;
  double errorCur = 0;
  for (const auto& errf : fn->getErrorFunctions()[iFrame]) {
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    const auto n = errf->getJacobianSize();

    int rows = 0;
    errorCur += errf->getJacobian(
        fn->getFrameParameters()[iFrame],
        skelState,
        meshState,
        jacobian.block(offset, 0, n, nFullParameters),
        residual.middleRows(offset, n),
        rows);

    MT_CHECK(rows >= 0 && rows <= static_cast<Eigen::Index>(n));
    offset += rows;
  }

  return {
      .jacobian = compactJacobianColumns<T>(
          jacobian.topRows(offset),
          fn->getPerFrameParameterIndices(),
          fn->getUniversalParameterIndices()),
      .residual = std::move(residual),
      .error = errorCur,
      .nFunctions = fn->getErrorFunctions()[iFrame].size(),
      .usedRows = offset,
      .nBandColumns = static_cast<Eigen::Index>(fn->getPerFrameParameterIndices().size())};
}

template <typename T>
JacobianResidual<T> computeSequenceJacobian(
    SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    size_t bandwidth,
    size_t chunkEnd,
    SequenceJacobianSelection selection) {
  const size_t maxBandwidthCur = std::min(fn->getNumFrames() - iFrame, bandwidth);
  size_t bandwidthCur = 0;
  size_t jacobianSizeRaw = 0;
  size_t nSelectedFunctions = 0;
  for (const auto& errf : fn->getSequenceErrorFunctions()[iFrame]) {
    if (!useSequenceErrorFunction(*errf, iFrame, chunkEnd, selection)) {
      continue;
    }
    ++nSelectedFunctions;
    if (errf->getWeight() <= 0) {
      continue;
    }
    bandwidthCur = std::max(bandwidthCur, errf->numFrames());
    jacobianSizeRaw += errf->getJacobianSize();
  }
  MT_CHECK(bandwidthCur <= maxBandwidthCur);

  if (bandwidthCur == 0) {
    return {
        .jacobian = Eigen::MatrixX<T>::Zero(
            0, static_cast<Eigen::Index>(fn->getUniversalParameterIndices().size())),
        .residual = Eigen::VectorX<T>::Zero(0),
        .error = 0.0,
        .nFunctions = nSelectedFunctions,
        .usedRows = 0,
        .nBandColumns = 0,
        .nFrameColumns = 0};
  }

  std::vector<SkeletonStateT<T>> skelStates(bandwidthCur);
  std::vector<MeshStateT<T>> meshStates(bandwidthCur);
  for (size_t kSubFrame = 0; kSubFrame < bandwidthCur; ++kSubFrame) {
    computeState(
        fn,
        iFrame + kSubFrame,
        fn->needsMeshSequence(),
        skelStates[kSubFrame],
        meshStates[kSubFrame]);
  }

  const auto nFullParameters = (*fn->getParameterTransform()).numAllModelParameters();
  const size_t jacobianSize = padToSimdAlignment(jacobianSizeRaw);
  Eigen::MatrixX<T> jacobian =
      Eigen::MatrixX<T>::Zero(jacobianSize, bandwidthCur * nFullParameters);
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);

  Eigen::Index offset = 0;
  double errorCur = 0;
  for (const auto& errf : fn->getSequenceErrorFunctions()[iFrame]) {
    if (!useSequenceErrorFunction(*errf, iFrame, chunkEnd, selection)) {
      continue;
    }
    if (errf->getWeight() <= 0.0f) {
      continue;
    }

    const auto nFrames = errf->numFrames();
    MT_CHECK(nFrames <= bandwidthCur);
    const size_t n = errf->getJacobianSize();
    int rows = 0;
    errorCur += errf->getJacobian(
        std::span(fn->getFrameParameters()).subspan(iFrame, nFrames),
        std::span<const SkeletonStateT<T>>(skelStates).subspan(0, nFrames),
        std::span<const MeshStateT<T>>(meshStates).subspan(0, nFrames),
        jacobian.block(offset, 0, n, nFrames * nFullParameters),
        residual.middleRows(offset, n),
        rows);

    MT_CHECK(rows >= 0 && rows <= static_cast<Eigen::Index>(n));
    offset += rows;
  }

  // Because we have only one copy of the universal parameters, merge all sub-frame
  // universal Jacobians into the first block.
  for (const auto universalIndex : fn->getUniversalParameterIndices()) {
    for (size_t kSubFrame = 1; kSubFrame < bandwidthCur; ++kSubFrame) {
      jacobian.col(universalIndex) += jacobian.col(kSubFrame * nFullParameters + universalIndex);
    }
  }

  const auto sequenceColumnIndices = detail::buildSequenceColumnIndices(fn, bandwidthCur);
  return {
      .jacobian = compactJacobianColumns<T>(
          jacobian.topRows(offset),
          std::span<const Eigen::Index>(sequenceColumnIndices.data(), sequenceColumnIndices.size()),
          fn->getUniversalParameterIndices()),
      .residual = std::move(residual),
      .error = errorCur,
      .nFunctions = nSelectedFunctions,
      .usedRows = offset,
      .nBandColumns = static_cast<Eigen::Index>(sequenceColumnIndices.size()),
      .nFrameColumns = bandwidthCur};
}

template <typename T, typename NormalEquationScalar>
struct AccumulationResult {
  std::unique_ptr<NormalEquationBlock<NormalEquationScalar>> normalEquations;
  double error = 0.0;
  size_t nFunctions = 0;
};

template <typename T, typename NormalEquationScalar>
AccumulationResult<T, NormalEquationScalar> processChunk(
    SequenceSolverFunctionT<T>* fn,
    size_t chunkStart,
    size_t chunkEnd,
    size_t bandwidthFrames,
    size_t targetRowsPerJtJChunk) {
  const Eigen::Index perFrameParams = fn->getPerFrameParameterIndices().size();
  const Eigen::Index storageBlockSize = std::max<Eigen::Index>(1, perFrameParams);
  const Eigen::Index nCommon = fn->getUniversalParameterIndices().size();
  const Eigen::Index nBand = (chunkEnd - chunkStart) * perFrameParams;
  const Eigen::Index bandwidth = bandwidthFrames * perFrameParams;
  auto block = std::make_unique<NormalEquationBlock<NormalEquationScalar>>(
      chunkStart * perFrameParams,
      nBand,
      nCommon,
      bandwidth,
      NormalEquationScalar(0),
      storageBlockSize);

  double error = 0.0;
  size_t nFunctions = 0;

  SkeletonStateT<T> skelState;
  MeshStateT<T> meshState;
  for (size_t iFrame = chunkStart; iFrame < chunkEnd; ++iFrame) {
    auto jacRes = computePerFrameJacobian(fn, iFrame, skelState, meshState);
    error += jacRes.error;
    nFunctions += jacRes.nFunctions;
    if (jacRes.usedRows > 0) {
      block->add(
          iFrame * perFrameParams,
          jacRes.jacobian,
          jacRes.nBandColumns,
          jacRes.residual.topRows(jacRes.usedRows),
          targetRowsPerJtJChunk);
    }
  }

  if (!fn->getSequenceErrorFunctions().empty()) {
    for (size_t iFrame = chunkStart; iFrame < chunkEnd; ++iFrame) {
      auto jacRes = computeSequenceJacobian(
          fn, iFrame, bandwidthFrames, chunkEnd, SequenceJacobianSelection::InternalToChunk);
      error += jacRes.error;
      nFunctions += jacRes.nFunctions;
      if (jacRes.usedRows > 0) {
        block->add(
            iFrame * perFrameParams,
            jacRes.jacobian,
            jacRes.nBandColumns,
            jacRes.residual.topRows(jacRes.usedRows),
            targetRowsPerJtJChunk);
      }
    }
  }

  return {.normalEquations = std::move(block), .error = error, .nFunctions = nFunctions};
}

template <typename T, typename NormalEquationScalar>
AccumulationResult<T, NormalEquationScalar> processBoundarySequenceFrame(
    SequenceSolverFunctionT<T>* fn,
    size_t iFrame,
    size_t bandwidthFrames,
    size_t chunkEnd,
    size_t targetRowsPerJtJChunk) {
  const Eigen::Index perFrameParams = fn->getPerFrameParameterIndices().size();
  const Eigen::Index storageBlockSize = std::max<Eigen::Index>(1, perFrameParams);
  const Eigen::Index nCommon = fn->getUniversalParameterIndices().size();

  auto jacRes = computeSequenceJacobian(
      fn, iFrame, bandwidthFrames, chunkEnd, SequenceJacobianSelection::CrossesChunk);
  auto block = std::make_unique<NormalEquationBlock<NormalEquationScalar>>(
      iFrame * perFrameParams,
      jacRes.nFrameColumns * perFrameParams,
      nCommon,
      bandwidthFrames * perFrameParams,
      NormalEquationScalar(0),
      storageBlockSize);

  if (jacRes.usedRows > 0) {
    block->add(
        iFrame * perFrameParams,
        jacRes.jacobian,
        jacRes.nBandColumns,
        jacRes.residual.topRows(jacRes.usedRows),
        targetRowsPerJtJChunk);
  }

  return {
      .normalEquations = std::move(block), .error = jacRes.error, .nFunctions = jacRes.nFunctions};
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> solveCommonNormalEquationsOnly(
    const NormalEquationBlock<T>& hessian);

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> solveNormalEquationsScalarLdlt(
    const NormalEquationBlock<T>& hessian) {
  using FactorScalar = double;
  using FactorMatrixType = Eigen::Matrix<FactorScalar, Eigen::Dynamic, Eigen::Dynamic>;
  using FactorVectorType = Eigen::Matrix<FactorScalar, Eigen::Dynamic, 1>;

  const Eigen::Index nBand = hessian.nBand();
  const Eigen::Index nCommon = hessian.nCommon();
  const Eigen::Index bandwidth = hessian.bandwidth();

  auto hessianBandEntry = [&](Eigen::Index iRow, Eigen::Index jCol) -> FactorScalar {
    MT_CHECK(iRow <= jCol);
    MT_CHECK(jCol - iRow < bandwidth);
    if (hessian.hasBlockBand()) {
      const Eigen::Index blockSize = hessian.blockSize();
      MT_CHECK(blockSize > 0);
      const Eigen::Index iBlock = iRow / blockSize;
      const Eigen::Index jBlock = jCol / blockSize;
      if (jBlock - iBlock >= hessian.blockBandwidth()) {
        return FactorScalar(0);
      }
      return hessian.blockBandBlock(jBlock, iBlock)(jCol % blockSize, iRow % blockSize);
    }
    return hessian.bandEntry(iRow, jCol);
  };

  FactorMatrixType lBand = FactorMatrixType::Zero(bandwidth, nBand);
  FactorVectorType diagonal = FactorVectorType::Zero(nBand);
  auto lEntry = [&](Eigen::Index iRow, Eigen::Index jCol) -> FactorScalar& {
    MT_CHECK(iRow >= jCol);
    MT_CHECK(iRow - jCol < bandwidth);
    return lBand(iRow - jCol, jCol);
  };
  auto lEntryConst = [&](Eigen::Index iRow, Eigen::Index jCol) -> FactorScalar {
    MT_CHECK(iRow >= jCol);
    MT_CHECK(iRow - jCol < bandwidth);
    return lBand(iRow - jCol, jCol);
  };

  for (Eigen::Index iCol = 0; iCol < nBand; ++iCol) {
    FactorScalar diag = hessianBandEntry(iCol, iCol);
    const Eigen::Index kStart = std::max<Eigen::Index>(0, iCol - bandwidth + 1);
    for (Eigen::Index k = kStart; k < iCol; ++k) {
      const FactorScalar lik = lEntryConst(iCol, k);
      diag -= lik * lik * diagonal(k);
    }

    if (std::abs(diag) <= std::numeric_limits<FactorScalar>::epsilon()) {
      MT_THROW("Banded LDLT factorization failed at column {}", iCol);
    }
    diagonal(iCol) = diag;
    lEntry(iCol, iCol) = FactorScalar(1);

    const Eigen::Index jEnd = std::min(nBand, iCol + bandwidth);
    for (Eigen::Index jRow = iCol + 1; jRow < jEnd; ++jRow) {
      FactorScalar value = hessianBandEntry(iCol, jRow);
      const auto kStartCur =
          std::max<Eigen::Index>({0, iCol - bandwidth + 1, jRow - bandwidth + 1});
      for (Eigen::Index k = kStartCur; k < iCol; ++k) {
        value -= lEntryConst(jRow, k) * lEntryConst(iCol, k) * diagonal(k);
      }
      lEntry(jRow, iCol) = value / diag;
    }
  }

  FactorMatrixType e = FactorMatrixType::Zero(nBand, nCommon);
  FactorVectorType zBand = hessian.rhs().head(nBand).template cast<FactorScalar>();

  for (Eigen::Index iRow = 0; iRow < nBand; ++iRow) {
    if (nCommon > 0) {
      e.row(iRow).noalias() = hessian.common().row(iRow).template cast<FactorScalar>();
    }
    const Eigen::Index kStart = std::max<Eigen::Index>(0, iRow - bandwidth + 1);
    for (Eigen::Index k = kStart; k < iRow; ++k) {
      if (nCommon > 0) {
        e.row(iRow).noalias() -= lEntryConst(iRow, k) * e.row(k);
      }
      zBand(iRow) -= lEntryConst(iRow, k) * zBand(k);
    }
  }

  FactorVectorType result = FactorVectorType::Zero(nBand + nCommon);
  if (nCommon > 0) {
    FactorMatrixType dInvE = e;
    FactorVectorType dInvZ = zBand;
    for (Eigen::Index iRow = 0; iRow < nBand; ++iRow) {
      dInvE.row(iRow) /= diagonal(iRow);
      dInvZ(iRow) /= diagonal(iRow);
    }

    FactorMatrixType schur = hessian.common().bottomRows(nCommon).template cast<FactorScalar>();
    schur.noalias() -= e.transpose() * dInvE;

    Eigen::LDLT<FactorMatrixType> ldlt(schur);
    if (ldlt.info() != Eigen::Success) {
      MT_THROW("Universal LDLT factorization failed");
    }
    result.tail(nCommon) = ldlt.solve(
        hessian.rhs().tail(nCommon).template cast<FactorScalar>() - e.transpose() * dInvZ);
  }

  FactorVectorType bandRhs = zBand;
  if (nCommon > 0) {
    bandRhs.noalias() -= e * result.tail(nCommon);
  }
  bandRhs.array() /= diagonal.array();

  for (Eigen::Index iRow = nBand - 1; iRow >= 0; --iRow) {
    auto dotProd = FactorScalar(0);
    const Eigen::Index jEnd = std::min(nBand, iRow + bandwidth);
    for (Eigen::Index jRow = iRow + 1; jRow < jEnd; ++jRow) {
      dotProd += lEntryConst(jRow, iRow) * result(jRow);
    }
    result(iRow) = bandRhs(iRow) - dotProd;
  }

  if (nBand == 0 && nCommon > 0) {
    result.tail(nCommon) = solveCommonNormalEquationsOnly(hessian).template cast<FactorScalar>();
  }

  return result.template cast<T>();
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> solveCommonNormalEquationsOnly(
    const NormalEquationBlock<T>& hessian) {
  using FactorScalar = double;
  using FactorMatrixType = Eigen::Matrix<FactorScalar, Eigen::Dynamic, Eigen::Dynamic>;

  const Eigen::Index nCommon = hessian.nCommon();
  FactorMatrixType common = hessian.common().bottomRows(nCommon).template cast<FactorScalar>();
  Eigen::LDLT<FactorMatrixType> ldlt(common);
  MT_THROW_IF(ldlt.info() != Eigen::Success, "Universal LDLT factorization failed");
  return ldlt.solve(hessian.rhs().tail(nCommon).template cast<FactorScalar>()).template cast<T>();
}

using BlockFactorScalar = double;
using BlockFactorMatrixType = Eigen::Matrix<BlockFactorScalar, Eigen::Dynamic, Eigen::Dynamic>;
using BlockFactorVectorType = Eigen::Matrix<BlockFactorScalar, Eigen::Dynamic, 1>;

template <typename T>
BlockFactorMatrixType getBlockNormalEquation(
    const NormalEquationBlock<T>& hessian,
    Eigen::Index blockSize,
    Eigen::Index iBlock,
    Eigen::Index jBlock) {
  if (hessian.hasBlockBand()) {
    return hessian.blockBandBlock(iBlock, jBlock).template cast<BlockFactorScalar>();
  }

  BlockFactorMatrixType result = BlockFactorMatrixType::Zero(blockSize, blockSize);
  const Eigen::Index iOffset = iBlock * blockSize;
  const Eigen::Index jOffset = jBlock * blockSize;
  for (Eigen::Index i = 0; i < blockSize; ++i) {
    for (Eigen::Index j = 0; j < blockSize; ++j) {
      const Eigen::Index iGlobal = iOffset + i;
      const Eigen::Index jGlobal = jOffset + j;
      if (iGlobal <= jGlobal) {
        if (jGlobal - iGlobal < hessian.bandwidth()) {
          result(i, j) = hessian.bandEntry(iGlobal, jGlobal);
        }
      } else if (iGlobal - jGlobal < hessian.bandwidth()) {
        result(i, j) = hessian.bandEntry(jGlobal, iGlobal);
      }
    }
  }
  return result;
}

struct BlockLdltFactors {
  Eigen::Index blockSize = 0;
  Eigen::Index nBlocks = 0;
  Eigen::Index bandwidthBlocks = 0;
  BlockFactorMatrixType lowerFactors;
  BlockFactorMatrixType diagonalFactors;
  std::vector<Eigen::LDLT<BlockFactorMatrixType>> diagonalLdltFactors;

  auto lowerFactor(Eigen::Index iBlock, Eigen::Index jBlock) {
    MT_CHECK(iBlock >= jBlock);
    const Eigen::Index offset = iBlock - jBlock;
    MT_CHECK(offset < bandwidthBlocks);
    return lowerFactors.block(iBlock * blockSize, offset * blockSize, blockSize, blockSize);
  }

  [[nodiscard]] auto lowerFactor(Eigen::Index iBlock, Eigen::Index jBlock) const {
    MT_CHECK(iBlock >= jBlock);
    const Eigen::Index offset = iBlock - jBlock;
    MT_CHECK(offset < bandwidthBlocks);
    return lowerFactors.block(iBlock * blockSize, offset * blockSize, blockSize, blockSize);
  }

  auto diagonalFactor(Eigen::Index iBlock) {
    return diagonalFactors.block(iBlock * blockSize, 0, blockSize, blockSize);
  }
};

template <typename T>
BlockLdltFactors factorBlockNormalEquations(
    const NormalEquationBlock<T>& hessian,
    Eigen::Index blockSize) {
  const Eigen::Index nBand = hessian.nBand();
  const Eigen::Index nBlocks = nBand / blockSize;
  const Eigen::Index bandwidthBlocks = hessian.blockBandwidth();
  BlockLdltFactors factors{
      .blockSize = blockSize,
      .nBlocks = nBlocks,
      .bandwidthBlocks = bandwidthBlocks,
      .lowerFactors = BlockFactorMatrixType::Zero(nBlocks * blockSize, bandwidthBlocks * blockSize),
      .diagonalFactors = BlockFactorMatrixType::Zero(nBand, blockSize),
      .diagonalLdltFactors = std::vector<Eigen::LDLT<BlockFactorMatrixType>>(nBlocks)};

  for (Eigen::Index iBlock = 0; iBlock < nBlocks; ++iBlock) {
    auto diagonal = getBlockNormalEquation(hessian, blockSize, iBlock, iBlock);
    const Eigen::Index kStart = std::max<Eigen::Index>(0, iBlock - bandwidthBlocks + 1);
    if (bandwidthBlocks == 2 && iBlock > 0) {
      // For block-tridiagonal systems, L_ik * D_k * L_ik^T is equivalent to
      // L_ik * A_ik^T and saves one dense block multiply per frame.
      diagonal.noalias() -= factors.lowerFactor(iBlock, iBlock - 1) *
          getBlockNormalEquation(hessian, blockSize, iBlock, iBlock - 1).transpose();
    } else {
      for (Eigen::Index kBlock = kStart; kBlock < iBlock; ++kBlock) {
        const auto& lik = factors.lowerFactor(iBlock, kBlock);
        diagonal.noalias() -= lik * factors.diagonalFactor(kBlock) * lik.transpose();
      }
    }

    factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock)).compute(diagonal);
    if (factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock)).info() != Eigen::Success) {
      MT_THROW("Block banded LDLT factorization failed at block {}", iBlock);
    }
    factors.diagonalFactor(iBlock) = diagonal;
    factors.lowerFactor(iBlock, iBlock).setIdentity();

    const Eigen::Index jEnd = std::min(nBlocks, iBlock + bandwidthBlocks);
    for (Eigen::Index jBlock = iBlock + 1; jBlock < jEnd; ++jBlock) {
      auto value = getBlockNormalEquation(hessian, blockSize, jBlock, iBlock);
      const auto kStartCur = std::max<Eigen::Index>({0, jBlock - bandwidthBlocks + 1, kStart});
      for (Eigen::Index kBlock = kStartCur; kBlock < iBlock; ++kBlock) {
        value.noalias() -= factors.lowerFactor(jBlock, kBlock) * factors.diagonalFactor(kBlock) *
            factors.lowerFactor(iBlock, kBlock).transpose();
      }

      factors.lowerFactor(jBlock, iBlock) =
          factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock))
              .solve(value.transpose())
              .transpose();
    }
  }

  return factors;
}

void applyBlockForwardSubstitution(
    const BlockLdltFactors& factors,
    BlockFactorVectorType& zBand,
    BlockFactorMatrixType& e) {
  for (Eigen::Index iBlock = 0; iBlock < factors.nBlocks; ++iBlock) {
    const Eigen::Index iOffset = iBlock * factors.blockSize;
    const Eigen::Index kStart = std::max<Eigen::Index>(0, iBlock - factors.bandwidthBlocks + 1);
    for (Eigen::Index kBlock = kStart; kBlock < iBlock; ++kBlock) {
      const Eigen::Index kOffset = kBlock * factors.blockSize;
      zBand.segment(iOffset, factors.blockSize).noalias() -=
          factors.lowerFactor(iBlock, kBlock) * zBand.segment(kOffset, factors.blockSize);
      if (e.cols() > 0) {
        e.middleRows(iOffset, factors.blockSize).noalias() -=
            factors.lowerFactor(iBlock, kBlock) * e.middleRows(kOffset, factors.blockSize);
      }
    }
  }
}

void applyBlockDiagonalInverse(
    const BlockLdltFactors& factors,
    BlockFactorVectorType& zBand,
    BlockFactorMatrixType& e) {
  for (Eigen::Index iBlock = 0; iBlock < factors.nBlocks; ++iBlock) {
    const Eigen::Index iOffset = iBlock * factors.blockSize;
    zBand.segment(iOffset, factors.blockSize) =
        factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock))
            .solve(zBand.segment(iOffset, factors.blockSize));
    if (e.cols() > 0) {
      e.middleRows(iOffset, factors.blockSize) =
          factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock))
              .solve(e.middleRows(iOffset, factors.blockSize));
    }
  }
}

template <typename T>
BlockFactorVectorType solveCommonBlock(
    const NormalEquationBlock<T>& hessian,
    const BlockFactorMatrixType& e,
    const BlockFactorMatrixType& dInvE,
    const BlockFactorVectorType& dInvZ) {
  const Eigen::Index nCommon = hessian.nCommon();
  BlockFactorMatrixType schur =
      hessian.common().bottomRows(nCommon).template cast<BlockFactorScalar>();
  schur.noalias() -= e.transpose() * dInvE;

  Eigen::LDLT<BlockFactorMatrixType> ldlt(schur);
  if (ldlt.info() != Eigen::Success) {
    MT_THROW("Universal LDLT factorization failed");
  }
  return ldlt.solve(
      hessian.rhs().tail(nCommon).template cast<BlockFactorScalar>() - e.transpose() * dInvZ);
}

BlockFactorVectorType solveBlockBandVariables(
    const BlockLdltFactors& factors,
    BlockFactorVectorType bandRhs) {
  for (Eigen::Index iBlock = 0; iBlock < factors.nBlocks; ++iBlock) {
    const Eigen::Index iOffset = iBlock * factors.blockSize;
    bandRhs.segment(iOffset, factors.blockSize) =
        factors.diagonalLdltFactors.at(static_cast<size_t>(iBlock))
            .solve(bandRhs.segment(iOffset, factors.blockSize));
  }

  BlockFactorVectorType result = BlockFactorVectorType::Zero(factors.nBlocks * factors.blockSize);
  for (Eigen::Index iBlock = factors.nBlocks - 1; iBlock >= 0; --iBlock) {
    const Eigen::Index iOffset = iBlock * factors.blockSize;
    const Eigen::Index jEnd = std::min(factors.nBlocks, iBlock + factors.bandwidthBlocks);
    for (Eigen::Index jBlock = iBlock + 1; jBlock < jEnd; ++jBlock) {
      const Eigen::Index jOffset = jBlock * factors.blockSize;
      bandRhs.segment(iOffset, factors.blockSize).noalias() -=
          factors.lowerFactor(jBlock, iBlock).transpose() *
          result.segment(jOffset, factors.blockSize);
    }

    result.segment(iOffset, factors.blockSize) = bandRhs.segment(iOffset, factors.blockSize);
  }
  return result;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> solveNormalEquationsBlock(
    const NormalEquationBlock<T>& hessian,
    Eigen::Index blockSize) {
  const Eigen::Index nBand = hessian.nBand();
  const Eigen::Index nCommon = hessian.nCommon();
  MT_CHECK(blockSize > 0);
  MT_CHECK(nBand % blockSize == 0);

  const BlockLdltFactors factors = factorBlockNormalEquations(hessian, blockSize);

  BlockFactorVectorType zBand = hessian.rhs().head(nBand).template cast<BlockFactorScalar>();
  BlockFactorMatrixType e = BlockFactorMatrixType::Zero(nBand, nCommon);
  if (nCommon > 0) {
    e = hessian.common().topRows(nBand).template cast<BlockFactorScalar>();
  }

  applyBlockForwardSubstitution(factors, zBand, e);

  BlockFactorVectorType result = BlockFactorVectorType::Zero(nBand + nCommon);
  if (nCommon > 0) {
    BlockFactorMatrixType dInvE = e;
    BlockFactorVectorType dInvZ = zBand;
    applyBlockDiagonalInverse(factors, dInvZ, dInvE);
    result.tail(nCommon) = solveCommonBlock(hessian, e, dInvE, dInvZ);
  }

  BlockFactorVectorType bandRhs = zBand;
  if (nCommon > 0) {
    bandRhs.noalias() -= e * result.tail(nCommon);
  }
  result.head(nBand) = solveBlockBandVariables(factors, bandRhs);

  return result.template cast<T>();
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> solveNormalEquations(
    const NormalEquationBlock<T>& hessian,
    Eigen::Index blockSize,
    bool useBlockLdlt) {
  if (!useBlockLdlt || hessian.nBand() == 0 || blockSize <= 1 || hessian.nBand() % blockSize != 0 ||
      hessian.bandwidth() % blockSize != 0 || !hessian.hasBlockBand()) {
    return solveNormalEquationsScalarLdlt(hessian);
  }

  return solveNormalEquationsBlock(hessian, blockSize);
}

} // namespace

template <typename T>
SequenceCholeskySolverT<T>::SequenceCholeskySolverT(
    const SolverOptions& options,
    SequenceSolverFunctionT<T>* function)
    : SequenceSolverBaseT<T>(options, function) {
  chunkSize_ = SequenceCholeskySolverOptions().chunkSize;
  targetRowsPerJtJChunk_ = SequenceCholeskySolverOptions().targetRowsPerJtJChunk;
  useBlockLdlt_ = SequenceCholeskySolverOptions().useBlockLdlt;

  setOptions(options);
}

template <typename T>
std::string_view SequenceCholeskySolverT<T>::getName() const {
  return "SequenceCholeskySolver";
}

template <typename T>
void SequenceCholeskySolverT<T>::setOptions(const SolverOptions& options) {
  SequenceSolverBaseT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const SequenceCholeskySolverOptions*>(&options)) {
    chunkSize_ = std::max<size_t>(1, derivedOptions->chunkSize);
    targetRowsPerJtJChunk_ = std::max<size_t>(1, derivedOptions->targetRowsPerJtJChunk);
    useBlockLdlt_ = derivedOptions->useBlockLdlt;
    useDoublePrecisionNormalEquations_ = derivedOptions->useDoublePrecisionNormalEquations;
  }
}

template <typename T>
void SequenceCholeskySolverT<T>::initializeSolver() {
  MT_PROFILE_FUNCTION();

  auto* fn = static_cast<SequenceSolverFunctionT<T>*>(this->solverFunction_);

  this->bandwidth_ = 1;
  for (const auto& frameErrf : fn->getSequenceErrorFunctions()) {
    for (const auto& errf : frameErrf) {
      this->bandwidth_ = std::max(this->bandwidth_, errf->numFrames());
    }
  }
}

template <typename T>
void SequenceCholeskySolverT<T>::doIteration() {
  if constexpr (std::is_same_v<T, double>) {
    doIterationWithNormalEquationScalar<double>();
  } else {
    if (useDoublePrecisionNormalEquations_) {
      doIterationWithNormalEquationScalar<double>();
    } else {
      doIterationWithNormalEquationScalar<T>();
    }
  }
}

template <typename T>
template <typename NormalEquationScalar>
void SequenceCholeskySolverT<T>::doIterationWithNormalEquationScalar() {
  MT_PROFILE_FUNCTION();

  const auto normalEquationStart = std::chrono::steady_clock::now();

  auto* fnPtr = static_cast<SequenceSolverFunctionT<T>*>(this->solverFunction_);
  MT_CHECK(fnPtr != nullptr);
  auto& fn = *fnPtr;

  fn.setFrameParametersFromJoinedParameterVector(this->parameters_);

  const Eigen::Index perFrameParams = fn.getPerFrameParameterIndices().size();
  const Eigen::Index storageBlockSize = std::max<Eigen::Index>(1, perFrameParams);
  const Eigen::Index nBand = fn.getNumFrames() * perFrameParams;
  const Eigen::Index nCommon = fn.getUniversalParameterIndices().size();
  const Eigen::Index bandwidth = this->bandwidth_ * perFrameParams;
  NormalEquationBlock<NormalEquationScalar> globalHessian(
      0, nBand, nCommon, bandwidth, NormalEquationScalar(this->regularization_), storageBlockSize);

  const size_t chunkSize = std::max<size_t>(1, chunkSize_);
  const size_t nChunks = (fn.getNumFrames() + chunkSize - 1) / chunkSize;
  std::vector<AccumulationResult<T, NormalEquationScalar>> chunkResults(nChunks);

  if (this->multithreaded_) {
    dispenso::parallel_for(size_t(0), nChunks, [&](size_t iChunk) {
      const size_t chunkStart = iChunk * chunkSize;
      const size_t chunkEnd = std::min(fn.getNumFrames(), chunkStart + chunkSize);
      chunkResults[iChunk] = processChunk<T, NormalEquationScalar>(
          &fn, chunkStart, chunkEnd, this->bandwidth_, targetRowsPerJtJChunk_);
    });
  } else {
    for (size_t iChunk = 0; iChunk < nChunks; ++iChunk) {
      const size_t chunkStart = iChunk * chunkSize;
      const size_t chunkEnd = std::min(fn.getNumFrames(), chunkStart + chunkSize);
      chunkResults[iChunk] = processChunk<T, NormalEquationScalar>(
          &fn, chunkStart, chunkEnd, this->bandwidth_, targetRowsPerJtJChunk_);
    }
  }

  std::unique_ptr<ProgressBar> progress;
  if (this->progressBar_) {
    progress = std::make_unique<ProgressBar>(
        "Solving sequence",
        fn.numTotalPerFrameErrorFunctions_ + fn.numTotalSequenceErrorFunctions_);
  }

  this->error_ = 0;
  for (auto& result : chunkResults) {
    globalHessian.mergeBlock(*result.normalEquations);
    this->error_ += result.error;
    if (progress) {
      progress->increment(result.nFunctions);
    }
  }

  if (fn.numTotalSequenceErrorFunctions_ != 0) {
    for (size_t iChunk = 0; iChunk < nChunks; ++iChunk) {
      const size_t chunkStart = iChunk * chunkSize;
      const size_t chunkEnd = std::min(fn.getNumFrames(), chunkStart + chunkSize);
      size_t boundaryStart = chunkEnd;
      if (this->bandwidth_ > 1) {
        const size_t boundaryFrames = this->bandwidth_ - 1;
        boundaryStart = chunkEnd > boundaryFrames ? std::max(chunkStart, chunkEnd - boundaryFrames)
                                                  : chunkStart;
      }
      for (size_t iFrame = boundaryStart; iFrame < chunkEnd; ++iFrame) {
        auto result = processBoundarySequenceFrame<T, NormalEquationScalar>(
            &fn, iFrame, this->bandwidth_, chunkEnd, targetRowsPerJtJChunk_);
        globalHessian.mergeBlock(*result.normalEquations);
        this->error_ += result.error;
        if (progress) {
          progress->increment(result.nFunctions);
        }
      }
    }
  }

  const auto normalEquationEnd = std::chrono::steady_clock::now();
  lastNormalEquationTimeMs_ =
      std::chrono::duration<double, std::milli>(normalEquationEnd - normalEquationStart).count();

  const auto linearSolveStart = std::chrono::steady_clock::now();
  const Eigen::Matrix<NormalEquationScalar, Eigen::Dynamic, 1> normalEquationSearchDir =
      solveNormalEquations(globalHessian, storageBlockSize, useBlockLdlt_);
  const Eigen::VectorX<T>& searchDir = normalEquationSearchDir.template cast<T>();
  const auto linearSolveEnd = std::chrono::steady_clock::now();
  lastLinearSolveTimeMs_ =
      std::chrono::duration<double, std::milli>(linearSolveEnd - linearSolveStart).count();

  const double errorOrig = this->error_;
  if (this->doLineSearch_) {
    const double innerProd = -globalHessian.rhs().dot(normalEquationSearchDir);

    const float c1 = 1e-4f;
    const float tau = 0.5f;
    float alpha = 1.0f;

    const Eigen::VectorX<T> parametersOrig = this->parameters_;
    for (size_t kStep = 0; kStep < 10 && std::fpclassify(alpha) == FP_NORMAL; ++kStep) {
      this->parameters_ = parametersOrig;
      this->solverFunction_->updateParameters(this->parameters_, alpha * searchDir);

      const double errorNew = this->solverFunction_->getError(this->parameters_);
      if ((errorOrig - errorNew) >= c1 * alpha * -innerProd) {
        break;
      }

      alpha = alpha * tau;
    }
  } else {
    this->solverFunction_->updateParameters(this->parameters_, searchDir);
  }
}

template class SequenceCholeskySolverT<float>;
template class SequenceCholeskySolverT<double>;

} // namespace momentum
