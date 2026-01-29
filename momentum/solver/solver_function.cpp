/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/solver_function.h"

#include "momentum/common/exception.h"

namespace momentum {

template <typename T>
void SolverFunctionT<T>::getHessian(const VectorX<T>& parameters, MatrixX<T>& hessian) {
  (void)parameters;
  (void)hessian;
  MT_THROW("SolverFunctionT::getHessian() is not implemented");
}

template <typename T>
double SolverFunctionT<T>::getJacobian(
    const VectorX<T>& parameters,
    MatrixX<T>& jacobian,
    VectorX<T>& residual,
    size_t& actualRows) {
  initializeJacobianComputation(parameters);

  const size_t numBlocks = getJacobianBlockCount();

  // Calculate total size
  size_t totalRows = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    totalRows += getJacobianBlockSize(i);
  }
  // Add padding for alignment
  totalRows += 8 - (totalRows % 8);

  // Resize if needed
  if (totalRows > static_cast<size_t>(jacobian.rows()) || parameters.size() != jacobian.cols()) {
    jacobian.resize(totalRows, parameters.size());
    residual.resize(totalRows);
  }

  jacobian.setZero();
  residual.setZero();

  double error = 0.0;
  size_t position = 0;
  actualRows = totalRows;

  // Assemble blocks into full Jacobian
  for (size_t i = 0; i < numBlocks; ++i) {
    const size_t blockSize = getJacobianBlockSize(i);
    if (blockSize == 0) {
      continue;
    }

    size_t blockActualRows = 0;

    auto jacBlock = jacobian.block(position, 0, blockSize, parameters.size());
    auto resBlock = residual.segment(position, blockSize);

    error += computeJacobianBlock(parameters, i, jacBlock, resBlock, blockActualRows);

    position += blockSize;
  }

  finalizeJacobianComputation();
  return error;
}

template <typename T>
double SolverFunctionT<T>::getJtJR(const VectorX<T>& parameters, MatrixX<T>& jtj, VectorX<T>& jtr) {
  initializeJacobianComputation(parameters);

  const size_t numBlocks = getJacobianBlockCount();

  // Calculate total Jacobian size
  size_t jacobianSize = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    jacobianSize += getJacobianBlockSize(i);
  }
  // Add padding for alignment
  jacobianSize += 8 - (jacobianSize % 8);

  const size_t numParams = parameters.size();

  // Pre-allocate temporary storage once (resize only if needed)
  if (jacobianSize > static_cast<size_t>(tJacobian_.rows()) ||
      numParams != static_cast<size_t>(tJacobian_.cols())) {
    tJacobian_.resize(jacobianSize, numParams);
    tResidual_.resize(jacobianSize);
  }

  // Zero out once at the start
  tJacobian_.topRows(jacobianSize).setZero();
  tResidual_.head(jacobianSize).setZero();
  jtj.setZero(actualParameters_, actualParameters_);
  jtr.setZero(actualParameters_);

  // Block compute JtJ and JtR on the fly
  size_t position = 0;
  double error = 0.0;

  for (size_t i = 0; i < numBlocks; ++i) {
    const size_t blockSize = getJacobianBlockSize(i);
    if (blockSize == 0) {
      continue;
    }

    size_t blockActualRows = 0;

    auto jacBlock = tJacobian_.block(position, 0, blockSize, numParams);
    auto resBlock = tResidual_.segment(position, blockSize);

    error += computeJacobianBlock(parameters, i, jacBlock, resBlock, blockActualRows);

    // Update JtJ and JtR
    if (blockActualRows > 0) {
      const auto JtBlock =
          tJacobian_.block(position, 0, blockActualRows, actualParameters_).transpose();

      // Efficiently update JtJ using selfadjointView with rankUpdate
      jtj.template selfadjointView<Eigen::Lower>().rankUpdate(JtBlock);

      // Update JtR
      jtr.noalias() += JtBlock * tResidual_.segment(position, blockActualRows);
    }
    position += blockSize;
  }

  finalizeJacobianComputation();
  return error;
}

template <typename T>
double SolverFunctionT<T>::getJtJR_Sparse(
    const VectorX<T>& parameters,
    SparseMatrix<T>& jtj,
    VectorX<T>& jtr) {
  // mostly a template here. See the actual derived function for implementation (e.g.,
  // batchSkeletonSolverFunction.cpp)
  (void)parameters;
  (void)jtj;
  (void)jtr;
  return 0.0;
};

template <typename T>
double SolverFunctionT<T>::getSolverDerivatives(
    const VectorX<T>& parameters,
    MatrixX<T>& hess,
    VectorX<T>& grad) {
  // default implementation returns JtJ, JtR
  return getJtJR(parameters, hess, grad);
}

template <typename T>
void SolverFunctionT<T>::setEnabledParameters(const ParameterSet& /* params */) {};

template <typename T>
size_t SolverFunctionT<T>::getNumParameters() const {
  return numParameters_;
}

template <typename T>
size_t SolverFunctionT<T>::getActualParameters() const {
  return actualParameters_;
}

template <typename T>
void SolverFunctionT<T>::storeHistory(
    std::unordered_map<std::string, MatrixX<T>>& history,
    size_t iteration,
    size_t maxIterations_) {
  // unused variables
  (void)history;
  (void)iteration;
  (void)maxIterations_;
}

template <typename T>
void SolverFunctionT<T>::finalizeJacobianComputation() {
  // Default implementation does nothing
}

template class SolverFunctionT<float>;
template class SolverFunctionT<double>;

} // namespace momentum
