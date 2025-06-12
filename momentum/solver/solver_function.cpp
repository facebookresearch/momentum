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
double SolverFunctionT<T>::getJtJR(const VectorX<T>& parameters, MatrixX<T>& jtj, VectorX<T>& jtr) {
  // Generic implementation, considering J in one block.
  // Vastly sub-optimal, but here for compatibility reasons
  size_t actualRows = 0;
  MatrixX<T> jacobian;
  VectorX<T> residual;

  const double error = getJacobian(parameters, jacobian, residual, actualRows);
  jtj = jacobian.topLeftCorner(actualRows, parameters.size()).transpose() *
      jacobian.topLeftCorner(actualRows, parameters.size());

  jtr = jacobian.topLeftCorner(actualRows, parameters.size()).transpose() * residual;

  return error;
};

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

template class SolverFunctionT<float>;
template class SolverFunctionT<double>;

} // namespace momentum
