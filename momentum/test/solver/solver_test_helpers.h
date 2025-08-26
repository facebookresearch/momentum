/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/common/log.h>
#include <momentum/solver/solver.h>
#include <momentum/solver/solver_function.h>

#include <chrono>

namespace momentum::test {

SolverOptions defaultSolverOptions() {
  SolverOptions result;
  result.minIterations = 4;
  result.maxIterations = 40;
  result.threshold = 1000.f;
  result.verbose = false;
  return result;
}

template <typename T>
T checkAndTimeSolver(
    SolverFunctionT<T>& solverFunction,
    SolverT<T>& solver,
    const Eigen::VectorX<T>& parametersInit,
    const ParameterSet& enabledParameters = allParams()) {
  auto start_time = std::chrono::high_resolution_clock::now();
  solver.setStoreHistory(true);
  solver.setEnabledParameters(enabledParameters);
  Eigen::VectorX<T> parameters = parametersInit;
  solver.solve(parameters);
  auto end_time = std::chrono::high_resolution_clock::now();
  MT_LOGD("Errors for block QR: {}", solver.getHistory().find("error")->second);

  const std::unordered_map<std::string, Eigen::MatrixX<T>>& iterHistory = solver.getHistory();
  const auto iterCountIterator = iterHistory.find("iterations");
  MT_CHECK(iterCountIterator != iterHistory.end());
  size_t iterCount = size_t(iterCountIterator->second(0, 0) + 0.5);

  const auto err = solverFunction.getError(parameters);
  MT_LOGD(
      "Solver {:20}; elapsed: {:3}us; nIter: {:2}; solver.error: {:.6}",
      solver.getName(),
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(),
      iterCount,
      err);

  return err;
}

} // namespace momentum::test
