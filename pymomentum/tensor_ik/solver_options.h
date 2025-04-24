/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

namespace pymomentum {

enum class LinearSolverType { Cholesky, QR, TrustRegionQR };

inline const char* toString(LinearSolverType solverType) {
  switch (solverType) {
    default:
    case LinearSolverType::Cholesky:
      return "Cholesky";
    case LinearSolverType::QR:
      return "QR";
    case LinearSolverType::TrustRegionQR:
      return "TrustRegionQR";
  }
}

struct SolverOptions {
  LinearSolverType linearSolverType = LinearSolverType::QR;
  float levmar_lambda = 0.01f;
  size_t minIter = 4;
  size_t maxIter = 50;
  float threshold = 10.0f;
  bool lineSearch = true;
  // Temporal smoothing of model parameters (only for the sequence solver)
  float sequenceSmoothingWeight = 0.0f;

  bool operator==(const SolverOptions& rhs) const {
    return linearSolverType == rhs.linearSolverType &&
        levmar_lambda == rhs.levmar_lambda && minIter == rhs.minIter &&
        maxIter == rhs.maxIter && threshold == rhs.threshold &&
        lineSearch == rhs.lineSearch;
  }
};

} // namespace pymomentum
