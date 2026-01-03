/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <pymomentum/array_utility/array_utility.h>

#include <pybind11/numpy.h>

namespace pymomentum {

namespace py = pybind11;

// BatchIndexer: Converts flat batch indices to multi-dimensional indices.
// This enables proper broadcasting support - all accessors using the same
// BatchIndexer will iterate in lockstep, and each accessor can handle
// broadcasting by checking if a dimension's stride is 0.
//
// Example:
//   BatchIndexer indexer(leadingDims);  // e.g., leadingDims = {2, 3}
//   for (int64_t iBatch = 0; iBatch < 6; ++iBatch) {
//     auto indices = indexer.decompose(iBatch);  // e.g., [0, 0], [0, 1], ..., [1, 2]
//     // All accessors use the same indices for this batch
//   }
class BatchIndexer {
 public:
  explicit BatchIndexer(const LeadingDimensions& leadingDims)
      : leadingDims_(leadingDims.dims), nDims_(leadingDims.ndim()) {}

  // Decompose a flat batch index into multi-dimensional indices
  [[nodiscard]] std::vector<py::ssize_t> decompose(py::ssize_t iBatch) const {
    std::vector<py::ssize_t> indices(nDims_);
    py::ssize_t remaining = iBatch;

    // Convert flat index to multi-dimensional indices (row-major order)
    for (int i = static_cast<int>(nDims_) - 1; i >= 0; --i) {
      const py::ssize_t dimSize = leadingDims_[i];
      indices[i] = remaining % dimSize;
      remaining /= dimSize;
    }

    return indices;
  }

  [[nodiscard]] size_t ndim() const {
    return nDims_;
  }

 private:
  std::vector<py::ssize_t> leadingDims_;
  size_t nDims_;
};

} // namespace pymomentum
