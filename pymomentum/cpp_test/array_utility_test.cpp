/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>

namespace py = pybind11;

// NOTE: Tests for geometry accessors (JointParametersAccessor, ModelParametersAccessor,
// SkeletonStateAccessor) require numpy arrays and are better tested via Python unit tests.
// The accessors themselves are designed to be called from Python code where numpy is
// available. The C++ tests below focus on the core array infrastructure that doesn't
// require numpy.

// ============================================================================
// LeadingDimensions Tests
// ============================================================================

TEST(ArrayUtility, LeadingDimensions_TotalBatchElements) {
  pymomentum::LeadingDimensions ld;

  // Empty dims = 1 batch element (scalar)
  EXPECT_EQ(1, ld.totalBatchElements());

  // Single dimension
  ld.dims = {5};
  EXPECT_EQ(5, ld.totalBatchElements());

  // Multiple dimensions
  ld.dims = {2, 3, 4};
  EXPECT_EQ(24, ld.totalBatchElements());

  // With ones
  ld.dims = {1, 5, 1, 3};
  EXPECT_EQ(15, ld.totalBatchElements());
}

TEST(ArrayUtility, LeadingDimensions_IsScalar) {
  pymomentum::LeadingDimensions ld;

  // Empty dims = scalar
  EXPECT_TRUE(ld.isScalar());

  // All ones = scalar
  ld.dims = {1, 1, 1};
  EXPECT_TRUE(ld.isScalar());

  // Any non-one = not scalar
  ld.dims = {1, 2, 1};
  EXPECT_FALSE(ld.isScalar());

  ld.dims = {5};
  EXPECT_FALSE(ld.isScalar());
}

TEST(ArrayUtility, LeadingDimensions_BroadcastCompatibility) {
  pymomentum::LeadingDimensions ld1, ld2;

  // Different number of dimensions = incompatible
  ld1.dims = {5, 3};
  ld2.dims = {5};
  EXPECT_FALSE(ld1.broadcastCompatibleWith(ld2));

  // Same dimensions = compatible
  ld1.dims = {5, 3};
  ld2.dims = {5, 3};
  EXPECT_TRUE(ld1.broadcastCompatibleWith(ld2));

  // One has dimension 1 = compatible
  ld1.dims = {5, 1};
  ld2.dims = {5, 3};
  EXPECT_TRUE(ld1.broadcastCompatibleWith(ld2));

  ld1.dims = {1, 3};
  ld2.dims = {5, 3};
  EXPECT_TRUE(ld1.broadcastCompatibleWith(ld2));

  // Both have dimension 1 = compatible
  ld1.dims = {1, 1};
  ld2.dims = {5, 3};
  EXPECT_TRUE(ld1.broadcastCompatibleWith(ld2));

  // Incompatible dimensions (neither is 1)
  ld1.dims = {5, 2};
  ld2.dims = {5, 3};
  EXPECT_FALSE(ld1.broadcastCompatibleWith(ld2));
}

TEST(ArrayUtility, LeadingDimensions_BroadcastWith) {
  pymomentum::LeadingDimensions ld1, ld2, result;

  // Same dimensions
  ld1.dims = {5, 3};
  ld2.dims = {5, 3};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3}), result.dims);

  // First has 1
  ld1.dims = {1, 3};
  ld2.dims = {5, 3};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3}), result.dims);

  // Second has 1
  ld1.dims = {5, 1};
  ld2.dims = {5, 3};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3}), result.dims);

  // Both have 1s in different positions
  ld1.dims = {1, 3, 1};
  ld2.dims = {5, 1, 7};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3, 7}), result.dims);

  // Incompatible dimensions should throw
  ld1.dims = {5, 2};
  ld2.dims = {5, 3};
  EXPECT_THROW((void)ld1.broadcastWith(ld2), std::runtime_error);

  // Different number of dimensions should throw
  ld1.dims = {5, 3};
  ld2.dims = {5};
  EXPECT_THROW((void)ld1.broadcastWith(ld2), std::runtime_error);
}

TEST(ArrayUtility, LeadingDimensions_EmptyLeadingDims) {
  pymomentum::LeadingDimensions ld1, ld2, result;

  // Both empty (scalars)
  EXPECT_TRUE(ld1.broadcastCompatibleWith(ld2));
  result = ld1.broadcastWith(ld2);
  EXPECT_TRUE(result.dims.empty());
  EXPECT_EQ(1, result.totalBatchElements());
  EXPECT_TRUE(result.isScalar());
}

TEST(ArrayUtility, LeadingDimensions_ComplexBroadcasting) {
  pymomentum::LeadingDimensions ld1, ld2, ld3, result;

  // Test case (1): (3, 1) broadcasts with (1, 4) -> (3, 4)
  ld1.dims = {3, 1};
  ld2.dims = {1, 4};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{3, 4}), result.dims);
  EXPECT_EQ(12, result.totalBatchElements());

  // Test case (2): (5, 1, 7) broadcasts with (1, 3, 1) -> (5, 3, 7)
  ld1.dims = {5, 1, 7};
  ld2.dims = {1, 3, 1};
  result = ld1.broadcastWith(ld2);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3, 7}), result.dims);
  EXPECT_EQ(105, result.totalBatchElements());

  // Test case (3): Chained broadcasting
  ld1.dims = {1, 1};
  ld2.dims = {5, 1};
  ld3.dims = {1, 3};
  result = ld1.broadcastWith(ld2);
  result = result.broadcastWith(ld3);
  EXPECT_EQ((std::vector<py::ssize_t>{5, 3}), result.dims);
}

// ============================================================================
// BatchIndexer Tests
// ============================================================================

TEST(ArrayUtility, BatchIndexer_SingleDimension) {
  pymomentum::LeadingDimensions ld;
  ld.dims = {5};
  pymomentum::BatchIndexer indexer(ld);

  EXPECT_EQ(1, indexer.ndim());

  for (py::ssize_t i = 0; i < 5; ++i) {
    auto indices = indexer.decompose(i);
    EXPECT_EQ(1, indices.size());
    EXPECT_EQ(i, indices[0]);
  }
}

TEST(ArrayUtility, BatchIndexer_TwoDimensions) {
  pymomentum::LeadingDimensions ld;
  ld.dims = {2, 3};
  pymomentum::BatchIndexer indexer(ld);

  EXPECT_EQ(2, indexer.ndim());

  // Expected indices for iBatch = 0..5:
  // 0 -> [0, 0]
  // 1 -> [0, 1]
  // 2 -> [0, 2]
  // 3 -> [1, 0]
  // 4 -> [1, 1]
  // 5 -> [1, 2]

  auto idx0 = indexer.decompose(0);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 0}), idx0);

  auto idx1 = indexer.decompose(1);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 1}), idx1);

  auto idx3 = indexer.decompose(3);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 0}), idx3);

  auto idx5 = indexer.decompose(5);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 2}), idx5);
}

TEST(ArrayUtility, BatchIndexer_ThreeDimensions) {
  pymomentum::LeadingDimensions ld;
  ld.dims = {2, 3, 4};
  pymomentum::BatchIndexer indexer(ld);

  EXPECT_EQ(3, indexer.ndim());

  // iBatch = 0 -> [0, 0, 0]
  auto idx0 = indexer.decompose(0);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 0, 0}), idx0);

  // iBatch = 1 -> [0, 0, 1]
  auto idx1 = indexer.decompose(1);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 0, 1}), idx1);

  // iBatch = 4 -> [0, 1, 0]
  auto idx4 = indexer.decompose(4);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 1, 0}), idx4);

  // iBatch = 12 -> [1, 0, 0]
  auto idx12 = indexer.decompose(12);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 0, 0}), idx12);

  // iBatch = 23 -> [1, 2, 3]
  auto idx23 = indexer.decompose(23);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 2, 3}), idx23);
}

TEST(ArrayUtility, BatchIndexer_FourDimensions) {
  pymomentum::LeadingDimensions ld;
  ld.dims = {2, 3, 4, 5};
  pymomentum::BatchIndexer indexer(ld);

  EXPECT_EQ(4, indexer.ndim());

  // Total batch elements = 2 * 3 * 4 * 5 = 120
  EXPECT_EQ(120, ld.totalBatchElements());

  // Test first and last indices
  auto idx0 = indexer.decompose(0);
  EXPECT_EQ((std::vector<py::ssize_t>{0, 0, 0, 0}), idx0);

  auto idx119 = indexer.decompose(119);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 2, 3, 4}), idx119);

  // Test middle index: 60 = 1 * 3 * 4 * 5 -> [1, 0, 0, 0]
  auto idx60 = indexer.decompose(60);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 0, 0, 0}), idx60);

  // Test arbitrary index: 77 = 1 * 60 + 17 = 1 * 60 + 0 * 20 + 3 * 5 + 2 -> [1, 0, 3, 2]
  auto idx77 = indexer.decompose(77);
  EXPECT_EQ((std::vector<py::ssize_t>{1, 0, 3, 2}), idx77);
}

TEST(ArrayUtility, BatchIndexer_EmptyDimensions) {
  pymomentum::LeadingDimensions ld;
  // Empty leading dimensions (scalar case)
  pymomentum::BatchIndexer indexer(ld);

  EXPECT_EQ(0, indexer.ndim());

  // There's only one element (scalar)
  auto idx0 = indexer.decompose(0);
  EXPECT_TRUE(idx0.empty());
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(ArrayUtility, FormatArrayDims_Vector) {
  std::vector<py::ssize_t> dims = {2, 3, 4};
  std::string result = pymomentum::formatArrayDims(dims);
  EXPECT_EQ("[2, 3, 4]", result);

  dims = {5};
  result = pymomentum::formatArrayDims(dims);
  EXPECT_EQ("[5]", result);

  dims = {};
  result = pymomentum::formatArrayDims(dims);
  EXPECT_EQ("[]", result);
}

TEST(ArrayUtility, FormatExpectedDims) {
  std::vector<int> trailingDims = {3, 4};
  std::vector<const char*> names = {"dim1", "dim2"};
  std::unordered_map<int, int64_t> boundVars;

  std::string result = pymomentum::formatExpectedDims(trailingDims, names, boundVars);
  EXPECT_EQ("[..., dim1(=3), dim2(=4)]", result);

  // With variable dimensions
  trailingDims = {-1, 4};
  names = {"numItems", "dim2"};
  boundVars[-1] = 10;

  result = pymomentum::formatExpectedDims(trailingDims, names, boundVars);
  EXPECT_EQ("[..., numItems(=10), dim2(=4)]", result);
}
