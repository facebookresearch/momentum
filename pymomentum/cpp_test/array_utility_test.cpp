/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/math/transform.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Helper to create buffer_info for testing accessors without numpy
template <typename T>
py::buffer_info createTestBuffer(const std::vector<py::ssize_t>& shape) {
  // Calculate total size and C-contiguous strides
  py::ssize_t totalSize = 1;
  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t stride = sizeof(T);

  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
    totalSize *= shape[i];
  }

  // Allocate data (note: this is a simplified test helper; in real usage
  // the data would need to be kept alive for the lifetime of the accessor)
  static std::vector<T> data;
  data.resize(totalSize, T(0));

  return py::buffer_info(
      data.data(), sizeof(T), py::format_descriptor<T>::format(), shape.size(), shape, strides);
}

// NOTE: Accessor tests demonstrate that accessors can now work with plain buffer_info
// without requiring numpy arrays. In production Python code, you would pass py::array
// which automatically delegates to buffer_info via the convenience constructors.

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

// ============================================================================
// Geometry Accessor Tests (using buffer_info directly, no numpy required!)
// ============================================================================

TEST(ArrayAccessors, JointParametersAccessor_BufferInfo_Roundtrip) {
  const int nJoints = 3;
  const int nJointParams = nJoints * 7;

  // Create buffer with shape (2, 3, nJointParams) - flat format
  auto bufferInfo = createTestBuffer<double>({2, 3, nJointParams});
  pymomentum::LeadingDimensions leadingDims;
  leadingDims.dims = {2, 3};

  // Create accessor using buffer_info directly (no numpy!)
  pymomentum::JointParametersAccessor<double> accessor(
      bufferInfo, leadingDims, nJoints, pymomentum::JointParamsShape::Flat);

  // Create test joint parameters
  Eigen::VectorXd jpVec(nJointParams);
  for (int i = 0; i < nJointParams; ++i) {
    jpVec(i) = static_cast<double>(i * 0.1 + 2.5);
  }
  momentum::JointParametersT<double> originalParams(jpVec);

  // Set and get at multiple indices
  accessor.set({0, 1}, originalParams);
  accessor.set({1, 0}, originalParams);

  auto retrieved01 = accessor.get({0, 1});
  auto retrieved10 = accessor.get({1, 0});

  // Verify lossless round-trip
  for (int i = 0; i < nJointParams; ++i) {
    EXPECT_DOUBLE_EQ(originalParams.v(i), retrieved01.v(i));
    EXPECT_DOUBLE_EQ(originalParams.v(i), retrieved10.v(i));
  }
}

TEST(ArrayAccessors, ModelParametersAccessor_BufferInfo_Roundtrip) {
  const int nModelParams = 5;

  // Create buffer with shape (4, nModelParams)
  auto bufferInfo = createTestBuffer<float>({4, nModelParams});
  pymomentum::LeadingDimensions leadingDims;
  leadingDims.dims = {4};

  // Create accessor using buffer_info directly
  pymomentum::ModelParametersAccessor<float> accessor(bufferInfo, leadingDims, nModelParams);

  // Create test model parameters
  Eigen::VectorXf mpVec(nModelParams);
  for (int i = 0; i < nModelParams; ++i) {
    mpVec(i) = static_cast<float>(i * 1.5f);
  }
  momentum::ModelParametersT<float> originalParams(mpVec);

  // Set and get
  accessor.set({2}, originalParams);
  auto retrieved = accessor.get({2});

  // Verify lossless round-trip
  for (int i = 0; i < nModelParams; ++i) {
    EXPECT_FLOAT_EQ(originalParams.v(i), retrieved.v(i));
  }
}

TEST(ArrayAccessors, SkeletonStateAccessor_BufferInfo_Roundtrip) {
  const int nJoints = 3;

  // Create buffer with shape (2, nJoints, 8)
  auto bufferInfo = createTestBuffer<float>({2, nJoints, 8});
  pymomentum::LeadingDimensions leadingDims;
  leadingDims.dims = {2};

  // Create accessor using buffer_info directly
  pymomentum::SkeletonStateAccessor<float> accessor(bufferInfo, leadingDims, nJoints);

  // Create test transforms
  momentum::TransformListT<float> originalTransforms(nJoints);
  for (int i = 0; i < nJoints; ++i) {
    originalTransforms[i].translation = Eigen::Vector3f(i * 1.0f, i * 2.0f, i * 3.0f);
    originalTransforms[i].rotation = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
    originalTransforms[i].scale = 1.0f + i * 0.1f;
  }

  // Set and get
  accessor.setTransforms({1}, originalTransforms);
  auto retrieved = accessor.getTransforms({1});

  // Verify lossless round-trip
  ASSERT_EQ(originalTransforms.size(), retrieved.size());
  for (size_t i = 0; i < originalTransforms.size(); ++i) {
    EXPECT_FLOAT_EQ(originalTransforms[i].translation.x(), retrieved[i].translation.x());
    EXPECT_FLOAT_EQ(originalTransforms[i].translation.y(), retrieved[i].translation.y());
    EXPECT_FLOAT_EQ(originalTransforms[i].translation.z(), retrieved[i].translation.z());
    EXPECT_FLOAT_EQ(originalTransforms[i].rotation.w(), retrieved[i].rotation.w());
    EXPECT_FLOAT_EQ(originalTransforms[i].rotation.x(), retrieved[i].rotation.x());
    EXPECT_FLOAT_EQ(originalTransforms[i].rotation.y(), retrieved[i].rotation.y());
    EXPECT_FLOAT_EQ(originalTransforms[i].rotation.z(), retrieved[i].rotation.z());
    EXPECT_FLOAT_EQ(originalTransforms[i].scale, retrieved[i].scale);
  }
}
