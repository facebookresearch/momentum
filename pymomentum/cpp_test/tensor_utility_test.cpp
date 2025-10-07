/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <pymomentum/tensor_utility/autograd_utility.h>
#include <pymomentum/tensor_utility/tensor_utility.h>

// Test toScalarType template function
TEST(TensorUtility, ToScalarType) {
  EXPECT_EQ(at::ScalarType::Float, pymomentum::toScalarType<float>());
  EXPECT_EQ(at::ScalarType::Double, pymomentum::toScalarType<double>());
  EXPECT_EQ(at::ScalarType::Int, pymomentum::toScalarType<int>());
  // Add more scalar types as needed
}

// Test formatTensorSizes functions
TEST(TensorUtility, FormatTensorSizes) {
  // Test formatTensorSizes with at::Tensor
  at::Tensor tensor = at::zeros({2, 3, 4});
  std::string tensorSizes = pymomentum::formatTensorSizes(tensor);
  EXPECT_EQ("[2, 3, 4]", tensorSizes);

  // Test formatTensorSizes with std::vector<pybind11::ssize_t>
  std::vector<pybind11::ssize_t> dims = {5, 6, 7};
  std::string vectorSizes = pymomentum::formatTensorSizes(dims);
  EXPECT_EQ("[5, 6, 7]", vectorSizes);

  // Test empty tensor
  at::Tensor emptyTensor = at::zeros({0});
  std::string emptySizes = pymomentum::formatTensorSizes(emptyTensor);
  EXPECT_EQ("[0]", emptySizes);
}

// Test isEmpty function
TEST(TensorUtility, IsEmpty) {
  at::Tensor emptyTensor = at::zeros({0});
  EXPECT_TRUE(pymomentum::isEmpty(emptyTensor));

  at::Tensor nonEmptyTensor = at::zeros({1, 2, 3});
  EXPECT_FALSE(pymomentum::isEmpty(nonEmptyTensor));
}

// Test denullify function
TEST(TensorUtility, Denullify) {
  // Test with nullopt
  std::optional<at::Tensor> nullTensor;
  at::Tensor result = pymomentum::denullify(nullTensor);
  EXPECT_TRUE(pymomentum::isEmpty(result));

  // Test with value
  at::Tensor tensor = at::ones({2, 3});
  std::optional<at::Tensor> optTensor(tensor);
  at::Tensor resultWithValue = pymomentum::denullify(optTensor);
  EXPECT_EQ(tensor.sizes(), resultWithValue.sizes());
  EXPECT_TRUE(at::equal(tensor, resultWithValue));
}

// Test toEigenMap function
TEST(TensorUtility, ToEigenMap) {
  // Test with float tensor
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  at::Tensor tensor = torch::tensor(data, torch::kFloat);
  auto eigenMap = pymomentum::toEigenMap<float>(tensor);

  EXPECT_EQ(data.size(), eigenMap.size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], eigenMap(i));
  }

  // Test with double tensor
  std::vector<double> doubleData = {1.0, 2.0, 3.0, 4.0};
  at::Tensor doubleTensor = torch::tensor(doubleData, torch::kDouble);
  auto doubleEigenMap = pymomentum::toEigenMap<double>(doubleTensor);

  EXPECT_EQ(doubleData.size(), doubleEigenMap.size());
  for (size_t i = 0; i < doubleData.size(); ++i) {
    EXPECT_DOUBLE_EQ(doubleData[i], doubleEigenMap(i));
  }

  // Test with int tensor
  std::vector<int> intData = {1, 2, 3, 4};
  at::Tensor intTensor = torch::tensor(intData, torch::kInt);
  auto intEigenMap = pymomentum::toEigenMap<int>(intTensor);

  EXPECT_EQ(intData.size(), intEigenMap.size());
  for (size_t i = 0; i < intData.size(); ++i) {
    EXPECT_EQ(intData[i], intEigenMap(i));
  }

  // Test type mismatch
  EXPECT_THROW(
      pymomentum::toEigenMap<float>(torch::tensor({1, 2, 3}, torch::kInt)), std::runtime_error);
}

// Test toMatrixList function
TEST(TensorUtility, ToMatrixList) {
  // Create a 3D tensor (2 x 4 x 4)
  at::Tensor tensor = torch::zeros({2, 4, 4}, torch::kFloat);

  // Fill with test data
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        tensor[i][j][k] = i * 16 + j * 4 + k;
      }
    }
  }

  // Convert to matrix list
  auto matrices = pymomentum::toMatrixList<float, 4, 4>(tensor);

  // Verify results
  EXPECT_EQ(2, matrices.size());

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_FLOAT_EQ(i * 16 + j * 4 + k, matrices[i](j, k));
      }
    }
  }

  // Test with wrong dimensions
  bool caught = false;
  try {
    at::Tensor wrongDimTensor = torch::zeros({2, 3}, torch::kFloat);
    pymomentum::toMatrixList<float, 4, 4>(wrongDimTensor);
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);

  // Test with wrong row count
  caught = false;
  try {
    at::Tensor wrongRowTensor = torch::zeros({2, 3, 4}, torch::kFloat);
    pymomentum::toMatrixList<float, 4, 4>(wrongRowTensor);
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);

  // Test with wrong column count
  caught = false;
  try {
    at::Tensor wrongColTensor = torch::zeros({2, 4, 3}, torch::kFloat);
    pymomentum::toMatrixList<float, 4, 4>(wrongColTensor);
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);
}

// Test to1DTensor functions
TEST(TensorUtility, To1DTensor) {
  // Test with raw pointer
  float rawData[] = {1.0f, 2.0f, 3.0f, 4.0f};
  at::Tensor rawTensor = pymomentum::to1DTensor(rawData, 4);
  EXPECT_EQ(1, rawTensor.ndimension());
  EXPECT_EQ(4, rawTensor.size(0));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(rawData[i], rawTensor[i].item<float>());
  }

  // Test with Eigen vector
  Eigen::Vector4f eigenVec;
  eigenVec << 5.0f, 6.0f, 7.0f, 8.0f;
  at::Tensor eigenTensor = pymomentum::to1DTensor(eigenVec);
  EXPECT_EQ(1, eigenTensor.ndimension());
  EXPECT_EQ(4, eigenTensor.size(0));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(eigenVec(i), eigenTensor[i].item<float>());
  }

  // Test with std::vector
  std::vector<float> vecData = {9.0f, 10.0f, 11.0f, 12.0f};
  at::Tensor vecTensor = pymomentum::to1DTensor(vecData);
  EXPECT_EQ(1, vecTensor.ndimension());
  EXPECT_EQ(4, vecTensor.size(0));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(vecData[i], vecTensor[i].item<float>());
  }

  // Test with different types
  std::vector<double> doubleData = {1.0, 2.0, 3.0};
  at::Tensor doubleTensor = pymomentum::to1DTensor(doubleData);
  EXPECT_EQ(at::ScalarType::Double, doubleTensor.scalar_type());

  std::vector<int> intData = {1, 2, 3};
  at::Tensor intTensor = pymomentum::to1DTensor(intData);
  EXPECT_EQ(at::ScalarType::Int, intTensor.scalar_type());
}

// Test to2DTensor functions
TEST(TensorUtility, To2DTensor) {
  // Test with row-major matrix
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> rowMajorMat;
  rowMajorMat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  at::Tensor rowMajorTensor = pymomentum::to2DTensor(rowMajorMat);
  EXPECT_EQ(2, rowMajorTensor.ndimension());
  EXPECT_EQ(3, rowMajorTensor.size(0));
  EXPECT_EQ(4, rowMajorTensor.size(1));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(rowMajorMat(i, j), rowMajorTensor[i][j].item<float>());
    }
  }

  // Test with column-major matrix
  Eigen::Matrix<float, 3, 4, Eigen::ColMajor> colMajorMat;
  colMajorMat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  at::Tensor colMajorTensor = pymomentum::to2DTensor(colMajorMat);
  EXPECT_EQ(2, colMajorTensor.ndimension());
  EXPECT_EQ(3, colMajorTensor.size(0));
  EXPECT_EQ(4, colMajorTensor.size(1));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(colMajorMat(i, j), colMajorTensor[i][j].item<float>());
    }
  }

  // Test with different types
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> doubleMat;
  doubleMat << 1.0, 2.0, 3.0, 4.0;
  at::Tensor doubleTensor = pymomentum::to2DTensor(doubleMat);
  EXPECT_EQ(at::ScalarType::Double, doubleTensor.scalar_type());

  Eigen::Matrix<int, 2, 2, Eigen::RowMajor> intMat;
  intMat << 1, 2, 3, 4;
  at::Tensor intTensor = pymomentum::to2DTensor(intMat);
  EXPECT_EQ(at::ScalarType::Int, intTensor.scalar_type());
}

// Test throwIfNaNOrINF function
TEST(TensorUtility, ThrowIfNaNOrINF) {
  // Test with valid tensor
  at::Tensor validTensor = torch::ones({2, 3}, torch::kFloat);
  EXPECT_NO_THROW(pymomentum::throwIfNaNOrINF(validTensor, "test_context", "test_tensor"));

  // Test with NaN tensor
  at::Tensor nanTensor = torch::ones({2, 3}, torch::kFloat);
  nanTensor[0][0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(
      pymomentum::throwIfNaNOrINF(nanTensor, "test_context", "test_tensor"), std::runtime_error);

  // Test with INF tensor
  at::Tensor infTensor = torch::ones({2, 3}, torch::kFloat);
  infTensor[0][0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(
      pymomentum::throwIfNaNOrINF(infTensor, "test_context", "test_tensor"), std::runtime_error);

  // Test with empty tensor
  at::Tensor emptyTensor = at::zeros({0});
  EXPECT_NO_THROW(pymomentum::throwIfNaNOrINF(emptyTensor, "test_context", "test_tensor"));
}

TEST(ValidateTensor, CheckBatchDim) {
  const size_t nBatch = 3;

  pymomentum::TensorChecker checker("testFun1");
  // Establish batch dimension:
  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, 5, 3}, at::kFloat), "arg1", {5, 3}, {"five", "three"}, at::kFloat);
    ASSERT_EQ(3, res.ndimension());
  }

  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, 6, 4}, at::kFloat), "arg2", {6, 4}, {"six", "four"}, at::kFloat);
    ASSERT_EQ(3, res.ndimension());
    ASSERT_EQ(nBatch, res.size(0));
  }

  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({2, 1}, at::kFloat), "arg2", {2, 1}, {"two", "one"}, at::kFloat);
    ASSERT_EQ(3, res.ndimension());
    ASSERT_EQ(nBatch, res.size(0));
  }

  {
    EXPECT_THROW(
        checker.validateAndFixTensor(
            at::zeros({nBatch + 1, 1}, at::kFloat), "arg3", {1}, {"one"}, at::kFloat),
        std::runtime_error);
  }
}

TEST(ValidateTensor, CheckBoundVariables) {
  const size_t nBatch = 3;

  const int v1_index = -1;
  const int v2_index = -2;
  const size_t v1_value = 5;
  const size_t v2_value = 3;

  pymomentum::TensorChecker checker("testFun1");
  // Establish v1:
  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, 5, v1_value}, at::kFloat),
        "arg1",
        {5, v1_index},
        {"five", "v2"},
        at::kFloat);
    ASSERT_EQ(3, res.ndimension());
  }

  // Establish v2:
  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, v2_value, 4}, at::kFloat),
        "arg2",
        {v2_index, 4},
        {"v2", "four"},
        at::kFloat);
    ASSERT_EQ(3, res.ndimension());
  }

  // Check v1 and v2:
  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, v2_value, 3, v1_value}, at::kFloat),
        "arg2",
        {v2_index, 3, v1_index},
        {"v2", "three", "v1"},
        at::kFloat);
    ASSERT_EQ(4, res.ndimension());
  }

  // Check mismatched v2:
  {
    EXPECT_THROW(
        checker.validateAndFixTensor(
            at::zeros({nBatch, 1, v2_value + 1}, at::kFloat),
            "arg3",
            {1, v2_index},
            {"one", "v2"},
            at::kFloat),
        std::runtime_error);
  }
}

// Additional tests for TensorChecker
TEST(ValidateTensor, AdditionalChecks) {
  pymomentum::TensorChecker checker("testFun2");

  // Test with empty tensor
  bool unsqueezed = false;
  EXPECT_THROW(
      checker.validateAndFixTensor(
          at::zeros({0}), "empty_tensor", {5}, {"five"}, at::kFloat, true, false, &unsqueezed),
      std::runtime_error);

  // Test with allowEmpty=true
  at::Tensor emptyResult = checker.validateAndFixTensor(
      at::zeros({0}), "empty_tensor", {5}, {"five"}, at::kFloat, true, true, &unsqueezed);
  EXPECT_TRUE(pymomentum::isEmpty(emptyResult));

  // Test needsSqueeze_out parameter with unbatched tensor
  bool needsSqueeze = false;
  at::Tensor unbatchedTensor = torch::ones({3}, torch::kFloat);
  at::Tensor result = checker.validateAndFixTensor(
      unbatchedTensor,
      "unbatched_tensor",
      {3},
      {"three"},
      at::kFloat,
      true, // allowUnbatched
      true, // allowEmpty
      &needsSqueeze);
  EXPECT_TRUE(needsSqueeze);
  EXPECT_EQ(2, result.ndimension()); // Should be batched now

  // Test needsSqueeze_out parameter with already batched tensor
  needsSqueeze = true; // Reset to check it gets set to false
  at::Tensor batchedTensor = torch::ones({1, 3}, torch::kFloat);
  result = checker.validateAndFixTensor(
      batchedTensor,
      "batched_tensor",
      {3},
      {"three"},
      at::kFloat,
      true, // allowUnbatched
      true, // allowEmpty
      &needsSqueeze);
  EXPECT_FALSE(needsSqueeze);
  EXPECT_EQ(2, result.ndimension());

  // Test with null needsSqueeze_out parameter
  result = checker.validateAndFixTensor(
      unbatchedTensor,
      "null_squeeze_param",
      {3},
      {"three"},
      at::kFloat,
      true, // allowUnbatched
      true, // allowEmpty
      nullptr); // Pass nullptr explicitly
  EXPECT_EQ(2, result.ndimension()); // Should still be batched

  // Test with nullptr in dimensionNames vector
  std::vector<int> expectedSizes = {5, 3};
  std::vector<const char*> dimensionNames = {"five", nullptr};
  result = checker.validateAndFixTensor(
      torch::ones({1, 5, 3}, torch::kFloat),
      "null_dimension_name",
      expectedSizes,
      dimensionNames,
      at::kFloat);
  EXPECT_EQ(3, result.ndimension());
  EXPECT_EQ(5, result.size(1));
  EXPECT_EQ(3, result.size(2));

  // Test with negative expected size and nullptr dimension name - this should
  // fail because we can't have a negative size with a null dimension name
  std::vector<int> negativeExpectedSizes = {5, -1};
  std::vector<const char*> nullDimensionNames = {"five", nullptr};

  // Create a tensor with dimensions that will cause validateAndFixTensor to
  // call formatExpectedDimensions with our negative expected size and null
  // dimension name This should trigger the MT_THROW_IF in
  // formatExpectedDimensions
  EXPECT_THROW(
      checker.validateAndFixTensor(
          torch::ones({0}), // Empty tensor to trigger the empty tensor check
          "negative_null_dimension",
          negativeExpectedSizes,
          nullDimensionNames,
          at::kFloat,
          true,
          false), // allowEmpty=false to force the error path
      std::runtime_error);

  // Test with wrong dimension count
  EXPECT_THROW(
      checker.validateAndFixTensor(
          at::zeros({2, 3, 4, 5}), "wrong_dim", {3, 4}, {"three", "four"}, at::kFloat),
      std::runtime_error);

  // Test with allowUnbatched=false
  EXPECT_THROW(
      checker.validateAndFixTensor(at::zeros({5}), "unbatched", {5}, {"five"}, at::kFloat, false),
      std::runtime_error);

  // Test type conversion
  at::Tensor intTensor = torch::ones({2, 3}, torch::kInt);
  at::Tensor convertedTensor =
      checker.validateAndFixTensor(intTensor, "convert_type", {2, 3}, {"two", "three"}, at::kFloat);
  EXPECT_EQ(at::kFloat, convertedTensor.scalar_type());

  // Test getBatchSize and getBoundValue
  pymomentum::TensorChecker checker2("testFun3");
  checker2.validateAndFixTensor(at::zeros({4, 5, 6}), "test", {5, -1}, {"five", "var"}, at::kFloat);

  EXPECT_EQ(4, checker2.getBatchSize());
  EXPECT_EQ(6, checker2.getBoundValue(-1));

  // Test invalid getBatchSize
  pymomentum::TensorChecker checker3("testFun4");
  EXPECT_THROW(checker3.getBatchSize(), std::runtime_error);

  // Test invalid getBoundValue
  EXPECT_THROW(checker2.getBoundValue(-2), std::runtime_error);
}

// Tests for autograd_utility.h

// Test hasFloat64 function
TEST(AutogradUtility, HasFloat64) {
  // Test with float tensor
  at::Tensor floatTensor = torch::ones({2, 3}, torch::kFloat);
  EXPECT_FALSE(pymomentum::hasFloat64(floatTensor));

  // Test with double tensor
  at::Tensor doubleTensor = torch::ones({2, 3}, torch::kDouble);
  EXPECT_TRUE(pymomentum::hasFloat64(doubleTensor));

  // Test with int tensor
  at::Tensor intTensor = torch::ones({2, 3}, torch::kInt);
  EXPECT_FALSE(pymomentum::hasFloat64(intTensor));

  // Test with non-tensor type
  int nonTensor = 5;
  EXPECT_FALSE(pymomentum::hasFloat64(nonTensor));

  // Test with multiple arguments, none are double
  EXPECT_FALSE(pymomentum::hasFloat64(floatTensor, intTensor, nonTensor));

  // Test with multiple arguments, one is double
  EXPECT_TRUE(pymomentum::hasFloat64(floatTensor, doubleTensor, intTensor));

  // Test with multiple arguments, all are double
  at::Tensor doubleTensor2 = torch::ones({4, 5}, torch::kDouble);
  EXPECT_TRUE(pymomentum::hasFloat64(doubleTensor, doubleTensor2));
}

// Mock autograd function for testing applyTemplatedAutogradFunction
template <typename T>
struct MockAutogradFunction {
  static torch::autograd::variable_list apply(
      const at::Tensor& /* input1 */,
      const at::Tensor& /* input2 */) {
    // Return a tensor with scalar type matching the template parameter
    at::Tensor result = torch::ones({1}, at::CppTypeToScalarType<T>::value);
    return {result};
  }
};

// Test applyTemplatedAutogradFunction
TEST(AutogradUtility, ApplyTemplatedAutogradFunction) {
  // Test with float tensors
  at::Tensor floatTensor1 = torch::ones({2, 3}, torch::kFloat);
  at::Tensor floatTensor2 = torch::ones({3, 4}, torch::kFloat);

  auto floatResult =
      pymomentum::applyTemplatedAutogradFunction<MockAutogradFunction>(floatTensor1, floatTensor2);

  EXPECT_EQ(1, floatResult.size());
  EXPECT_EQ(at::kFloat, floatResult[0].scalar_type());

  // Test with one double tensor
  at::Tensor doubleTensor = torch::ones({2, 3}, torch::kDouble);

  auto mixedResult =
      pymomentum::applyTemplatedAutogradFunction<MockAutogradFunction>(floatTensor1, doubleTensor);

  EXPECT_EQ(1, mixedResult.size());
  EXPECT_EQ(at::kDouble, mixedResult[0].scalar_type());

  // Test with both double tensors
  at::Tensor doubleTensor2 = torch::ones({3, 4}, torch::kDouble);

  auto doubleResult =
      pymomentum::applyTemplatedAutogradFunction<MockAutogradFunction>(doubleTensor, doubleTensor2);

  EXPECT_EQ(1, doubleResult.size());
  EXPECT_EQ(at::kDouble, doubleResult[0].scalar_type());
}
