/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <pymomentum/tensor_utility/tensor_utility.h>

TEST(ValidateTensor, CheckBatchDim) {
  const int nBatch = 3;

  pymomentum::TensorChecker checker("testFun1");
  // Establish batch dimension:
  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, 5, 3}, at::kFloat),
        "arg1",
        {5, 3},
        {"five", "three"},
        at::kFloat);
    ASSERT_EQ(3, res.ndimension());
  }

  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({nBatch, 6, 4}, at::kFloat),
        "arg2",
        {6, 4},
        {"six", "four"},
        at::kFloat);
    ASSERT_EQ(3, res.ndimension());
    ASSERT_EQ(nBatch, res.size(0));
  }

  {
    at::Tensor res = checker.validateAndFixTensor(
        at::zeros({2, 1}, at::kFloat),
        "arg2",
        {2, 1},
        {"two", "one"},
        at::kFloat);
    ASSERT_EQ(3, res.ndimension());
    ASSERT_EQ(nBatch, res.size(0));
  }

  {
    EXPECT_THROW(
        checker.validateAndFixTensor(
            at::zeros({nBatch + 1, 1}, at::kFloat),
            "arg3",
            {1},
            {"one"},
            at::kFloat),
        std::runtime_error);
  }
}

TEST(ValidateTensor, CheckBoundVariables) {
  const int nBatch = 3;

  const int v1_index = -1;
  const int v2_index = -2;
  const int v1_value = 5;
  const int v2_value = 3;

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
