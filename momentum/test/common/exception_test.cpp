/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/exception.h"

#include <gtest/gtest.h>

using namespace momentum;

TEST(ExceptionTest, ThrowImplFormattedMessage) {
  try {
    detail::throwImpl<std::runtime_error>("Error: {}", "test error");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ("Error: test error", e.what());
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, ThrowImplNoArguments) {
  try {
    detail::throwImpl<std::bad_array_new_length>();
    FAIL() << "Expected std::bad_array_new_length";
  } catch (const std::bad_array_new_length&) {
    EXPECT_TRUE(true) << "std::bad_array_new_length thrown as expected";
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, MT_THROW) {
  try {
    MT_THROW("Error occurred: {}", "critical");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ("Error occurred: critical", e.what());
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, MT_THROW_NoDefaultException) {
  try {
    MT_THROW_T(std::invalid_argument, "Error occurred: {}", "critical");
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ("Error occurred: critical", e.what());
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, MT_THROW_IFTrueCondition) {
  try {
    MT_THROW_IF(true, "Error because condition is {}", true);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ("Error because condition is true", e.what());
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, MT_THROW_IFFalseCondition) {
  try {
    MT_THROW_IF(false, "This should not throw");
    SUCCEED();
  } catch (...) {
    FAIL() << "No exception should be thrown";
  }
}

TEST(ExceptionTest, MT_THROW_IF_TrueConditionNoDefaultException) {
  try {
    MT_THROW_IF_T(true, std::invalid_argument, "Error because condition is {}", true);
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ("Error because condition is true", e.what());
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST(ExceptionTest, MT_THROW_IF_FalseConditionNoDefaultException) {
  try {
    MT_THROW_IF_T(false, std::invalid_argument, "This should not throw");
    SUCCEED();
  } catch (...) {
    FAIL() << "No exception should be thrown";
  }
}
