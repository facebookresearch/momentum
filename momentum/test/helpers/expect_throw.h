/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <string>

#define EXPECT_THROW_WITH_MESSAGE(callable, ExceptionType, matcher)            \
  {                                                                            \
    EXPECT_THROW(                                                              \
        try { callable(); } catch (const ExceptionType& e) {                   \
          EXPECT_THAT(std::string{e.what()}, matcher);                         \
          /* re-throw the current exception to use the gtest provided macro */ \
          throw;                                                               \
        },                                                                     \
        ExceptionType);                                                        \
  }

#if defined(MOMENTUM_WITH_XR_LOGGER)
#define MOMENTUM_EXPECT_DEATH(statement, message) EXPECT_DEATH_IF_SUPPORTED(statement, message)
#else
#if defined(NDEBUG) // assert() is disabled
#define MOMENTUM_EXPECT_DEATH(statement, message)                                 \
  do {                                                                            \
    GTEST_MESSAGE_("Death test is skipped", ::testing::TestPartResult::kSuccess); \
  } while (0)
#else
#define MOMENTUM_EXPECT_DEATH(statement, message) EXPECT_DEATH_IF_SUPPORTED(statement, message)
#endif // NDEBUG
#endif // MOMENTUM_WITH_XR_LOGGER
