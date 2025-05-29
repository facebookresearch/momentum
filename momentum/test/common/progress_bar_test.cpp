/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/progress_bar.h"

#include <gtest/gtest.h>

using namespace momentum;

TEST(ProgressBarTest, Constructor) {
  // Test with different prefixes and operation counts
  ProgressBar bar1("Test Bar", 100);
  EXPECT_EQ(bar1.getCurrentProgress(), 0);

  ProgressBar bar2("", 50);
  EXPECT_EQ(bar2.getCurrentProgress(), 0);

  ProgressBar bar3("Very Long Prefix That Might Exceed Buffer Limits", 1000);
  EXPECT_EQ(bar3.getCurrentProgress(), 0);

  // Test with edge cases
  ProgressBar bar4("Zero Operations", 0);
  EXPECT_EQ(bar4.getCurrentProgress(), 0);

  // Note: We don't test with negative operations since size_t is unsigned
  // and the underlying indicators library uses size_t
}

TEST(ProgressBarTest, Increment) {
  ProgressBar bar("Increment Test", 100);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  // Test default increment (1)
  bar.increment();
  EXPECT_EQ(bar.getCurrentProgress(), 1);

  // Test increment with specific values
  bar.increment(5);
  EXPECT_EQ(bar.getCurrentProgress(), 6);

  bar.increment(10);
  EXPECT_EQ(bar.getCurrentProgress(), 16);

  // Test increment with zero
  bar.increment(0);
  EXPECT_EQ(bar.getCurrentProgress(), 16); // Should remain unchanged

  // Note: We don't test with negative increments since size_t is unsigned
  // and the underlying indicators library uses size_t

  // Test increment beyond max
  int64_t currentProgress = bar.getCurrentProgress();
  bar.increment(200); // Should handle gracefully
  // Progress should be clamped to max or at least not decrease
  EXPECT_GE(bar.getCurrentProgress(), currentProgress);
}

TEST(ProgressBarTest, Set) {
  ProgressBar bar("Set Test", 100);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  // Test setting to specific values
  bar.set(0);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  bar.set(50);
  EXPECT_EQ(bar.getCurrentProgress(), 50);

  bar.set(100);
  EXPECT_EQ(bar.getCurrentProgress(), 100);

  // Test setting to values outside the range
  // Note: We don't test with negative values since size_t is unsigned
  // and the underlying indicators library uses size_t

  bar.set(150); // Should handle gracefully

  // Test setting to the same value multiple times
  bar.set(75);
  EXPECT_EQ(bar.getCurrentProgress(), 75);

  bar.set(75);
  EXPECT_EQ(bar.getCurrentProgress(), 75);
}

TEST(ProgressBarTest, MixedOperations) {
  ProgressBar bar("Mixed Test", 100);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  // Mix increment and set operations
  bar.set(10);
  EXPECT_EQ(bar.getCurrentProgress(), 10);

  bar.increment(5);
  EXPECT_EQ(bar.getCurrentProgress(), 15);

  bar.set(0);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  bar.increment(20);
  EXPECT_EQ(bar.getCurrentProgress(), 20);

  bar.set(100);
  EXPECT_EQ(bar.getCurrentProgress(), 100);

  bar.increment(); // Beyond max
  // Progress should be clamped to max or at least not decrease
  EXPECT_GE(bar.getCurrentProgress(), 100);
}

TEST(ProgressBarTest, LargeValues) {
  // Test with large operation counts
  ProgressBar bar("Large Values", 1000000);
  EXPECT_EQ(bar.getCurrentProgress(), 0);

  bar.increment(10000);
  EXPECT_EQ(bar.getCurrentProgress(), 10000);

  bar.set(500000);
  EXPECT_EQ(bar.getCurrentProgress(), 500000);

  bar.increment(500000);
  EXPECT_EQ(bar.getCurrentProgress(), 1000000);
}
