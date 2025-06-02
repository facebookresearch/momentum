/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/locator.h"

namespace momentum {

class LocatorTest : public ::testing::Test {};

// Test default constructor
TEST_F(LocatorTest, DefaultConstructor) {
  Locator locator;

  EXPECT_EQ(locator.name, "uninitialized");
  EXPECT_EQ(locator.parent, kInvalidIndex);
  EXPECT_TRUE(locator.offset.isZero());
  EXPECT_TRUE(locator.locked.isZero());
  EXPECT_FLOAT_EQ(locator.weight, 1.0f);
  EXPECT_TRUE(locator.limitOrigin.isZero());
  EXPECT_TRUE(locator.limitWeight.isZero());
}

// Test constructor with custom parameters
TEST_F(LocatorTest, CustomConstructor) {
  std::string name = "test_locator";
  size_t parent = 1;
  Vector3f offset(1.0f, 2.0f, 3.0f);
  Vector3i locked(1, 0, 1);
  float weight = 0.5f;
  Vector3f limitOrigin(0.1f, 0.2f, 0.3f);
  Vector3f limitWeight(0.4f, 0.5f, 0.6f);

  Locator locator(name, parent, offset, locked, weight, limitOrigin, limitWeight);

  EXPECT_EQ(locator.name, name);
  EXPECT_EQ(locator.parent, parent);
  EXPECT_TRUE(locator.offset.isApprox(offset));
  EXPECT_TRUE(locator.locked.isApprox(locked));
  EXPECT_FLOAT_EQ(locator.weight, weight);
  EXPECT_TRUE(locator.limitOrigin.isApprox(limitOrigin));
  EXPECT_TRUE(locator.limitWeight.isApprox(limitWeight));
}

// Test equality operator with identical locators
TEST_F(LocatorTest, EqualityOperatorIdentical) {
  std::string name = "test_locator";
  size_t parent = 1;
  Vector3f offset(1.0f, 2.0f, 3.0f);
  Vector3i locked(1, 0, 1);
  float weight = 0.5f;
  Vector3f limitOrigin(0.1f, 0.2f, 0.3f);
  Vector3f limitWeight(0.4f, 0.5f, 0.6f);

  Locator locator1(name, parent, offset, locked, weight, limitOrigin, limitWeight);
  Locator locator2(name, parent, offset, locked, weight, limitOrigin, limitWeight);

  EXPECT_TRUE(locator1 == locator2);
}

// Test equality operator with different locators
TEST_F(LocatorTest, EqualityOperatorDifferent) {
  Locator locator1("locator1", 1, Vector3f(1.0f, 0.0f, 0.0f));
  Locator locator2("locator2", 1, Vector3f(1.0f, 0.0f, 0.0f));
  Locator locator3("locator1", 2, Vector3f(1.0f, 0.0f, 0.0f));
  Locator locator4("locator1", 1, Vector3f(2.0f, 0.0f, 0.0f));
  Locator locator5("locator1", 1, Vector3f(1.0f, 0.0f, 0.0f), Vector3i(1, 0, 0));
  Locator locator6("locator1", 1, Vector3f(1.0f, 0.0f, 0.0f), Vector3i::Zero(), 0.5f);
  Locator locator7(
      "locator1",
      1,
      Vector3f(1.0f, 0.0f, 0.0f),
      Vector3i::Zero(),
      1.0f,
      Vector3f(0.1f, 0.0f, 0.0f));
  Locator locator8(
      "locator1",
      1,
      Vector3f(1.0f, 0.0f, 0.0f),
      Vector3i::Zero(),
      1.0f,
      Vector3f::Zero(),
      Vector3f(0.1f, 0.0f, 0.0f));

  // Different name
  EXPECT_FALSE(locator1 == locator2);
  // Different parent
  EXPECT_FALSE(locator1 == locator3);
  // Different offset
  EXPECT_FALSE(locator1 == locator4);
  // Different locked
  EXPECT_FALSE(locator1 == locator5);
  // Different weight
  EXPECT_FALSE(locator1 == locator6);
  // Different limitOrigin
  EXPECT_FALSE(locator1 == locator7);
  // Different limitWeight
  EXPECT_FALSE(locator1 == locator8);
}

// Test equality operator with approximately equal values
TEST_F(LocatorTest, EqualityOperatorApproxEqual) {
  // Create two locators with very close but not identical values
  Vector3f offset1(1.0f, 2.0f, 3.0f);
  Vector3f offset2(1.0f + 1e-7f, 2.0f - 1e-7f, 3.0f + 1e-7f);

  Vector3f limitOrigin1(0.1f, 0.2f, 0.3f);
  Vector3f limitOrigin2(0.1f + 1e-7f, 0.2f - 1e-7f, 0.3f + 1e-7f);

  Vector3f limitWeight1(0.4f, 0.5f, 0.6f);
  Vector3f limitWeight2(0.4f + 1e-7f, 0.5f - 1e-7f, 0.6f + 1e-7f);

  float weight1 = 0.5f;
  float weight2 = 0.5f + 1e-7f;

  Locator locator1("test", 1, offset1, Vector3i(1, 0, 1), weight1, limitOrigin1, limitWeight1);
  Locator locator2("test", 1, offset2, Vector3i(1, 0, 1), weight2, limitOrigin2, limitWeight2);

  // They should be considered equal due to approximate comparison
  EXPECT_TRUE(locator1 == locator2);
}

} // namespace momentum
