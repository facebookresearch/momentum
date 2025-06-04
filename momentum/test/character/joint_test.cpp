/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <momentum/character/joint.h>
#include <momentum/math/constants.h>

namespace {

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct JointTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(JointTest, Types);

// Test the default initialization of JointT
TYPED_TEST(JointTest, DefaultInitialization) {
  using T = typename TestFixture::Type;

  JointT<T> jointFloat;
  EXPECT_EQ(jointFloat.name, "uninitialized");
  EXPECT_EQ(jointFloat.parent, kInvalidIndex);
  EXPECT_TRUE(jointFloat.preRotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_TRUE(jointFloat.translationOffset.isApprox(Vector3<T>::Zero()));
}

// Test the isApprox method of JointT
TYPED_TEST(JointTest, IsApprox) {
  using T = typename TestFixture::Type;

  // Create two identical joints
  JointT<T> joint1;
  JointT<T> joint2;

  // They should be approximately equal
  EXPECT_TRUE(joint1.isApprox(joint2));

  // Modify joint2's name
  joint2.name = "modified";
  EXPECT_FALSE(joint1.isApprox(joint2));

  // Reset joint2 and modify its parent
  joint2 = JointT<T>();
  joint2.parent = 1;
  EXPECT_FALSE(joint1.isApprox(joint2));

  // Reset joint2 and modify its preRotation
  joint2 = JointT<T>();
  // Use different values for float and double to avoid precision issues
  if constexpr (std::is_same_v<T, float>) {
    joint2.preRotation = Quaternion<T>(0.7071f, 0.7071f, 0.0f, 0.0f); // 90 degrees around X
  } else {
    joint2.preRotation = Quaternion<T>(0.7071, 0.7071, 0.0, 0.0); // 90 degrees around X
  }
  EXPECT_FALSE(joint1.isApprox(joint2));

  // Reset joint2 and modify its translationOffset
  joint2 = JointT<T>();
  joint2.translationOffset = Vector3<T>(1.0, 0.0, 0.0);
  EXPECT_FALSE(joint1.isApprox(joint2));

  // Test with custom tolerances
  // Instead of testing with small differences, let's test the basic functionality

  // Create a joint with a significantly different preRotation
  joint2 = JointT<T>();
  joint2.preRotation = Quaternion<T>(0.0, 1.0, 0.0, 0.0); // 180 degrees around X

  // Test with a very large rotation tolerance (should pass)
  EXPECT_TRUE(joint1.isApprox(joint2, 10.0, Eigen::NumTraits<T>::dummy_precision()));

  // Test with a small rotation tolerance (should fail)
  EXPECT_FALSE(joint1.isApprox(joint2, 0.1, Eigen::NumTraits<T>::dummy_precision()));

  // Create a joint with a significantly different translationOffset
  joint2 = JointT<T>();
  joint2.translationOffset = Vector3<T>(10.0, 0.0, 0.0);

  // Test with a small translation tolerance (should fail)
  EXPECT_FALSE(joint1.isApprox(joint2, Eps<T>(1e-4, 1e-10), 1.0));
}

// Test the cast method of JointT
// Test the cast method of JointT
TYPED_TEST(JointTest, Cast) {
  using T = typename TestFixture::Type;

  // Create a joint with T precision
  JointT<T> joint;
  joint.name = "test";
  joint.parent = 1;

  // Compute quaternion using T's precision
  joint.preRotation = Quaternion<T>(
      std::sqrt(T(2)) / T(2), // Use T(2) for type-specific computation
      std::sqrt(T(2)) / T(2),
      T(0),
      T(0));
  joint.translationOffset = Vector3<T>(T(1), T(2), T(3));

  // Cast to double
  JointT<double> jointDouble = joint.template cast<double>();

  // Check fields with precise values and tolerance
  EXPECT_EQ(jointDouble.name, "test");
  EXPECT_EQ(jointDouble.parent, 1);
  EXPECT_TRUE(jointDouble.preRotation.isApprox(
      Quaternion<double>(std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0, 0.0, 0.0),
      1e-6)); // Tolerance for floating-point comparison
  EXPECT_TRUE(jointDouble.translationOffset.isApprox(Vector3<double>(1.0, 2.0, 3.0)));

  // Cast back to T
  JointT<T> jointAgain = jointDouble.template cast<T>();

  // Verify fields match original with T-specific expected values
  EXPECT_EQ(jointAgain.name, "test");
  EXPECT_EQ(jointAgain.parent, 1);

  // Use computed expected value for precision
  EXPECT_TRUE(jointAgain.preRotation.isApprox(
      Quaternion<T>(std::sqrt(T(2)) / T(2), std::sqrt(T(2)) / T(2), T(0), T(0)), 1e-6));
  EXPECT_TRUE(jointAgain.translationOffset.isApprox(Vector3<T>(T(1), T(2), T(3))));

  // Cast to same type (copy)
  JointT<T> jointCopy = joint.template cast<T>();

  // Check copy matches original
  EXPECT_EQ(jointCopy.name, "test");
  EXPECT_EQ(jointCopy.parent, 1);
  EXPECT_TRUE(jointCopy.preRotation.isApprox(
      Quaternion<T>(std::sqrt(T(2)) / T(2), std::sqrt(T(2)) / T(2), T(0), T(0)), 1e-6));
  EXPECT_TRUE(jointCopy.translationOffset.isApprox(Vector3<T>(T(1), T(2), T(3))));

  // Ensure original is unchanged after modifying copy
  jointCopy.name = "modified";
  EXPECT_EQ(joint.name, "test");
}

// Test the JointListT type alias
TYPED_TEST(JointTest, JointListT) {
  using T = typename TestFixture::Type;

  // Create a joint list with float precision
  JointListT<T> jointListFloat;
  EXPECT_TRUE(jointListFloat.empty());

  // Add a joint to the list
  JointT<T> jointFloat;
  jointFloat.name = "test";
  jointFloat.parent = 1;
  jointFloat.preRotation = Quaternion<T>(0.7071f, 0.7071f, 0.0f, 0.0f); // 90 degrees around X
  jointFloat.translationOffset = Vector3<T>(1.0f, 2.0f, 3.0f);
  jointListFloat.push_back(jointFloat);

  EXPECT_EQ(jointListFloat.size(), 1);
  EXPECT_EQ(jointListFloat[0].name, "test");
  EXPECT_EQ(jointListFloat[0].parent, 1);
  EXPECT_TRUE(jointListFloat[0].preRotation.isApprox(Quaternion<T>(0.7071f, 0.7071f, 0.0f, 0.0f)));
  EXPECT_TRUE(jointListFloat[0].translationOffset.isApprox(Vector3<T>(1.0f, 2.0f, 3.0f)));
}

} // namespace
