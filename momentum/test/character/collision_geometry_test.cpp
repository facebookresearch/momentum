/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/collision_geometry.h>
#include <momentum/math/constants.h>

#include <gtest/gtest.h>

namespace {

using namespace momentum;

// Test the default constructor of TaperedCapsuleT
TEST(CollisionGeometryTest, DefaultConstructor) {
  // Test with float precision
  TaperedCapsuleT<float> capsuleFloat;
  EXPECT_TRUE(capsuleFloat.transformation.isApprox(Affine3<float>::Identity()));
  EXPECT_TRUE(capsuleFloat.radius.isApprox(Vector2<float>::Zero()));
  EXPECT_EQ(capsuleFloat.parent, kInvalidIndex);
  EXPECT_FLOAT_EQ(capsuleFloat.length, 0.0f);

  // Test with double precision
  TaperedCapsuleT<double> capsuleDouble;
  EXPECT_TRUE(capsuleDouble.transformation.isApprox(Affine3<double>::Identity()));
  EXPECT_TRUE(capsuleDouble.radius.isApprox(Vector2<double>::Zero()));
  EXPECT_EQ(capsuleDouble.parent, kInvalidIndex);
  EXPECT_DOUBLE_EQ(capsuleDouble.length, 0.0);
}

// Test the isApprox method of TaperedCapsuleT
TEST(CollisionGeometryTest, IsApprox) {
  // Create two identical capsules
  TaperedCapsuleT<float> capsule1;
  TaperedCapsuleT<float> capsule2;

  // They should be approximately equal
  EXPECT_TRUE(capsule1.isApprox(capsule2));

  // Modify capsule2's transformation
  capsule2.transformation.translation() = Vector3<float>(1.0f, 0.0f, 0.0f);
  EXPECT_FALSE(capsule1.isApprox(capsule2));

  // Reset capsule2 and modify its radius
  capsule2 = TaperedCapsuleT<float>();
  capsule2.radius = Vector2<float>(1.0f, 2.0f);
  EXPECT_FALSE(capsule1.isApprox(capsule2));

  // Reset capsule2 and modify its parent
  capsule2 = TaperedCapsuleT<float>();
  capsule2.parent = 1;
  EXPECT_FALSE(capsule1.isApprox(capsule2));

  // Reset capsule2 and modify its length
  capsule2 = TaperedCapsuleT<float>();
  capsule2.length = 1.0f;
  EXPECT_FALSE(capsule1.isApprox(capsule2));

  // Test with a custom tolerance
  capsule2 = TaperedCapsuleT<float>();
  capsule2.transformation.translation() = Vector3<float>(1e-5f, 0.0f, 0.0f);
  EXPECT_TRUE(
      capsule1.isApprox(capsule2, 1e-4f)); // Should be approximately equal with this tolerance
  EXPECT_FALSE(
      capsule1.isApprox(capsule2, 1e-6f)); // Should not be approximately equal with this tolerance
}

// Test the CollisionGeometryT type alias
TEST(CollisionGeometryTest, CollisionGeometryT) {
  // Create a collision geometry with float precision
  CollisionGeometryT<float> collisionGeometryFloat;
  EXPECT_TRUE(collisionGeometryFloat.empty());

  // Add a capsule to the collision geometry
  TaperedCapsuleT<float> capsuleFloat;
  capsuleFloat.radius = Vector2<float>(1.0f, 2.0f);
  capsuleFloat.length = 3.0f;
  capsuleFloat.parent = 0;
  collisionGeometryFloat.push_back(capsuleFloat);

  EXPECT_EQ(collisionGeometryFloat.size(), 1);
  EXPECT_TRUE(collisionGeometryFloat[0].radius.isApprox(Vector2<float>(1.0f, 2.0f)));
  EXPECT_FLOAT_EQ(collisionGeometryFloat[0].length, 3.0f);
  EXPECT_EQ(collisionGeometryFloat[0].parent, 0);

  // Create a collision geometry with double precision
  CollisionGeometryT<double> collisionGeometryDouble;
  EXPECT_TRUE(collisionGeometryDouble.empty());

  // Add a capsule to the collision geometry
  TaperedCapsuleT<double> capsuleDouble;
  capsuleDouble.radius = Vector2<double>(1.0, 2.0);
  capsuleDouble.length = 3.0;
  capsuleDouble.parent = 0;
  collisionGeometryDouble.push_back(capsuleDouble);

  EXPECT_EQ(collisionGeometryDouble.size(), 1);
  EXPECT_TRUE(collisionGeometryDouble[0].radius.isApprox(Vector2<double>(1.0, 2.0)));
  EXPECT_DOUBLE_EQ(collisionGeometryDouble[0].length, 3.0);
  EXPECT_EQ(collisionGeometryDouble[0].parent, 0);
}

// Test the CollisionGeometry and CollisionGeometryd type aliases
TEST(CollisionGeometryTest, TypeAliases) {
  // Test CollisionGeometry (float precision)
  CollisionGeometry collisionGeometry;
  EXPECT_TRUE(collisionGeometry.empty());

  // Add a capsule to the collision geometry
  TaperedCapsuleT<float> capsule;
  capsule.radius = Vector2<float>(1.0f, 2.0f);
  capsule.length = 3.0f;
  capsule.parent = 0;
  collisionGeometry.push_back(capsule);

  EXPECT_EQ(collisionGeometry.size(), 1);
  EXPECT_TRUE(collisionGeometry[0].radius.isApprox(Vector2<float>(1.0f, 2.0f)));
  EXPECT_FLOAT_EQ(collisionGeometry[0].length, 3.0f);
  EXPECT_EQ(collisionGeometry[0].parent, 0);

  // Test CollisionGeometryd (double precision)
  CollisionGeometryd collisionGeometryd;
  EXPECT_TRUE(collisionGeometryd.empty());

  // Add a capsule to the collision geometry
  TaperedCapsuleT<double> capsuled;
  capsuled.radius = Vector2<double>(1.0, 2.0);
  capsuled.length = 3.0;
  capsuled.parent = 0;
  collisionGeometryd.push_back(capsuled);

  EXPECT_EQ(collisionGeometryd.size(), 1);
  EXPECT_TRUE(collisionGeometryd[0].radius.isApprox(Vector2<double>(1.0, 2.0)));
  EXPECT_DOUBLE_EQ(collisionGeometryd[0].length, 3.0);
  EXPECT_EQ(collisionGeometryd[0].parent, 0);
}

} // namespace
