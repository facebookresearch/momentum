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
  EXPECT_TRUE(capsuleFloat.transformation.isApprox(TransformT<float>()));
  EXPECT_TRUE(capsuleFloat.radius.isApprox(Vector2<float>::Zero()));
  EXPECT_EQ(capsuleFloat.parent, kInvalidIndex);
  EXPECT_FLOAT_EQ(capsuleFloat.length, 0.0f);

  // Test with double precision
  TaperedCapsuleT<double> capsuleDouble;
  EXPECT_TRUE(capsuleDouble.transformation.isApprox(TransformT<double>()));
  EXPECT_TRUE(capsuleDouble.radius.isApprox(Vector2<double>::Zero()));
  EXPECT_EQ(capsuleDouble.parent, kInvalidIndex);
  EXPECT_DOUBLE_EQ(capsuleDouble.length, 0.0);

  CollisionEllipsoidT<float> ellipsoidFloat;
  EXPECT_TRUE(ellipsoidFloat.transformation.isApprox(TransformT<float>()));
  EXPECT_TRUE(ellipsoidFloat.radii.isApprox(Vector3<float>::Zero()));
  EXPECT_EQ(ellipsoidFloat.parent, kInvalidIndex);

  CollisionBoxT<float> boxFloat;
  EXPECT_TRUE(boxFloat.transformation.isApprox(TransformT<float>()));
  EXPECT_TRUE(boxFloat.halfExtents.isApprox(Vector3<float>::Zero()));
  EXPECT_EQ(boxFloat.parent, kInvalidIndex);

  CollisionPrimitiveT<float> primitiveFloat;
  EXPECT_EQ(primitiveFloat.type, CollisionPrimitiveType::TaperedCapsule);
  EXPECT_TRUE(primitiveFloat.transformation.isApprox(TransformT<float>()));
  EXPECT_TRUE(primitiveFloat.radius.isApprox(Vector2<float>::Zero()));
  EXPECT_TRUE(primitiveFloat.ellipsoidRadii.isApprox(Vector3<float>::Zero()));
  EXPECT_TRUE(primitiveFloat.boxHalfExtents.isApprox(Vector3<float>::Zero()));
  EXPECT_EQ(primitiveFloat.parent, kInvalidIndex);
  EXPECT_FLOAT_EQ(primitiveFloat.length, 0.0f);
}

// Test the isApprox method of TaperedCapsuleT
TEST(CollisionGeometryTest, IsApprox) {
  // Create two identical capsules
  TaperedCapsuleT<float> capsule1;
  TaperedCapsuleT<float> capsule2;

  // They should be approximately equal
  EXPECT_TRUE(capsule1.isApprox(capsule2));

  // Modify capsule2's transformation
  capsule2.transformation.translation = Vector3<float>(1.0f, 0.0f, 0.0f);
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
  capsule2.transformation.translation = Vector3<float>(1e-5f, 0.0f, 0.0f);
  EXPECT_TRUE(
      capsule1.isApprox(capsule2, 1e-4f)); // Should be approximately equal with this tolerance
  EXPECT_FALSE(
      capsule1.isApprox(capsule2, 1e-6f)); // Should not be approximately equal with this tolerance

  CollisionEllipsoidT<float> ellipsoid1;
  CollisionEllipsoidT<float> ellipsoid2;
  EXPECT_TRUE(ellipsoid1.isApprox(ellipsoid2));

  ellipsoid2.radii = Vector3<float>(1.0f, 2.0f, 3.0f);
  EXPECT_FALSE(ellipsoid1.isApprox(ellipsoid2));

  CollisionBoxT<float> box1;
  CollisionBoxT<float> box2;
  EXPECT_TRUE(box1.isApprox(box2));

  box2.halfExtents = Vector3<float>(1.0f, 2.0f, 3.0f);
  EXPECT_FALSE(box1.isApprox(box2));

  box2 = CollisionBoxT<float>();
  box2.transformation.translation = Vector3<float>(1.0f, 0.0f, 0.0f);
  EXPECT_FALSE(box1.isApprox(box2));

  box2 = CollisionBoxT<float>();
  box2.parent = 1;
  EXPECT_FALSE(box1.isApprox(box2));

  CollisionBoxT<float> box3;
  box3.halfExtents = Vector3<float>(1.0f, 2.0f, 3.0f);
  CollisionPrimitiveT<float> boxPrimitive1(box1);
  CollisionPrimitiveT<float> boxPrimitive2(box1);
  EXPECT_TRUE(boxPrimitive1.isApprox(boxPrimitive2));

  boxPrimitive2 = box3;
  EXPECT_FALSE(boxPrimitive1.isApprox(boxPrimitive2));

  CollisionPrimitiveT<float> primitive1(capsule1);
  CollisionPrimitiveT<float> primitive2(capsule1);
  EXPECT_TRUE(primitive1.isApprox(primitive2));
  EXPECT_TRUE(primitive1.isTaperedCapsule());
  EXPECT_FALSE(primitive1.isEllipsoid());
  EXPECT_TRUE(primitive1.toTaperedCapsule().isApprox(capsule1));

  primitive2 = ellipsoid2;
  EXPECT_FALSE(primitive1.isApprox(primitive2));
  EXPECT_TRUE(primitive2.isEllipsoid());
  EXPECT_TRUE(primitive2.toEllipsoid().isApprox(ellipsoid2));

  primitive2 = box3;
  EXPECT_FALSE(primitive1.isApprox(primitive2));
  EXPECT_TRUE(primitive2.isBox());
  EXPECT_TRUE(primitive2.toBox().isApprox(box3));
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

  CollisionEllipsoidT<float> ellipsoidFloat;
  ellipsoidFloat.radii = Vector3<float>(1.0f, 2.0f, 3.0f);
  ellipsoidFloat.parent = 1;
  collisionGeometryFloat.push_back(ellipsoidFloat);

  EXPECT_EQ(collisionGeometryFloat.size(), 2);
  EXPECT_TRUE(collisionGeometryFloat[1].isEllipsoid());
  EXPECT_TRUE(collisionGeometryFloat[1].ellipsoidRadii.isApprox(Vector3<float>(1.0f, 2.0f, 3.0f)));
  EXPECT_EQ(collisionGeometryFloat[1].parent, 1);

  CollisionBoxT<float> boxFloat;
  boxFloat.halfExtents = Vector3<float>(0.5f, 1.0f, 1.5f);
  boxFloat.parent = 2;
  collisionGeometryFloat.push_back(boxFloat);

  EXPECT_EQ(collisionGeometryFloat.size(), 3);
  EXPECT_TRUE(collisionGeometryFloat[2].isBox());
  EXPECT_TRUE(collisionGeometryFloat[2].boxHalfExtents.isApprox(Vector3<float>(0.5f, 1.0f, 1.5f)));
  EXPECT_EQ(collisionGeometryFloat[2].parent, 2);

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
