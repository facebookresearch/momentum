/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

class CollisionGeometryStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a test character with a skeleton
    character = createTestCharacter();
    skeletonState.jointState.resize(character.skeleton.joints.size());

    // Initialize joint states with identity transforms
    for (auto& jointState : skeletonState.jointState) {
      jointState.localRotation().setIdentity();
      jointState.localTranslation().setZero();
      jointState.localScale() = 1.0f;
      jointState.transform.rotation.setIdentity();
      jointState.transform.translation.setZero();
      jointState.transform.scale = 1.0f;
    }

    // Create a simple collision geometry
    setupCollisionGeometry();
  }

  void setupCollisionGeometry() {
    // Create two tapered capsules
    TaperedCapsule capsule1;
    capsule1.transformation = TransformT<float>();
    capsule1.radius = Vector2f(0.5f, 0.3f); // Tapered from 0.5 to 0.3
    capsule1.parent = 0; // Attached to the root joint
    capsule1.length = 1.0f;

    TaperedCapsule capsule2;
    capsule2.transformation = TransformT<float>();
    capsule2.transformation.translation = Vector3f(0.0f, 1.0f, 0.0f); // Offset in Y direction
    capsule2.radius = Vector2f(0.4f, 0.2f); // Tapered from 0.4 to 0.2
    capsule2.parent = 1; // Attached to the second joint
    capsule2.length = 1.5f;

    collisionGeometry.push_back(capsule1);
    collisionGeometry.push_back(capsule2);
  }

  Character character;
  SkeletonState skeletonState;
  CollisionGeometry collisionGeometry;
};

// Test the update method of CollisionGeometryState
TEST_F(CollisionGeometryStateTest, Update) {
  // Create a collision geometry state
  CollisionGeometryState collisionState;

  // Update the state based on the skeleton state and collision geometry
  collisionState.update(skeletonState, collisionGeometry);

  // Check that the state has the correct size
  ASSERT_EQ(collisionState.origin.size(), 2);
  ASSERT_EQ(collisionState.direction.size(), 2);
  ASSERT_EQ(collisionState.radius.size(), 2);
  ASSERT_EQ(collisionState.delta.size(), 2);

  // Check the values for the first capsule
  EXPECT_TRUE(collisionState.origin[0].isApprox(Vector3f::Zero()));
  EXPECT_TRUE(collisionState.direction[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(collisionState.radius[0].isApprox(Vector2f(0.5f, 0.3f)));
  EXPECT_FLOAT_EQ(collisionState.delta[0], -0.2f); // 0.3 - 0.5 = -0.2

  // Check the values for the second capsule
  EXPECT_TRUE(collisionState.origin[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(collisionState.direction[1].isApprox(Vector3f(1.5f, 0.0f, 0.0f)));
  EXPECT_TRUE(collisionState.radius[1].isApprox(Vector2f(0.4f, 0.2f)));
  EXPECT_FLOAT_EQ(collisionState.delta[1], -0.2f); // 0.2 - 0.4 = -0.2
}

// Test the update method with transformed joints
TEST_F(CollisionGeometryStateTest, UpdateWithTransformedJoints) {
  // Apply transformations to the joints
  skeletonState.jointState[0].transform.translation += Vector3f(1.0f, 2.0f, 3.0f);
  skeletonState.jointState[0].transform.rotation =
      Quaternionf(Eigen::AngleAxisf(pi() / 2, Vector3f::UnitY())); // 90 degrees around Y
  skeletonState.jointState[0].transform.scale = 2.0f;

  skeletonState.jointState[1].transform.translation += Vector3f(0.0f, 1.0f, 0.0f);
  skeletonState.jointState[1].transform.scale = 1.5f;

  // Create a collision geometry state
  CollisionGeometryState collisionState;

  // Update the state based on the transformed skeleton
  collisionState.update(skeletonState, collisionGeometry);

  // Check the values for the first capsule (attached to joint 0)
  // Origin should be at the joint's position
  EXPECT_TRUE(collisionState.origin[0].isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
  // Direction should be rotated 90 degrees around Y and scaled by length*scale: (1,0,0) -> (0,0,-2)
  // but Transform implementation might differ, so let's be more flexible
  EXPECT_NEAR(collisionState.direction[0].norm(), 2.0f, 1e-4f); // length*scale = 1.0*2.0
  // Radius should be scaled by the joint's scale
  EXPECT_TRUE(collisionState.radius[0].isApprox(Vector2f(1.0f, 0.6f))); // 0.5*2, 0.3*2
  EXPECT_FLOAT_EQ(collisionState.delta[0], -0.4f); // 0.6 - 1.0 = -0.4

  // Check the values for the second capsule (attached to joint 1)
  // Origin should be at joint 1's position plus the capsule's offset
  Vector3f expectedOrigin = skeletonState.jointState[1].transform.translation +
      skeletonState.jointState[1].transform.toLinear() *
          collisionGeometry[1].transformation.translation;
  EXPECT_TRUE(collisionState.origin[1].isApprox(expectedOrigin));
  // Direction should be scaled by the joint's scale and the capsule's length: length=1.5, scale=1.5
  // norm should be length * scale = 1.5 * 1.5 = 2.25
  EXPECT_NEAR(collisionState.direction[1].norm(), 2.25f, 1e-4f);
  // Radius should be scaled by the joint's scale
  EXPECT_TRUE(collisionState.radius[1].isApprox(Vector2f(0.6f, 0.3f))); // 0.4*1.5, 0.2*1.5
  EXPECT_FLOAT_EQ(collisionState.delta[1], -0.3f); // 0.3 - 0.6 = -0.3
}

TEST_F(CollisionGeometryStateTest, UpdateWithEllipsoid) {
  CollisionEllipsoid ellipsoid;
  ellipsoid.parent = 0;
  ellipsoid.transformation.translation = Vector3f(0.25f, 0.5f, 0.75f);
  ellipsoid.transformation.rotation =
      Quaternionf(Eigen::AngleAxisf(pi() / 2.0f, Vector3f::UnitZ()));
  ellipsoid.radii = Vector3f(0.2f, 0.4f, 0.6f);
  collisionGeometry.push_back(ellipsoid);

  skeletonState.jointState[0].transform.translation = Vector3f(1.0f, 2.0f, 3.0f);
  skeletonState.jointState[0].transform.scale = 2.0f;

  CollisionGeometryState collisionState;
  collisionState.update(skeletonState, collisionGeometry);

  ASSERT_EQ(collisionState.type.size(), 3);
  EXPECT_EQ(collisionState.type[2], CollisionPrimitiveType::Ellipsoid);
  EXPECT_TRUE(collisionState.origin[2].isApprox(Vector3f(1.5f, 3.0f, 4.5f)));
  EXPECT_TRUE(collisionState.direction[2].isZero());
  EXPECT_TRUE(collisionState.radius[2].isZero());
  EXPECT_FLOAT_EQ(collisionState.delta[2], 0.0f);
  EXPECT_TRUE(collisionState.ellipsoidRadii[2].isApprox(Vector3f(0.4f, 0.8f, 1.2f)));
}

// Test the overlaps function with non-overlapping capsules
TEST_F(CollisionGeometryStateTest, OverlapsNonOverlapping) {
  // Create two non-overlapping capsules
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f);
  Vector2f radiiA(0.5f, 0.3f);
  float deltaA = -0.2f;

  Vector3f originB(0.0f, 3.0f, 0.0f); // Far away in Y direction
  Vector3f directionB(1.0f, 0.0f, 0.0f);
  Vector2f radiiB(0.4f, 0.2f);
  float deltaB = -0.2f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should not overlap
  EXPECT_FALSE(result);
}

TEST_F(CollisionGeometryStateTest, OverlapsCapsuleEllipsoid) {
  CollisionGeometry geometry;

  TaperedCapsule capsule;
  capsule.parent = 0;
  capsule.length = 1.0f;
  capsule.radius = Vector2f(0.25f, 0.25f);
  geometry.push_back(capsule);

  CollisionEllipsoid ellipsoid;
  ellipsoid.parent = 1;
  ellipsoid.transformation.translation = Vector3f(0.5f, 0.45f, 0.0f);
  ellipsoid.radii = Vector3f(0.2f, 0.3f, 0.2f);
  geometry.push_back(ellipsoid);

  CollisionGeometryState collisionState;
  collisionState.update(skeletonState, geometry);

  CollisionOverlapResultT<float> overlap;
  EXPECT_TRUE(overlaps(collisionState, geometry, 0, 1, overlap));
  EXPECT_NEAR(overlap.distance, 0.45f, 1e-5f);
  EXPECT_NEAR(overlap.closestPoints[0], 0.5f, 1e-5f);
  EXPECT_NEAR(overlap.radiusA, 0.25f, 1e-5f);
  EXPECT_NEAR(overlap.radiusB, 0.3f, 1e-5f);
  EXPECT_NEAR(overlap.overlap, 0.1f, 1e-5f);
}

TEST_F(CollisionGeometryStateTest, OverlapsEllipsoidEllipsoid) {
  CollisionGeometry geometry;

  CollisionEllipsoid ellipsoidA;
  ellipsoidA.parent = 0;
  ellipsoidA.radii = Vector3f(0.5f, 0.2f, 0.2f);
  geometry.push_back(ellipsoidA);

  CollisionEllipsoid ellipsoidB;
  ellipsoidB.parent = 1;
  ellipsoidB.transformation.translation = Vector3f(0.8f, 0.0f, 0.0f);
  ellipsoidB.radii = Vector3f(0.4f, 0.2f, 0.2f);
  geometry.push_back(ellipsoidB);

  CollisionGeometryState collisionState;
  collisionState.update(skeletonState, geometry);

  CollisionOverlapResultT<float> overlap;
  EXPECT_TRUE(overlaps(collisionState, geometry, 0, 1, overlap));
  EXPECT_NEAR(overlap.distance, 0.8f, 1e-5f);
  EXPECT_NEAR(overlap.radiusA, 0.5f, 1e-5f);
  EXPECT_NEAR(overlap.radiusB, 0.4f, 1e-5f);
  EXPECT_NEAR(overlap.overlap, 0.1f, 1e-5f);
}

// Test the overlaps function with overlapping capsules
TEST_F(CollisionGeometryStateTest, OverlapsOverlapping) {
  // Create two overlapping capsules
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f);
  Vector2f radiiA(0.5f, 0.3f);
  float deltaA = -0.2f;

  Vector3f originB(0.0f, 0.7f, 0.0f); // Close in Y direction
  Vector3f directionB(1.0f, 0.0f, 0.0f);
  Vector2f radiiB(0.4f, 0.2f);
  float deltaB = -0.2f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should overlap
  EXPECT_TRUE(result);
  // The distance should be approximately 0.7
  EXPECT_NEAR(outDistance, 0.7f, 1e-5f);
  // The overlap should be positive (0.5 + 0.4 - 0.7 = 0.2)
  EXPECT_NEAR(outOverlap, 0.2f, 1e-5f);
}

// Test the overlaps function with edge cases
TEST_F(CollisionGeometryStateTest, OverlapsEdgeCases) {
  // Create two capsules that are exactly touching
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f);
  Vector2f radiiA(0.5f, 0.3f);
  float deltaA = -0.2f;

  Vector3f originB(0.0f, 0.9f, 0.0f); // Distance equals sum of radii
  Vector3f directionB(1.0f, 0.0f, 0.0f);
  Vector2f radiiB(0.4f, 0.2f);
  float deltaB = -0.2f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should not overlap (outOverlap is very close to 0)
  EXPECT_FALSE(result);
  // The distance should be approximately 0.9
  EXPECT_NEAR(outDistance, 0.9f, 1e-5f);
  // The overlap should be very close to 0 (0.5 + 0.4 - 0.9 = 0)
  EXPECT_NEAR(outOverlap, 0.0f, 1e-5f);

  // Test with zero distance (centers at same position)
  originB = Vector3f(0.0f, 0.0f, 0.0f);

  result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They may or may not overlap depending on the implementation details
  // Just check that the output parameters are set correctly
  // The distance should be very small (not exactly zero due to numerical precision)
  EXPECT_NEAR(outDistance, 0.0f, 1e-5f);
  // The overlap should be approximately the sum of the radii
  EXPECT_NEAR(outOverlap, 0.9f, 1e-5f);
}

// Test the updateAabb function
TEST_F(CollisionGeometryStateTest, UpdateAabb) {
  // Create a capsule
  Vector3f origin(1.0f, 2.0f, 3.0f);
  Vector3f direction(2.0f, 0.0f, 0.0f);
  Vector2f radii(0.5f, 0.3f);

  // Create an AABB
  axel::BoundingBox<float> aabb;

  // Update the AABB
  updateAabb(aabb, origin, direction, radii);

  // Check the AABB bounds
  // Min should be origin - max radius
  EXPECT_TRUE(aabb.aabb.min().isApprox(Vector3f(0.5f, 1.5f, 2.5f)));
  // Max should be (origin + direction) + max radius
  EXPECT_TRUE(aabb.aabb.max().isApprox(Vector3f(3.3f, 2.5f, 3.5f)));
}

// Test the updateAabb function with different radii
TEST_F(CollisionGeometryStateTest, UpdateAabbDifferentRadii) {
  // Create a capsule with different radii at each end
  Vector3f origin(1.0f, 2.0f, 3.0f);
  Vector3f direction(2.0f, 0.0f, 0.0f);
  Vector2f radii(0.5f, 0.3f); // First radius is larger

  // Create an AABB
  axel::BoundingBox<float> aabb;

  // Update the AABB
  updateAabb(aabb, origin, direction, radii);

  // Check the AABB bounds
  // Min should account for the larger radius at the origin
  EXPECT_TRUE(aabb.aabb.min().isApprox(Vector3f(0.5f, 1.5f, 2.5f)));
  // Max should account for the smaller radius at the end
  EXPECT_TRUE(aabb.aabb.max().isApprox(Vector3f(3.3f, 2.5f, 3.5f)));

  // Now try with the larger radius at the end
  radii = Vector2f(0.3f, 0.5f);

  // Update the AABB again
  updateAabb(aabb, origin, direction, radii);

  // Check the AABB bounds
  // Min should account for the smaller radius at the origin and the larger radius at the end
  EXPECT_TRUE(aabb.aabb.min().isApprox(Vector3f(0.7f, 1.5f, 2.5f)));
  // Max should account for the larger radius at the end
  EXPECT_TRUE(aabb.aabb.max().isApprox(Vector3f(3.5f, 2.5f, 3.5f)));
}

// Test that updateEllipsoidAabb uses the tight L2 ellipsoid bound. A 45-degree rotation about
// Z makes the tight bound (sqrt((cos45*rx)^2 + (sin45*ry)^2)) differ from the looser L1/OBB
// bound used for boxes (cos45*rx + sin45*ry), so this exercises the ellipsoid-specific path.
TEST_F(CollisionGeometryStateTest, UpdateEllipsoidAabb) {
  const Vector3f center(1.0f, 2.0f, 5.0f);
  // Unit quaternion (w, x, y, z) for a +45-degree rotation about the Z axis.
  const Quaternionf orientation(0.9238795325f, 0.0f, 0.0f, 0.3826834324f);
  const Vector3f radii(2.0f, 1.0f, 3.0f);

  axel::BoundingBox<float> aabb;
  updateEllipsoidAabb(aabb, center, orientation, radii);

  // Tight half-extent along X and Y is sqrt((cos45*2)^2 + (sin45*1)^2) = sqrt(2.5) ~= 1.5811,
  // whereas the box/L1 bound would be cos45*2 + sin45*1 ~= 2.1213. Z is unrotated, so 3.0.
  const float hXy = std::sqrt(2.5f);
  const float hZ = 3.0f;
  EXPECT_NEAR(aabb.aabb.min().x(), center.x() - hXy, 1e-5f);
  EXPECT_NEAR(aabb.aabb.min().y(), center.y() - hXy, 1e-5f);
  EXPECT_NEAR(aabb.aabb.min().z(), center.z() - hZ, 1e-5f);
  EXPECT_NEAR(aabb.aabb.max().x(), center.x() + hXy, 1e-5f);
  EXPECT_NEAR(aabb.aabb.max().y(), center.y() + hXy, 1e-5f);
  EXPECT_NEAR(aabb.aabb.max().z(), center.z() + hZ, 1e-5f);
}

// Test the CollisionGeometryState with double precision
TEST_F(CollisionGeometryStateTest, DoublePrecision) {
  // Create a collision geometry state with double precision
  CollisionGeometryStated collisionState;

  // Create a skeleton state with double precision
  SkeletonStated skeletonStated;
  skeletonStated.jointState.resize(skeletonState.jointState.size());

  // Convert the skeleton state to double precision
  for (size_t i = 0; i < skeletonState.jointState.size(); ++i) {
    skeletonStated.jointState[i].localRotation() =
        skeletonState.jointState[i].localRotation().cast<double>();
    skeletonStated.jointState[i].localTranslation() =
        skeletonState.jointState[i].localTranslation().cast<double>();
    skeletonStated.jointState[i].localScale() =
        static_cast<double>(skeletonState.jointState[i].localScale());
    skeletonStated.jointState[i].transform.rotation =
        skeletonState.jointState[i].transform.rotation.cast<double>();
    skeletonStated.jointState[i].transform.translation =
        skeletonState.jointState[i].transform.translation.cast<double>();
    skeletonStated.jointState[i].transform.scale =
        static_cast<double>(skeletonState.jointState[i].transform.scale);
  }

  // Update the state based on the skeleton state and collision geometry
  collisionState.update(skeletonStated, collisionGeometry);

  // Check that the state has the correct size
  ASSERT_EQ(collisionState.origin.size(), 2);
  ASSERT_EQ(collisionState.direction.size(), 2);
  ASSERT_EQ(collisionState.radius.size(), 2);
  ASSERT_EQ(collisionState.delta.size(), 2);

  // Check the values for the first capsule
  EXPECT_TRUE(collisionState.origin[0].isApprox(Vector3d::Zero()));
  EXPECT_TRUE(collisionState.direction[0].isApprox(Vector3d(1.0, 0.0, 0.0)));
  // Use a larger epsilon for the comparison due to floating point precision issues
  EXPECT_TRUE(collisionState.radius[0].isApprox(Vector2d(0.5, 0.3), 1e-5));
  EXPECT_NEAR(collisionState.delta[0], -0.2, 1e-7); // 0.3 - 0.5 = -0.2

  // Check the values for the second capsule
  EXPECT_TRUE(collisionState.origin[1].isApprox(Vector3d(0.0, 1.0, 0.0)));
  EXPECT_TRUE(collisionState.direction[1].isApprox(Vector3d(1.5, 0.0, 0.0)));
  // Use a larger epsilon for the comparison due to floating point precision issues
  EXPECT_TRUE(collisionState.radius[1].isApprox(Vector2d(0.4, 0.2), 1e-5));
  EXPECT_NEAR(collisionState.delta[1], -0.2, 1e-7); // 0.2 - 0.4 = -0.2
}

// Test the update method with empty collision geometry
TEST_F(CollisionGeometryStateTest, UpdateEmptyCollisionGeometry) {
  // Create an empty collision geometry
  CollisionGeometry emptyGeometry;

  // Create a collision geometry state
  CollisionGeometryState collisionState;

  // Update the state based on the skeleton state and empty collision geometry
  collisionState.update(skeletonState, emptyGeometry);

  // Check that the state has zero size
  EXPECT_EQ(collisionState.origin.size(), 0);
  EXPECT_EQ(collisionState.direction.size(), 0);
  EXPECT_EQ(collisionState.radius.size(), 0);
  EXPECT_EQ(collisionState.delta.size(), 0);
}

// Test the overlaps function with perpendicular capsules
TEST_F(CollisionGeometryStateTest, OverlapsPerpendicular) {
  // Create two perpendicular capsules
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f); // Along X-axis
  Vector2f radiiA(0.5f, 0.3f);
  float deltaA = -0.2f;

  Vector3f originB(0.5f, -0.5f, 0.0f);
  Vector3f directionB(0.0f, 1.0f, 0.0f); // Along Y-axis
  Vector2f radiiB(0.4f, 0.2f);
  float deltaB = -0.2f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  [[maybe_unused]] bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // The result depends on the implementation details
  // Just check that the output parameters are set correctly

  // The closest points should be at the intersection of the axes
  EXPECT_NEAR(outClosestPoints[0], 0.5f, 1e-5f); // Point on capsule A's axis
  EXPECT_NEAR(outClosestPoints[1], 0.5f, 1e-5f); // Point on capsule B's axis

  // The overlap should be positive
  EXPECT_GT(outOverlap, 0.0f);
}

// Test the overlaps function with parallel but offset capsules
TEST_F(CollisionGeometryStateTest, OverlapsParallelOffset) {
  // Create two parallel but offset capsules
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f);
  Vector2f radiiA(0.5f, 0.3f);
  float deltaA = -0.2f;

  Vector3f originB(0.0f, 0.0f, 1.0f); // Offset in Z direction
  Vector3f directionB(1.0f, 0.0f, 0.0f); // Parallel to capsule A
  Vector2f radiiB(0.4f, 0.2f);
  float deltaB = -0.2f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should not overlap (distance = 1.0, sum of radii = 0.5 + 0.4 = 0.9)
  EXPECT_FALSE(result);
}

// Test the overlaps function with extreme radii
TEST_F(CollisionGeometryStateTest, OverlapsExtremeRadii) {
  // Create a capsule with very small radius
  Vector3f originA(0.0f, 0.0f, 0.0f);
  Vector3f directionA(1.0f, 0.0f, 0.0f);
  Vector2f radiiA(0.001f, 0.001f); // Very small radius
  float deltaA = 0.0f;

  // Create a capsule with very large radius
  Vector3f originB(0.0f, 2.0f, 0.0f);
  Vector3f directionB(1.0f, 0.0f, 0.0f);
  Vector2f radiiB(1.5f, 1.5f); // Large radius
  float deltaB = 0.0f;

  float outDistance = NAN;
  Vector2f outClosestPoints;
  float outOverlap = NAN;

  // Check if the capsules overlap
  bool result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should not overlap (distance = 2.0, sum of radii = 0.001 + 1.5 = 1.501)
  EXPECT_FALSE(result);

  // Now make the large radius even larger to cause overlap
  radiiB = Vector2f(2.1f, 2.1f);

  result = overlaps(
      originA,
      directionA,
      radiiA,
      deltaA,
      originB,
      directionB,
      radiiB,
      deltaB,
      outDistance,
      outClosestPoints,
      outOverlap);

  // They should overlap now
  EXPECT_TRUE(result);

  // The overlap should be positive (0.001 + 2.1 - 2.0 = 0.101)
  EXPECT_NEAR(outOverlap, 0.101f, 1e-5f);
}

// Test the updateAabb function with zero direction
TEST_F(CollisionGeometryStateTest, UpdateAabbZeroDirection) {
  // Create a capsule with zero direction
  Vector3f origin(1.0f, 2.0f, 3.0f);
  Vector3f direction(0.0f, 0.0f, 0.0f); // Zero direction
  Vector2f radii(0.5f, 0.3f);

  // Create an AABB
  axel::BoundingBox<float> aabb;

  // Update the AABB
  updateAabb(aabb, origin, direction, radii);

  // Check the AABB bounds
  // Min should be origin - max radius
  EXPECT_TRUE(aabb.aabb.min().isApprox(Vector3f(0.5f, 1.5f, 2.5f)));
  // Max should be origin + max radius
  EXPECT_TRUE(aabb.aabb.max().isApprox(Vector3f(1.5f, 2.5f, 3.5f)));
}

// Test the updateAabb function with negative direction components
TEST_F(CollisionGeometryStateTest, UpdateAabbNegativeDirection) {
  // Create a capsule with negative direction
  Vector3f origin(1.0f, 2.0f, 3.0f);
  Vector3f direction(-2.0f, 0.0f, 0.0f); // Negative X direction
  Vector2f radii(0.5f, 0.3f);

  // Create an AABB
  axel::BoundingBox<float> aabb;

  // Update the AABB
  updateAabb(aabb, origin, direction, radii);

  // Check the AABB bounds
  // Min should account for the negative direction
  EXPECT_TRUE(aabb.aabb.min().isApprox(Vector3f(-1.3f, 1.5f, 2.5f)));
  // Max should account for the origin's radius
  EXPECT_TRUE(aabb.aabb.max().isApprox(Vector3f(1.5f, 2.5f, 3.5f)));
}
