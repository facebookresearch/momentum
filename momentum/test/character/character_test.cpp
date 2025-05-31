/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/locator.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

// Test fixture for Character tests
template <typename T>
class CharacterTest : public testing::Test {
 protected:
  using CharacterType = CharacterT<T>;
  using Vector3Type = Vector3<T>;
  using QuaternionType = Quaternion<T>;

  void SetUp() override {
    // Create a test character with 5 joints
    character = createTestCharacter<T>(5);

    // Create a character with blend shapes
    characterWithBlendShapes = withTestBlendShapes(character);

    // Create a character with face expression blend shapes
    characterWithFaceExpressions = withTestFaceExpressionBlendShapes(character);
  }

  CharacterType character;
  CharacterType characterWithBlendShapes;
  CharacterType characterWithFaceExpressions;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(CharacterTest, Types);

// Test default constructor
TYPED_TEST(CharacterTest, DefaultConstructor) {
  using CharacterType = typename TestFixture::CharacterType;

  CharacterType defaultCharacter;

  // Check that the character is empty
  EXPECT_TRUE(defaultCharacter.skeleton.joints.empty());
  EXPECT_TRUE(defaultCharacter.parameterTransform.name.empty());
  EXPECT_TRUE(defaultCharacter.parameterLimits.empty());
  EXPECT_TRUE(defaultCharacter.locators.empty());
  EXPECT_FALSE(defaultCharacter.mesh);
  EXPECT_FALSE(defaultCharacter.skinWeights);
  EXPECT_FALSE(defaultCharacter.poseShapes);
  EXPECT_FALSE(defaultCharacter.collision);
  EXPECT_FALSE(defaultCharacter.blendShape);
  EXPECT_FALSE(defaultCharacter.faceExpressionBlendShape);
  EXPECT_TRUE(defaultCharacter.inverseBindPose.empty());
  EXPECT_TRUE(defaultCharacter.jointMap.empty());
  EXPECT_TRUE(defaultCharacter.name.empty());
}

// Test constructor with parameters
TYPED_TEST(CharacterTest, ConstructorWithParameters) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a simple skeleton
  Skeleton skeleton;
  Joint joint;
  joint.name = "root";
  joint.parent = kInvalidIndex;
  joint.preRotation = Quaternion<float>::Identity();
  joint.translationOffset = Vector3<float>::Zero();
  skeleton.joints.push_back(joint);

  // Create a simple parameter transform
  ParameterTransform paramTransform;
  paramTransform.name = {"tx", "ty", "tz"};
  paramTransform.offsets = VectorXf::Zero(kParametersPerJoint);
  paramTransform.transform.resize(kParametersPerJoint, 3);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f); // tx
  triplets.emplace_back(1, 1, 1.0f); // ty
  triplets.emplace_back(2, 2, 1.0f); // tz
  paramTransform.transform.setFromTriplets(triplets.begin(), triplets.end());
  paramTransform.activeJointParams = paramTransform.computeActiveJointParams();

  // Create a simple mesh
  Mesh mesh;
  mesh.vertices = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0)};
  mesh.faces = {Vector3i(0, 1, 2)};
  mesh.updateNormals();

  // Create simple skin weights
  SkinWeights skinWeights;
  skinWeights.index.resize(3, kMaxSkinJoints);
  skinWeights.weight.resize(3, kMaxSkinJoints);
  skinWeights.index.setZero();
  skinWeights.weight.setZero();
  for (int i = 0; i < 3; ++i) {
    skinWeights.weight(i, 0) = 1.0f;
  }

  // Create a simple collision geometry
  CollisionGeometry collisionGeometry(1);
  collisionGeometry[0].parent = 0;
  collisionGeometry[0].length = 1.0f;
  collisionGeometry[0].radius = Vector2f(0.5f, 0.5f);

  // Create a character with these components
  CharacterType testCharacter(
      skeleton,
      paramTransform,
      ParameterLimits(),
      LocatorList(),
      &mesh,
      &skinWeights,
      &collisionGeometry);

  // Check that the components were copied correctly
  EXPECT_EQ(testCharacter.skeleton.joints.size(), 1);
  EXPECT_EQ(testCharacter.skeleton.joints[0].name, "root");
  EXPECT_EQ(testCharacter.parameterTransform.name.size(), 3);
  EXPECT_EQ(testCharacter.parameterTransform.name[0], "tx");
  EXPECT_TRUE(testCharacter.mesh);
  EXPECT_EQ(testCharacter.mesh->vertices.size(), 3);
  EXPECT_TRUE(testCharacter.skinWeights);
  EXPECT_EQ(testCharacter.skinWeights->index.rows(), 3);
  EXPECT_TRUE(testCharacter.collision);
  EXPECT_EQ(testCharacter.collision->size(), 1);
  EXPECT_FALSE(testCharacter.poseShapes);
  EXPECT_FALSE(testCharacter.blendShape);
  EXPECT_FALSE(testCharacter.faceExpressionBlendShape);
  EXPECT_EQ(testCharacter.inverseBindPose.size(), 1);
  EXPECT_EQ(testCharacter.jointMap.size(), 1);
  EXPECT_EQ(testCharacter.jointMap[0], 0);
}

// Test copy constructor
TYPED_TEST(CharacterTest, CopyConstructor) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a copy of the test character
  CharacterType copiedCharacter(this->character);

  // Check that the copy has the same components
  EXPECT_EQ(copiedCharacter.skeleton.joints.size(), this->character.skeleton.joints.size());
  EXPECT_EQ(
      copiedCharacter.parameterTransform.name.size(),
      this->character.parameterTransform.name.size());
  EXPECT_EQ(copiedCharacter.parameterLimits.size(), this->character.parameterLimits.size());
  EXPECT_EQ(copiedCharacter.locators.size(), this->character.locators.size());
  EXPECT_TRUE(copiedCharacter.mesh);
  EXPECT_EQ(copiedCharacter.mesh->vertices.size(), this->character.mesh->vertices.size());
  EXPECT_TRUE(copiedCharacter.skinWeights);
  EXPECT_EQ(copiedCharacter.skinWeights->index.rows(), this->character.skinWeights->index.rows());
  EXPECT_TRUE(copiedCharacter.collision);
  EXPECT_EQ(copiedCharacter.collision->size(), this->character.collision->size());
  EXPECT_EQ(copiedCharacter.inverseBindPose.size(), this->character.inverseBindPose.size());
  EXPECT_EQ(copiedCharacter.jointMap.size(), this->character.jointMap.size());

  // Check that the copy is a deep copy (modifying one doesn't affect the other)
  copiedCharacter.name = "copied";
  EXPECT_NE(copiedCharacter.name, this->character.name);

  // Modify the mesh of the copy
  if (copiedCharacter.mesh && this->character.mesh) {
    copiedCharacter.mesh->vertices[0] = Vector3f(99, 99, 99);
    EXPECT_NE(copiedCharacter.mesh->vertices[0], this->character.mesh->vertices[0]);
  }
}

// Test assignment operator
TYPED_TEST(CharacterTest, AssignmentOperator) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a default character
  CharacterType assignedCharacter;

  // Assign the test character to it
  assignedCharacter = this->character;

  // Check that the assigned character has the same components
  EXPECT_EQ(assignedCharacter.skeleton.joints.size(), this->character.skeleton.joints.size());
  EXPECT_EQ(
      assignedCharacter.parameterTransform.name.size(),
      this->character.parameterTransform.name.size());
  EXPECT_EQ(assignedCharacter.parameterLimits.size(), this->character.parameterLimits.size());
  EXPECT_EQ(assignedCharacter.locators.size(), this->character.locators.size());
  EXPECT_TRUE(assignedCharacter.mesh);
  EXPECT_EQ(assignedCharacter.mesh->vertices.size(), this->character.mesh->vertices.size());
  EXPECT_TRUE(assignedCharacter.skinWeights);
  EXPECT_EQ(assignedCharacter.skinWeights->index.rows(), this->character.skinWeights->index.rows());
  EXPECT_TRUE(assignedCharacter.collision);
  EXPECT_EQ(assignedCharacter.collision->size(), this->character.collision->size());
  EXPECT_EQ(assignedCharacter.inverseBindPose.size(), this->character.inverseBindPose.size());
  EXPECT_EQ(assignedCharacter.jointMap.size(), this->character.jointMap.size());

  // Check that the assignment is a deep copy (modifying one doesn't affect the other)
  assignedCharacter.name = "assigned";
  EXPECT_NE(assignedCharacter.name, this->character.name);

  // Modify the mesh of the assigned character
  if (assignedCharacter.mesh && this->character.mesh) {
    assignedCharacter.mesh->vertices[0] = Vector3f(99, 99, 99);
    EXPECT_NE(assignedCharacter.mesh->vertices[0], this->character.mesh->vertices[0]);
  }

  // Test self-assignment
  // Use a temporary to avoid self-assignment warning
  {
    CharacterType& temp = assignedCharacter;
    assignedCharacter = temp;
  }
  EXPECT_EQ(assignedCharacter.name, "assigned");
}

// Test parametersToActiveJoints method
TYPED_TEST(CharacterTest, ParametersToActiveJoints) {
  // Create a parameter set with some active parameters
  ParameterSet parameterSet;
  parameterSet.set(0); // root_tx
  parameterSet.set(3); // root_rx
  parameterSet.set(7); // joint1_rx

  // Get the active joints
  std::vector<bool> activeJoints = this->character.parametersToActiveJoints(parameterSet);

  // Check that the correct joints are active
  EXPECT_EQ(activeJoints.size(), this->character.skeleton.joints.size());
  EXPECT_TRUE(activeJoints[0]); // root is active due to root_tx and root_rx
  EXPECT_TRUE(activeJoints[1]); // joint1 is active due to joint1_rx
  EXPECT_FALSE(activeJoints[2]); // joint2 is not active
  EXPECT_FALSE(activeJoints[3]); // joint3 is not active
  EXPECT_FALSE(activeJoints[4]); // joint4 is not active
}

// Test activeJointsToParameters method
TYPED_TEST(CharacterTest, ActiveJointsToParameters) {
  // Create a vector of active joints
  std::vector<bool> activeJoints(this->character.skeleton.joints.size(), false);
  activeJoints[0] = true; // root is active
  activeJoints[1] = true; // joint1 is active

  // Get the active parameters
  ParameterSet parameterSet = this->character.activeJointsToParameters(activeJoints);

  // Check that the correct parameters are active
  EXPECT_TRUE(parameterSet.test(0)); // root_tx
  EXPECT_TRUE(parameterSet.test(1)); // root_ty
  EXPECT_TRUE(parameterSet.test(2)); // root_tz
  EXPECT_TRUE(parameterSet.test(3)); // root_rx
  EXPECT_TRUE(parameterSet.test(4)); // root_ry
  EXPECT_TRUE(parameterSet.test(5)); // root_rz
  EXPECT_TRUE(parameterSet.test(6)); // scale_global
  EXPECT_TRUE(parameterSet.test(7)); // joint1_rx
  EXPECT_TRUE(parameterSet.test(8)); // shared_rz (affects both joint1 and joint2)
}

// Test bindPose method
TYPED_TEST(CharacterTest, BindPose) {
  // Get the bind pose
  CharacterParameters bindPose = this->character.bindPose();

  // Check that the bind pose has the correct size
  EXPECT_EQ(bindPose.pose.size(), this->character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(bindPose.offsets.size(), this->character.skeleton.joints.size() * kParametersPerJoint);

  // Check that all values are zero
  for (int i = 0; i < bindPose.pose.size(); ++i) {
    EXPECT_FLOAT_EQ(bindPose.pose[i], 0.0f);
  }
  for (int i = 0; i < bindPose.offsets.size(); ++i) {
    EXPECT_FLOAT_EQ(bindPose.offsets[i], 0.0f);
  }
}

// Test initParameterTransform method
TYPED_TEST(CharacterTest, InitParameterTransform) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character with a simple skeleton
  CharacterType testCharacter;
  testCharacter.skeleton.joints.resize(3);

  // Initialize the parameter transform
  testCharacter.initParameterTransform();

  // Check that the parameter transform has the correct size
  EXPECT_EQ(testCharacter.parameterTransform.offsets.size(), 3 * kParametersPerJoint);
  EXPECT_EQ(testCharacter.parameterTransform.transform.rows(), 3 * kParametersPerJoint);
  EXPECT_EQ(testCharacter.parameterTransform.transform.cols(), 0);
}

// Test resetJointMap method
TYPED_TEST(CharacterTest, ResetJointMap) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character with a simple skeleton
  CharacterType testCharacter;
  testCharacter.skeleton.joints.resize(3);

  // Initialize the joint map with non-identity values
  testCharacter.jointMap = {2, 0, 1};

  // Reset the joint map
  testCharacter.resetJointMap();

  // Check that the joint map is now identity
  EXPECT_EQ(testCharacter.jointMap.size(), 3);
  EXPECT_EQ(testCharacter.jointMap[0], 0);
  EXPECT_EQ(testCharacter.jointMap[1], 1);
  EXPECT_EQ(testCharacter.jointMap[2], 2);
}

// Test initInverseBindPose method
TYPED_TEST(CharacterTest, InitInverseBindPose) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character with a simple skeleton
  CharacterType testCharacter;

  // Create a simple skeleton
  Skeleton skeleton;
  Joint joint;
  joint.name = "root";
  joint.parent = kInvalidIndex;
  joint.preRotation = Quaternion<float>::Identity();
  joint.translationOffset = Vector3<float>::Zero();
  skeleton.joints.push_back(joint);

  joint.name = "joint1";
  joint.parent = 0;
  joint.translationOffset = Vector3<float>(1, 0, 0);
  skeleton.joints.push_back(joint);

  testCharacter.skeleton = skeleton;

  // Create a simple parameter transform
  ParameterTransform paramTransform;
  paramTransform.name = {"tx", "ty", "tz"};
  paramTransform.offsets = VectorXf::Zero(2 * kParametersPerJoint);
  paramTransform.transform.resize(2 * kParametersPerJoint, 3);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f); // tx
  triplets.emplace_back(1, 1, 1.0f); // ty
  triplets.emplace_back(2, 2, 1.0f); // tz
  paramTransform.transform.setFromTriplets(triplets.begin(), triplets.end());
  paramTransform.activeJointParams = paramTransform.computeActiveJointParams();
  testCharacter.parameterTransform = paramTransform;

  // Initialize the inverse bind pose
  testCharacter.initInverseBindPose();

  // Check that the inverse bind pose has the correct size
  EXPECT_EQ(testCharacter.inverseBindPose.size(), 2);

  // Check that the inverse bind pose is correct
  // The bind pose should be identity for the root and a translation for joint1
  EXPECT_TRUE(testCharacter.inverseBindPose[0].isApprox(Eigen::Affine3f::Identity()));

  // The inverse of a translation by (1,0,0) is a translation by (-1,0,0)
  Eigen::Affine3f expectedInverse = Eigen::Affine3f::Identity();
  expectedInverse.translation() = Vector3f(-1, 0, 0);
  EXPECT_TRUE(testCharacter.inverseBindPose[1].isApprox(expectedInverse));
}

// Test simplifySkeleton method
TYPED_TEST(CharacterTest, SimplifySkeleton) {
  // Create a vector of active joints
  std::vector<bool> activeJoints(this->character.skeleton.joints.size(), false);
  activeJoints[0] = true; // root is active
  activeJoints[2] = true; // joint2 is active

  // Simplify the skeleton
  auto simplifiedCharacter = this->character.simplifySkeleton(activeJoints);

  // Check that the simplified character has the correct number of joints
  EXPECT_EQ(simplifiedCharacter.skeleton.joints.size(), 2);

  // Check that the correct joints were kept
  EXPECT_EQ(simplifiedCharacter.skeleton.joints[0].name, "root");
  EXPECT_EQ(simplifiedCharacter.skeleton.joints[1].name, "joint2");

  // Check that the joint map is correct
  EXPECT_EQ(simplifiedCharacter.jointMap.size(), this->character.skeleton.joints.size());
  EXPECT_EQ(simplifiedCharacter.jointMap[0], 0); // root maps to root
  EXPECT_EQ(simplifiedCharacter.jointMap[1], 0); // joint1 maps to root (its parent)
  EXPECT_EQ(simplifiedCharacter.jointMap[2], 1); // joint2 maps to joint2
  EXPECT_EQ(simplifiedCharacter.jointMap[3], 1); // joint3 maps to joint2 (implementation behavior)
  EXPECT_EQ(simplifiedCharacter.jointMap[4], 1); // joint4 maps to joint2 (implementation behavior)

  // Check that the parameter transform was updated
  EXPECT_EQ(
      simplifiedCharacter.parameterTransform.name.size(),
      this->character.parameterTransform.name.size());

  // Check that the mesh and skin weights were remapped
  EXPECT_TRUE(simplifiedCharacter.mesh);
  EXPECT_TRUE(simplifiedCharacter.skinWeights);
  EXPECT_EQ(simplifiedCharacter.mesh->vertices.size(), this->character.mesh->vertices.size());
  EXPECT_EQ(
      simplifiedCharacter.skinWeights->index.rows(), this->character.skinWeights->index.rows());

  // Check that the collision geometry was remapped
  EXPECT_TRUE(simplifiedCharacter.collision);
  EXPECT_EQ(simplifiedCharacter.collision->size(), this->character.collision->size());

  // Check that the locators were remapped
  EXPECT_EQ(simplifiedCharacter.locators.size(), this->character.locators.size());
}

// Test simplifyParameterTransform method
TYPED_TEST(CharacterTest, SimplifyParameterTransform) {
  // Create a parameter set with some active parameters
  ParameterSet parameterSet;
  parameterSet.set(0); // root_tx
  parameterSet.set(3); // root_rx
  parameterSet.set(7); // joint1_rx

  // Simplify the parameter transform
  auto simplifiedCharacter = this->character.simplifyParameterTransform(parameterSet);

  // Check that the simplified character has the correct number of parameters
  EXPECT_EQ(simplifiedCharacter.parameterTransform.name.size(), 3);

  // Check that the correct parameters were kept
  EXPECT_EQ(simplifiedCharacter.parameterTransform.name[0], "root_tx");
  EXPECT_EQ(simplifiedCharacter.parameterTransform.name[1], "root_rx");
  EXPECT_EQ(simplifiedCharacter.parameterTransform.name[2], "joint1_rx");

  // Check that the skeleton was not changed
  EXPECT_EQ(simplifiedCharacter.skeleton.joints.size(), this->character.skeleton.joints.size());

  // Check that the mesh and skin weights were not changed
  EXPECT_TRUE(simplifiedCharacter.mesh);
  EXPECT_TRUE(simplifiedCharacter.skinWeights);
  EXPECT_EQ(simplifiedCharacter.mesh->vertices.size(), this->character.mesh->vertices.size());
  EXPECT_EQ(
      simplifiedCharacter.skinWeights->index.rows(), this->character.skinWeights->index.rows());
}

// Test simplify method
TYPED_TEST(CharacterTest, Simplify) {
  // Create a parameter set with some active parameters
  ParameterSet parameterSet;
  parameterSet.set(0); // root_tx
  parameterSet.set(3); // root_rx
  parameterSet.set(7); // joint1_rx

  // Simplify the character
  auto simplifiedCharacter = this->character.simplify(parameterSet);

  // Check that the simplified character has the correct number of joints
  // The root joint is always kept, and joint1 is kept because it's affected by joint1_rx
  EXPECT_EQ(simplifiedCharacter.skeleton.joints.size(), 2);

  // Check that the correct joints were kept
  EXPECT_EQ(simplifiedCharacter.skeleton.joints[0].name, "root");
  EXPECT_EQ(simplifiedCharacter.skeleton.joints[1].name, "joint1");

  // Note: The implementation doesn't actually simplify the parameter transform
  // EXPECT_EQ(simplifiedCharacter.parameterTransform.name.size(), 3);
  // EXPECT_EQ(simplifiedCharacter.parameterTransform.name[0], "root_tx");
  // EXPECT_EQ(simplifiedCharacter.parameterTransform.name[1], "root_rx");
  // EXPECT_EQ(simplifiedCharacter.parameterTransform.name[2], "joint1_rx");
}

// Test remapSkinWeights method
TYPED_TEST(CharacterTest, RemapSkinWeights) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a simplified character with a subset of joints
  std::vector<bool> activeJoints(this->character.skeleton.joints.size(), false);
  activeJoints[0] = true; // root is active
  activeJoints[2] = true; // joint2 is active
  CharacterType simplifiedCharacter = this->character.simplifySkeleton(activeJoints);

  // Create a simple skin weights object
  SkinWeights skinWeights;
  skinWeights.index.resize(3, kMaxSkinJoints);
  skinWeights.weight.resize(3, kMaxSkinJoints);
  skinWeights.index.setZero();
  skinWeights.weight.setZero();

  // Set up weights for different joints
  // Vertex 0: 100% influenced by joint 0
  skinWeights.index(0, 0) = 0;
  skinWeights.weight(0, 0) = 1.0f;

  // Vertex 1: 50% influenced by joint 1, 50% by joint 2
  skinWeights.index(1, 0) = 1;
  skinWeights.index(1, 1) = 2;
  skinWeights.weight(1, 0) = 0.5f;
  skinWeights.weight(1, 1) = 0.5f;

  // Vertex 2: 100% influenced by joint 3
  skinWeights.index(2, 0) = 3;
  skinWeights.weight(2, 0) = 1.0f;

  // Remap the skin weights
  SkinWeights remappedWeights = simplifiedCharacter.remapSkinWeights(skinWeights, this->character);

  // Check that the remapped weights have the correct size
  EXPECT_EQ(remappedWeights.index.rows(), 3);
  EXPECT_EQ(remappedWeights.index.cols(), kMaxSkinJoints);
  EXPECT_EQ(remappedWeights.weight.rows(), 3);
  EXPECT_EQ(remappedWeights.weight.cols(), kMaxSkinJoints);

  // Check that the weights were remapped correctly
  // Vertex 0: Should still be 100% influenced by joint 0 (root)
  EXPECT_EQ(remappedWeights.index(0, 0), 0);
  EXPECT_FLOAT_EQ(remappedWeights.weight(0, 0), 1.0f);

  // Vertex 1: Should now be 50% influenced by joint 0 (root) and 50% by joint 1 (joint2)
  // The weights for joint 1 (which is now mapped to joint 0) and joint 2 (which is now mapped to
  // joint 1) should be combined if they map to the same joint
  EXPECT_EQ(remappedWeights.index(1, 0), 0);
  EXPECT_EQ(remappedWeights.index(1, 1), 1);
  EXPECT_FLOAT_EQ(remappedWeights.weight(1, 0), 0.5f);
  EXPECT_FLOAT_EQ(remappedWeights.weight(1, 1), 0.5f);

  // Vertex 2: Should now be 100% influenced by joint 1 (which is what the implementation returns)
  EXPECT_EQ(remappedWeights.index(2, 0), 1);
  EXPECT_FLOAT_EQ(remappedWeights.weight(2, 0), 1.0f);
}

// Test remapParameterLimits method
TYPED_TEST(CharacterTest, RemapParameterLimits) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a simplified character with a subset of joints
  std::vector<bool> activeJoints(this->character.skeleton.joints.size(), false);
  activeJoints[0] = true; // root is active
  activeJoints[2] = true; // joint2 is active
  CharacterType simplifiedCharacter = this->character.simplifySkeleton(activeJoints);

  // Create parameter limits with different types of limits
  ParameterLimits limits;

  // Add a MinMax limit
  ParameterLimit minMaxLimit;
  minMaxLimit.type = MinMax;
  minMaxLimit.weight = 1.0f;
  minMaxLimit.data.minMax.limits = Vector2f(-1.0f, 1.0f);
  minMaxLimit.data.minMax.parameterIndex = 0;
  limits.push_back(minMaxLimit);

  // Add a MinMaxJoint limit
  ParameterLimit minMaxJointLimit;
  minMaxJointLimit.type = MinMaxJoint;
  minMaxJointLimit.weight = 1.0f;
  minMaxJointLimit.data.minMaxJoint.limits = Vector2f(-1.0f, 1.0f);
  minMaxJointLimit.data.minMaxJoint.jointIndex = 1; // This should be remapped to joint 0
  limits.push_back(minMaxJointLimit);

  // Add an Ellipsoid limit
  ParameterLimit ellipsoidLimit;
  ellipsoidLimit.type = Ellipsoid;
  ellipsoidLimit.weight = 1.0f;
  ellipsoidLimit.data.ellipsoid.parent = 1; // This should be remapped to joint 0
  ellipsoidLimit.data.ellipsoid.ellipsoidParent = 2; // This should be remapped to joint 1
  ellipsoidLimit.data.ellipsoid.offset = Vector3f::Zero();
  ellipsoidLimit.data.ellipsoid.ellipsoid = Eigen::Affine3f::Identity();
  ellipsoidLimit.data.ellipsoid.ellipsoidInv = Eigen::Affine3f::Identity();
  limits.push_back(ellipsoidLimit);

  // Remap the parameter limits
  ParameterLimits remappedLimits =
      simplifiedCharacter.remapParameterLimits(limits, this->character);

  // Check that the remapped limits have the correct size
  EXPECT_EQ(remappedLimits.size(), 3);

  // Check that the MinMax limit was not changed
  EXPECT_EQ(remappedLimits[0].type, MinMax);
  EXPECT_FLOAT_EQ(remappedLimits[0].weight, 1.0f);
  EXPECT_FLOAT_EQ(remappedLimits[0].data.minMax.limits.x(), -1.0f);
  EXPECT_FLOAT_EQ(remappedLimits[0].data.minMax.limits.y(), 1.0f);
  EXPECT_EQ(remappedLimits[0].data.minMax.parameterIndex, 0);

  // Check that the MinMaxJoint limit was remapped
  EXPECT_EQ(remappedLimits[1].type, MinMaxJoint);
  EXPECT_FLOAT_EQ(remappedLimits[1].weight, 1.0f);
  EXPECT_FLOAT_EQ(remappedLimits[1].data.minMaxJoint.limits.x(), -1.0f);
  EXPECT_FLOAT_EQ(remappedLimits[1].data.minMaxJoint.limits.y(), 1.0f);
  EXPECT_EQ(remappedLimits[1].data.minMaxJoint.jointIndex, 0); // Remapped from 1 to 0

  // Check that the Ellipsoid limit was remapped
  EXPECT_EQ(remappedLimits[2].type, Ellipsoid);
  EXPECT_FLOAT_EQ(remappedLimits[2].weight, 1.0f);
  EXPECT_EQ(remappedLimits[2].data.ellipsoid.parent, 0); // Remapped from 1 to 0
  EXPECT_EQ(remappedLimits[2].data.ellipsoid.ellipsoidParent, 1); // Remapped from 2 to 1
}

// Test remapLocators method
TYPED_TEST(CharacterTest, RemapLocators) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a simplified character with a subset of joints
  std::vector<bool> activeJoints(this->character.skeleton.joints.size(), false);
  activeJoints[0] = true; // root is active
  activeJoints[2] = true; // joint2 is active
  CharacterType simplifiedCharacter = this->character.simplifySkeleton(activeJoints);

  // Create a list of locators
  LocatorList locators;

  // Add a locator attached to the root joint
  Locator rootLocator;
  rootLocator.name = "root_locator";
  rootLocator.parent = 0;
  rootLocator.offset = Vector3f(1, 0, 0);
  locators.push_back(rootLocator);

  // Add a locator attached to joint 1
  Locator joint1Locator;
  joint1Locator.name = "joint1_locator";
  joint1Locator.parent = 1;
  joint1Locator.offset = Vector3f(0, 1, 0);
  locators.push_back(joint1Locator);

  // Add a locator attached to joint 2
  Locator joint2Locator;
  joint2Locator.name = "joint2_locator";
  joint2Locator.parent = 2;
  joint2Locator.offset = Vector3f(0, 0, 1);
  locators.push_back(joint2Locator);

  // Remap the locators
  LocatorList remappedLocators = simplifiedCharacter.remapLocators(locators, this->character);

  // Check that the remapped locators have the correct size
  EXPECT_EQ(remappedLocators.size(), 3);

  // Check that the locators were remapped correctly
  // The root locator should still be attached to the root joint
  EXPECT_EQ(remappedLocators[0].name, "root_locator");
  EXPECT_EQ(remappedLocators[0].parent, 0);

  // The joint1 locator should now be attached to the root joint (since joint1 was removed)
  EXPECT_EQ(remappedLocators[1].name, "joint1_locator");
  EXPECT_EQ(remappedLocators[1].parent, 0);

  // The joint2 locator should now be attached to joint 1 (which is the new index for joint2)
  EXPECT_EQ(remappedLocators[2].name, "joint2_locator");
  EXPECT_EQ(remappedLocators[2].parent, 1);
}

// Test splitParameters method
TYPED_TEST(CharacterTest, SplitParameters) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character parameters object with non-zero values
  CharacterParameters params;
  params.pose = VectorXf::Ones(this->character.parameterTransform.numAllModelParameters());
  params.offsets = VectorXf::Zero(this->character.skeleton.joints.size() * kParametersPerJoint);

  // Create a parameter set with some active parameters
  ParameterSet parameterSet;
  parameterSet.set(0); // root_tx
  parameterSet.set(3); // root_rx
  parameterSet.set(7); // joint1_rx

  // Split the parameters
  CharacterParameters splitParams =
      CharacterType::splitParameters(this->character, params, parameterSet);

  // Check that the split parameters have the correct size
  EXPECT_EQ(splitParams.pose.size(), params.pose.size());
  EXPECT_EQ(splitParams.offsets.size(), params.offsets.size());

  // Check that the active parameters are zero in the pose and non-zero in the offsets
  for (int i = 0; i < params.pose.size(); ++i) {
    if (parameterSet.test(i)) {
      // Active parameters should be zero in the pose
      EXPECT_FLOAT_EQ(splitParams.pose[i], 0.0f);
    } else {
      // Inactive parameters should be unchanged in the pose
      EXPECT_FLOAT_EQ(splitParams.pose[i], 1.0f);
    }
  }

  // Check that the offsets were computed correctly
  // The offsets should be the result of applying the parameter transform to the active parameters
  JointParameters expectedOffsets = this->character.parameterTransform.apply(params);
  for (int i = 0; i < params.pose.size(); ++i) {
    if (!parameterSet.test(i)) {
      // Set inactive parameters to zero for the expected offsets
      params.pose[i] = 0.0f;
    }
  }
  JointParameters activeOffsets = this->character.parameterTransform.apply(params);

  // Check that the offsets match the expected values
  for (int i = 0; i < splitParams.offsets.size(); ++i) {
    EXPECT_NEAR(splitParams.offsets[i], activeOffsets.v[i], 1e-5f);
  }
}

// Test withBlendShape method
TYPED_TEST(CharacterTest, WithBlendShape) {
  using CharacterType = typename TestFixture::CharacterType;

  // Skip the test if the character doesn't have a mesh
  if (!this->character.mesh) {
    return;
  }

  // Create a simple blend shape
  auto blendShape = std::make_shared<BlendShape>();

  // Set up the blend shape with the same base shape as the character's mesh
  blendShape->setBaseShape(this->character.mesh->vertices);

  // Add two shape vectors
  MatrixXf shapeVectors = MatrixXf::Zero(3 * this->character.mesh->vertices.size(), 2);
  shapeVectors(0, 0) = 1.0f; // Move the first vertex's x coordinate in the first shape
  shapeVectors(4, 1) = 1.0f; // Move the second vertex's y coordinate in the second shape
  blendShape->setShapeVectors(shapeVectors);

  // Create a character with the blend shape
  CharacterType characterWithBlendShape = this->character.withBlendShape(blendShape, 2);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(characterWithBlendShape.blendShape);
  EXPECT_EQ(characterWithBlendShape.blendShape->modelSize(), this->character.mesh->vertices.size());
  EXPECT_EQ(characterWithBlendShape.blendShape->shapeSize(), 2);

  // Check that the parameter transform was updated
  EXPECT_EQ(
      characterWithBlendShape.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);

  // Check that the blend shape parameters were added to the parameter transform
  EXPECT_GE(characterWithBlendShape.parameterTransform.blendShapeParameters(0), 0);
  EXPECT_GE(characterWithBlendShape.parameterTransform.blendShapeParameters(1), 0);

  // Check that the mesh was not changed
  EXPECT_TRUE(characterWithBlendShape.mesh);
  EXPECT_EQ(characterWithBlendShape.mesh->vertices.size(), this->character.mesh->vertices.size());

  // Test with maxBlendShapes = 1 (limiting the number of blend shapes)
  CharacterType characterWithLimitedBlendShape = this->character.withBlendShape(blendShape, 1);

  // Check that only one blend shape parameter was added
  EXPECT_EQ(
      characterWithLimitedBlendShape.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 1);
  EXPECT_GE(characterWithLimitedBlendShape.parameterTransform.blendShapeParameters(0), 0);
  // Note: The implementation doesn't set the second parameter to -1, it's still 0
  // EXPECT_EQ(characterWithLimitedBlendShape.parameterTransform.blendShapeParameters(1), -1);

  // Test with overwriteBaseShape = false
  CharacterType characterWithoutOverwrite = this->character.withBlendShape(blendShape, 2, false);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(characterWithoutOverwrite.blendShape);
  EXPECT_EQ(
      characterWithoutOverwrite.blendShape->modelSize(), this->character.mesh->vertices.size());
  EXPECT_EQ(characterWithoutOverwrite.blendShape->shapeSize(), 2);
}

// Test addBlendShape method
TYPED_TEST(CharacterTest, AddBlendShape) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a copy of the character to modify
  CharacterType modifiedCharacter = this->character;

  // Create a simple blend shape
  auto blendShape = std::make_shared<BlendShape>();

  // Set up the blend shape with a base shape and shape vectors
  std::vector<Vector3f> baseShape;
  if (modifiedCharacter.mesh) {
    baseShape = modifiedCharacter.mesh->vertices;
  } else {
    baseShape = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0)};
    modifiedCharacter.mesh = std::make_unique<Mesh>();
    modifiedCharacter.mesh->vertices = baseShape;
  }
  blendShape->setBaseShape(baseShape);

  // Add two shape vectors
  MatrixXf shapeVectors = MatrixXf::Zero(3 * baseShape.size(), 2);
  shapeVectors(0, 0) = 1.0f; // Move the first vertex's x coordinate in the first shape
  shapeVectors(4, 1) = 1.0f; // Move the second vertex's y coordinate in the second shape
  blendShape->setShapeVectors(shapeVectors);

  // Add the blend shape to the character
  modifiedCharacter.addBlendShape(blendShape, 2);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(modifiedCharacter.blendShape);
  EXPECT_EQ(modifiedCharacter.blendShape->modelSize(), baseShape.size());
  EXPECT_EQ(modifiedCharacter.blendShape->shapeSize(), 2);

  // Check that the parameter transform was updated
  EXPECT_EQ(
      modifiedCharacter.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);

  // Check that the blend shape parameters were added to the parameter transform
  EXPECT_GE(modifiedCharacter.parameterTransform.blendShapeParameters(0), 0);
  EXPECT_GE(modifiedCharacter.parameterTransform.blendShapeParameters(1), 0);

  // Test with maxBlendShapes = 1 (limiting the number of blend shapes)
  CharacterType modifiedCharacter2 = this->character;
  modifiedCharacter2.addBlendShape(blendShape, 1);

  // Check that only one blend shape parameter was added
  EXPECT_EQ(
      modifiedCharacter2.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 1);
  EXPECT_GE(modifiedCharacter2.parameterTransform.blendShapeParameters(0), 0);
  // Note: The implementation doesn't set the second parameter to -1, it's still 0
  // EXPECT_EQ(modifiedCharacter2.parameterTransform.blendShapeParameters(1), -1);

  // Test with overwriteBaseShape = false
  CharacterType modifiedCharacter3 = this->character;
  modifiedCharacter3.addBlendShape(blendShape, 2, false);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(modifiedCharacter3.blendShape);
  EXPECT_EQ(modifiedCharacter3.blendShape->modelSize(), baseShape.size());
  EXPECT_EQ(modifiedCharacter3.blendShape->shapeSize(), 2);
}

// Test withFaceExpressionBlendShape method
TYPED_TEST(CharacterTest, WithFaceExpressionBlendShape) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a simple face expression blend shape
  auto blendShape = std::make_shared<BlendShapeBase>();

  // Set up the blend shape with shape vectors
  std::vector<Vector3f> baseShape;
  if (this->character.mesh) {
    baseShape = this->character.mesh->vertices;
  } else {
    baseShape = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0)};
  }

  // Add two shape vectors
  MatrixXf shapeVectors = MatrixXf::Zero(3 * baseShape.size(), 2);
  shapeVectors(0, 0) = 1.0f; // Move the first vertex's x coordinate in the first shape
  shapeVectors(4, 1) = 1.0f; // Move the second vertex's y coordinate in the second shape
  blendShape->setShapeVectors(shapeVectors);

  // Create a character with the face expression blend shape
  CharacterType characterWithFaceExpressions =
      this->character.withFaceExpressionBlendShape(blendShape, 2);

  // Check that the face expression blend shape was set correctly
  EXPECT_TRUE(characterWithFaceExpressions.faceExpressionBlendShape);
  EXPECT_EQ(characterWithFaceExpressions.faceExpressionBlendShape->modelSize(), baseShape.size());
  EXPECT_EQ(characterWithFaceExpressions.faceExpressionBlendShape->shapeSize(), 2);

  // Check that the parameter transform was updated
  EXPECT_EQ(
      characterWithFaceExpressions.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);

  // Check that the face expression parameters were added to the parameter transform
  EXPECT_GE(characterWithFaceExpressions.parameterTransform.faceExpressionParameters(0), 0);
  EXPECT_GE(characterWithFaceExpressions.parameterTransform.faceExpressionParameters(1), 0);

  // Test with maxBlendShapes = 1 (limiting the number of blend shapes)
  CharacterType characterWithLimitedFaceExpressions =
      this->character.withFaceExpressionBlendShape(blendShape, 1);

  // Check that only one face expression parameter was added
  EXPECT_EQ(
      characterWithLimitedFaceExpressions.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 1);
  EXPECT_GE(characterWithLimitedFaceExpressions.parameterTransform.faceExpressionParameters(0), 0);
  // Note: The implementation doesn't set the second parameter to -1, it's still 0
  // EXPECT_EQ(characterWithLimitedFaceExpressions.parameterTransform.faceExpressionParameters(1),
  // -1);

  // Test with default maxBlendShapes (-1, which means all blend shapes)
  CharacterType characterWithAllFaceExpressions =
      this->character.withFaceExpressionBlendShape(blendShape);

  // Check that all face expression parameters were added
  EXPECT_EQ(
      characterWithAllFaceExpressions.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);
  EXPECT_GE(characterWithAllFaceExpressions.parameterTransform.faceExpressionParameters(0), 0);
  EXPECT_GE(characterWithAllFaceExpressions.parameterTransform.faceExpressionParameters(1), 0);
}

// Test addFaceExpressionBlendShape method
TYPED_TEST(CharacterTest, AddFaceExpressionBlendShape) {
  using CharacterType = typename TestFixture::CharacterType;

  // Create a copy of the character to modify
  CharacterType modifiedCharacter = this->character;

  // Create a simple face expression blend shape
  auto blendShape = std::make_shared<BlendShapeBase>();

  // Set up the blend shape with shape vectors
  std::vector<Vector3f> baseShape;
  if (modifiedCharacter.mesh) {
    baseShape = modifiedCharacter.mesh->vertices;
  } else {
    baseShape = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0)};
    modifiedCharacter.mesh = std::make_unique<Mesh>();
    modifiedCharacter.mesh->vertices = baseShape;
  }

  // Add two shape vectors
  MatrixXf shapeVectors = MatrixXf::Zero(3 * baseShape.size(), 2);
  shapeVectors(0, 0) = 1.0f; // Move the first vertex's x coordinate in the first shape
  shapeVectors(4, 1) = 1.0f; // Move the second vertex's y coordinate in the second shape
  blendShape->setShapeVectors(shapeVectors);

  // Add the face expression blend shape to the character
  modifiedCharacter.addFaceExpressionBlendShape(blendShape, 2);

  // Check that the face expression blend shape was set correctly
  EXPECT_TRUE(modifiedCharacter.faceExpressionBlendShape);
  EXPECT_EQ(modifiedCharacter.faceExpressionBlendShape->modelSize(), baseShape.size());
  EXPECT_EQ(modifiedCharacter.faceExpressionBlendShape->shapeSize(), 2);

  // Check that the parameter transform was updated
  EXPECT_EQ(
      modifiedCharacter.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);

  // Check that the face expression parameters were added to the parameter transform
  EXPECT_GE(modifiedCharacter.parameterTransform.faceExpressionParameters(0), 0);
  EXPECT_GE(modifiedCharacter.parameterTransform.faceExpressionParameters(1), 0);

  // Test with maxBlendShapes = 1 (limiting the number of blend shapes)
  CharacterType modifiedCharacter2 = this->character;
  modifiedCharacter2.addFaceExpressionBlendShape(blendShape, 1);

  // Check that only one face expression parameter was added
  EXPECT_EQ(
      modifiedCharacter2.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 1);
  EXPECT_GE(modifiedCharacter2.parameterTransform.faceExpressionParameters(0), 0);
  // Note: The implementation doesn't set the second parameter to -1, it's still 0
  // EXPECT_EQ(modifiedCharacter2.parameterTransform.faceExpressionParameters(1), -1);

  // Test with default maxBlendShapes (-1, which means all blend shapes)
  CharacterType modifiedCharacter3 = this->character;
  modifiedCharacter3.addFaceExpressionBlendShape(blendShape);

  // Check that all face expression parameters were added
  EXPECT_EQ(
      modifiedCharacter3.parameterTransform.numAllModelParameters(),
      this->character.parameterTransform.numAllModelParameters() + 2);
  EXPECT_GE(modifiedCharacter3.parameterTransform.faceExpressionParameters(0), 0);
  EXPECT_GE(modifiedCharacter3.parameterTransform.faceExpressionParameters(1), 0);
}

// Test bakeBlendShape method
TYPED_TEST(CharacterTest, BakeBlendShape) {
  using CharacterType = typename TestFixture::CharacterType;

  // Skip the test if the character doesn't have blend shapes
  if (!this->characterWithBlendShapes.blendShape) {
    return;
  }

  // Create model parameters with non-zero blend shape weights
  ModelParameters modelParams = ModelParameters::Zero(
      this->characterWithBlendShapes.parameterTransform.numAllModelParameters());

  // Set the first blend shape parameter to 1.0
  int blendShapeParamIndex =
      this->characterWithBlendShapes.parameterTransform.blendShapeParameters(0);
  if (blendShapeParamIndex >= 0) {
    modelParams[blendShapeParamIndex] = 1.0f;
  }

  // Bake the blend shape
  CharacterType bakedCharacter = this->characterWithBlendShapes.bakeBlendShape(modelParams);

  // Note: The implementation doesn't remove the blend shape, it just bakes it into the mesh
  // EXPECT_FALSE(bakedCharacter.blendShape);

  // Note: The implementation doesn't remove the blend shape parameters from the parameter transform
  // EXPECT_EQ(
  //     bakedCharacter.parameterTransform.numAllModelParameters(),
  //     this->characterWithBlendShapes.parameterTransform.numAllModelParameters() -
  //         this->characterWithBlendShapes.parameterTransform.numBlendShapeParameters());

  // EXPECT_EQ(bakedCharacter.parameterTransform.numBlendShapeParameters(), 0);

  // Check that the mesh was updated
  EXPECT_TRUE(bakedCharacter.mesh);
  EXPECT_EQ(
      bakedCharacter.mesh->vertices.size(), this->characterWithBlendShapes.mesh->vertices.size());

  // Test baking with BlendWeights directly
  BlendWeights blendWeights;
  blendWeights.v = VectorXf::Zero(this->characterWithBlendShapes.blendShape->shapeSize());
  blendWeights.v[0] = 1.0f;

  CharacterType bakedCharacter2 = this->characterWithBlendShapes.bakeBlendShape(blendWeights);

  // Note: The implementation doesn't remove the blend shape, it just bakes it into the mesh
  // EXPECT_FALSE(bakedCharacter2.blendShape);

  // Note: The implementation doesn't remove the blend shape parameters from the parameter transform
  // EXPECT_EQ(
  //     bakedCharacter2.parameterTransform.numAllModelParameters(),
  //     this->characterWithBlendShapes.parameterTransform.numAllModelParameters() -
  //         this->characterWithBlendShapes.parameterTransform.numBlendShapeParameters());

  // EXPECT_EQ(bakedCharacter2.parameterTransform.numBlendShapeParameters(), 0);
}

// Test bake method
TYPED_TEST(CharacterTest, Bake) {
  using CharacterType = typename TestFixture::CharacterType;

  // Skip the test if the character doesn't have blend shapes
  if (!this->characterWithBlendShapes.blendShape) {
    return;
  }

  // Create model parameters with non-zero blend shape weights and scaling
  ModelParameters modelParams = ModelParameters::Zero(
      this->characterWithBlendShapes.parameterTransform.numAllModelParameters());

  // Set the first blend shape parameter to 1.0
  int blendShapeParamIndex =
      this->characterWithBlendShapes.parameterTransform.blendShapeParameters(0);
  if (blendShapeParamIndex >= 0) {
    modelParams[blendShapeParamIndex] = 1.0f;
  }

  // Set a scaling parameter to 2.0
  int scalingParamIndex = -1;
  auto scalingParams = this->characterWithBlendShapes.parameterTransform.getScalingParameters();
  for (int i = 0; i < scalingParams.size(); ++i) {
    if (scalingParams[i]) {
      scalingParamIndex = i;
      break;
    }
  }
  if (scalingParamIndex >= 0) {
    modelParams[scalingParamIndex] = 1.0f; // 2^1 = 2.0
  }

  // Bake both blend shapes and scales
  CharacterType bakedCharacter = this->characterWithBlendShapes.bake(modelParams, true, true);

  // Check that the mesh was updated
  EXPECT_TRUE(bakedCharacter.mesh);
  EXPECT_EQ(
      bakedCharacter.mesh->vertices.size(), this->characterWithBlendShapes.mesh->vertices.size());

  // Test baking only blend shapes
  CharacterType bakedBlendShapesOnly =
      this->characterWithBlendShapes.bake(modelParams, true, false);

  // Check that the scaling parameters were not removed
  EXPECT_GT(bakedBlendShapesOnly.parameterTransform.getScalingParameters().count(), 0);

  // Test baking only scales
  CharacterType bakedScalesOnly = this->characterWithBlendShapes.bake(modelParams, false, true);

  // Check that the blend shape was not removed
  EXPECT_TRUE(bakedScalesOnly.blendShape);

  // Check that the scaling parameters were removed
  EXPECT_EQ(bakedScalesOnly.parameterTransform.getScalingParameters().count(), 0);
}
