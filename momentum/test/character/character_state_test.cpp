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
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/locator.h"
#include "momentum/character/locator_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/pose_shape.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

// Test fixture for CharacterState tests
template <typename T>
class CharacterStateTest : public testing::Test {
 protected:
  using Type = T;
  using CharacterType = CharacterT<T>;
  using CharacterStateType = CharacterStateT<T>;
  using Vector3Type = Vector3<T>;
  using QuaternionType = Quaternion<T>;

  void SetUp() override {
    // Create a test character with 5 joints
    character = createTestCharacter<T>(5);

    // Create a character with blend shapes
    characterWithBlendShapes = withTestBlendShapes(character);
  }

  CharacterType character;
  CharacterType characterWithBlendShapes;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(CharacterStateTest, Types);

// Helper functions to reduce test verbosity
template <typename T>
void expectStateHasCorrectComponents(
    const CharacterStateT<T>& state,
    const CharacterT<T>& character,
    bool expectMesh = true,
    bool expectCollision = true) {
  EXPECT_EQ(state.parameters.pose.size(), character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(state.parameters.offsets.size(), character.parameterTransform.numJointParameters());
  EXPECT_EQ(state.skeletonState.jointState.size(), character.skeleton.joints.size());
  EXPECT_EQ(!!state.meshState, expectMesh);
  EXPECT_EQ(!!state.collisionState, expectCollision);

  if (expectMesh) {
    EXPECT_EQ(state.meshState->vertices.size(), character.mesh->vertices.size());
  }

  if (expectCollision) {
    EXPECT_EQ(state.collisionState->origin.size(), character.collision->size());
  }
}

template <typename T>
bool hasVerticesChangedFromBindPose(
    const CharacterStateT<T>& state,
    const CharacterStateT<T>& bindPoseState) {
  if (!state.meshState || !bindPoseState.meshState) {
    return false;
  }

  for (size_t i = 0; i < state.meshState->vertices.size(); ++i) {
    if (!state.meshState->vertices[i].isApprox(bindPoseState.meshState->vertices[i])) {
      return true;
    }
  }
  return false;
}

// Test default constructor
TYPED_TEST(CharacterStateTest, DefaultConstructor) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  CharacterStateType defaultState;

  // Check that the state is empty
  EXPECT_TRUE(defaultState.parameters.pose.size() == 0);
  EXPECT_TRUE(defaultState.parameters.offsets.size() == 0);
  EXPECT_TRUE(defaultState.skeletonState.jointState.empty());
  EXPECT_TRUE(defaultState.locatorState.position.empty());
  EXPECT_FALSE(defaultState.collisionState);
}

// Test copy constructor
TYPED_TEST(CharacterStateTest, CopyConstructor) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a state from the test character
  CharacterStateType state(this->character);

  // Create a copy of the state
  CharacterStateType copiedState(state);

  // Check that the copy has the same components
  EXPECT_EQ(copiedState.parameters.pose.size(), state.parameters.pose.size());
  EXPECT_EQ(copiedState.parameters.offsets.size(), state.parameters.offsets.size());
  EXPECT_EQ(copiedState.skeletonState.jointState.size(), state.skeletonState.jointState.size());

  // Check that meshState was deep copied
  EXPECT_TRUE(copiedState.meshState);
  EXPECT_TRUE(state.meshState);
  EXPECT_NE(copiedState.meshState.get(), state.meshState.get());
  EXPECT_EQ(copiedState.meshState->vertices.size(), state.meshState->vertices.size());

  // Check that collisionState was deep copied
  EXPECT_TRUE(copiedState.collisionState);
  EXPECT_TRUE(state.collisionState);
  EXPECT_NE(copiedState.collisionState.get(), state.collisionState.get());
  EXPECT_EQ(copiedState.collisionState->origin.size(), state.collisionState->origin.size());

  // Check that the copy is a deep copy (modifying one doesn't affect the other)
  if (copiedState.meshState && state.meshState && !copiedState.meshState->vertices.empty()) {
    Vector3f originalVertex = state.meshState->vertices[0];
    copiedState.meshState->vertices[0] = Vector3f(99, 99, 99);
    EXPECT_NE(copiedState.meshState->vertices[0], state.meshState->vertices[0]);
    EXPECT_EQ(state.meshState->vertices[0], originalVertex);
  }
}

// Test move constructor
TYPED_TEST(CharacterStateTest, MoveConstructor) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a state from the test character
  CharacterStateType state(this->character);

  // Store pointers to the original resources
  auto originalMeshState = state.meshState.get();
  auto originalCollisionState = state.collisionState.get();

  // Create a moved state
  CharacterStateType movedState(std::move(state));

  // Check that the moved state has the resources
  EXPECT_TRUE(movedState.meshState);
  EXPECT_TRUE(movedState.collisionState);

  // Check that the resources were moved (not copied)
  EXPECT_EQ(movedState.meshState.get(), originalMeshState);
  EXPECT_EQ(movedState.collisionState.get(), originalCollisionState);
}

// Test move assignment operator
TYPED_TEST(CharacterStateTest, MoveAssignmentOperator) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a state from the test character
  CharacterStateType state(this->character);

  // Store pointers to the original resources
  auto originalMeshState = state.meshState.get();
  auto originalCollisionState = state.collisionState.get();

  // Create a default state
  CharacterStateType assignedState;

  // Assign the state using move assignment
  assignedState = std::move(state);

  // Check that the assigned state has the resources
  EXPECT_TRUE(assignedState.meshState);
  EXPECT_TRUE(assignedState.collisionState);

  // Check that the resources were moved (not copied)
  EXPECT_EQ(assignedState.meshState.get(), originalMeshState);
  EXPECT_EQ(assignedState.collisionState.get(), originalCollisionState);
}

// Test constructor with reference character
TYPED_TEST(CharacterStateTest, ConstructorWithReferenceCharacter) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a state from the test character
  CharacterStateType state(this->character);

  // Check that the state has the correct components
  EXPECT_EQ(
      state.parameters.pose.size(), this->character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(
      state.parameters.offsets.size(), this->character.parameterTransform.numJointParameters());
  EXPECT_EQ(state.skeletonState.jointState.size(), this->character.skeleton.joints.size());
  // LocatorState doesn't have a size() method, so we don't check it
  EXPECT_TRUE(state.meshState);
  EXPECT_TRUE(state.collisionState);

  // Check that the mesh was skinned correctly
  EXPECT_EQ(state.meshState->vertices.size(), this->character.mesh->vertices.size());

  // Check that the collision state was updated correctly
  EXPECT_EQ(state.collisionState->origin.size(), this->character.collision->size());

  // Test with updateMesh = false
  CharacterStateType stateNoMesh(this->character, false, true);
  EXPECT_FALSE(stateNoMesh.meshState);
  EXPECT_TRUE(stateNoMesh.collisionState);

  // Test with updateCollision = false
  CharacterStateType stateNoCollision(this->character, true, false);
  EXPECT_TRUE(stateNoCollision.meshState);
  EXPECT_FALSE(stateNoCollision.collisionState);

  // Test with both updateMesh and updateCollision = false
  CharacterStateType stateNoMeshNoCollision(this->character, false, false);
  EXPECT_FALSE(stateNoMeshNoCollision.meshState);
  EXPECT_FALSE(stateNoMeshNoCollision.collisionState);
}

// Test constructor with parameters and reference character
TYPED_TEST(CharacterStateTest, ConstructorWithParametersAndReferenceCharacter) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create parameters with non-zero values
  CharacterParameters params;
  params.pose = VectorXf::Ones(this->character.parameterTransform.numAllModelParameters());
  params.offsets = VectorXf::Zero(this->character.parameterTransform.numJointParameters());

  // Create a state with the parameters
  CharacterStateType state(params, this->character);

  // Check that the state has the correct components
  EXPECT_EQ(state.parameters.pose.size(), params.pose.size());
  EXPECT_EQ(state.parameters.offsets.size(), params.offsets.size());
  EXPECT_EQ(state.skeletonState.jointState.size(), this->character.skeleton.joints.size());
  // LocatorState doesn't have a size() method, so we don't check it
  EXPECT_TRUE(state.meshState);
  EXPECT_TRUE(state.collisionState);

  // Check that the parameters were set correctly
  for (int i = 0; i < params.pose.size(); ++i) {
    EXPECT_FLOAT_EQ(state.parameters.pose[i], params.pose[i]);
  }

  // Test with updateMesh = false
  CharacterStateType stateNoMesh(params, this->character, false, true);
  EXPECT_FALSE(stateNoMesh.meshState);
  EXPECT_TRUE(stateNoMesh.collisionState);

  // Test with updateCollision = false
  CharacterStateType stateNoCollision(params, this->character, true, false);
  EXPECT_TRUE(stateNoCollision.meshState);
  EXPECT_FALSE(stateNoCollision.collisionState);

  // Test with applyLimits = false
  // Create parameters that exceed limits
  CharacterParameters paramsExceedingLimits;
  paramsExceedingLimits.pose =
      VectorXf::Constant(this->character.parameterTransform.numAllModelParameters(), 100.0f);
  paramsExceedingLimits.offsets =
      VectorXf::Zero(this->character.parameterTransform.numJointParameters());

  // Create a state with the parameters and applyLimits = false
  CharacterStateType stateNoLimits(paramsExceedingLimits, this->character, true, true, false);

  // The parameters should not be limited
  for (int i = 0; i < paramsExceedingLimits.pose.size(); ++i) {
    EXPECT_FLOAT_EQ(stateNoLimits.parameters.pose[i], paramsExceedingLimits.pose[i]);
  }
}

// Test set method
TYPED_TEST(CharacterStateTest, Set) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a default state
  CharacterStateType state;

  // Create parameters with non-zero values
  CharacterParameters params;
  params.pose = VectorXf::Ones(this->character.parameterTransform.numAllModelParameters());
  params.offsets = VectorXf::Zero(this->character.parameterTransform.numJointParameters());

  // Set the state with the parameters
  state.set(params, this->character);

  // Check that the state has the correct components
  EXPECT_EQ(state.parameters.pose.size(), params.pose.size());
  EXPECT_EQ(state.parameters.offsets.size(), params.offsets.size());
  EXPECT_EQ(state.skeletonState.jointState.size(), this->character.skeleton.joints.size());
  // LocatorState doesn't have a size() method, so we don't check it
  EXPECT_TRUE(state.meshState);
  EXPECT_TRUE(state.collisionState);

  // Check that the parameters were set correctly
  for (int i = 0; i < params.pose.size(); ++i) {
    EXPECT_FLOAT_EQ(state.parameters.pose[i], params.pose[i]);
  }

  // Test with updateMesh = false
  CharacterStateType stateNoMesh;
  stateNoMesh.set(params, this->character, false, true);
  EXPECT_FALSE(stateNoMesh.meshState);
  EXPECT_TRUE(stateNoMesh.collisionState);

  // Test with updateCollision = false
  CharacterStateType stateNoCollision;
  stateNoCollision.set(params, this->character, true, false);
  EXPECT_TRUE(stateNoCollision.meshState);
  EXPECT_FALSE(stateNoCollision.collisionState);

  // Test with empty offsets
  CharacterParameters paramsNoOffsets;
  paramsNoOffsets.pose = VectorXf::Ones(this->character.parameterTransform.numAllModelParameters());
  // Don't set offsets, let the set method handle it

  CharacterStateType stateNoOffsets;
  stateNoOffsets.set(paramsNoOffsets, this->character);

  // Check that offsets were initialized to zero
  EXPECT_EQ(
      stateNoOffsets.parameters.offsets.size(),
      this->character.parameterTransform.numJointParameters());
  for (int i = 0; i < stateNoOffsets.parameters.offsets.size(); ++i) {
    EXPECT_FLOAT_EQ(stateNoOffsets.parameters.offsets[i], 0.0f);
  }
}

// Test setBindPose method
TYPED_TEST(CharacterStateTest, SetBindPose) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Create a default state
  CharacterStateType state;

  // Set the state to the bind pose
  state.setBindPose(this->character);

  // Check that the state has the correct components
  EXPECT_EQ(
      state.parameters.pose.size(), this->character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(
      state.parameters.offsets.size(), this->character.parameterTransform.numJointParameters());
  EXPECT_EQ(state.skeletonState.jointState.size(), this->character.skeleton.joints.size());
  // LocatorState doesn't have a size() method, so we don't check it
  EXPECT_TRUE(state.meshState);
  EXPECT_TRUE(state.collisionState);

  // Check that the parameters are the bind pose (all zeros)
  for (int i = 0; i < state.parameters.pose.size(); ++i) {
    EXPECT_FLOAT_EQ(state.parameters.pose[i], 0.0f);
  }
  for (int i = 0; i < state.parameters.offsets.size(); ++i) {
    EXPECT_FLOAT_EQ(state.parameters.offsets[i], 0.0f);
  }

  // Test with updateMesh = false
  CharacterStateType stateNoMesh;
  stateNoMesh.setBindPose(this->character, false, true);
  EXPECT_FALSE(stateNoMesh.meshState);
  EXPECT_TRUE(stateNoMesh.collisionState);

  // Test with updateCollision = false
  CharacterStateType stateNoCollision;
  stateNoCollision.setBindPose(this->character, true, false);
  EXPECT_TRUE(stateNoCollision.meshState);
  EXPECT_FALSE(stateNoCollision.collisionState);
}

// Test with blend shapes
TYPED_TEST(CharacterStateTest, WithBlendShapes) {
  using CharacterStateType = typename TestFixture::CharacterStateType;

  // Skip the test if the character doesn't have blend shapes
  if (!this->characterWithBlendShapes.blendShape) {
    return;
  }

  // Create parameters with non-zero blend shape weights
  CharacterParameters params;
  params.pose =
      VectorXf::Zero(this->characterWithBlendShapes.parameterTransform.numAllModelParameters());
  params.offsets =
      VectorXf::Zero(this->characterWithBlendShapes.parameterTransform.numJointParameters());

  // Set the first blend shape parameter to 1.0
  int blendShapeParamIndex =
      this->characterWithBlendShapes.parameterTransform.blendShapeParameters(0);
  if (blendShapeParamIndex >= 0) {
    params.pose[blendShapeParamIndex] = 1.0f;
  }

  // Create a state with the parameters
  CharacterStateType state(params, this->characterWithBlendShapes);

  // Check that the state has the correct components
  expectStateHasCorrectComponents(state, this->characterWithBlendShapes);

  // Create a bind pose state for comparison
  CharacterStateType bindPoseState(this->characterWithBlendShapes);

  // Verify that the blend shapes affected the mesh
  EXPECT_TRUE(hasVerticesChangedFromBindPose(state, bindPoseState));
}

// Test with pose shapes
TYPED_TEST(CharacterStateTest, WithPoseShapes) {
  using CharacterStateType = typename TestFixture::CharacterStateType;
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character with pose shapes
  CharacterType characterWithPoseShapes = this->character;

  // Add a simple pose shape
  auto poseShapes = std::make_unique<PoseShape>();
  poseShapes->baseJoint = 1;
  poseShapes->baseRot = Quaternionf::Identity();
  poseShapes->jointMap = {1}; // Joint 1

  // Set base shape from mesh vertices
  VectorXf baseShape(characterWithPoseShapes.mesh->vertices.size() * 3);
  for (size_t i = 0; i < characterWithPoseShapes.mesh->vertices.size(); ++i) {
    baseShape[i * 3] = characterWithPoseShapes.mesh->vertices[i].x();
    baseShape[i * 3 + 1] = characterWithPoseShapes.mesh->vertices[i].y();
    baseShape[i * 3 + 2] = characterWithPoseShapes.mesh->vertices[i].z();
  }
  poseShapes->baseShape = baseShape;

  // Create shape vectors and modify first vertex
  MatrixXf shapeVectors = MatrixXf::Zero(baseShape.size(), 4);
  if (!characterWithPoseShapes.mesh->vertices.empty()) {
    shapeVectors(0, 0) = 1.0f; // Move x coordinate for x component of quaternion
  }
  poseShapes->shapeVectors = shapeVectors;
  characterWithPoseShapes.poseShapes = std::move(poseShapes);

  // Create parameters with joint rotation
  CharacterParameters params;
  params.pose = VectorXf::Zero(characterWithPoseShapes.parameterTransform.numAllModelParameters());
  params.offsets = VectorXf::Zero(characterWithPoseShapes.parameterTransform.numJointParameters());

  // Set rotation parameter directly
  int rotXParamIndex = 1; // Parameter controlling joint 1's X rotation
  params.pose[rotXParamIndex] = 0.5f;

  // Create a state with the parameters
  CharacterStateType state(params, characterWithPoseShapes);

  // Check that the state has the correct components
  expectStateHasCorrectComponents(state, characterWithPoseShapes);

  // Create a bind pose state for comparison
  CharacterStateType bindPoseState(characterWithPoseShapes);

  // Verify that the pose shapes affected the mesh
  EXPECT_TRUE(hasVerticesChangedFromBindPose(state, bindPoseState));
}

// Test with both blend shapes and pose shapes
TYPED_TEST(CharacterStateTest, WithBlendShapesAndPoseShapes) {
  using CharacterStateType = typename TestFixture::CharacterStateType;
  using CharacterType = typename TestFixture::CharacterType;

  // Create a character with both blend shapes and pose shapes
  CharacterType characterWithBoth = withTestBlendShapes(this->character);

  // Skip the test if the character doesn't have blend shapes
  if (!characterWithBoth.blendShape) {
    return;
  }

  // Add pose shapes
  auto poseShapes = std::make_unique<PoseShape>();
  poseShapes->baseJoint = 1;
  poseShapes->baseRot = Quaternionf::Identity();
  poseShapes->jointMap = {1}; // Joint 1

  // Set base shape from mesh vertices
  VectorXf baseShape(characterWithBoth.mesh->vertices.size() * 3);
  for (size_t i = 0; i < characterWithBoth.mesh->vertices.size(); ++i) {
    baseShape[i * 3] = characterWithBoth.mesh->vertices[i].x();
    baseShape[i * 3 + 1] = characterWithBoth.mesh->vertices[i].y();
    baseShape[i * 3 + 2] = characterWithBoth.mesh->vertices[i].z();
  }
  poseShapes->baseShape = baseShape;

  // Create shape vectors and modify first vertex
  MatrixXf shapeVectors = MatrixXf::Zero(baseShape.size(), 4);
  if (!characterWithBoth.mesh->vertices.empty()) {
    shapeVectors(0, 0) = 1.0f; // Move x coordinate for x component of quaternion
  }
  poseShapes->shapeVectors = shapeVectors;
  characterWithBoth.poseShapes = std::move(poseShapes);

  // Create parameters with non-zero blend shape weights and joint rotation
  CharacterParameters params;
  params.pose = VectorXf::Zero(characterWithBoth.parameterTransform.numAllModelParameters());
  params.offsets = VectorXf::Zero(characterWithBoth.parameterTransform.numJointParameters());

  // Set the first blend shape parameter to 1.0
  int blendShapeParamIndex = characterWithBoth.parameterTransform.blendShapeParameters(0);
  if (blendShapeParamIndex >= 0) {
    params.pose[blendShapeParamIndex] = 1.0f;
  }

  // Set joint rotation parameter
  int rotXParamIndex = 1; // Parameter controlling joint 1's X rotation
  params.pose[rotXParamIndex] = 0.5f;

  // Create a state with the parameters
  CharacterStateType state(params, characterWithBoth);

  // Check that the state has the correct components
  expectStateHasCorrectComponents(state, characterWithBoth);

  // Create a bind pose state for comparison
  CharacterStateType bindPoseState(characterWithBoth);

  // Verify that the blend shapes and pose shapes affected the mesh
  EXPECT_TRUE(hasVerticesChangedFromBindPose(state, bindPoseState));
}
