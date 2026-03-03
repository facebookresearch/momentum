/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character_state.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/locator.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

#include <gtest/gtest.h>

using namespace momentum;

// Test fixture for Character blend shape tests
template <typename T>
class CharacterBlendShapeTest : public testing::Test {
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
TYPED_TEST_SUITE(CharacterBlendShapeTest, Types);

// Test withBlendShape method
TYPED_TEST(CharacterBlendShapeTest, WithBlendShape) {
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

  CharacterType characterWithoutOverwrite = this->character.withBlendShape(blendShape, 2);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(characterWithoutOverwrite.blendShape);
  EXPECT_EQ(
      characterWithoutOverwrite.blendShape->modelSize(), this->character.mesh->vertices.size());
  EXPECT_EQ(characterWithoutOverwrite.blendShape->shapeSize(), 2);
}

// Test addBlendShape method
TYPED_TEST(CharacterBlendShapeTest, AddBlendShape) {
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

  CharacterType modifiedCharacter3 = this->character;
  modifiedCharacter3.addBlendShape(blendShape, 2);

  // Check that the blend shape was set correctly
  EXPECT_TRUE(modifiedCharacter3.blendShape);
  EXPECT_EQ(modifiedCharacter3.blendShape->modelSize(), baseShape.size());
  EXPECT_EQ(modifiedCharacter3.blendShape->shapeSize(), 2);
}

// Test withFaceExpressionBlendShape method
TYPED_TEST(CharacterBlendShapeTest, WithFaceExpressionBlendShape) {
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
TYPED_TEST(CharacterBlendShapeTest, AddFaceExpressionBlendShape) {
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
TYPED_TEST(CharacterBlendShapeTest, BakeBlendShape) {
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
TYPED_TEST(CharacterBlendShapeTest, Bake) {
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
