/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape.h>
#include <momentum/character/blend_shape_skinning.h>
#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/checks.h>
#include <momentum/math/mesh.h>
#include <momentum/math/types.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/helpers/expect_throw.h>

#include <gtest/gtest.h>

namespace momentum {

template <typename T>
struct BlendShapeSkinningTest : testing::Test {
  using Type = T;

  void SetUp() override {
    // Create a test character
    character = createTestCharacter();

    // Add blend shapes to the character
    character = withTestBlendShapes(character);

    // Set up skeleton state
    skeletonState.jointState.resize(character.skeleton.joints.size());
    for (auto& jointState : skeletonState.jointState) {
      jointState.localRotation() = Eigen::Quaternion<T>::Identity();
      jointState.localTranslation().setZero();
      jointState.localScale() = T(1);
      jointState.transform.rotation = Eigen::Quaternion<T>::Identity();
      jointState.transform.translation.setZero();
      jointState.transform.scale = T(1);
      jointState.transformation = Affine3<T>::Identity();
    }

    // Set up blend weights
    const size_t numBlendShapes = character.blendShape->shapeSize();
    blendWeights.v = Eigen::VectorX<T>::Zero(numBlendShapes);

    // Set up model parameters - make sure it's large enough for all possible indices
    int maxParamIdx = -1;
    for (Eigen::Index i = 0; i < character.parameterTransform.blendShapeParameters.size(); ++i) {
      const auto paramIdx = character.parameterTransform.blendShapeParameters[i];
      if (paramIdx >= 0) {
        maxParamIdx = std::max(maxParamIdx, static_cast<int>(paramIdx));
      }
    }

    // Add some buffer to ensure we have enough space
    int size = std::max(
        static_cast<int>(character.parameterTransform.blendShapeParameters.size()),
        maxParamIdx + 1);
    modelParams = ModelParametersT<T>(Eigen::VectorX<T>::Zero(size));

    // Set up output mesh
    outputMesh.vertices.resize(character.mesh->vertices.size(), Eigen::Vector3<T>::Zero());
    outputMesh.normals.resize(character.mesh->normals.size(), Eigen::Vector3<T>::Zero());
    outputMesh.faces = character.mesh->faces;
  }

  Character character;
  SkeletonStateT<T> skeletonState;
  BlendWeightsT<T> blendWeights;
  ModelParametersT<T> modelParams;
  MeshT<T> outputMesh;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(BlendShapeSkinningTest, Types);

// Test skinWithBlendShapes with BlendWeights
TYPED_TEST(BlendShapeSkinningTest, SkinWithBlendShapesUsingBlendWeights) {
  using T = typename TestFixture::Type;

  // Set non-zero blend weights
  this->blendWeights.v = Eigen::VectorX<T>::Ones(this->blendWeights.v.size()) * T(0.5);

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());
  EXPECT_EQ(this->outputMesh.normals.size(), this->character.mesh->normals.size());
  EXPECT_EQ(this->outputMesh.faces.size(), this->character.mesh->faces.size());

  // Verify that the output mesh is different from the input mesh due to blend shapes
  bool anyDifferent = false;
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(
            this->character.mesh->vertices[i].template cast<T>())) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test skinWithBlendShapes with ModelParameters
TYPED_TEST(BlendShapeSkinningTest, SkinWithBlendShapesUsingModelParameters) {
  using T = typename TestFixture::Type;

  // Resize model parameters if needed to accommodate blend shape parameters
  int maxParamIdx = -1;
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.blendShapeParameters[i];
    if (paramIdx >= 0) {
      maxParamIdx = std::max(maxParamIdx, static_cast<int>(paramIdx));
    }
  }

  // Set non-zero model parameters for blend shape parameters
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.blendShapeParameters[i];
    if (paramIdx >= 0) {
      this->modelParams(paramIdx) = T(0.5);
    }
  }

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->modelParams, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());
  EXPECT_EQ(this->outputMesh.normals.size(), this->character.mesh->normals.size());
  EXPECT_EQ(this->outputMesh.faces.size(), this->character.mesh->faces.size());

  // Verify that the output mesh is different from the input mesh due to blend shapes
  bool anyDifferent = false;
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(
            this->character.mesh->vertices[i].template cast<T>())) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test extractBlendWeights
TYPED_TEST(BlendShapeSkinningTest, ExtractBlendWeights) {
  using T = typename TestFixture::Type;

  // Set non-zero model parameters for blend shape parameters
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.blendShapeParameters[i];
    if (paramIdx >= 0) {
      this->modelParams(paramIdx) = T(0.5) * (i + 1);
    }
  }

  // Extract blend weights
  BlendWeightsT<T> extractedWeights =
      extractBlendWeights(this->character.parameterTransform, this->modelParams);

  // Verify extracted weights have correct size
  EXPECT_EQ(
      extractedWeights.v.size(), this->character.parameterTransform.blendShapeParameters.size());

  // Verify extracted weights match expected values
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    EXPECT_NEAR(extractedWeights.v(i), T(0.5) * (i + 1), 1e-5);
  }
}

// Test extractFaceExpressionBlendWeights
TYPED_TEST(BlendShapeSkinningTest, ExtractFaceExpressionBlendWeights) {
  using T = typename TestFixture::Type;

  // First, add face expression blend shapes to the character
  this->character = withTestFaceExpressionBlendShapes(this->character);

  // Resize model parameters if needed to accommodate face expression parameters
  int maxParamIdx = -1;
  for (Eigen::Index i = 0; i < this->character.parameterTransform.faceExpressionParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.faceExpressionParameters[i];
    if (paramIdx >= 0) {
      maxParamIdx = std::max(maxParamIdx, static_cast<int>(paramIdx));
    }
  }

  if (maxParamIdx >= this->modelParams.size()) {
    // Create a new vector with the desired size
    Eigen::VectorX<T> newParams = Eigen::VectorX<T>::Zero(maxParamIdx + 1);

    // Copy existing values
    for (Eigen::Index i = 0; i < this->modelParams.size(); ++i) {
      newParams[i] = this->modelParams[i];
    }

    // Assign the new vector to modelParams
    this->modelParams = newParams;
  }

  // Set non-zero model parameters for face expression parameters
  for (Eigen::Index i = 0; i < this->character.parameterTransform.faceExpressionParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.faceExpressionParameters[i];
    if (paramIdx >= 0) {
      this->modelParams(paramIdx) = T(0.25) * (i + 1);
    }
  }

  // Extract face expression blend weights
  BlendWeightsT<T> extractedWeights =
      extractFaceExpressionBlendWeights(this->character.parameterTransform, this->modelParams);

  // Verify extracted weights have correct size
  EXPECT_EQ(
      extractedWeights.v.size(),
      this->character.parameterTransform.faceExpressionParameters.size());

  // Verify extracted weights match expected values
  for (Eigen::Index i = 0; i < this->character.parameterTransform.faceExpressionParameters.size();
       ++i) {
    EXPECT_NEAR(extractedWeights.v(i), T(0.25) * (i + 1), 1e-5);
  }
}

// Test skinWithBlendShapes with joint transformations
TYPED_TEST(BlendShapeSkinningTest, SkinWithBlendShapesAndJointTransformations) {
  using T = typename TestFixture::Type;

  // Set non-zero blend weights
  this->blendWeights.v = Eigen::VectorX<T>::Ones(this->blendWeights.v.size()) * T(0.5);

  // Apply joint transformations
  // Rotate first joint around Y axis by 90 degrees
  this->skeletonState.jointState[0].transformation =
      Affine3<T>::Identity() * Eigen::AngleAxis<T>(pi<T>() / 2, Eigen::Vector3<T>::UnitY());

  // Translate second joint
  this->skeletonState.jointState[1].transformation = Affine3<T>::Identity();
  this->skeletonState.jointState[1].transformation.translate(Eigen::Vector3<T>(1, 2, 3));

  // Store original vertices for comparison
  MeshT<T> originalMesh;
  originalMesh.vertices.resize(this->character.mesh->vertices.size());
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    originalMesh.vertices[i] = this->character.mesh->vertices[i].template cast<T>();
  }

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());

  // Verify that the output mesh is different from the original mesh due to both
  // blend shapes and joint transformations
  bool anyDifferent = false;
  for (size_t i = 0; i < originalMesh.vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(originalMesh.vertices[i])) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test skinWithBlendShapes with zero blend weights
TYPED_TEST(BlendShapeSkinningTest, SkinWithZeroBlendWeights) {
  using T = typename TestFixture::Type;

  // Keep blend weights at zero

  // Apply joint transformations
  // Rotate first joint around Y axis by 90 degrees
  this->skeletonState.jointState[0].transformation =
      Affine3<T>::Identity() * Eigen::AngleAxis<T>(pi<T>() / 2, Eigen::Vector3<T>::UnitY());

  // Store original vertices for comparison
  MeshT<T> originalMesh;
  originalMesh.vertices.resize(this->character.mesh->vertices.size());
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    originalMesh.vertices[i] = this->character.mesh->vertices[i].template cast<T>();
  }

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());

  // Verify that the output mesh is different from the original mesh due to joint transformations
  bool anyDifferent = false;
  for (size_t i = 0; i < originalMesh.vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(originalMesh.vertices[i])) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test skinWithBlendShapes with identity joint transformations
TYPED_TEST(BlendShapeSkinningTest, SkinWithIdentityJointTransformations) {
  using T = typename TestFixture::Type;

  // Set non-zero blend weights
  this->blendWeights.v = Eigen::VectorX<T>::Ones(this->blendWeights.v.size()) * T(0.5);

  // Keep joint transformations at identity

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());

  // Verify that the output mesh is different from the input mesh due to blend shapes
  bool anyDifferent = false;
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(
            this->character.mesh->vertices[i].template cast<T>())) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test skinWithBlendShapes with character that has no blend shapes
TYPED_TEST(BlendShapeSkinningTest, SkinWithNoBlendShapes) {
  using T = typename TestFixture::Type;

  // Create a character without blend shapes
  Character characterWithoutBlendShapes = createTestCharacter();
  characterWithoutBlendShapes.blendShape = nullptr;

  // Set non-zero blend weights
  this->blendWeights.v = Eigen::VectorX<T>::Ones(this->blendWeights.v.size()) * T(0.5);

  // Apply joint transformations
  this->skeletonState.jointState[0].transformation =
      Affine3<T>::Identity() * Eigen::AngleAxis<T>(pi<T>() / 2, Eigen::Vector3<T>::UnitY());

  // Store original vertices for comparison
  MeshT<T> originalMesh;
  originalMesh.vertices.resize(characterWithoutBlendShapes.mesh->vertices.size());
  for (size_t i = 0; i < characterWithoutBlendShapes.mesh->vertices.size(); ++i) {
    originalMesh.vertices[i] = characterWithoutBlendShapes.mesh->vertices[i].template cast<T>();
  }

  // Apply skinning
  skinWithBlendShapes(
      characterWithoutBlendShapes, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), characterWithoutBlendShapes.mesh->vertices.size());

  // Verify that the output mesh is different from the original mesh due to joint transformations
  bool anyDifferent = false;
  for (size_t i = 0; i < originalMesh.vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(originalMesh.vertices[i])) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test error cases for skinWithBlendShapes
TYPED_TEST(BlendShapeSkinningTest, SkinWithBlendShapesErrors) {
  using T = typename TestFixture::Type;

  // Test with mismatched state and inverse bind pose sizes
  SkeletonStateT<T> smallState;
  smallState.jointState.resize(1); // Only one joint

  MOMENTUM_EXPECT_DEATH(
      skinWithBlendShapes(this->character, smallState, this->blendWeights, this->outputMesh), ".*");

  // Test with mismatched skin weights and mesh vertices sizes
  Character characterWithSmallSkin = this->character;
  characterWithSmallSkin.skinWeights = std::make_unique<SkinWeights>();
  characterWithSmallSkin.skinWeights->index.resize(1, kMaxSkinJoints); // Only one vertex
  characterWithSmallSkin.skinWeights->weight.resize(1, kMaxSkinJoints);

  MOMENTUM_EXPECT_DEATH(
      skinWithBlendShapes(
          characterWithSmallSkin, this->skeletonState, this->blendWeights, this->outputMesh),
      ".*");

  // Test with blend weights size exceeding shape vectors columns
  if (this->character.blendShape) {
    BlendWeightsT<T> tooManyWeights;
    tooManyWeights.v =
        Eigen::VectorX<T>::Ones(this->character.blendShape->getShapeVectors().cols() + 1);

    MOMENTUM_EXPECT_DEATH(
        skinWithBlendShapes(this->character, this->skeletonState, tooManyWeights, this->outputMesh),
        ".*");
  }

  // Test with null skinWeights
  Character characterWithNullSkin = this->character;
  characterWithNullSkin.skinWeights = nullptr;

  MOMENTUM_EXPECT_DEATH(
      skinWithBlendShapes(
          characterWithNullSkin, this->skeletonState, this->blendWeights, this->outputMesh),
      ".*");

  // Test with null mesh
  Character characterWithNullMesh = this->character;
  characterWithNullMesh.mesh = nullptr;

  MOMENTUM_EXPECT_DEATH(
      skinWithBlendShapes(
          characterWithNullMesh, this->skeletonState, this->blendWeights, this->outputMesh),
      ".*");
}

// Test with different blend weight values
TYPED_TEST(BlendShapeSkinningTest, DifferentBlendWeightValues) {
  using T = typename TestFixture::Type;

  // Set blend weights with different values
  for (Eigen::Index i = 0; i < this->blendWeights.v.size(); ++i) {
    this->blendWeights.v(i) = T(0.1) * (i + 1);
  }

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());

  // Verify that the output mesh is different from the input mesh due to blend shapes
  bool anyDifferent = false;
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(
            this->character.mesh->vertices[i].template cast<T>())) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);

  // Store the first result
  MeshT<T> firstResult = this->outputMesh;

  // Change blend weights
  for (Eigen::Index i = 0; i < this->blendWeights.v.size(); ++i) {
    this->blendWeights.v(i) = T(0.2) * (i + 1);
  }

  // Apply skinning again
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify that the second result is different from the first result
  bool resultsDifferent = false;
  for (size_t i = 0; i < this->outputMesh.vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(firstResult.vertices[i])) {
      resultsDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(resultsDifferent);
}

// Test consistency between the two skinWithBlendShapes overloads
TYPED_TEST(BlendShapeSkinningTest, ConsistencyBetweenOverloads) {
  using T = typename TestFixture::Type;

  // Resize model parameters if needed to accommodate blend shape parameters
  int maxParamIdx = -1;
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.blendShapeParameters[i];
    if (paramIdx >= 0) {
      maxParamIdx = std::max(maxParamIdx, static_cast<int>(paramIdx));
    }
  }

  // Set non-zero model parameters for blend shape parameters
  for (Eigen::Index i = 0; i < this->character.parameterTransform.blendShapeParameters.size();
       ++i) {
    const auto paramIdx = this->character.parameterTransform.blendShapeParameters[i];
    if (paramIdx >= 0) {
      this->modelParams(paramIdx) = T(0.5) * (i + 1);
    }
  }

  // Extract blend weights
  BlendWeightsT<T> extractedWeights =
      extractBlendWeights(this->character.parameterTransform, this->modelParams);

  // Apply skinning using the first overload (with blend weights)
  MeshT<T> result1;
  result1.vertices.resize(this->character.mesh->vertices.size(), Eigen::Vector3<T>::Zero());
  result1.normals.resize(this->character.mesh->normals.size(), Eigen::Vector3<T>::Zero());
  result1.faces = this->character.mesh->faces;

  skinWithBlendShapes(this->character, this->skeletonState, extractedWeights, result1);

  // Apply skinning using the second overload (with model parameters)
  MeshT<T> result2;
  result2.vertices.resize(this->character.mesh->vertices.size(), Eigen::Vector3<T>::Zero());
  result2.normals.resize(this->character.mesh->normals.size(), Eigen::Vector3<T>::Zero());
  result2.faces = this->character.mesh->faces;

  skinWithBlendShapes(this->character, this->skeletonState, this->modelParams, result2);

  // Verify that both results are the same
  for (size_t i = 0; i < result1.vertices.size(); ++i) {
    EXPECT_TRUE(result1.vertices[i].isApprox(result2.vertices[i]));
  }

  for (size_t i = 0; i < result1.normals.size(); ++i) {
    EXPECT_TRUE(result1.normals[i].isApprox(result2.normals[i]));
  }
}

// Test with extreme blend weight values
TYPED_TEST(BlendShapeSkinningTest, ExtremeBlendWeightValues) {
  using T = typename TestFixture::Type;

  // Set extreme blend weights
  for (Eigen::Index i = 0; i < this->blendWeights.v.size(); ++i) {
    this->blendWeights.v(i) = (i % 2 == 0) ? T(10.0) : T(-10.0);
  }

  // Apply skinning
  skinWithBlendShapes(this->character, this->skeletonState, this->blendWeights, this->outputMesh);

  // Verify output mesh has correct dimensions
  EXPECT_EQ(this->outputMesh.vertices.size(), this->character.mesh->vertices.size());

  // Verify that the output mesh is different from the input mesh due to extreme blend shapes
  bool anyDifferent = false;
  for (size_t i = 0; i < this->character.mesh->vertices.size(); ++i) {
    if (!this->outputMesh.vertices[i].isApprox(
            this->character.mesh->vertices[i].template cast<T>())) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// Test with empty parameter transform
TYPED_TEST(BlendShapeSkinningTest, EmptyParameterTransform) {
  using T = typename TestFixture::Type;

  // Create a parameter transform with empty blend shape parameters
  ParameterTransform emptyParamTransform;

  // Extract blend weights from empty parameter transform
  BlendWeightsT<T> extractedWeights = extractBlendWeights(emptyParamTransform, this->modelParams);

  // Verify extracted weights have size 0
  EXPECT_EQ(extractedWeights.v.size(), 0);

  // Extract face expression blend weights from empty parameter transform
  BlendWeightsT<T> extractedFaceWeights =
      extractFaceExpressionBlendWeights(emptyParamTransform, this->modelParams);

  // Verify extracted face weights have size 0
  EXPECT_EQ(extractedFaceWeights.v.size(), 0);
}

// Test with negative parameter indices
TYPED_TEST(BlendShapeSkinningTest, NegativeParameterIndices) {
  using T = typename TestFixture::Type;

  // Create a parameter transform with negative indices
  ParameterTransform paramTransform;
  paramTransform.blendShapeParameters.resize(5);
  paramTransform.blendShapeParameters << -1, 0, -1, 1, -1;

  // Set up model parameters
  ModelParametersT<T> modelParams = Eigen::VectorX<T>::Ones(2);

  // Extract blend weights
  BlendWeightsT<T> extractedWeights = extractBlendWeights(paramTransform, modelParams);

  // Verify extracted weights have correct size
  EXPECT_EQ(extractedWeights.v.size(), 5);

  // Verify extracted weights match expected values
  EXPECT_NEAR(extractedWeights.v(0), T(0), 1e-5);
  EXPECT_NEAR(extractedWeights.v(1), T(1), 1e-5);
  EXPECT_NEAR(extractedWeights.v(2), T(0), 1e-5);
  EXPECT_NEAR(extractedWeights.v(3), T(1), 1e-5);
  EXPECT_NEAR(extractedWeights.v(4), T(0), 1e-5);
}

// Test with parameter indices out of bounds
TYPED_TEST(BlendShapeSkinningTest, ParameterIndicesOutOfBounds) {
  using T = typename TestFixture::Type;

  // Create a parameter transform with an out-of-bounds index
  ParameterTransform paramTransform;
  paramTransform.blendShapeParameters.resize(3);
  paramTransform.blendShapeParameters << 0, 1, 2;

  // Set up model parameters with fewer elements than the max index
  ModelParametersT<T> modelParams = Eigen::VectorX<T>::Ones(2);

  // This should not crash, but the out-of-bounds index should be ignored
  // Extract blend weights
  BlendWeightsT<T> extractedWeights = extractBlendWeights(paramTransform, modelParams);

  // Verify extracted weights have correct size
  EXPECT_EQ(extractedWeights.v.size(), 3);

  // Verify extracted weights match expected values
  EXPECT_NEAR(extractedWeights.v(0), T(1), 1e-5); // Index 0 is valid
  EXPECT_NEAR(extractedWeights.v(1), T(1), 1e-5); // Index 1 is valid
  EXPECT_NEAR(extractedWeights.v(2), T(0), 1e-5); // Index 2 is out of bounds, should be 0
}

} // namespace momentum
