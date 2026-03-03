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

// Test fixture for Character mesh reduction and skinning tests
template <typename T>
class CharacterMeshTest : public testing::Test {
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
TYPED_TEST_SUITE(CharacterMeshTest, Types);

// Test mesh reduction triangle correctness
TYPED_TEST(CharacterMeshTest, MeshReductionTriangleCorrectness) {
  using CharacterType = typename TestFixture::CharacterType;

  const auto& originalMesh = *this->character.mesh;
  const size_t originalVertexCount = originalMesh.vertices.size();

  // Create active vertices (keep every other vertex)
  std::vector<bool> activeVertices(originalVertexCount, false);
  for (size_t i = 0; i < originalVertexCount / 2; i++) {
    activeVertices[i] = true;
  }
  const size_t expectedNewVertexCount = (originalVertexCount + 1) / 2;

  // Reduce mesh by vertices
  CharacterType reducedCharacter = reduceMeshByVertices(this->character, activeVertices);

  // Verify basic properties
  EXPECT_TRUE(reducedCharacter.mesh);
  EXPECT_EQ(reducedCharacter.mesh->vertices.size(), expectedNewVertexCount);

  // Verify that only triangles where all vertices are active are included
  std::vector<bool> expectedActiveFaces = verticesToFaces(*this->character.mesh, activeVertices);
  size_t expectedFaceCount = 0;
  for (bool active : expectedActiveFaces) {
    if (active) {
      expectedFaceCount++;
    }
  }

  EXPECT_GT(expectedFaceCount, 0);

  EXPECT_EQ(reducedCharacter.mesh->faces.size(), expectedFaceCount);

  // Verify all face indices are valid for the reduced mesh
  for (const auto& face : reducedCharacter.mesh->faces) {
    EXPECT_LT(face[0], static_cast<int>(expectedNewVertexCount));
    EXPECT_LT(face[1], static_cast<int>(expectedNewVertexCount));
    EXPECT_LT(face[2], static_cast<int>(expectedNewVertexCount));
    EXPECT_GE(face[0], 0);
    EXPECT_GE(face[1], 0);
    EXPECT_GE(face[2], 0);
  }
}

// Test skinning consistency after mesh reduction
TYPED_TEST(CharacterMeshTest, MeshReductionSkinningConsistency) {
  using CharacterType = typename TestFixture::CharacterType;

  const auto& originalMesh = *this->character.mesh;
  const size_t originalVertexCount = originalMesh.vertices.size();

  // Create active vertices (keep every other vertex)
  std::vector<bool> activeVertices(originalVertexCount, false);
  std::vector<size_t> activeVertexIndices;
  for (size_t i = 0; i < originalVertexCount; i += 2) {
    activeVertices[i] = true;
    activeVertexIndices.push_back(i);
  }

  // Reduce mesh by vertices
  CharacterType reducedCharacter = reduceMeshByVertices(this->character, activeVertices);

  // Create a test pose with some joint transformations
  ModelParameters modelParams =
      ModelParameters::Zero(this->character.parameterTransform.numAllModelParameters());
  if (modelParams.size() > 0) {
    modelParams[0] = 0.1f; // root_tx
  }
  if (modelParams.size() > 3) {
    modelParams[3] = 0.2f; // root_rx
  }

  // Apply skinning to original character
  SkeletonState originalState(
      this->character.parameterTransform.apply(modelParams), this->character.skeleton, false);

  Mesh originalSkinnedMesh = *this->character.mesh;
  applySSD(
      this->character.inverseBindPose,
      *this->character.skinWeights,
      originalSkinnedMesh,
      originalState,
      originalSkinnedMesh);

  // Apply skinning to reduced character
  SkeletonState reducedState(
      reducedCharacter.parameterTransform.apply(modelParams), reducedCharacter.skeleton, false);

  Mesh reducedSkinnedMesh = *reducedCharacter.mesh;
  applySSD(
      reducedCharacter.inverseBindPose,
      *reducedCharacter.skinWeights,
      reducedSkinnedMesh,
      reducedState,
      reducedSkinnedMesh);

  // Compare skinned vertex positions
  // The reduced character's skinned vertices should match the corresponding
  // vertices from the original character's skinned mesh
  EXPECT_EQ(reducedSkinnedMesh.vertices.size(), activeVertexIndices.size());

  for (size_t i = 0; i < activeVertexIndices.size(); ++i) {
    const size_t originalIdx = activeVertexIndices[i];
    const Vector3f& originalVertex = originalSkinnedMesh.vertices[originalIdx];
    const Vector3f& reducedVertex = reducedSkinnedMesh.vertices[i];

    // Allow small floating point differences
    EXPECT_TRUE(originalVertex.isApprox(reducedVertex, 1e-5f))
        << "Mismatch at vertex " << i << " (original idx " << originalIdx
        << "): " << "original=" << originalVertex.transpose()
        << ", reduced=" << reducedVertex.transpose();
  }
}
