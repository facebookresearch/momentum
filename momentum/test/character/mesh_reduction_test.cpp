/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

#include <gtest/gtest.h>

using namespace momentum;

template <typename T>
class MeshReductionTest : public testing::Test {
 protected:
  using CharacterType = CharacterT<T>;

  void SetUp() override {
    character = createTestCharacter<T>(5);
  }

  CharacterType character;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(MeshReductionTest, Types);

// Test mesh reduction triangle correctness
TYPED_TEST(MeshReductionTest, TriangleCorrectness) {
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
TYPED_TEST(MeshReductionTest, SkinningConsistency) {
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

// Helper to add polygon data to a mesh for testing.
// Creates polygons from existing triangle faces by merging adjacent triangle pairs into quads.
void addTestPolyData(Mesh& mesh) {
  mesh.polyFaces.clear();
  mesh.polyFaceSizes.clear();
  mesh.polyTexcoordFaces.clear();

  // Merge pairs of triangles into quads; leave any leftover triangle as-is.
  // For each pair of adjacent triangles (2*i, 2*i+1), create a quad from the unique vertices.
  for (size_t i = 0; i + 1 < mesh.faces.size(); i += 2) {
    const auto& f0 = mesh.faces[i];
    const auto& f1 = mesh.faces[i + 1];

    // Quad from the 4 unique vertices of two triangles sharing an edge
    mesh.polyFaceSizes.push_back(4);
    mesh.polyFaces.push_back(static_cast<uint32_t>(f0[0]));
    mesh.polyFaces.push_back(static_cast<uint32_t>(f0[2]));
    mesh.polyFaces.push_back(static_cast<uint32_t>(f1[2]));
    mesh.polyFaces.push_back(static_cast<uint32_t>(f1[0]));
  }

  // If odd number of triangles, add the last one as a 3-gon
  if (mesh.faces.size() % 2 == 1) {
    const auto& f = mesh.faces.back();
    mesh.polyFaceSizes.push_back(3);
    mesh.polyFaces.push_back(static_cast<uint32_t>(f[0]));
    mesh.polyFaces.push_back(static_cast<uint32_t>(f[1]));
    mesh.polyFaces.push_back(static_cast<uint32_t>(f[2]));
  }
}

// Compute which original polygons should survive a reduction given activeVertices,
// and collect the original vertex positions for each surviving polygon.
// Returns {expectedPolySizes, expectedPolyVertexPositions}.
std::pair<std::vector<uint32_t>, std::vector<Vector3f>> expectedSurvivingPolygons(
    const Mesh& mesh,
    const std::vector<bool>& activeVertices) {
  std::vector<uint32_t> expectedSizes;
  std::vector<Vector3f> expectedPositions;
  uint32_t offset = 0;
  for (const auto polySize : mesh.polyFaceSizes) {
    bool allActive = true;
    for (uint32_t i = 0; i < polySize; ++i) {
      const auto vtx = mesh.polyFaces[offset + i];
      if (vtx >= activeVertices.size() || !activeVertices[vtx]) {
        allActive = false;
        break;
      }
    }
    if (allActive) {
      expectedSizes.push_back(polySize);
      for (uint32_t i = 0; i < polySize; ++i) {
        expectedPositions.push_back(mesh.vertices[mesh.polyFaces[offset + i]]);
      }
    }
    offset += polySize;
  }
  return {expectedSizes, expectedPositions};
}

// Verify that a reduced mesh's polygon data matches expected sizes and vertex positions.
void verifyReducedPolygons(
    const Mesh& reduced,
    const std::vector<uint32_t>& expectedSizes,
    const std::vector<Vector3f>& expectedPositions) {
  // Correct number of polygons survived
  ASSERT_EQ(reduced.polyFaceSizes.size(), expectedSizes.size());

  // Polygon sizes are preserved
  uint32_t offset = 0;
  uint32_t expectedPosIdx = 0;
  for (size_t polyIdx = 0; polyIdx < reduced.polyFaceSizes.size(); ++polyIdx) {
    EXPECT_EQ(reduced.polyFaceSizes[polyIdx], expectedSizes[polyIdx]);
    const auto polySize = reduced.polyFaceSizes[polyIdx];

    for (uint32_t i = 0; i < polySize; ++i) {
      // Remapped index is valid
      const auto remappedIdx = reduced.polyFaces[offset + i];
      ASSERT_LT(remappedIdx, static_cast<uint32_t>(reduced.vertices.size()));

      // Remapped vertex position matches the original vertex position
      EXPECT_TRUE(reduced.vertices[remappedIdx].isApprox(expectedPositions[expectedPosIdx], 1e-6f))
          << "Polygon " << polyIdx << " vertex " << i << ": expected position "
          << expectedPositions[expectedPosIdx].transpose() << " but got "
          << reduced.vertices[remappedIdx].transpose();
      ++expectedPosIdx;
    }
    offset += polySize;
  }
  EXPECT_EQ(offset, reduced.polyFaces.size());
}

// Test that reduceMeshByVertices preserves the correct polygons with correct vertex positions
TYPED_TEST(MeshReductionTest, ReduceMeshByVerticesPreservesPolygonData) {
  auto meshWithPolys = std::make_unique<Mesh>(*this->character.mesh);
  addTestPolyData(*meshWithPolys);
  ASSERT_GT(meshWithPolys->polyFaceSizes.size(), 0);

  // Keep the first half of vertices
  const size_t vertexCount = meshWithPolys->vertices.size();
  std::vector<bool> activeVertices(vertexCount, false);
  for (size_t i = 0; i < vertexCount / 2; ++i) {
    activeVertices[i] = true;
  }

  // Compute expected surviving polygons before reduction
  auto [expectedSizes, expectedPositions] =
      expectedSurvivingPolygons(*meshWithPolys, activeVertices);
  // Some polygons should be filtered out (not all vertices in low half)
  EXPECT_LT(expectedSizes.size(), meshWithPolys->polyFaceSizes.size());

  MeshT<float> reduced = reduceMeshByVertices(*meshWithPolys, activeVertices);
  verifyReducedPolygons(reduced, expectedSizes, expectedPositions);
}

// Test that reduceMeshByFaces preserves the correct polygons with correct vertex positions
TYPED_TEST(MeshReductionTest, ReduceMeshByFacesPreservesPolygonData) {
  auto meshWithPolys = std::make_unique<Mesh>(*this->character.mesh);
  addTestPolyData(*meshWithPolys);
  ASSERT_GT(meshWithPolys->polyFaceSizes.size(), 0);

  // Keep the first half of faces
  std::vector<bool> activeFaces(meshWithPolys->faces.size(), false);
  for (size_t i = 0; i < meshWithPolys->faces.size() / 2; ++i) {
    activeFaces[i] = true;
  }

  // reduceMeshByFaces converts face selection to vertex selection internally;
  // compute the resulting active vertices to predict which polygons survive
  const auto activeVertices = facesToVertices(*meshWithPolys, activeFaces);
  auto [expectedSizes, expectedPositions] =
      expectedSurvivingPolygons(*meshWithPolys, activeVertices);

  MeshT<float> reduced = reduceMeshByFaces(*meshWithPolys, activeFaces);
  verifyReducedPolygons(reduced, expectedSizes, expectedPositions);
}

// Test reduceMeshByPolys: verify selected polygons are preserved with correct geometry
TYPED_TEST(MeshReductionTest, ReduceMeshByPolys) {
  auto meshWithPolys = std::make_unique<Mesh>(*this->character.mesh);
  addTestPolyData(*meshWithPolys);
  const size_t polyCount = meshWithPolys->polyFaceSizes.size();
  ASSERT_GT(polyCount, 2u);

  // Select the first two polygons only
  std::vector<bool> activePolys(polyCount, false);
  activePolys[0] = true;
  activePolys[1] = true;

  // Collect original vertex positions for the two selected polygons
  std::vector<uint32_t> expectedSizes;
  std::vector<Vector3f> expectedPositions;
  uint32_t offset = 0;
  for (size_t polyIdx = 0; polyIdx < polyCount; ++polyIdx) {
    const auto polySize = meshWithPolys->polyFaceSizes[polyIdx];
    if (activePolys[polyIdx]) {
      expectedSizes.push_back(polySize);
      for (uint32_t i = 0; i < polySize; ++i) {
        expectedPositions.push_back(meshWithPolys->vertices[meshWithPolys->polyFaces[offset + i]]);
      }
    }
    offset += polySize;
  }

  MeshT<float> reduced = reduceMeshByPolys(*meshWithPolys, activePolys);

  // Should have exactly 2 polygons with correct geometry
  verifyReducedPolygons(reduced, expectedSizes, expectedPositions);

  // Should also have triangle faces and all vertices should be referenced
  EXPECT_GT(reduced.faces.size(), 0u);
  std::vector<bool> usedVertices(reduced.vertices.size(), false);
  for (const auto& face : reduced.faces) {
    usedVertices[face[0]] = true;
    usedVertices[face[1]] = true;
    usedVertices[face[2]] = true;
  }
  offset = 0;
  for (size_t polyIdx = 0; polyIdx < reduced.polyFaceSizes.size(); ++polyIdx) {
    for (uint32_t i = 0; i < reduced.polyFaceSizes[polyIdx]; ++i) {
      usedVertices[reduced.polyFaces[offset + i]] = true;
    }
    offset += reduced.polyFaceSizes[polyIdx];
  }
  for (size_t i = 0; i < usedVertices.size(); ++i) {
    EXPECT_TRUE(usedVertices[i]) << "Vertex " << i << " is unused in reduced mesh";
  }
}
