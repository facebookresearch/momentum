/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/character/marker.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skin_weights.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/usd/usd_io.h>
#include <momentum/math/mesh.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/character/character_helpers_gtest.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <algorithm>
#include <filesystem>
#include <map>
#include <vector>

using namespace momentum;

namespace {

/// Compares two skeletons including joint names, parents, translationOffset, and preRotation.
void compareSkeletons(const Skeleton& a, const Skeleton& b) {
  ASSERT_EQ(a.joints.size(), b.joints.size());
  for (size_t i = 0; i < a.joints.size(); ++i) {
    EXPECT_EQ(a.joints[i].name, b.joints[i].name) << "Joint name mismatch at index " << i;
    EXPECT_EQ(a.joints[i].parent, b.joints[i].parent)
        << "Parent mismatch at joint " << a.joints[i].name;
    EXPECT_TRUE(a.joints[i].translationOffset.isApprox(b.joints[i].translationOffset, 1e-4f))
        << "translationOffset mismatch at joint " << a.joints[i].name;
    EXPECT_TRUE(a.joints[i].preRotation.toRotationMatrix().isApprox(
        b.joints[i].preRotation.toRotationMatrix(), 1e-4f))
        << "preRotation mismatch at joint " << a.joints[i].name;
  }
}

/// Compares mesh data across formats. Skips polyFaces (GLTF doesn't populate them).
void compareMeshesCrossFormat(const Mesh_u& a, const Mesh_u& b) {
  ASSERT_TRUE(a && b) << "One or both meshes are null";

  ASSERT_EQ(a->vertices.size(), b->vertices.size());
  for (size_t i = 0; i < a->vertices.size(); ++i) {
    EXPECT_TRUE(a->vertices[i].isApprox(b->vertices[i], 1e-4f)) << "Vertex mismatch at index " << i;
  }

  ASSERT_EQ(a->normals.size(), b->normals.size());
  for (size_t i = 0; i < a->normals.size(); ++i) {
    EXPECT_TRUE(a->normals[i].isApprox(b->normals[i], 0.01f)) << "Normal mismatch at index " << i;
  }

  ASSERT_EQ(a->faces.size(), b->faces.size());
  EXPECT_EQ(a->faces, b->faces);

  ASSERT_EQ(a->colors.size(), b->colors.size());
  for (size_t i = 0; i < a->colors.size(); ++i) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(a->colors[i][c], b->colors[i][c], 1)
          << "Color mismatch at vertex " << i << ", channel " << c;
    }
  }

  ASSERT_EQ(a->texcoords.size(), b->texcoords.size());
  for (size_t i = 0; i < a->texcoords.size(); ++i) {
    EXPECT_TRUE(a->texcoords[i].isApprox(b->texcoords[i], 1e-4f))
        << "Texcoord mismatch at index " << i;
  }

  EXPECT_EQ(a->texcoord_faces, b->texcoord_faces);

  ASSERT_EQ(a->confidence.size(), b->confidence.size());
  for (size_t i = 0; i < a->confidence.size(); ++i) {
    EXPECT_NEAR(a->confidence[i], b->confidence[i], 1e-4f) << "Confidence mismatch at index " << i;
  }
  // NOTE: polyFaces/polyFaceSizes/polyTexcoordFaces are skipped — GLTF doesn't populate them.
}

/// Compares skin weights using order-independent per-vertex comparison of nonzero influences.
void compareSkinWeightsData(const SkinWeights& a, const SkinWeights& b) {
  ASSERT_EQ(a.index.rows(), b.index.rows());
  ASSERT_EQ(a.weight.rows(), b.weight.rows());
  for (Eigen::Index v = 0; v < a.index.rows(); ++v) {
    // Collect nonzero influences sorted by joint index
    std::map<uint32_t, float> aInfluences, bInfluences;
    for (Eigen::Index j = 0; j < a.weight.cols(); ++j) {
      if (a.weight(v, j) > 1e-6f) {
        aInfluences[a.index(v, j)] = a.weight(v, j);
      }
    }
    for (Eigen::Index j = 0; j < b.weight.cols(); ++j) {
      if (b.weight(v, j) > 1e-6f) {
        bInfluences[b.index(v, j)] = b.weight(v, j);
      }
    }
    ASSERT_EQ(aInfluences.size(), bInfluences.size())
        << "Different number of nonzero skin weight influences at vertex " << v;
    for (const auto& [jointIdx, weight] : aInfluences) {
      auto it = bInfluences.find(jointIdx);
      ASSERT_NE(it, bInfluences.end())
          << "Joint " << jointIdx << " influence missing at vertex " << v;
      EXPECT_NEAR(weight, it->second, 1e-4f)
          << "Weight mismatch at vertex " << v << ", joint " << jointIdx;
    }
  }
}

} // namespace

class IoUsdGltfRoundtripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    testCharacter = createTestCharacter();
  }

  Character testCharacter;
};

TEST_F(IoUsdGltfRoundtripTest, BasicCharacter_GltfToUsd) {
  // Save to GLTF, load, save to USD, load back, compare
  auto gltfFile = temporaryFile("roundtrip_gltf_to_usd", "glb");
  saveGltfCharacter(gltfFile.path(), testCharacter);
  auto gltfChar = loadGltfCharacter(gltfFile.path());

  auto usdFile = temporaryFile("roundtrip_gltf_to_usd", "usda");
  saveUsd(usdFile.path(), gltfChar);
  auto usdChar = loadUsdCharacter(usdFile.path());

  compareSkeletons(usdChar.skeleton, gltfChar.skeleton);
  compareMeshesCrossFormat(usdChar.mesh, gltfChar.mesh);

  ASSERT_TRUE(usdChar.skinWeights);
  ASSERT_TRUE(gltfChar.skinWeights);
  compareSkinWeightsData(*usdChar.skinWeights, *gltfChar.skinWeights);
}

TEST_F(IoUsdGltfRoundtripTest, BasicCharacter_UsdToGltf) {
  // Save to USD, load, save to GLTF, load back, compare
  auto usdFile = temporaryFile("roundtrip_usd_to_gltf", "usda");
  saveUsd(usdFile.path(), testCharacter);
  auto usdChar = loadUsdCharacter(usdFile.path());

  auto gltfFile = temporaryFile("roundtrip_usd_to_gltf", "glb");
  saveGltfCharacter(gltfFile.path(), usdChar);
  auto gltfChar = loadGltfCharacter(gltfFile.path());

  compareSkeletons(gltfChar.skeleton, usdChar.skeleton);
  compareMeshesCrossFormat(gltfChar.mesh, usdChar.mesh);

  ASSERT_TRUE(gltfChar.skinWeights);
  ASSERT_TRUE(usdChar.skinWeights);
  compareSkinWeightsData(*gltfChar.skinWeights, *usdChar.skinWeights);
}

TEST_F(IoUsdGltfRoundtripTest, BasicCharacter_IndependentSave) {
  // Save the same character to both formats independently and compare loaded results
  auto usdFile = temporaryFile("roundtrip_independent", "usda");
  saveUsd(usdFile.path(), testCharacter);

  auto gltfFile = temporaryFile("roundtrip_independent", "glb");
  saveGltfCharacter(gltfFile.path(), testCharacter);

  auto usdChar = loadUsdCharacter(usdFile.path());
  auto gltfChar = loadGltfCharacter(gltfFile.path());

  compareSkeletons(gltfChar.skeleton, usdChar.skeleton);
  compareMeshesCrossFormat(gltfChar.mesh, usdChar.mesh);

  ASSERT_TRUE(gltfChar.skinWeights);
  ASSERT_TRUE(usdChar.skinWeights);
  compareSkinWeightsData(*gltfChar.skinWeights, *usdChar.skinWeights);
}

TEST_F(IoUsdGltfRoundtripTest, CharacterWithSkeletonStates) {
  const auto& skeleton = testCharacter.skeleton;
  const auto numJoints = skeleton.joints.size();
  const float fps = 30.0f;

  // Create skeleton states with varying translations and rotations
  std::vector<SkeletonState> states;
  for (int frame = 0; frame < 5; ++frame) {
    JointParameters params = JointParameters::Zero(numJoints * kParametersPerJoint);
    params[0] = static_cast<float>(frame) * 0.1f; // root tx
    params[1] = static_cast<float>(frame) * 0.05f; // root ty
    params[kParametersPerJoint + 3] = static_cast<float>(frame) * 0.2f; // joint1 rx
    states.emplace_back(params, skeleton, false);
  }

  // Save to USD with skeleton states
  auto usdFile = temporaryFile("roundtrip_skelstates", "usda");
  saveUsdCharacter(usdFile.path(), testCharacter, fps, states);
  auto [usdChar, usdStates, usdTimes] = loadUsdCharacterWithSkeletonStates(usdFile.path());

  // Save to GLTF with skeleton states
  auto gltfFile = temporaryFile("roundtrip_skelstates", "glb");
  saveGltfCharacter(gltfFile.path(), testCharacter, fps, states);
  auto [gltfChar, gltfStates, gltfTimes] = loadCharacterWithSkeletonStates(gltfFile.path());

  ASSERT_EQ(usdStates.size(), states.size());
  ASSERT_EQ(gltfStates.size(), states.size());

  // Compare global transforms between USD and GLTF loaded states
  for (size_t f = 0; f < states.size(); ++f) {
    ASSERT_EQ(usdStates[f].jointState.size(), numJoints);
    ASSERT_EQ(gltfStates[f].jointState.size(), numJoints);
    for (size_t j = 0; j < numJoints; ++j) {
      const auto& usdTransform = usdStates[f].jointState[j].transform;
      const auto& gltfTransform = gltfStates[f].jointState[j].transform;

      EXPECT_TRUE(usdTransform.translation.isApprox(gltfTransform.translation, 0.01f))
          << "Translation mismatch at frame " << f << ", joint " << j << " ("
          << skeleton.joints[j].name << ")";

      EXPECT_TRUE(usdTransform.rotation.toRotationMatrix().isApprox(
          gltfTransform.rotation.toRotationMatrix(), 0.01f))
          << "Rotation mismatch at frame " << f << ", joint " << j << " ("
          << skeleton.joints[j].name << ")";

      EXPECT_NEAR(usdTransform.scale, gltfTransform.scale, 0.01f)
          << "Scale mismatch at frame " << f << ", joint " << j << " (" << skeleton.joints[j].name
          << ")";
    }
  }
}

TEST_F(IoUsdGltfRoundtripTest, BlendShapes) {
  auto character = withTestBlendShapes(testCharacter);
  ASSERT_TRUE(character.blendShape);
  const auto& origVectors = character.blendShape->getShapeVectors();
  const auto& origNames = character.blendShape->getShapeNames();

  // Save to USD
  auto usdFile = temporaryFile("roundtrip_blendshapes", "usda");
  saveUsd(usdFile.path(), character);
  auto usdChar = loadUsdCharacter(usdFile.path());

  // Save to GLTF
  auto gltfFile = temporaryFile("roundtrip_blendshapes", "glb");
  saveGltfCharacter(gltfFile.path(), character);
  auto gltfChar = loadGltfCharacter(gltfFile.path());

  // Both should have blend shapes
  ASSERT_TRUE(usdChar.blendShape);
  ASSERT_TRUE(gltfChar.blendShape);
  EXPECT_EQ(usdChar.blendShape->shapeSize(), origVectors.cols());
  EXPECT_EQ(gltfChar.blendShape->shapeSize(), origVectors.cols());
  EXPECT_EQ(usdChar.blendShape->modelSize(), character.blendShape->modelSize());
  EXPECT_EQ(gltfChar.blendShape->modelSize(), character.blendShape->modelSize());

  // Compare shape vectors against original
  const auto& usdVectors = usdChar.blendShape->getShapeVectors();
  const auto& gltfVectors = gltfChar.blendShape->getShapeVectors();

  ASSERT_EQ(usdVectors.rows(), origVectors.rows());
  ASSERT_EQ(usdVectors.cols(), origVectors.cols());
  EXPECT_TRUE(usdVectors.isApprox(origVectors, 1e-3f)) << "USD shape vectors differ from original";

  ASSERT_EQ(gltfVectors.rows(), origVectors.rows());
  ASSERT_EQ(gltfVectors.cols(), origVectors.cols());
  EXPECT_TRUE(gltfVectors.isApprox(origVectors, 1e-3f))
      << "GLTF shape vectors differ from original";

  // Compare shape vectors between formats
  EXPECT_TRUE(usdVectors.isApprox(gltfVectors, 1e-3f))
      << "USD and GLTF shape vectors differ from each other";

  // Compare shape names
  const auto& usdNames = usdChar.blendShape->getShapeNames();
  const auto& gltfNames = gltfChar.blendShape->getShapeNames();
  ASSERT_EQ(usdNames.size(), origNames.size());
  ASSERT_EQ(gltfNames.size(), origNames.size());
  for (size_t i = 0; i < origNames.size(); ++i) {
    EXPECT_EQ(usdNames[i], origNames[i]) << "USD shape name mismatch at index " << i;
    EXPECT_EQ(gltfNames[i], origNames[i]) << "GLTF shape name mismatch at index " << i;
  }
}

TEST_F(IoUsdGltfRoundtripTest, MarkerSequence_SparseData) {
  const float testFps = 60.0f;
  const size_t numFrames = 10;

  std::vector<std::vector<Marker>> originalSequence(numFrames);

  // Marker1: Present in even frames only
  for (size_t frame : {0, 2, 4, 6, 8}) {
    Marker m;
    m.name = "marker1";
    m.pos = Eigen::Vector3d(frame * 1.0, frame * 2.0, frame * 3.0);
    m.occluded = false;
    originalSequence[frame].push_back(m);
  }

  // Marker2: Present in odd frames only
  for (size_t frame : {1, 3, 5, 7, 9}) {
    Marker m;
    m.name = "marker2";
    m.pos = Eigen::Vector3d(frame * 10.0, frame * 20.0, frame * 30.0);
    m.occluded = false;
    originalSequence[frame].push_back(m);
  }

  // Marker3: Present in all frames
  for (size_t frame = 0; frame < numFrames; ++frame) {
    Marker m;
    m.name = "marker3";
    m.pos = Eigen::Vector3d(frame * 100.0, frame * 200.0, frame * 300.0);
    m.occluded = false;
    originalSequence[frame].push_back(m);
  }

  // Save to USD
  auto usdFile = temporaryFile("roundtrip_markers", "usda");
  saveUsdCharacterWithMotion(usdFile.path(), testCharacter, testFps, {}, {}, originalSequence);

  // Save to GLTF
  auto gltfFile = temporaryFile("roundtrip_markers", "glb");
  saveGltfCharacter(gltfFile.path(), testCharacter, testFps, {}, {}, originalSequence);

  // Load marker sequences back
  auto usdLoaded = loadUsdMarkerSequence(usdFile.path());
  auto gltfLoaded = loadMarkerSequence(gltfFile.path());

  EXPECT_GT(usdLoaded.fps, 0.0f);
  EXPECT_GT(gltfLoaded.fps, 0.0f);
  ASSERT_EQ(usdLoaded.frames.size(), numFrames);
  ASSERT_EQ(gltfLoaded.frames.size(), numFrames);

  // Compare marker positions against original for both formats
  for (size_t f = 0; f < numFrames; ++f) {
    std::map<std::string, const Marker*> usdMarkers, gltfMarkers;
    for (const auto& m : usdLoaded.frames[f]) {
      if (!m.occluded) {
        usdMarkers[m.name] = &m;
      }
    }
    for (const auto& m : gltfLoaded.frames[f]) {
      if (!m.occluded) {
        gltfMarkers[m.name] = &m;
      }
    }

    for (const auto& orig : originalSequence[f]) {
      if (orig.occluded) {
        continue;
      }

      // Check USD
      auto usdIt = usdMarkers.find(orig.name);
      ASSERT_NE(usdIt, usdMarkers.end())
          << "USD: marker " << orig.name << " missing at frame " << f;
      EXPECT_NEAR(usdIt->second->pos.x(), orig.pos.x(), 1e-4)
          << "USD: " << orig.name << " X mismatch at frame " << f;
      EXPECT_NEAR(usdIt->second->pos.y(), orig.pos.y(), 1e-4)
          << "USD: " << orig.name << " Y mismatch at frame " << f;
      EXPECT_NEAR(usdIt->second->pos.z(), orig.pos.z(), 1e-4)
          << "USD: " << orig.name << " Z mismatch at frame " << f;

      // Check GLTF
      auto gltfIt = gltfMarkers.find(orig.name);
      ASSERT_NE(gltfIt, gltfMarkers.end())
          << "GLTF: marker " << orig.name << " missing at frame " << f;
      EXPECT_NEAR(gltfIt->second->pos.x(), orig.pos.x(), 1e-4)
          << "GLTF: " << orig.name << " X mismatch at frame " << f;
      EXPECT_NEAR(gltfIt->second->pos.y(), orig.pos.y(), 1e-4)
          << "GLTF: " << orig.name << " Y mismatch at frame " << f;
      EXPECT_NEAR(gltfIt->second->pos.z(), orig.pos.z(), 1e-4)
          << "GLTF: " << orig.name << " Z mismatch at frame " << f;
    }
  }

  // Verify sparse patterns via non-occluded marker counts
  auto countNonOccludedMarkers = [](const MarkerSequence& seq) {
    std::map<std::string, size_t> counts;
    for (const auto& frame : seq.frames) {
      for (const auto& marker : frame) {
        if (!marker.occluded) {
          counts[marker.name]++;
        }
      }
    }
    return counts;
  };

  auto usdCounts = countNonOccludedMarkers(usdLoaded);
  auto gltfCounts = countNonOccludedMarkers(gltfLoaded);

  EXPECT_EQ(usdCounts["marker1"], 5) << "USD: marker1 should appear in 5 frames";
  EXPECT_EQ(usdCounts["marker2"], 5) << "USD: marker2 should appear in 5 frames";
  EXPECT_EQ(usdCounts["marker3"], 10) << "USD: marker3 should appear in all 10 frames";
  EXPECT_EQ(gltfCounts["marker1"], 5) << "GLTF: marker1 should appear in 5 frames";
  EXPECT_EQ(gltfCounts["marker2"], 5) << "GLTF: marker2 should appear in 5 frames";
  EXPECT_EQ(gltfCounts["marker3"], 10) << "GLTF: marker3 should appear in all 10 frames";
}

TEST_F(IoUsdGltfRoundtripTest, NonTrivialPreRotation) {
  // Create character with non-identity preRotations to verify they survive cross-format roundtrip
  auto character = createTestCharacter();
  character.skeleton.joints[0].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitX()));
  character.skeleton.joints[1].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(-0.5f, Eigen::Vector3f::UnitZ()));
  character.skeleton.joints[2].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(0.7f, Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized()));

  // Save to USD, load back
  auto usdFile = temporaryFile("roundtrip_prerot", "usda");
  saveUsd(usdFile.path(), character);
  auto usdChar = loadUsdCharacter(usdFile.path());

  // Save to GLTF, load back
  auto gltfFile = temporaryFile("roundtrip_prerot", "glb");
  saveGltfCharacter(gltfFile.path(), character);
  auto gltfChar = loadGltfCharacter(gltfFile.path());

  // Compare both against original
  compareSkeletons(usdChar.skeleton, character.skeleton);
  compareSkeletons(gltfChar.skeleton, character.skeleton);

  // Compare between formats
  compareSkeletons(usdChar.skeleton, gltfChar.skeleton);
}
