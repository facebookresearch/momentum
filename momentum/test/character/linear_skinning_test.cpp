/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/helpers/expect_throw.h"

#include <gtest/gtest.h>

namespace momentum {

class LinearSkinningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a test character with a skeleton
    character = createTestCharacter();
    skeletonState.jointState.resize(character.skeleton.joints.size());

    // Initialize joint states with identity transforms
    for (auto& jointState : skeletonState.jointState) {
      jointState.localRotation() = Quaternionf::Identity();
      jointState.localTranslation().setZero();
      jointState.localScale() = 1.0f;
      jointState.transform.rotation = Quaternionf::Identity();
      jointState.transform.translation.setZero();
      jointState.transform.scale = 1.0f;
      jointState.transformation = Eigen::Affine3f::Identity();
    }

    // Create inverse bind pose transformations
    inverseBindPose.resize(character.skeleton.joints.size());
    for (auto& i : inverseBindPose) {
      i = Eigen::Affine3f::Identity();
    }

    // Set up skin weights for testing
    setupSkinWeights();
  }

  void setupSkinWeights() {
    // Create skin weights for 3 vertices with 2 joints each
    skin.index.resize(3, kMaxSkinJoints);
    skin.weight.resize(3, kMaxSkinJoints);

    // Initialize all indices and weights to 0
    skin.index.setZero();
    skin.weight.setZero();

    // Vertex 0: 100% joint 0
    skin.index(0, 0) = 0;
    skin.weight(0, 0) = 1.0f;

    // Vertex 1: 50% joint 0, 50% joint 1
    skin.index(1, 0) = 0;
    skin.weight(1, 0) = 0.5f;
    skin.index(1, 1) = 1;
    skin.weight(1, 1) = 0.5f;

    // Vertex 2: 25% joint 0, 75% joint 1
    skin.index(2, 0) = 0;
    skin.weight(2, 0) = 0.25f;
    skin.index(2, 1) = 1;
    skin.weight(2, 1) = 0.75f;
  }

  Character character;
  SkeletonState skeletonState;
  TransformationList inverseBindPose;
  SkinWeights skin;
};

// Test applySSD with points (float version)
TEST_F(LinearSkinningTest, ApplySSDPointsFloat) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Apply identity transforms - points should remain the same
  auto result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(result[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(result[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));

  // Apply translation to joint 0
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));

  result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  ASSERT_EQ(result.size(), 3);
  // Vertex 0: 100% influenced by joint 0
  EXPECT_TRUE(result[0].isApprox(Vector3f(2.0f, 2.0f, 3.0f)));
  // Vertex 1: 50% influenced by joint 0, 50% by joint 1 (which is still at identity)
  // Check each component separately with appropriate tolerance
  EXPECT_NEAR(result[1].x(), 0.5f, 1e-3f);
  EXPECT_NEAR(result[1].y(), 2.0f, 1e-3f);
  EXPECT_NEAR(result[1].z(), 1.5f, 1e-3f);
  // Vertex 2: 25% influenced by joint 0, 75% by joint 1
  EXPECT_NEAR(result[2].x(), 0.25f, 1e-3f);
  EXPECT_NEAR(result[2].y(), 0.5f, 1e-3f);
  EXPECT_NEAR(result[2].z(), 1.75f, 1e-3f);

  // Apply translation to joint 1
  skeletonState.jointState[1].transformation.translate(Vector3f(2.0f, 0.0f, 1.0f));

  result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  ASSERT_EQ(result.size(), 3);
  // Vertex 0: 100% influenced by joint 0
  EXPECT_TRUE(result[0].isApprox(Vector3f(2.0f, 2.0f, 3.0f)));
  // Vertex 1: 50% influenced by joint 0, 50% by joint 1
  EXPECT_TRUE(result[1].isApprox(Vector3f(1.5f, 2.0f, 2.0f)));
  // Vertex 2: 25% influenced by joint 0, 75% by joint 1
  EXPECT_NEAR(result[2].x(), 1.75f, 1e-3f);
  EXPECT_NEAR(result[2].y(), 0.5f, 1e-3f);
  EXPECT_NEAR(result[2].z(), 2.5f, 1e-3f);
}

// Test applySSD with points (double version)
TEST_F(LinearSkinningTest, ApplySSDPointsDouble) {
  // Create test points
  std::vector<Vector3d> points = {
      Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 1.0, 0.0), Vector3d(0.0, 0.0, 1.0)};

  // Create double versions of the transforms
  TransformationListT<double> inverseBindPoseDouble;
  inverseBindPoseDouble.resize(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPoseDouble.size(); ++i) {
    inverseBindPoseDouble[i] = inverseBindPose[i].cast<double>();
  }

  // Create double version of skeleton state
  SkeletonStateT<double> skeletonStateDouble;
  skeletonStateDouble.jointState.resize(skeletonState.jointState.size());
  for (size_t i = 0; i < skeletonStateDouble.jointState.size(); ++i) {
    skeletonStateDouble.jointState[i].transformation =
        skeletonState.jointState[i].transformation.cast<double>();
  }

  // Apply identity transforms - points should remain the same
  auto result =
      applySSD(inverseBindPoseDouble, skin, gsl::span<const Vector3d>(points), skeletonStateDouble);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].isApprox(Vector3d(1.0, 0.0, 0.0)));
  EXPECT_TRUE(result[1].isApprox(Vector3d(0.0, 1.0, 0.0)));
  EXPECT_TRUE(result[2].isApprox(Vector3d(0.0, 0.0, 1.0)));

  // Apply translation to joint 0
  skeletonStateDouble.jointState[0].transformation.translate(Vector3d(1.0, 2.0, 3.0));

  result =
      applySSD(inverseBindPoseDouble, skin, gsl::span<const Vector3d>(points), skeletonStateDouble);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].isApprox(Vector3d(2.0, 2.0, 3.0)));
  EXPECT_NEAR(result[1].x(), 0.5, 1e-3);
  EXPECT_NEAR(result[1].y(), 2.0, 1e-3);
  EXPECT_NEAR(result[1].z(), 1.5, 1e-3);
  EXPECT_NEAR(result[2].x(), 0.25, 1e-3);
  EXPECT_NEAR(result[2].y(), 0.5, 1e-3);
  EXPECT_NEAR(result[2].z(), 1.75, 1e-3);
}

// Test applySSD with mesh (float version)
TEST_F(LinearSkinningTest, ApplySSDMeshFloat) {
  // Create test mesh
  MeshT<float> mesh;
  mesh.vertices = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.normals = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.faces = {Vector3i(0, 1, 2)};

  // Create output mesh with same structure
  MeshT<float> outputMesh = mesh;

  // Apply identity transforms - mesh should remain the same
  applySSD(inverseBindPose, skin, mesh, skeletonState, outputMesh);

  ASSERT_EQ(outputMesh.vertices.size(), 3);
  EXPECT_TRUE(outputMesh.vertices[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.vertices[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.vertices[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));

  EXPECT_TRUE(outputMesh.normals[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.normals[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.normals[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));

  // Apply rotation to joint 0 (90 degrees around Y)
  Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
  rotation.rotate(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY()));
  skeletonState.jointState[0].transformation = rotation;

  applySSD(inverseBindPose, skin, mesh, skeletonState, outputMesh);

  ASSERT_EQ(outputMesh.vertices.size(), 3);
  // Vertex 0: 100% influenced by joint 0, rotated 90 degrees around Y
  EXPECT_TRUE(outputMesh.vertices[0].isApprox(Vector3f(0.0f, 0.0f, -1.0f)));
  // Normals should also be rotated
  EXPECT_TRUE(outputMesh.normals[0].isApprox(Vector3f(0.0f, 0.0f, -1.0f)));
}

// Test applySSD with mesh and JointStateList (float version)
TEST_F(LinearSkinningTest, ApplySSDMeshJointStateFloat) {
  // Create test mesh
  MeshT<float> mesh;
  mesh.vertices = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.normals = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.faces = {Vector3i(0, 1, 2)};

  // Create output mesh with same structure
  MeshT<float> outputMesh = mesh;

  // Apply identity transforms - mesh should remain the same
  applySSD(inverseBindPose, skin, mesh, skeletonState.jointState, outputMesh);

  ASSERT_EQ(outputMesh.vertices.size(), 3);
  EXPECT_TRUE(outputMesh.vertices[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.vertices[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(outputMesh.vertices[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));

  // Apply translation to joint 1
  skeletonState.jointState[1].transformation.translate(Vector3f(2.0f, 3.0f, 4.0f));

  applySSD(inverseBindPose, skin, mesh, skeletonState.jointState, outputMesh);

  ASSERT_EQ(outputMesh.vertices.size(), 3);
  // Vertex 0: 100% influenced by joint 0 (still at identity)
  EXPECT_TRUE(outputMesh.vertices[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  // Vertex 1: 50% influenced by joint 0, 50% by joint 1
  EXPECT_TRUE(outputMesh.vertices[1].isApprox(Vector3f(1.0f, 2.5f, 2.0f)));
  // Vertex 2: 25% influenced by joint 0, 75% by joint 1
  EXPECT_NEAR(outputMesh.vertices[2].x(), 1.5f, 1e-2f);
  EXPECT_NEAR(outputMesh.vertices[2].y(), 2.25f, 1e-2f);
  EXPECT_NEAR(outputMesh.vertices[2].z(), 4.0f, 1e-2f);
}

// Test applySSD with mesh (double version)
TEST_F(LinearSkinningTest, ApplySSDMeshDouble) {
  // Create test mesh
  MeshT<double> mesh;
  mesh.vertices = {Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 1.0, 0.0), Vector3d(0.0, 0.0, 1.0)};
  mesh.normals = {Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 1.0, 0.0), Vector3d(0.0, 0.0, 1.0)};
  mesh.faces = {Vector3i(0, 1, 2)};

  // Create output mesh with same structure
  MeshT<double> outputMesh = mesh;

  // Create double versions of the transforms
  TransformationListT<double> inverseBindPoseDouble;
  inverseBindPoseDouble.resize(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPoseDouble.size(); ++i) {
    inverseBindPoseDouble[i] = inverseBindPose[i].cast<double>();
  }

  // Create double version of skeleton state
  SkeletonStateT<double> skeletonStateDouble;
  skeletonStateDouble.jointState.resize(skeletonState.jointState.size());
  for (size_t i = 0; i < skeletonStateDouble.jointState.size(); ++i) {
    skeletonStateDouble.jointState[i].transformation =
        skeletonState.jointState[i].transformation.cast<double>();
  }

  // Apply identity transforms - mesh should remain the same
  applySSD(inverseBindPoseDouble, skin, mesh, skeletonStateDouble, outputMesh);

  ASSERT_EQ(outputMesh.vertices.size(), 3);
  EXPECT_TRUE(outputMesh.vertices[0].isApprox(Vector3d(1.0, 0.0, 0.0)));
  EXPECT_TRUE(outputMesh.vertices[1].isApprox(Vector3d(0.0, 1.0, 0.0)));
  EXPECT_TRUE(outputMesh.vertices[2].isApprox(Vector3d(0.0, 0.0, 1.0)));
}

// Test getInverseSSDTransformation
TEST_F(LinearSkinningTest, GetInverseSSDTransformation) {
  // Apply identity transforms - should get identity inverse transform
  Affine3f inverseTransform = getInverseSSDTransformation(inverseBindPose, skin, skeletonState, 0);

  EXPECT_TRUE(inverseTransform.matrix().isApprox(Eigen::Matrix4f::Identity()));

  // Apply translation to joint 0
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));

  // For vertex 0 (100% influenced by joint 0), the inverse transform should be the inverse of joint
  // 0's transform
  inverseTransform = getInverseSSDTransformation(inverseBindPose, skin, skeletonState, 0);
  Affine3f expectedInverse = skeletonState.jointState[0].transformation.inverse();

  EXPECT_TRUE(inverseTransform.matrix().isApprox(expectedInverse.matrix()));

  // For vertex 1 (50% joint 0, 50% joint 1), the inverse is more complex
  // Apply translation to joint 1
  skeletonState.jointState[1].transformation.translate(Vector3f(2.0f, 0.0f, 1.0f));

  inverseTransform = getInverseSSDTransformation(inverseBindPose, skin, skeletonState, 1);

  // Test by applying the inverse transform to a transformed point
  // Create a single-point skin weight for this test
  SkinWeights singlePointSkin;
  singlePointSkin.index.resize(1, kMaxSkinJoints);
  singlePointSkin.weight.resize(1, kMaxSkinJoints);

  // Initialize all indices and weights to 0
  singlePointSkin.index.setZero();
  singlePointSkin.weight.setZero();

  // Copy only the non-zero weights from vertex 1
  singlePointSkin.index(0, 0) = skin.index(1, 0);
  singlePointSkin.weight(0, 0) = skin.weight(1, 0);
  singlePointSkin.index(0, 1) = skin.index(1, 1);
  singlePointSkin.weight(0, 1) = skin.weight(1, 1);

  Vector3f originalPoint(0.0f, 1.0f, 0.0f);
  Vector3f transformedPoint = applySSD(
      inverseBindPose,
      singlePointSkin,
      gsl::span<const Vector3f>(&originalPoint, 1),
      skeletonState)[0];
  Vector3f recoveredPoint = inverseTransform * transformedPoint;

  EXPECT_TRUE(recoveredPoint.isApprox(originalPoint));
}

// Test applyInverseSSD with points
TEST_F(LinearSkinningTest, ApplyInverseSSDPoints) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Apply identity transforms - points should remain the same
  auto result =
      applyInverseSSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(result[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(result[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));

  // Apply translation to joint 0
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));

  // First apply forward skinning to get transformed points
  auto transformedPoints =
      applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  // Then apply inverse skinning to get back original points
  result = applyInverseSSD(
      inverseBindPose, skin, gsl::span<const Vector3f>(transformedPoints), skeletonState);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(result[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(result[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));
}

// Test applyInverseSSD with mesh
TEST_F(LinearSkinningTest, ApplyInverseSSDMesh) {
  // Create test mesh
  Mesh mesh;
  mesh.vertices = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.normals = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.faces = {Vector3i(0, 1, 2)};

  // Apply translation to joint 0
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));

  // First apply forward skinning to get transformed points
  std::vector<Vector3f> transformedPoints =
      applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(mesh.vertices), skeletonState);

  // Then apply inverse skinning to mesh
  Mesh resultMesh = mesh;
  applyInverseSSD(
      inverseBindPose,
      skin,
      gsl::span<const Vector3f>(transformedPoints),
      skeletonState,
      resultMesh);

  ASSERT_EQ(resultMesh.vertices.size(), 3);
  EXPECT_TRUE(resultMesh.vertices[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(resultMesh.vertices[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(resultMesh.vertices[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));
}

// Test error cases for applySSD
TEST_F(LinearSkinningTest, ApplySSDErrors) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Test with mismatched joint state and inverse bind pose sizes
  SkeletonState smallState;
  smallState.jointState.resize(1); // Only one joint

  MOMENTUM_EXPECT_DEATH(
      applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), smallState), ".*");

  // Test with mismatched skin weights and points sizes
  SkinWeights smallSkin;
  smallSkin.index.resize(2, kMaxSkinJoints); // Only two vertices
  smallSkin.weight.resize(2, kMaxSkinJoints);

  MOMENTUM_EXPECT_DEATH(
      applySSD(inverseBindPose, smallSkin, gsl::span<const Vector3f>(points), skeletonState), ".*");

  // Test with invalid joint index in skin weights
  SkinWeights invalidSkin = skin;
  invalidSkin.index(0, 0) = 99; // Invalid joint index

  MOMENTUM_EXPECT_DEATH(
      applySSD(inverseBindPose, invalidSkin, gsl::span<const Vector3f>(points), skeletonState),
      ".*");
}

// Test error cases for applySSD with mesh
TEST_F(LinearSkinningTest, ApplySSDMeshErrors) {
  // Create test mesh
  MeshT<float> mesh;
  mesh.vertices = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.normals = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};
  mesh.faces = {Vector3i(0, 1, 2)};

  // Create output mesh with different number of vertices
  MeshT<float> smallOutputMesh;
  smallOutputMesh.vertices = {Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f)};
  smallOutputMesh.normals = {Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f)};
  smallOutputMesh.faces = {Vector3i(0, 1, 0)};

  MOMENTUM_EXPECT_DEATH(
      applySSD(inverseBindPose, skin, mesh, skeletonState, smallOutputMesh), ".*");

  // Create output mesh with different number of normals
  MeshT<float> badNormalsMesh = mesh;
  badNormalsMesh.normals.pop_back();

  MOMENTUM_EXPECT_DEATH(applySSD(inverseBindPose, skin, mesh, skeletonState, badNormalsMesh), ".*");

  // Create output mesh with different number of faces
  MeshT<float> badFacesMesh = mesh;
  badFacesMesh.faces.emplace_back(0, 1, 2);

  MOMENTUM_EXPECT_DEATH(applySSD(inverseBindPose, skin, mesh, skeletonState, badFacesMesh), ".*");
}

// Test error cases for getInverseSSDTransformation
TEST_F(LinearSkinningTest, GetInverseSSDTransformationErrors) {
  // Test with mismatched joint state and inverse bind pose sizes
  SkeletonState smallState;
  smallState.jointState.resize(1); // Only one joint

  MOMENTUM_EXPECT_DEATH(getInverseSSDTransformation(inverseBindPose, skin, smallState, 0), ".*");

  // Test with invalid vertex index
  MOMENTUM_EXPECT_DEATH(
      getInverseSSDTransformation(inverseBindPose, skin, skeletonState, 99), ".*");
}

// Test error cases for applyInverseSSD
TEST_F(LinearSkinningTest, ApplyInverseSSDErrors) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Test with mismatched joint state and inverse bind pose sizes
  SkeletonState smallState;
  smallState.jointState.resize(1); // Only one joint

  MOMENTUM_EXPECT_DEATH(
      applyInverseSSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), smallState), ".*");

  // Test with mismatched skin weights and points sizes
  SkinWeights smallSkin;
  smallSkin.index.resize(2, kMaxSkinJoints); // Only two vertices
  smallSkin.weight.resize(2, kMaxSkinJoints);

  MOMENTUM_EXPECT_DEATH(
      applyInverseSSD(inverseBindPose, smallSkin, gsl::span<const Vector3f>(points), skeletonState),
      ".*");

  // Test with invalid joint index in skin weights
  SkinWeights invalidSkin = skin;
  invalidSkin.index(0, 0) = 99; // Invalid joint index

  MOMENTUM_EXPECT_DEATH(
      applyInverseSSD(
          inverseBindPose, invalidSkin, gsl::span<const Vector3f>(points), skeletonState),
      ".*");
}

// Test error cases for applyInverseSSD with mesh
TEST_F(LinearSkinningTest, ApplyInverseSSDMeshErrors) {
  // Create test points and mesh
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  Mesh mesh;
  mesh.vertices = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Test with mismatched points and mesh vertices sizes
  Mesh smallMesh;
  smallMesh.vertices = {Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f)};

  MOMENTUM_EXPECT_DEATH(
      applyInverseSSD(
          inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState, smallMesh),
      ".*");
}

// Test with complex transformations
TEST_F(LinearSkinningTest, ComplexTransformations) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Apply complex transformations to joints
  // Joint 0: Rotation + Translation
  Eigen::Affine3f transform0 = Eigen::Affine3f::Identity();
  transform0.rotate(Eigen::AngleAxisf(M_PI / 4, Vector3f::UnitY())); // 45 degrees around Y
  transform0.translate(Vector3f(1.0f, 2.0f, 3.0f));
  skeletonState.jointState[0].transformation = transform0;

  // Joint 1: Rotation + Translation + Scale
  Eigen::Affine3f transform1 = Eigen::Affine3f::Identity();
  transform1.rotate(Eigen::AngleAxisf(M_PI / 6, Vector3f::UnitZ())); // 30 degrees around Z
  transform1.translate(Vector3f(2.0f, 1.0f, -1.0f));
  transform1.scale(1.5f);
  skeletonState.jointState[1].transformation = transform1;

  // Apply SSD
  auto result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  // Apply inverse SSD
  auto inverseResult =
      applyInverseSSD(inverseBindPose, skin, gsl::span<const Vector3f>(result), skeletonState);

  // Check that we get back the original points
  ASSERT_EQ(inverseResult.size(), 3);
  EXPECT_TRUE(inverseResult[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(inverseResult[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(inverseResult[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));
}

// Test with non-identity inverse bind pose
TEST_F(LinearSkinningTest, NonIdentityInverseBindPose) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Set up non-identity inverse bind pose
  inverseBindPose[0].translate(Vector3f(-0.5f, -1.0f, -0.5f));
  inverseBindPose[1].rotate(Eigen::AngleAxisf(M_PI / 3, Vector3f::UnitX())); // 60 degrees around X

  // Apply identity transforms to joints
  for (auto& jointState : skeletonState.jointState) {
    jointState.transformation = Eigen::Affine3f::Identity();
  }

  // Apply SSD
  auto result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  // Points should be transformed by inverse bind pose
  ASSERT_EQ(result.size(), 3);

  // Vertex 0: 100% influenced by joint 0's inverse bind pose
  // Check each component separately with appropriate tolerance
  EXPECT_NEAR(result[0].x(), 0.5f, 1e-2f);
  EXPECT_NEAR(result[0].y(), -1.0f, 1e-2f);
  EXPECT_NEAR(result[0].z(), -0.5f, 1e-2f);

  // Apply joint transformations
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));
  skeletonState.jointState[1].transformation.rotate(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY()));

  // Apply SSD again
  result = applySSD(inverseBindPose, skin, gsl::span<const Vector3f>(points), skeletonState);

  // Apply inverse SSD
  auto inverseResult =
      applyInverseSSD(inverseBindPose, skin, gsl::span<const Vector3f>(result), skeletonState);

  // Check that we get back the original points
  ASSERT_EQ(inverseResult.size(), 3);
  EXPECT_TRUE(inverseResult[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(inverseResult[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(inverseResult[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));
}

// Test with zero weights
TEST_F(LinearSkinningTest, ZeroWeights) {
  // Create test points
  std::vector<Vector3f> points = {
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f)};

  // Create skin weights with zero weights for some vertices
  SkinWeights zeroSkin = skin;
  zeroSkin.weight(1, 0) = 0.0f; // Set all weights for vertex 1 to zero
  zeroSkin.weight(1, 1) = 0.0f;

  // Apply transformations to joints
  skeletonState.jointState[0].transformation.translate(Vector3f(1.0f, 2.0f, 3.0f));
  skeletonState.jointState[1].transformation.translate(Vector3f(2.0f, 0.0f, 1.0f));

  // Apply SSD
  auto result =
      applySSD(inverseBindPose, zeroSkin, gsl::span<const Vector3f>(points), skeletonState);

  ASSERT_EQ(result.size(), 3);
  // Vertex 0: 100% influenced by joint 0
  EXPECT_TRUE(result[0].isApprox(Vector3f(2.0f, 2.0f, 3.0f)));
  // Vertex 1: 0% influenced by any joint, should remain at origin
  EXPECT_TRUE(result[1].isApprox(Vector3f(0.0f, 0.0f, 0.0f)));
  // Vertex 2: 25% influenced by joint 0, 75% by joint 1
  EXPECT_NEAR(result[2].x(), 1.75f, 1e-2f);
  EXPECT_NEAR(result[2].y(), 0.5f, 1e-2f);
  EXPECT_NEAR(result[2].z(), 2.5f, 1e-2f);
}

} // namespace momentum
