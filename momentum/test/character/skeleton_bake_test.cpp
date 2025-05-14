/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape_skinning.h>
#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/mesh.h>
#include <momentum/math/random.h>
#include <momentum/test/character/character_helpers.h>

#include <gtest/gtest.h>

namespace {

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct BakeTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(BakeTest, Types);

// Helper: ensure at least one scaling & one blend-shape parameter != 0
template <typename T>
void activateSomeBakeableParams(const Character& c, ModelParametersT<T>& mp) {
  // Set the first scaling parameter we can find.
  for (size_t i = 0; i < mp.size(); ++i) {
    if (c.parameterTransform.getScalingParameters().test(i)) {
      mp[i] = T(0.07); // 7 % scale tweak
      break;
    }
  }
  // And the first blend-shape parameter.
  for (size_t i = 0; i < mp.size(); ++i) {
    if (c.parameterTransform.getBlendShapeParameters().test(i)) {
      mp[i] = T(0.25); // 25 % weight
      break;
    }
  }
}

TYPED_TEST(BakeTest, Geometry) {
  using T = typename TestFixture::Type;

  // Build a character with both scale & blend-shape capabilities
  const Character characterBlend = withTestBlendShapes(createTestCharacter());
  ASSERT_TRUE(characterBlend.mesh);
  ASSERT_TRUE(characterBlend.blendShape);
  ASSERT_TRUE(characterBlend.skinWeights);

  const ParameterTransformT<T> paramX = characterBlend.parameterTransform.cast<T>();

  // Random (but reproducible) model parameters
  ModelParametersT<T> modelParams = uniform<VectorX<T>>(paramX.numAllModelParameters(), 0, 1);

  activateSomeBakeableParams(characterBlend, modelParams);

  // Runtime path: skinning with blend shapes
  SkeletonStateT<T> skelState(paramX.apply(modelParams), characterBlend.skeleton);

  MeshT<T> posedMesh = characterBlend.mesh->cast<T>();
  skinWithBlendShapes(characterBlend, skelState, modelParams, posedMesh);

  // Bake-time path
  Character baked = characterBlend.bake(modelParams.template cast<float>());
  ASSERT_TRUE(baked.mesh); // sanity

  MeshT<T> bakedMesh = baked.mesh->cast<T>(); // convert verts to current precision

  // Compare vertex positions
  ASSERT_EQ(posedMesh.vertices.size(), bakedMesh.vertices.size());
  for (size_t i = 0; i < posedMesh.vertices.size(); ++i) {
    const Eigen::Vector3<T> vRun = posedMesh.vertices[i];
    const Eigen::Vector3<T> vBake = bakedMesh.vertices[i];
    EXPECT_LE((vRun - vBake).norm(), Eps<T>(1e-5f, 5e-6)) << "Vertex " << i << " mismatch";
  }
}

TYPED_TEST(BakeTest, ParameterStripping) {
  using T = typename TestFixture::Type;

  const Character characterBlend = withTestBlendShapes(createTestCharacter());
  const ParameterTransformT<T> paramX = characterBlend.parameterTransform.cast<T>();

  ModelParametersT<T> modelParams = VectorX<T>::Zero(paramX.numAllModelParameters());
  activateSomeBakeableParams(characterBlend, modelParams);

  Character baked = characterBlend.bake(modelParams.template cast<float>());

  // Blend-shape and scale parameter sets are gone
  EXPECT_EQ(baked.parameterTransform.getBlendShapeParameters().count(), 0);
  EXPECT_EQ(baked.parameterTransform.getScalingParameters().count(), 0);

  // Overall DOF count strictly smaller (unless the original had none to strip)
  EXPECT_LT(
      baked.parameterTransform.numAllModelParameters(),
      characterBlend.parameterTransform.numAllModelParameters());
}

} // namespace
