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
#include "momentum/character/linear_skinning.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/gauss_newton_solver_qr.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/character_solver/vertex_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct BlendShapesTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(BlendShapesTest, Types);

// Verify that skinWithBlendShapes is equivalent to applying blend shapes then skinning.
TYPED_TEST(BlendShapesTest, Skinning) {
  using T = typename TestFixture::Type;

  const Character characterBlend = withTestBlendShapes(createTestCharacter());
  const ParameterTransformT<T> castedCharacterBlendParameterTransform =
      characterBlend.parameterTransform.cast<T>();
  ASSERT_TRUE(characterBlend.mesh);
  ASSERT_TRUE(characterBlend.blendShape);
  ASSERT_TRUE(characterBlend.skinWeights);

  const ModelParametersT<T> modelParams =
      0.25 * VectorX<T>::Random(castedCharacterBlendParameterTransform.numAllModelParameters());
  const SkeletonStateT<T> skelState(
      castedCharacterBlendParameterTransform.apply(modelParams), characterBlend.skeleton);

  const BlendWeightsT<T> blendWeights =
      extractBlendWeights(characterBlend.parameterTransform, modelParams);
  MeshT<T> restMesh = characterBlend.mesh->cast<T>();
  if constexpr (std::is_same_v<T, float>) {
    restMesh.vertices = characterBlend.blendShape->computeShape(blendWeights);
  } else if constexpr (std::is_same_v<T, double>) // TODO: Remove this conditional check once
                                                  // Character is templatized
  {
    const auto vertices = characterBlend.blendShape->computeShape(blendWeights);
    for (auto i = 0; i < vertices.size(); ++i) {
      restMesh.vertices[i] = vertices[i].template cast<T>();
    }
  }

  MeshT<T> posedMesh = characterBlend.mesh->cast<T>();
  applySSD(
      cast<T>(characterBlend.inverseBindPose),
      *characterBlend.skinWeights,
      restMesh,
      skelState,
      posedMesh);

  MeshT<T> posedMesh2 = characterBlend.mesh->cast<T>();
  skinWithBlendShapes(characterBlend, skelState, blendWeights, posedMesh2);

  ASSERT_EQ(posedMesh.vertices.size(), posedMesh2.vertices.size());
  for (size_t i = 0; i < posedMesh.vertices.size(); ++i) {
    const Eigen::Vector3<T> v1 = posedMesh.vertices[i];
    const Eigen::Vector3<T> v2 = posedMesh2.vertices[i];
    EXPECT_LE((v1 - v2).norm(), Eps<T>(1e-6f, 5e-7));
  }
}

// Verify that applying face expressions first and then shape blendshapes is equivalent
// to apply shape blendshapes first and then face expressions
TYPED_TEST(BlendShapesTest, FaceExpressions) {
  using T = typename TestFixture::Type;

  Character characterBlend = withTestBlendShapes(createTestCharacter());
  characterBlend = withTestFaceExpressionBlendShapes(characterBlend);
  const ParameterTransformT<T> castedCharacterBlendParameterTransform =
      characterBlend.parameterTransform.cast<T>();
  ASSERT_TRUE(characterBlend.mesh);
  ASSERT_TRUE(characterBlend.blendShape);
  ASSERT_TRUE(characterBlend.faceExpressionBlendShape);
  ASSERT_TRUE(characterBlend.skinWeights);

  const ModelParametersT<T> modelParams =
      0.25 * VectorX<T>::Random(castedCharacterBlendParameterTransform.numAllModelParameters());

  // Extract body shape blendshape weights
  const BlendWeightsT<T> blendWeights =
      extractBlendWeights(characterBlend.parameterTransform, modelParams);

  // Extract face blendshape weights
  const BlendWeightsT<T> faceExpressionBlendWeights =
      extractFaceExpressionBlendWeights(characterBlend.parameterTransform, modelParams);

  // Mesh to which we'll apply shape deltas first, then facial expression deltas
  MeshT<T> restMeshShapeFirst = characterBlend.mesh->cast<T>();
  // Mesh to which we'll apply face expression deltas first, then shape deltas
  MeshT<T> restMeshFaceExprFirst = characterBlend.mesh->cast<T>();

  // First apply shape blendshapes, then apply face expression blendshapes.
  characterBlend.blendShape->applyDeltas(blendWeights, restMeshShapeFirst.vertices);
  characterBlend.faceExpressionBlendShape->applyDeltas(
      faceExpressionBlendWeights, restMeshShapeFirst.vertices);

  // First apply face expression blendshapes, then apply shape blendshapes.
  characterBlend.faceExpressionBlendShape->applyDeltas(
      faceExpressionBlendWeights, restMeshFaceExprFirst.vertices);
  characterBlend.blendShape->applyDeltas(blendWeights, restMeshFaceExprFirst.vertices);

  SCOPED_TRACE("Checking face expression and shape blendshapes consistency");
  for (int i = 0; i < restMeshShapeFirst.vertices.size(); ++i) {
    EXPECT_LE(
        (restMeshShapeFirst.vertices[i] - restMeshFaceExprFirst.vertices[i]).norm(),
        Eps<T>(1e-3f, 1e-6));
  }
}

TYPED_TEST(BlendShapesTest, Fitting) {
  using T = typename TestFixture::Type;

  const Character characterOrig = createTestCharacter();

  const Character characterBlend = withTestBlendShapes(characterOrig);
  const ParameterTransformT<T> castedCharacterBlendParameterTransform =
      characterBlend.parameterTransform.cast<T>();
  ASSERT_TRUE(characterBlend.mesh);
  ASSERT_TRUE(characterBlend.blendShape);
  ASSERT_TRUE(characterBlend.skinWeights);

  const ModelParametersT<T> modelParamsTarget =
      0.5 * VectorX<T>::Random(characterBlend.parameterTransform.numAllModelParameters());
  // Make sure there's actually a valid blend shape in there:
  EXPECT_GT(
      extractBlendWeights(characterBlend.parameterTransform, modelParamsTarget).v.norm(), 0.05);

  MeshT<T> targetMesh = characterBlend.mesh->cast<T>();
  skinWithBlendShapes(
      characterBlend,
      SkeletonStateT<T>(
          castedCharacterBlendParameterTransform.apply(modelParamsTarget), characterBlend.skeleton),
      modelParamsTarget,
      targetMesh);

  // Use IK to fit the pose + blend shapes.
  SkeletonSolverFunctionT<T> solverFunction(
      &characterBlend.skeleton, &castedCharacterBlendParameterTransform);

  auto errorFunction = std::make_shared<VertexErrorFunctionT<T>>(characterBlend);
  for (size_t iVert = 0; iVert < targetMesh.vertices.size(); ++iVert) {
    errorFunction->addConstraint(iVert, 1.0, targetMesh.vertices[iVert], targetMesh.normals[iVert]);
  }
  solverFunction.addErrorFunction(errorFunction);

  // create solver
  GaussNewtonSolverQROptions options;
  options.maxIterations = 6;
  options.minIterations = 6;
  options.threshold = 1.0;
  options.regularization = 1e-7;
  options.doLineSearch = true;
  GaussNewtonSolverQRT<T> solver(options, &solverFunction);

  ModelParametersT<T> modelParamsOptimized =
      ModelParametersT<T>::Zero(characterBlend.parameterTransform.numAllModelParameters());
  solver.solve(modelParamsOptimized.v);

  MeshT<T> optimizedMesh = characterBlend.mesh->cast<T>();
  skinWithBlendShapes(
      characterBlend,
      SkeletonStateT<T>(
          castedCharacterBlendParameterTransform.apply(modelParamsOptimized),
          characterBlend.skeleton),
      modelParamsOptimized,
      optimizedMesh);

  ASSERT_EQ(targetMesh.vertices.size(), optimizedMesh.vertices.size());
  for (size_t i = 0; i < targetMesh.vertices.size(); ++i) {
    const Eigen::Vector3<T> v1 = targetMesh.vertices[i];
    const Eigen::Vector3<T> v2 = optimizedMesh.vertices[i];
    EXPECT_LE((v1 - v2).norm(), Eps<T>(1e-3f, 1e-3));
  }

  Character characterBaked =
      characterBlend.bakeBlendShape(modelParamsOptimized.template cast<float>());
  const ParameterTransformT<T> castedCharacterBakedParameterTransform =
      characterBaked.parameterTransform.cast<T>();
  // blend shape parameters stripped:
  EXPECT_EQ(
      characterBaked.parameterTransform.numAllModelParameters(),
      characterOrig.parameterTransform.numAllModelParameters());

  const ModelParametersT<T> modelParamsBaked =
      modelParamsOptimized.v.head(characterBaked.parameterTransform.numAllModelParameters());
  MeshT<T> bakedMesh = characterBlend.mesh->cast<T>();
  skinWithBlendShapes(
      characterBaked,
      SkeletonStateT<T>(
          castedCharacterBakedParameterTransform.apply(modelParamsBaked), characterBaked.skeleton),
      modelParamsBaked,
      optimizedMesh);
}
