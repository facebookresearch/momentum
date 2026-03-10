/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/vertex_position_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexPositionErrorFunction_Serial_WithoutBlendShapes) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  const size_t nConstraints = 10;

  const Character character_orig = createTestCharacter();
  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;
  const ModelParametersT<T> modelParamsTarget = refParams +
      0.05 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const Skeleton& skeleton = character_orig.skeleton;
  const ParameterTransformT<T> transform = character_orig.parameterTransform.cast<T>();
  const momentum::SkeletonStateT<T> skelState(transform.apply(modelParamsTarget), skeleton);
  momentum::TransformationListT<T> ibp;
  for (const auto& js : character_orig.inverseBindPose) {
    ibp.push_back(js.cast<T>());
  }
  const auto mesh = character_orig.mesh->cast<T>();
  const auto& skin = *character_orig.skinWeights;
  momentum::MeshT<T> targetMesh = character_orig.mesh->cast<T>();
  applySSD(ibp, skin, mesh, skelState, targetMesh);

  VertexPositionErrorFunctionT<T> errorFunction(character_orig, character_orig.parameterTransform);
  errorFunction.setMaxThreads(0);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    const size_t index =
        uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
    errorFunction.addConstraint(
        VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-2)));
  }
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character_orig,
      Eps<T>(5e-2f, 1e-5),
      Eps<T>(1e-6f, 1e-14),
      true,
      false);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexPositionErrorFunction_Serial_WithBlendShapes) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  const size_t nConstraints = 10;

  const Character character_blend = withTestBlendShapes(createTestCharacter());
  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;
  const ModelParametersT<T> modelParamsTarget = refParams +
      0.05 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
  const Skeleton& skeleton = character_blend.skeleton;
  const ParameterTransformT<T> transform = character_blend.parameterTransform.cast<T>();
  const momentum::SkeletonStateT<T> skelState(transform.apply(modelParamsTarget), skeleton);
  momentum::TransformationListT<T> ibp;
  for (const auto& js : character_blend.inverseBindPose) {
    ibp.push_back(js.cast<T>());
  }
  const auto mesh = character_blend.mesh->cast<T>();
  const auto& skin = *character_blend.skinWeights;
  momentum::MeshT<T> targetMesh = character_blend.mesh->cast<T>();
  applySSD(ibp, skin, mesh, skelState, targetMesh);

  VertexPositionErrorFunctionT<T> errorFunction(
      character_blend, character_blend.parameterTransform);
  errorFunction.setMaxThreads(0);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    const size_t index =
        uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));
    errorFunction.addConstraint(
        VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-2)));
  }
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character_blend,
      Eps<T>(1e-2f, 1e-5),
      Eps<T>(1e-6f, 1e-14),
      true,
      false);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexPositionErrorFunction_Parallel_WithoutBlendShapes) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  const size_t nConstraints = 1000;

  const Character character_orig = createTestCharacter();
  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;
  const ModelParametersT<T> modelParamsTarget = refParams +
      0.05 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const Skeleton& skeleton = character_orig.skeleton;
  const ParameterTransformT<T> transform = character_orig.parameterTransform.cast<T>();
  const momentum::SkeletonStateT<T> skelState(transform.apply(modelParamsTarget), skeleton);
  momentum::TransformationListT<T> ibp;
  for (const auto& js : character_orig.inverseBindPose) {
    ibp.push_back(js.cast<T>());
  }
  const auto mesh = character_orig.mesh->cast<T>();
  const auto& skin = *character_orig.skinWeights;
  momentum::MeshT<T> targetMesh = character_orig.mesh->cast<T>();
  applySSD(ibp, skin, mesh, skelState, targetMesh);

  VertexPositionErrorFunctionT<T> errorFunction(character_orig, character_orig.parameterTransform);
  errorFunction.setMaxThreads(100000);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    const size_t index =
        uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
    errorFunction.addConstraint(
        VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-4)));
  }
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character_orig,
      Eps<T>(5e-2f, 1e-5),
      Eps<T>(1e-6f, 1e-15),
      true,
      false);
}

TYPED_TEST(
    Momentum_ErrorFunctionsTest,
    VertexPositionErrorFunctionFaceParameters_FaceExpressionOnly) {
  using T = typename TestFixture::Type;

  const size_t nConstraints = 10;

  const Character character_blend = withTestFaceExpressionBlendShapes(createTestCharacter());
  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);

  VertexPositionErrorFunctionT<T> errorFunction(
      character_blend, character_blend.parameterTransform);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    errorFunction.addConstraint(
        VertexPositionDataT<T>(
            static_cast<size_t>(
                uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size()) - 1)),
            uniform<Vector3<T>>(0, 1),
            uniform<T>(0, 1)));
  }

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character_blend,
      Eps<T>(5e-2f, 1e-5),
      Eps<T>(1e-6f, 5e-16),
      true,
      false);
}

TYPED_TEST(
    Momentum_ErrorFunctionsTest,
    VertexPositionErrorFunctionFaceParameters_FaceExpressionAndShape) {
  using T = typename TestFixture::Type;

  const size_t nConstraints = 10;

  const Character character_blend =
      withTestBlendShapes(withTestFaceExpressionBlendShapes(createTestCharacter()));
  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);

  VertexPositionErrorFunctionT<T> errorFunction(
      character_blend, character_blend.parameterTransform);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    errorFunction.addConstraint(
        VertexPositionDataT<T>(
            static_cast<size_t>(
                uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size()) - 1)),
            uniform<Vector3<T>>(0, 1),
            uniform<T>(0, 1)));
  }

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character_blend,
      Eps<T>(1e-2f, 1e-5),
      Eps<T>(1e-6f, 5e-16),
      true,
      false);
}
