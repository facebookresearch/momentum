/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/pose_prior_error_function.h"
#include "momentum/character_solver/state_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, StateError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  StateErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  {
    SCOPED_TRACE("State Test");
    SkeletonStateT<T> reference(transform.bindPose(), skeleton);
    errorFunction.setTargetState(reference);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(2e-5f, 1e-3));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(1e-2f, 5e-6));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, StateErrorLogMap_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  StateErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, RotationErrorType::QuaternionLogMap);
  {
    SCOPED_TRACE("State Test");
    SkeletonStateT<T> reference(transform.bindPose(), skeleton);
    errorFunction.setTargetState(reference);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(2e-5f, 1e-3));
    EXPECT_EQ(errorFunction.getJacobianSize(), 6 * character.skeleton.joints.size());
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(1e-2f, 5e-6));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, ModelParametersError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  ModelParametersErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  {
    SCOPED_TRACE("Motion Test");
    SkeletonStateT<T> reference(transform.bindPose(), skeleton);
    VectorX<T> weights = VectorX<T>::Ones(transform.numAllModelParameters());
    weights(0) = 4.0;
    weights(1) = 5.0;
    weights(2) = 0.0;
    errorFunction.setTargetParameters(
        ModelParametersT<T>::Zero(transform.numAllModelParameters()), weights);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(5e-3f, 1e-10));
    }
  }
}

// Show an example of regularizing the blend weights:
TYPED_TEST(Momentum_ErrorFunctionsTest, ModelParametersError_RegularizeBlendWeights) {
  using T = typename TestFixture::Type;

  const Character character = withTestBlendShapes(createTestCharacter());
  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);
  ModelParametersErrorFunctionT<T> errorFunction(
      character, character.parameterTransform.getBlendShapeParameters());
  EXPECT_GT(
      errorFunction.getError(
          modelParams,
          SkeletonStateT<T>(
              character.parameterTransform.cast<T>().apply(modelParams), character.skeleton),
          MeshStateT<T>()),
      0);
  TEST_GRADIENT_AND_JACOBIAN(
      T, &errorFunction, modelParams, character, Eps<T>(1e-3f, 1e-10), Eps<T>(1e-7f, 1e-7), true);

  ModelParametersErrorFunctionT<T> errorFunction2(
      character, character.parameterTransform.getScalingParameters());
  TEST_GRADIENT_AND_JACOBIAN(
      T, &errorFunction2, modelParams, character, Eps<T>(1e-3f, 5e-12), Eps<T>(1e-6f, 1e-7), true);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PosePriorError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PosePriorErrorFunctionT<T> errorFunction(
      character.skeleton, character.parameterTransform, createDefaultPosePrior<T>());

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      ModelParametersT<T>::Zero(transform.numAllModelParameters()),
      character,
      Eps<T>(1e-1f, 1e-10),
      Eps<T>(1e-5f, 5e-6));

  for (size_t i = 0; i < 3; i++) {
    ModelParametersT<T> parameters = uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
    TEST_GRADIENT_AND_JACOBIAN(
        T, &errorFunction, parameters, character, Eps<T>(5e-1f, 1e-9), Eps<T>(5e-5f, 5e-6));
  }
}
