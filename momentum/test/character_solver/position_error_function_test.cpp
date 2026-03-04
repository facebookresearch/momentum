/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL2, uniform<T>(0.2, 10));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionL2 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(5e-2f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL1_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL1, uniform<T>(0.5, 2));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionL1 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorCauchy_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kCauchy, uniform<T>(0.5, 2));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionCauchy Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorWelsch_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kWelsch);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionWelsch Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(1e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorGeneral_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, uniform<T>(0.1, 10));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionGeneral Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-1f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}
