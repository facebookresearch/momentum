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
#include "momentum/character_solver/aim_error_function.h"
#include "momentum/character_solver/fixed_axis_error_function.h"
#include "momentum/character_solver/normal_error_function.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, OrientationErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  OrientationErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;

  {
    SCOPED_TRACE("Orientation Constraint Test");
    std::vector<OrientationDataT<T>> cl{
        OrientationDataT<T>(uniformQuaternion<T>(), uniformQuaternion<T>(), 2, kTestWeightValue),
        OrientationDataT<T>(uniformQuaternion<T>(), uniformQuaternion<T>(), 1, kTestWeightValue)};
    errorFunction.setConstraints(std::move(cl));

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(0.03f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(0.05f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisDiffErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  Random<> rng(10455);

  // create constraints
  FixedAxisDiffErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("FixedAxisDiffL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(
            rng.uniform<Vector3<T>>(-1, 1), rng.uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        FixedAxisDataT<T>(
            rng.uniform<Vector3<T>>(-1, 1), rng.uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          rng.uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(9e-2f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisCosErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;

  Random<> rng(43926);

  // create constraints
  FixedAxisCosErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("FixedAxisCosL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(
            rng.uniform<Vector3<T>>(-1, 1), rng.uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        FixedAxisDataT<T>(
            rng.uniform<Vector3<T>>(-1, 1), rng.uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters()),
        character,
        Eps<T>(2e-2f, 1e-4));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          rng.uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(5e-2f, 1e-4), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisAngleErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  Random<> rng(37482);

  // create constraints
  FixedAxisAngleErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("FixedAxisAngleL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(
            rng.uniform<Vector3<T>>(-1, 1), rng.uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        // corner case when the angle is close to zero
        FixedAxisDataT<T>(
            Vector3<T>::UnitY(), Vector3<T>(1e-16, 1 + 1e-16, 1e-16), 1, kTestWeightValue),
        FixedAxisDataT<T>(Vector3<T>::UnitX(), Vector3<T>::UnitX(), 2, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          5e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          2e-2f,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          rng.uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(2e-1f, 5e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, NormalError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();

  // create constraints
  NormalErrorFunctionT<T> errorFunction(character.skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("Normal Constraint Test");
    std::vector<NormalDataT<T>> cl{
        NormalDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            1,
            kTestWeightValue),
        NormalDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            2,
            kTestWeightValue)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters()),
          character,
          2e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters()),
          character,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 2e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, AimDistError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  AimDistErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("AimDist Constraint Test");
    std::vector<AimDataT<T>> cl{
        AimDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            1,
            kTestWeightValue),
        AimDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            2,
            kTestWeightValue)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          1e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 2e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, AimDirError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  AimDirErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("AimDir Constraint Test");
    std::vector<AimDataT<T>> cl{
        AimDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            1,
            kTestWeightValue),
        AimDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            2,
            kTestWeightValue)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          5e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 3e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}
