/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/aim_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

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
