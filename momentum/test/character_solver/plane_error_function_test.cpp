/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PlaneErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PlaneL2 Constraint Test");
    std::vector<PlaneDataT<T>> cl{
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0.1, 1).normalized(),
            uniform<T>(0, 1),
            2,
            kTestWeightValue),
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0.1, 1).normalized(),
            uniform<T>(0, 1),
            1,
            kTestWeightValue)};
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
          T, &errorFunction, parameters, character, Eps<T>(5e-2f, 1e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, HalfPlaneErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PlaneErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform, true);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PlaneL2 Constraint Test");
    std::vector<PlaneDataT<T>> cl{
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0.1, 1).normalized(),
            uniform<T>(0, 1),
            2,
            kTestWeightValue),
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0.1, 1).normalized(),
            uniform<T>(0, 1),
            1,
            kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-2f, 5e-6));

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(5e-2f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}
