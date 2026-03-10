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
#include "momentum/character_solver/joint_to_joint_distance_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, JointToJointDistanceError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  JointToJointDistanceErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 2.5;

  {
    SCOPED_TRACE("JointToJoint Distance Constraint Test");

    ASSERT_GE(skeleton.joints.size(), 3);

    errorFunction.addConstraint(
        1,
        uniform<Vector3<T>>(-1, 1),
        2,
        uniform<Vector3<T>>(-1, 1),
        uniform<T>(0.5, 2.0),
        kTestWeightValue);

    errorFunction.addConstraint(
        0, Vector3<T>::Zero(), 2, Vector3<T>::UnitX(), uniform<T>(1.0, 3.0), kTestWeightValue);

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
          false);
    }

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1.0f, 1.0f);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 2e-5), Eps<T>(1e-6f, 1e-7));
    }
  }

  {
    SCOPED_TRACE("JointToJoint Distance Zero Error Test");

    const ModelParametersT<T> modelParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    JointToJointDistanceErrorFunctionT<T> errorFunctionZero(skeleton, character.parameterTransform);

    const auto& jointState1 = skelState.jointState[1];
    const auto& jointState2 = skelState.jointState[2];

    const auto offset1 = uniform<Vector3<T>>(-1, 1);
    const auto offset2 = uniform<Vector3<T>>(-1, 1);

    const Vector3<T> worldPos1 = jointState1.transform * offset1;
    const Vector3<T> worldPos2 = jointState2.transform * offset2;

    const T actualDistance = (worldPos1 - worldPos2).norm();

    errorFunctionZero.addConstraint(1, offset1, 2, offset2, actualDistance, T(1.0));

    const double error = errorFunctionZero.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
  }
}
