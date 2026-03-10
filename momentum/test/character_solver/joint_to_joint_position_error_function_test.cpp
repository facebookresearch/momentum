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
#include "momentum/character_solver/joint_to_joint_position_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, JointToJointPositionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  JointToJointPositionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 2.5;

  {
    SCOPED_TRACE("JointToJoint Position Constraint Test");

    ASSERT_GE(skeleton.joints.size(), 3);

    // Add constraints with random offsets and targets
    errorFunction.addConstraint(
        1, // sourceJoint
        uniform<Vector3<T>>(-1, 1), // sourceOffset
        2, // referenceJoint
        uniform<Vector3<T>>(-1, 1), // referenceOffset
        uniform<Vector3<T>>(-1, 1), // target (in reference frame)
        kTestWeightValue);

    errorFunction.addConstraint(
        0,
        Vector3<T>::Zero(),
        2,
        Vector3<T>::UnitX(),
        uniform<Vector3<T>>(-0.5, 0.5),
        kTestWeightValue);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1.0f, 1.0f);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(6e-2f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }

  {
    SCOPED_TRACE("JointToJoint Position Zero Error Test");

    // Test that error is zero when the relative position matches the target
    const ModelParametersT<T> modelParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    JointToJointPositionErrorFunctionT<T> errorFunctionZero(skeleton, character.parameterTransform);

    const auto& srcJointState = skelState.jointState[1];
    const auto& refJointState = skelState.jointState[2];

    const auto sourceOffset = uniform<Vector3<T>>(-1, 1);
    const auto referenceOffset = uniform<Vector3<T>>(-1, 1);

    // Compute world positions
    const Vector3<T> srcWorldPos = srcJointState.transform * sourceOffset;
    const Vector3<T> refWorldPos = refJointState.transform * referenceOffset;

    // Compute the relative position in reference frame (this is what the error function computes)
    const Vector3<T> diff = srcWorldPos - refWorldPos;
    const Vector3<T> relativePos = refJointState.transform.toRotationMatrix().transpose() * diff;

    // Set the target to exactly match the current relative position
    errorFunctionZero.addConstraint(1, sourceOffset, 2, referenceOffset, relativePos, T(1.0));

    const double error = errorFunctionZero.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
  }

  {
    SCOPED_TRACE("JointToJoint Position Relative Frame Test");

    // Test that the constraint correctly transforms to the reference frame
    // by verifying that rotating the reference joint doesn't change the error
    // when the target is set to the current relative position

    const ModelParametersT<T> modelParams =
        0.5 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1.0f, 1.0f);
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    JointToJointPositionErrorFunctionT<T> errorFunctionRelative(
        skeleton, character.parameterTransform);

    const auto& srcJointState = skelState.jointState[1];
    const auto& refJointState = skelState.jointState[2];

    const Vector3<T> sourceOffset = Vector3<T>::Zero();
    const Vector3<T> referenceOffset = Vector3<T>::Zero();

    // Compute relative position in reference frame
    const Vector3<T> srcWorldPos = srcJointState.transform * sourceOffset;
    const Vector3<T> refWorldPos = refJointState.transform * referenceOffset;
    const Vector3<T> diff = srcWorldPos - refWorldPos;
    const Vector3<T> relativePos = refJointState.transform.toRotationMatrix().transpose() * diff;

    // Add constraint with target matching current relative position
    errorFunctionRelative.addConstraint(1, sourceOffset, 2, referenceOffset, relativePos, T(1.0));

    const double error = errorFunctionRelative.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-6f, 1e-12));
  }
}
