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
#include "momentum/character_solver/joint_to_joint_orientation_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, JointToJointOrientationError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  JointToJointOrientationErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 2.5;

  {
    SCOPED_TRACE("JointToJoint Orientation Constraint Test");

    ASSERT_GE(skeleton.joints.size(), 3);

    // Add constraints with random target orientations
    const Eigen::Quaternion<T> target1(
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitX()) *
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitY()) *
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitZ()));
    errorFunction.addConstraint(1, 2, target1, kTestWeightValue);

    const Eigen::Quaternion<T> target2(
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitX()) *
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitY()) *
        Eigen::AngleAxis<T>(uniform<T>(-1, 1), Vector3<T>::UnitZ()));
    errorFunction.addConstraint(0, 2, target2, kTestWeightValue);

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
    SCOPED_TRACE("JointToJoint Orientation Zero Error Test");

    // Test that error is zero when the relative orientation matches the target
    const ModelParametersT<T> modelParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    JointToJointOrientationErrorFunctionT<T> errorFunctionZero(
        skeleton, character.parameterTransform);

    const auto& srcJointState = skelState.jointState[1];
    const auto& refJointState = skelState.jointState[2];

    // Compute the relative orientation (this is what the error function computes)
    const Matrix3<T> relativeRot = refJointState.rotation().toRotationMatrix().transpose() *
        srcJointState.rotation().toRotationMatrix();
    const Eigen::Quaternion<T> relativeQuat(relativeRot);

    // Set the target to exactly match the current relative orientation
    errorFunctionZero.addConstraint(1, 2, relativeQuat, T(1.0));

    const double error = errorFunctionZero.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
  }

  {
    SCOPED_TRACE("JointToJoint Orientation Relative Frame Test");

    // Test that the constraint correctly computes relative orientation
    // with random model parameters
    const ModelParametersT<T> modelParams =
        0.5 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1.0f, 1.0f);
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    JointToJointOrientationErrorFunctionT<T> errorFunctionRelative(
        skeleton, character.parameterTransform);

    const auto& srcJointState = skelState.jointState[1];
    const auto& refJointState = skelState.jointState[2];

    // Compute relative orientation in reference frame
    const Matrix3<T> relativeRot = refJointState.rotation().toRotationMatrix().transpose() *
        srcJointState.rotation().toRotationMatrix();
    const Eigen::Quaternion<T> relativeQuat(relativeRot);

    // Add constraint with target matching current relative orientation
    errorFunctionRelative.addConstraint(1, 2, relativeQuat, T(1.0));

    const double error = errorFunctionRelative.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-6f, 1e-12));
  }
}
