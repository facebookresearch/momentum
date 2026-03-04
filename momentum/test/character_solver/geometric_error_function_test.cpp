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
#include "momentum/character_solver/distance_error_function.h"
#include "momentum/character_solver/height_error_function.h"
#include "momentum/character_solver/joint_to_joint_distance_error_function.h"
#include "momentum/character_solver/joint_to_joint_position_error_function.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver/projection_error_function.h"
#include "momentum/math/random.h"
#include "momentum/math/utility.h"
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

TYPED_TEST(Momentum_ErrorFunctionsTest, ProjectionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  ProjectionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform, 0.01);
  Eigen::Matrix4<T> projection = Eigen::Matrix4<T>::Identity();
  projection(2, 3) = 10;
  {
    SCOPED_TRACE("Projection Constraint Test");
    // Make a few projection constraints to ensure that at least one of them is active, since
    // projections are ignored behind the camera
    for (int i = 0; i < 5; ++i) {
      errorFunction.addConstraint(
          ProjectionConstraintDataT<T>{
              (projection + uniformAffine3<T>().matrix()).topRows(3),
              uniform<size_t>(0, 2),
              normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
              uniform<T>(0.1, 2.0),
              normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones())});
    }

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
          5e-3,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
      const momentum::SkeletonStateT<T> skelState(transform.apply(parameters), skeleton);
      ASSERT_GT(errorFunction.getError(parameters, skelState, MeshStateT<T>()), 0);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 5e-3), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, DistanceConstraint_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  DistanceErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("Distance Constraint Test");
    DistanceConstraintDataT<T> constraintData;
    constraintData.parent = 1;
    constraintData.offset = normal<Vector3<T>>(0, 1);
    constraintData.origin = normal<Vector3<T>>(0, 1);
    constraintData.target = 2.3f;
    constraintData.weight = kTestWeightValue;
    errorFunction.setConstraints({constraintData});

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
          uniform<VectorX<T>>(transform.numAllModelParameters(), 1, 0.0f, 1.0f);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 5e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

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

TYPED_TEST(Momentum_ErrorFunctionsTest, HeightError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  SCOPED_TRACE("Height Error Function Test");

  // Test with blend shapes enabled (so we have shape and scale parameters to work with)
  const Character character = withTestBlendShapes(createTestCharacter());
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Create error function with target height (it automatically uses blend shape and scale
  // parameters)
  HeightErrorFunctionT<T> errorFunction(character, T(1.8), Eigen::Vector3<T>::UnitY(), 4);

  // Test 1: Validate gradients and Jacobians
  {
    SCOPED_TRACE("Test standard gradient/Jacobian validation");

    // Test with zero parameters
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-14),
        true,
        false);

    // Test with random parameters
    for (size_t i = 0; i < 5; i++) {
      ModelParametersT<T> parameters =
          0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 1e-5),
          Eps<T>(1e-6f, 1e-14),
          true,
          false);
    }
  }

  // Test 2: Verify error does not depend on inactive parameters (pose parameters)
  {
    SCOPED_TRACE("Test that error does not depend on inactive parameters");

    // Get the set of active parameters (blend shapes + scale)
    const ParameterSet activeParams = transform.getBlendShapeParameters() |
        transform.getFaceExpressionParameters() | transform.getScalingParameters();

    // Create a base parameter set
    ModelParametersT<T> baseParameters =
        0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

    // Compute error with base parameters
    const SkeletonStateT<T> skelState(transform.apply(baseParameters), character.skeleton);
    const double baseError = errorFunction.getError(
        baseParameters, skelState, MeshStateT<T>(baseParameters, skelState, character));

    // Now perturb ONLY inactive parameters and verify error doesn't change
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> perturbedParameters = baseParameters;

      // Perturb only inactive parameters (pose parameters)
      for (Eigen::Index p = 0; p < perturbedParameters.size(); ++p) {
        if (!activeParams.test(p)) {
          perturbedParameters(p) = uniform<T>(-2.0, 2.0);
        }
      }

      const SkeletonStateT<T> perturbedSkelState(
          transform.apply(perturbedParameters), character.skeleton);
      const double perturbedError = errorFunction.getError(
          perturbedParameters,
          perturbedSkelState,
          MeshStateT<T>(perturbedParameters, perturbedSkelState, character));

      // Error should be identical (or very close) since we only changed inactive parameters
      EXPECT_NEAR(perturbedError, baseError, Eps<T>(1e-10f, 1e-15))
          << "Error changed when modifying inactive parameters. "
          << "Base error: " << baseError << ", Perturbed error: " << perturbedError;
    }
  }

  // Test 3: Verify gradients are zero for inactive parameters
  {
    SCOPED_TRACE("Test that gradients are zero for inactive parameters");

    // Get the set of active parameters (blend shapes + scale)
    const ParameterSet activeParams = transform.getBlendShapeParameters() |
        transform.getFaceExpressionParameters() | transform.getScalingParameters();

    ModelParametersT<T> parameters =
        0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

    const SkeletonStateT<T> skelState(transform.apply(parameters), character.skeleton);
    Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(transform.numAllModelParameters());

    errorFunction.getGradient(
        parameters, skelState, MeshStateT<T>(parameters, skelState, character), gradient);

    // Check that gradient is zero for all inactive parameters
    for (Eigen::Index p = 0; p < gradient.size(); ++p) {
      if (!activeParams.test(p)) {
        EXPECT_NEAR(gradient(p), T(0), Eps<T>(1e-10f, 1e-15))
            << "Gradient for inactive parameter " << p << "("
            << character.parameterTransform.name.at(p) << ") is non-zero: " << gradient(p);
      }
    }
  }
}
