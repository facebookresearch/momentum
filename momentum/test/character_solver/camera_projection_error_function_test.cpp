/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/camera_projection_error_function.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraProjectionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create a pinhole intrinsics model
  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  {
    SCOPED_TRACE("CameraProjection Static Camera Test");

    // Static camera (kInvalidIndex parent): camera is placed 10 units along Z
    Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
        Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

    CameraProjectionErrorFunctionT<T> errorFunction(
        skeleton, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

    for (int i = 0; i < 5; ++i) {
      errorFunction.addConstraint(
          ProjectionConstraintT<T>{
              uniform<size_t>(0, skeleton.joints.size() - 1),
              normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
              uniform<T>(0.1, 2.0),
              normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100))});
    }

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-3));

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 5e-3), Eps<T>(1e-5f, 1e-6));
    }
  }

  {
    SCOPED_TRACE("CameraProjection Bone-Parented Camera Test");

    // Camera parented to joint 2, offset along -Z so the skeleton is in front of the camera
    Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
        Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(-10));

    CameraProjectionErrorFunctionT<T> errorFunction(
        skeleton, character.parameterTransform, intrinsics, 2, cameraOffset);

    for (int i = 0; i < 5; ++i) {
      errorFunction.addConstraint(
          ProjectionConstraintT<T>{
              uniform<size_t>(0, skeleton.joints.size() - 1),
              normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
              uniform<T>(0.1, 2.0),
              normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100))});
    }

    // Bone-parented camera involves inverse transforms and two walks, so float
    // precision is lower than the static camera case. The threshold is raised
    // for float to accommodate platform-specific floating-point differences
    // (e.g. Mac vs Linux).
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-1f, 5e-3));

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(3e-1f, 5e-3), Eps<T>(1e-5f, 1e-6));
    }
  }

  {
    SCOPED_TRACE("CameraProjection Zero Error Test");

    // Verify that projecting a point and using its projected position as target gives zero error
    Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
        Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(5));

    const ModelParametersT<T> modelParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

    // Get world position of joint 1
    const size_t parentJoint = 1;
    const Eigen::Vector3<T> offset = Eigen::Vector3<T>(T(0.1), T(0.2), T(0.3));
    const Eigen::Vector3<T> p_world = skelState.jointState[parentJoint].transform * offset;

    // Project through the camera
    const Eigen::Vector3<T> p_eye = cameraOffset * p_world;
    const auto [projected, valid] = intrinsics->project(p_eye);
    ASSERT_TRUE(valid);

    CameraProjectionErrorFunctionT<T> errorFunction(
        skeleton, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

    errorFunction.addConstraint(
        ProjectionConstraintT<T>{parentJoint, offset, T(1.0), projected.template head<2>()});

    const double error = errorFunction.getError(modelParams, skelState, MeshStateT<T>());
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
  }
}
