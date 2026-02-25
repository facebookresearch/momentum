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
#include "momentum/character_solver/camera_intrinsics_parameters.h"
#include "momentum/character_solver/camera_projection_error_function.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

#include <set>

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

// ---- Tests for getParameterNames() and IntrinsicsModel name ----

TEST(CameraIntrinsicsParameters, GetParameterNames) {
  // Pinhole: 4 params
  {
    auto model = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);
    const auto names = model->getParameterNames();
    ASSERT_EQ(names.size(), 4);
    EXPECT_EQ(names[0], "fx");
    EXPECT_EQ(names[1], "fy");
    EXPECT_EQ(names[2], "cx");
    EXPECT_EQ(names[3], "cy");
  }

  // OpenCV Fisheye: 8 params
  {
    auto model =
        std::make_shared<OpenCVFisheyeIntrinsicsModel>(640, 480, 500.0f, 500.0f, 320.0f, 240.0f);
    const auto names = model->getParameterNames();
    ASSERT_EQ(names.size(), 8);
    EXPECT_EQ(names[0], "fx");
    EXPECT_EQ(names[4], "k1");
    EXPECT_EQ(names[7], "k4");
  }

  // OpenCV: 14 params
  {
    auto model = std::make_shared<OpenCVIntrinsicsModel>(640, 480, 500.0f, 500.0f, 320.0f, 240.0f);
    const auto names = model->getParameterNames();
    ASSERT_EQ(names.size(), 14);
    EXPECT_EQ(names[0], "fx");
    EXPECT_EQ(names[4], "k1");
    EXPECT_EQ(names[10], "p1");
    EXPECT_EQ(names[13], "p4");
  }
}

TEST(CameraIntrinsicsParameters, AddCameraIntrinsicsParameters_Pinhole) {
  const Character character = createTestCharacter();
  auto intrinsics = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);
  intrinsics->setName("cam0");

  const auto origSize = character.parameterTransform.name.size();

  auto [pt, pl] = addCameraIntrinsicsParameters(
      character.parameterTransform, character.parameterLimits, *intrinsics);

  // Should have added 4 params (fx, fy, cx, cy)
  EXPECT_EQ(pt.name.size(), origSize + 4);
  EXPECT_EQ(pt.name[origSize], "intrinsics_cam0_fx");
  EXPECT_EQ(pt.name[origSize + 1], "intrinsics_cam0_fy");
  EXPECT_EQ(pt.name[origSize + 2], "intrinsics_cam0_cx");
  EXPECT_EQ(pt.name[origSize + 3], "intrinsics_cam0_cy");

  // Transform matrix should have grown
  EXPECT_EQ(pt.transform.cols(), static_cast<Eigen::Index>(pt.name.size()));

  // Named parameter set should exist
  EXPECT_TRUE(pt.parameterSets.count("intrinsics_cam0") > 0);

  // Verify clone preserves name (guards against forgetting setName in clone())
  auto cloned = intrinsics->clone();
  EXPECT_EQ(cloned->name(), "cam0");
}

TEST(CameraIntrinsicsParameters, AddCameraIntrinsicsParameters_Idempotent) {
  const Character character = createTestCharacter();
  auto intrinsics = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);
  intrinsics->setName("cam0");

  auto [pt1, pl1] = addCameraIntrinsicsParameters(
      character.parameterTransform, character.parameterLimits, *intrinsics);

  // Call again — should not duplicate
  auto [pt2, pl2] = addCameraIntrinsicsParameters(pt1, pl1, *intrinsics);

  EXPECT_EQ(pt1.name.size(), pt2.name.size());
  EXPECT_EQ(pt1.transform.cols(), pt2.transform.cols());
}

TEST(CameraIntrinsicsParameters, AddCameraIntrinsicsParameters_MultipleCameras) {
  const Character character = createTestCharacter();
  auto intrinsics0 = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);
  intrinsics0->setName("cam0");

  auto intrinsics1 =
      std::make_shared<OpenCVFisheyeIntrinsicsModel>(640, 480, 500.0f, 500.0f, 320.0f, 240.0f);
  intrinsics1->setName("cam1");

  const auto origSize = character.parameterTransform.name.size();

  auto [pt1, pl1] = addCameraIntrinsicsParameters(
      character.parameterTransform, character.parameterLimits, *intrinsics0);
  auto [pt2, pl2] = addCameraIntrinsicsParameters(pt1, pl1, *intrinsics1);

  // cam0 has 4 params, cam1 has 8 params
  EXPECT_EQ(pt2.name.size(), origSize + 4 + 8);

  // Verify all params are distinct
  std::set<std::string> allNames(pt2.name.begin(), pt2.name.end());
  EXPECT_EQ(allNames.size(), pt2.name.size());

  // Per-camera lookup returns correct count
  EXPECT_EQ(getCameraIntrinsicsParameterSet(pt2, "cam0").count(), 4);
  EXPECT_EQ(getCameraIntrinsicsParameterSet(pt2, "cam1").count(), 8);
  EXPECT_EQ(getCameraIntrinsicsParameterSet(pt2, "nonexistent").count(), 0);

  // All-cameras lookup returns union
  EXPECT_EQ(getAllCameraIntrinsicsParameterSet(pt2).count(), 4 + 8);
}

TEST(CameraIntrinsicsParameters, ExtractAndSetCameraIntrinsics) {
  const Character character = createTestCharacter();
  auto intrinsics =
      std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f, 320.0f, 240.0f);
  intrinsics->setName("cam0");

  auto [pt, pl] = addCameraIntrinsicsParameters(
      character.parameterTransform, character.parameterLimits, *intrinsics);

  // setCameraIntrinsics: write intrinsics model values into model parameters
  ModelParameters modelParams = ModelParameters::Zero(pt.numAllModelParameters());
  setCameraIntrinsics(pt, *intrinsics, modelParams);

  // The pinhole params should now be (500, 500, 320, 240) at the right indices
  const auto fxIdx = pt.getParameterIdByName("intrinsics_cam0_fx");
  const auto fyIdx = pt.getParameterIdByName("intrinsics_cam0_fy");
  const auto cxIdx = pt.getParameterIdByName("intrinsics_cam0_cx");
  const auto cyIdx = pt.getParameterIdByName("intrinsics_cam0_cy");
  EXPECT_FLOAT_EQ(modelParams(fxIdx), 500.0f);
  EXPECT_FLOAT_EQ(modelParams(fyIdx), 500.0f);
  EXPECT_FLOAT_EQ(modelParams(cxIdx), 320.0f);
  EXPECT_FLOAT_EQ(modelParams(cyIdx), 240.0f);

  // Modify model parameters and extract back
  modelParams(fxIdx) = 600.0f;
  modelParams(cyIdx) = 250.0f;

  const auto extracted = extractCameraIntrinsics(pt, modelParams, *intrinsics);
  ASSERT_EQ(extracted.size(), 4);
  EXPECT_FLOAT_EQ(extracted(0), 600.0f); // fx — modified
  EXPECT_FLOAT_EQ(extracted(1), 500.0f); // fy — unchanged
  EXPECT_FLOAT_EQ(extracted(2), 320.0f); // cx — unchanged
  EXPECT_FLOAT_EQ(extracted(3), 250.0f); // cy — modified

  // Apply extracted values to a cloned model and verify round-trip
  auto cloned = intrinsics->clone();
  cloned->setIntrinsicParameters(extracted);
  EXPECT_FLOAT_EQ(cloned->fx(), 600.0f);
}

TEST(CameraIntrinsicsParameters, ExtractCameraIntrinsics_MissingParams) {
  const Character character = createTestCharacter();
  auto intrinsics =
      std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f, 320.0f, 240.0f);
  intrinsics->setName("cam0");

  auto [pt, pl] = addCameraIntrinsicsParameters(
      character.parameterTransform, character.parameterLimits, *intrinsics);

  // Downselect to only include fx and cy (simulating a user restricting the parameter set)
  ParameterSet subset;
  // Keep all non-intrinsics params
  for (size_t i = 0; i < pt.name.size(); ++i) {
    if (pt.name[i].substr(0, std::string(kIntrinsicsParamPrefix).size()) !=
        std::string(kIntrinsicsParamPrefix)) {
      subset.set(i);
    }
  }
  // Also keep fx and cy
  subset.set(pt.getParameterIdByName("intrinsics_cam0_fx"));
  subset.set(pt.getParameterIdByName("intrinsics_cam0_cy"));

  auto [ptSubset, plSubset] = subsetParameterTransform(pt, pl, subset);

  // Set fx=600 in the reduced model parameters
  ModelParameters modelParams = ModelParameters::Zero(ptSubset.numAllModelParameters());
  const auto fxIdx = ptSubset.getParameterIdByName("intrinsics_cam0_fx");
  modelParams(fxIdx) = 600.0f;
  const auto cyIdx = ptSubset.getParameterIdByName("intrinsics_cam0_cy");
  modelParams(cyIdx) = 250.0f;

  // Extract: fx and cy come from modelParams, fy and cx fall back to intrinsics model defaults
  const auto extracted = extractCameraIntrinsics(ptSubset, modelParams, *intrinsics);
  ASSERT_EQ(extracted.size(), 4);
  EXPECT_FLOAT_EQ(extracted(0), 600.0f); // fx — from modelParams
  EXPECT_FLOAT_EQ(extracted(1), 500.0f); // fy — fallback to model default
  EXPECT_FLOAT_EQ(extracted(2), 320.0f); // cx — fallback to model default
  EXPECT_FLOAT_EQ(extracted(3), 250.0f); // cy — from modelParams
}
