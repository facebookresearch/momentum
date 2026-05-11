/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/camera_intrinsics_parameters.h"
#include "momentum/character_solver/camera_vertex_projection_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_StaticCamera) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

  const size_t numVertices = character.mesh->vertices.size();
  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraVertexProjectionDataT<T>{
            uniform<size_t>(0, numVertices - 1),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<T>(T(0.1), T(2.0))});
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

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_BoneParentedCamera) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  // Camera offset along +Z so the skeleton stays in front of the camera
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, 2, cameraOffset);

  const size_t numVertices = character.mesh->vertices.size();
  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraVertexProjectionDataT<T>{
            uniform<size_t>(0, numVertices - 1),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<T>(T(0.1), T(2.0))});
  }

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

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_WithBlendShapes) {
  using T = typename TestFixture::Type;

  const Character character = withTestBlendShapes(createTestCharacter());
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

  const size_t numVertices = character.mesh->vertices.size();
  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraVertexProjectionDataT<T>{
            uniform<size_t>(0, numVertices - 1),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<T>(T(0.1), T(2.0))});
  }

  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character,
      Eps<T>(1e-1f, 5e-3),
      Eps<T>(1e-5f, 1e-6),
      true,
      false);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_ZeroError) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(5));

  const ModelParametersT<T> modelParams =
      ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);

  MeshStateT<T> meshState;
  meshState.update(modelParams, skelState, character);

  const size_t vertexIndex = 0;
  const Eigen::Vector3<T> p_world = meshState.posedMesh_->vertices[vertexIndex].template cast<T>();
  const Eigen::Vector3<T> p_eye = cameraOffset * p_world;
  const auto [projected, valid] = intrinsics->project(p_eye);
  ASSERT_TRUE(valid);

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

  errorFunction.addConstraint(
      CameraVertexProjectionDataT<T>{vertexIndex, projected.template head<2>(), T(1.0)});

  const double error = errorFunction.getError(modelParams, skelState, meshState);
  EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  MeshStateT<T> meshState;
  meshState.update(parameters, state, character);

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(5));

  const size_t vertexIndex = 0;
  const Eigen::Vector3<T> p_world = meshState.posedMesh_->vertices[vertexIndex].template cast<T>();
  const Eigen::Vector3<T> p_eye = cameraOffset * p_world;
  const auto [projected, valid] = intrinsics->project(p_eye);
  ASSERT_TRUE(valid);

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);
  errorFunction.addConstraint(
      CameraVertexProjectionDataT<T>{vertexIndex, projected.template head<2>(), T(1.0)});

  const size_t jacobianSize = errorFunction.getJacobianSize();
  Eigen::MatrixX<T> jacobian =
      Eigen::MatrixX<T>::Zero(jacobianSize, transform.numAllModelParameters());
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);
  int usedRows = 0;
  const double error =
      errorFunction.getJacobian(parameters, state, meshState, jacobian, residual, usedRows);

  EXPECT_NEAR(error, 0.0, Eps<T>(1e-6f, 1e-15));
  EXPECT_NEAR(residual.norm(), 0.0, Eps<T>(1e-3f, 1e-10));
  EXPECT_GT(jacobian.norm(), 0.0);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, CameraVertexProjectionError_IntrinsicsOptimization) {
  using T = typename TestFixture::Type;

  const Character baseCharacter = createTestCharacter();
  const Skeleton& skeleton = baseCharacter.skeleton;

  auto intrinsicsF = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);
  intrinsicsF->setName("cam0");

  auto ptAndPl = addCameraIntrinsicsParameters(
      baseCharacter.parameterTransform, baseCharacter.parameterLimits, *intrinsicsF);
  const auto& pt = std::get<0>(ptAndPl);
  const auto& pl = std::get<1>(ptAndPl);
  const Character character(
      skeleton,
      pt,
      pl,
      baseCharacter.locators,
      baseCharacter.mesh.get(),
      baseCharacter.skinWeights.get(),
      baseCharacter.collision.get(),
      baseCharacter.poseShapes.get(),
      baseCharacter.blendShape,
      baseCharacter.faceExpressionBlendShape,
      baseCharacter.name,
      baseCharacter.inverseBindPose);

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));
  intrinsics->setName("cam0");

  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  const auto paramNames = intrinsics->getParameterNames();
  auto setIntrinsicsParams = [&](ModelParametersT<T>& params) {
    const auto intrinsicValues = intrinsics->getIntrinsicParameters();
    for (size_t i = 0; i < paramNames.size(); ++i) {
      const auto idx =
          pt.getParameterIdByName(std::string(kIntrinsicsParamPrefix) + "cam0_" + paramNames[i]);
      params(idx) = T(intrinsicValues(static_cast<Eigen::Index>(i)));
    }
  };

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraVertexProjectionErrorFunctionT<T> errorFunction(
      character, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);

  const size_t numVertices = character.mesh->vertices.size();
  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraVertexProjectionDataT<T>{
            uniform<size_t>(0, numVertices - 1),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<T>(T(0.1), T(2.0))});
  }

  ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  setIntrinsicsParams(params);

  {
    const SkeletonStateT<T> state(transform.apply(params), skeleton);
    MeshStateT<T> meshState;
    meshState.update(params, state, character);
    const double err = errorFunction.getError(params, state, meshState);
    EXPECT_GT(err, 0.0) << "Error should be non-zero to validate gradient test is meaningful";
  }

  TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, params, character, Eps<T>(5e-2f, 5e-3));

  for (size_t i = 0; i < 10; i++) {
    ModelParametersT<T> parameters =
        uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
    setIntrinsicsParams(parameters);
    TEST_GRADIENT_AND_JACOBIAN(
        T, &errorFunction, parameters, character, Eps<T>(1e-1f, 5e-3), Eps<T>(1e-5f, 1e-6));
  }
}
