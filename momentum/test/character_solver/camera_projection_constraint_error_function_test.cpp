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
#include "momentum/character_solver/camera_projection_constraint_error_function.h"
#include "momentum/character_solver/camera_projection_error_function.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

// ===========================================================================
// Helper: cross-validate legacy CameraProjectionErrorFunctionT vs new
// CameraProjectionConstraintErrorFunctionT (error, gradient, Jacobian).
// ===========================================================================
template <typename T>
void crossValidateCameraProjectionErrorFunctions(
    CameraProjectionErrorFunctionT<T>& legacy,
    CameraProjectionConstraintErrorFunctionT<T>& newFunc,
    const Character& character,
    const Eigen::VectorX<T>& parameters) {
  const auto& pt = character.parameterTransform;
  const ModelParametersT<T> modelParams = parameters;
  const SkeletonStateT<T> skelState(pt.cast<T>().apply(modelParams), character.skeleton);
  const MeshStateT<T> meshState;

  // --- Compare getError ---
  const double errLegacy = legacy.getError(modelParams, skelState, meshState);
  const double errNew = newFunc.getError(modelParams, skelState, meshState);
  EXPECT_NEAR(errLegacy, errNew, std::abs(errLegacy) * 1e-4 + 1e-10)
      << "getError mismatch: legacy=" << errLegacy << " new=" << errNew;

  // --- Compare getGradient (element-wise) ---
  const auto numParams = pt.numAllModelParameters();
  Eigen::VectorX<T> gradLegacy = Eigen::VectorX<T>::Zero(numParams);
  Eigen::VectorX<T> gradNew = Eigen::VectorX<T>::Zero(numParams);
  legacy.getGradient(modelParams, skelState, meshState, gradLegacy);
  newFunc.getGradient(modelParams, skelState, meshState, gradNew);

  // Use absolute + relative tolerance:
  // The decomposition into 5 contributions introduces numerical differences
  // compared to the monolithic two-chain walk, especially for float precision.
  const T absTol = std::is_same_v<T, float> ? T(0.5) : T(1e-8);
  const T relTol = std::is_same_v<T, float> ? T(5e-3) : T(1e-6);

  for (Eigen::Index j = 0; j < numParams; ++j) {
    const T diff = std::abs(gradLegacy[j] - gradNew[j]);
    const T scale = std::max(std::abs(gradLegacy[j]), std::abs(gradNew[j]));
    EXPECT_TRUE(diff < absTol || diff < relTol * scale)
        << "Gradient element " << j << " mismatch: legacy=" << gradLegacy[j]
        << " new=" << gradNew[j] << " diff=" << diff;
  }

  // --- Compare getJacobian element-wise ---
  const size_t jacSizeLegacy = legacy.getJacobianSize();
  const size_t jacSizeNew = newFunc.getJacobianSize();
  EXPECT_EQ(jacSizeLegacy, jacSizeNew) << "Jacobian size mismatch";

  if (jacSizeLegacy > 0 && jacSizeNew > 0) {
    Eigen::MatrixX<T> jacLegacy = Eigen::MatrixX<T>::Zero(jacSizeLegacy, numParams);
    Eigen::VectorX<T> resLegacy = Eigen::VectorX<T>::Zero(jacSizeLegacy);
    int usedRowsLegacy = 0;
    legacy.getJacobian(modelParams, skelState, meshState, jacLegacy, resLegacy, usedRowsLegacy);

    Eigen::MatrixX<T> jacNew = Eigen::MatrixX<T>::Zero(jacSizeNew, numParams);
    Eigen::VectorX<T> resNew = Eigen::VectorX<T>::Zero(jacSizeNew);
    int usedRowsNew = 0;
    newFunc.getJacobian(modelParams, skelState, meshState, jacNew, resNew, usedRowsNew);

    EXPECT_EQ(usedRowsLegacy, usedRowsNew) << "Used rows mismatch";

    // Compare residuals element-wise
    for (int r = 0; r < usedRowsLegacy; ++r) {
      const T diff = std::abs(resLegacy[r] - resNew[r]);
      const T scale = std::max(std::abs(resLegacy[r]), std::abs(resNew[r]));
      EXPECT_TRUE(diff < absTol || diff < relTol * scale)
          << "Residual element " << r << " mismatch: legacy=" << resLegacy[r]
          << " new=" << resNew[r] << " diff=" << diff;
    }

    // Compare Jacobian entries element-wise
    for (int r = 0; r < usedRowsLegacy; ++r) {
      for (Eigen::Index c = 0; c < numParams; ++c) {
        const T diff = std::abs(jacLegacy(r, c) - jacNew(r, c));
        const T scale = std::max(std::abs(jacLegacy(r, c)), std::abs(jacNew(r, c)));
        EXPECT_TRUE(diff < absTol || diff < relTol * scale)
            << "Jacobian element (" << r << "," << c << ") mismatch: legacy=" << jacLegacy(r, c)
            << " new=" << jacNew(r, c) << " diff=" << diff;
      }
    }
  }
}

// ===========================================================================
// GradientsAndJacobians_BoneParented
// ===========================================================================
TYPED_TEST(
    Momentum_ErrorFunctionsTest,
    CameraProjectionConstraint_GradientsAndJacobians_BoneParented) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  // Camera parented to joint 2, offset along -Z
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(-10));

  CameraProjectionConstraintErrorFunctionT<T> errorFunction(character, intrinsics, 2, cameraOffset);

  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraProjectionConstraintDataT<T>(
            uniform<size_t>(0, skeleton.joints.size() - 1),
            normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<float>(0.1f, 2.0f)));
  }

  // The bone-parented camera accumulates floating-point differently from
  // finite-difference numerical gradients, causing larger gradient mismatch
  // for float precision — up to ~88% on mac-arm/opt due to FMA rounding.
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      ModelParametersT<T>::Zero(transform.numAllModelParameters()),
      character,
      Eps<T>(1.0f, 1e-5));

  for (size_t i = 0; i < 10; i++) {
    ModelParametersT<T> parameters =
        uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
    TEST_GRADIENT_AND_JACOBIAN(
        T, &errorFunction, parameters, character, Eps<T>(1.0f, 1e-5), Eps<T>(1e-5f, 1e-5));
  }
}

// ===========================================================================
// GradientsAndJacobians_StaticCamera
// ===========================================================================
TYPED_TEST(
    Momentum_ErrorFunctionsTest,
    CameraProjectionConstraint_GradientsAndJacobians_StaticCamera) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  // Static camera placed 10 units along Z
  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  eyeFromWorld.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraProjectionConstraintErrorFunctionT<T> errorFunction(character, intrinsics, eyeFromWorld);

  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(
        CameraProjectionConstraintDataT<T>(
            uniform<size_t>(0, skeleton.joints.size() - 1),
            normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
            normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100)),
            uniform<float>(0.1f, 2.0f)));
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

// ===========================================================================
// CrossVal_BoneParented — element-wise comparison with legacy
// ===========================================================================
TYPED_TEST(Momentum_ErrorFunctionsTest, CameraProjectionConstraint_CrossVal_BoneParented) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(-10));

  CameraProjectionErrorFunctionT<T> legacy(
      skeleton, character.parameterTransform, intrinsics, 2, cameraOffset);
  CameraProjectionConstraintErrorFunctionT<T> newFunc(character, intrinsics, 2, cameraOffset);

  for (int i = 0; i < 5; ++i) {
    const auto parent = uniform<size_t>(0, skeleton.joints.size() - 1);
    const auto offset = normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones());
    const auto target = normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100));
    const T weight = uniform<T>(0.1, 2.0);

    legacy.addConstraint(ProjectionConstraintT<T>{parent, offset, weight, target});
    newFunc.addConstraint(
        CameraProjectionConstraintDataT<T>(parent, offset, target, static_cast<float>(weight)));
  }

  // Test at zero parameters
  crossValidateCameraProjectionErrorFunctions<T>(
      legacy, newFunc, character, Eigen::VectorX<T>::Zero(transform.numAllModelParameters()));

  // Test at multiple random parameter sets
  for (size_t i = 0; i < 5; i++) {
    auto parameters = uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
    crossValidateCameraProjectionErrorFunctions<T>(legacy, newFunc, character, parameters);
  }
}

// ===========================================================================
// CrossVal_StaticCamera — element-wise comparison with legacy
// ===========================================================================
TYPED_TEST(Momentum_ErrorFunctionsTest, CameraProjectionConstraint_CrossVal_StaticCamera) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  cameraOffset.translation() = Eigen::Vector3<T>(0, 0, T(10));

  CameraProjectionErrorFunctionT<T> legacy(
      skeleton, character.parameterTransform, intrinsics, kInvalidIndex, cameraOffset);
  CameraProjectionConstraintErrorFunctionT<T> newFunc(character, intrinsics, cameraOffset);

  for (int i = 0; i < 5; ++i) {
    const auto parent = uniform<size_t>(0, skeleton.joints.size() - 1);
    const auto offset = normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones());
    const auto target = normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones() * T(100));
    const T weight = uniform<T>(0.1, 2.0);

    legacy.addConstraint(ProjectionConstraintT<T>{parent, offset, weight, target});
    newFunc.addConstraint(
        CameraProjectionConstraintDataT<T>(parent, offset, target, static_cast<float>(weight)));
  }

  // Test at zero parameters
  crossValidateCameraProjectionErrorFunctions<T>(
      legacy, newFunc, character, Eigen::VectorX<T>::Zero(transform.numAllModelParameters()));

  // Test at multiple random parameter sets
  for (size_t i = 0; i < 5; i++) {
    auto parameters = uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
    crossValidateCameraProjectionErrorFunctions<T>(legacy, newFunc, character, parameters);
  }
}

// ===========================================================================
// NearClip — verify behind-camera constraints produce zero error
// ===========================================================================
TYPED_TEST(Momentum_ErrorFunctionsTest, CameraProjectionConstraint_NearClip) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  auto intrinsics = std::make_shared<PinholeIntrinsicsModelT<T>>(640, 480, T(500), T(500));

  // Place camera far behind the skeleton so the skeleton is behind the camera
  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  eyeFromWorld.translation() = Eigen::Vector3<T>(0, 0, T(-100));

  CameraProjectionConstraintErrorFunctionT<T> errorFunction(character, intrinsics, eyeFromWorld);

  // Add a constraint at a point that will be behind the camera
  errorFunction.addConstraint(
      CameraProjectionConstraintDataT<T>(
          0, Eigen::Vector3<T>::Zero(), Eigen::Vector2<T>(T(100), T(200)), 1.0f));

  const ModelParametersT<T> modelParams =
      ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> skelState(transform.apply(modelParams), skeleton);
  const MeshStateT<T> meshState;

  const double error = errorFunction.getError(modelParams, skelState, meshState);
  EXPECT_NEAR(error, 0.0, 1e-10) << "Behind-camera constraint should produce zero error";
}
