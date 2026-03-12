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
#include "momentum/character_solver/position_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL2, uniform<T>(0.2, 10));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionL2 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
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
          T, &errorFunction, parameters, character, Eps<T>(5e-2f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL1_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL1, uniform<T>(0.5, 2));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionL1 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorCauchy_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kCauchy, uniform<T>(0.5, 2));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionCauchy Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorWelsch_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kWelsch);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionWelsch Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(1e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorGeneral_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, uniform<T>(0.1, 10));
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("PositionGeneral Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 2, kTestWeightValue),
        PositionDataT<T>(
            uniform<Vector3<T>>(-1, 1), uniform<Vector3<T>>(-1, 1), 1, kTestWeightValue)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-1f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

// Verify that the Jacobian is correctly computed even when the function value is zero.
// A zero function value (f = 0) does not imply a zero derivative (df/dq = 0). The Jacobian matrix
// should be non-zero at a point where the constraint is exactly satisfied, because small
// perturbations to the parameters will change the function value. This is important for the
// Gauss-Newton solver to correctly identify the configuration as a minimum (positive definite
// Hessian approximation J^T * J).
TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL2_JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use non-zero parameters to create a non-trivial joint configuration
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  // Create a position constraint where the target is set to the current position, so f = 0
  const Vector3<T> offset = Vector3<T>::UnitY();
  const Vector3<T> currentPosition = state.jointState[2].transform * offset;

  PositionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeight = 1.0;
  std::vector<PositionDataT<T>> cl{PositionDataT<T>(offset, currentPosition, 2, kTestWeight)};
  errorFunction.setConstraints(cl);

  // Compute Jacobian
  const size_t jacobianSize = errorFunction.getJacobianSize();
  Eigen::MatrixX<T> jacobian =
      Eigen::MatrixX<T>::Zero(jacobianSize, transform.numAllModelParameters());
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);
  MeshStateT<T> meshState{};
  int usedRows = 0;
  const double error =
      errorFunction.getJacobian(parameters, state, meshState, jacobian, residual, usedRows);

  // The error and residual should be zero since f = 0
  EXPECT_NEAR(error, 0.0, 1e-10);
  EXPECT_NEAR(residual.norm(), 0.0, Eps<T>(1e-6f, 1e-10));

  // The Jacobian matrix should be non-zero: even though f = 0, df/dq != 0
  EXPECT_GT(jacobian.norm(), 0.0);

  // Also verify consistency with numerical derivatives using the standard helper
  TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(5e-2f, 5e-6));
}
