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

// Verify that the Jacobian is correctly computed even when the function value is zero.
// A zero function value (f = 0) does not imply a zero derivative (df/dq = 0).
TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneErrorL2_JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use non-zero parameters to create a non-trivial joint configuration
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  // Create a plane constraint where the point lies exactly on the plane, so f = 0
  const Vector3<T> offset = Vector3<T>::UnitY();
  const Vector3<T> worldPoint = state.jointState[2].transform * offset;
  const Vector3<T> normal = Vector3<T>::UnitZ();
  const T d = normal.dot(worldPoint);

  PlaneErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeight = 1.0;
  std::vector<PlaneDataT<T>> cl{PlaneDataT<T>(offset, normal, d, 2, kTestWeight)};
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
