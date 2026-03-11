/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/normal_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, NormalError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();

  // create constraints
  NormalErrorFunctionT<T> errorFunction(character.skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  {
    SCOPED_TRACE("Normal Constraint Test");
    std::vector<NormalDataT<T>> cl{
        NormalDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            1,
            kTestWeightValue),
        NormalDataT<T>(
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            uniform<Vector3<T>>(-1, 1),
            2,
            kTestWeightValue)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters()),
          character,
          2e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters()),
          character,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 2e-5), Eps<T>(1e-6f, 1e-7));
    }
  }
}

// Verify that the Jacobian is correctly computed even when the function value is zero.
// A zero function value (f = 0) does not imply a zero derivative (df/dq = 0).
TYPED_TEST(Momentum_ErrorFunctionsTest, NormalError_JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use non-zero parameters to create a non-trivial joint configuration
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  // Create a normal constraint where the global point equals the world point, so f = 0.
  // The error is dot(worldNormal, globalPoint - worldPoint) = dot(worldNormal, 0) = 0.
  const Vector3<T> localPoint = Vector3<T>::UnitX();
  const Vector3<T> localNormal = Vector3<T>::UnitY();
  const Vector3<T> worldPoint = state.jointState[2].transform * localPoint;

  NormalErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeight = 1.0;
  std::vector<NormalDataT<T>> cl{
      NormalDataT<T>(localPoint, localNormal, worldPoint, 2, kTestWeight)};
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
}
