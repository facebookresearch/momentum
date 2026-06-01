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
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

namespace {
// Random rotation with a bounded angle. Full-range uniformQuaternion() routes to Eigen's unseeded
// std::rand and can yield near-pi rotations, where the float finite-difference gradient check is
// unreliable and diverges across platforms/std libs. A bounded angle keeps the orientation
// difference well away from the pi singularity, so the float FD check stays well-conditioned and
// reproducible on every platform.
template <typename T>
Quaternion<T> boundedRandomRotation(Random<>& rng) {
  const Vector3<T> axis = rng.uniform<Vector3<T>>(T(-1), T(1)).normalized();
  const T angle = rng.uniform<T>(T(-0.7), T(0.7));
  return Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
}
} // namespace

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, OrientationErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  OrientationErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  Random<>& rng = Random<>::GetSingleton();

  {
    SCOPED_TRACE("Orientation Constraint Test");
    const Quaternion<T> offset0 = boundedRandomRotation<T>(rng);
    const Quaternion<T> target0 = boundedRandomRotation<T>(rng);
    const Quaternion<T> offset1 = boundedRandomRotation<T>(rng);
    const Quaternion<T> target1 = boundedRandomRotation<T>(rng);
    std::vector<OrientationDataT<T>> cl{
        OrientationDataT<T>(offset0, target0, 2, kTestWeightValue),
        OrientationDataT<T>(offset1, target1, 1, kTestWeightValue)};
    errorFunction.setConstraints(std::move(cl));

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(0.03f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          rng.uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(0.05f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}

// Verify that the Jacobian is correctly computed even when the function value is zero.
// A zero function value (f = 0) does not imply a zero derivative (df/dq = 0).
TYPED_TEST(Momentum_ErrorFunctionsTest, OrientationErrorL2_JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use non-zero parameters to create a non-trivial joint configuration
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  // Create an orientation constraint where the target matches the current joint orientation,
  // so f = 0
  const Eigen::Quaternion<T> offset(Eigen::AngleAxis<T>(T(0.3), Vector3<T>::UnitY()));
  const Eigen::Quaternion<T> jointQuat(state.jointState[2].transform.toRotationMatrix());
  const Eigen::Quaternion<T> target = jointQuat * offset;

  OrientationErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeight = 1.0;
  std::vector<OrientationDataT<T>> cl{OrientationDataT<T>(offset, target, 2, kTestWeight)};
  errorFunction.setConstraints(std::move(cl));

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

TYPED_TEST(Momentum_ErrorFunctionsTest, OrientationRotDiffErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  OrientationRotDiffErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T kTestWeightValue = 4.5;
  Random<>& rng = Random<>::GetSingleton();

  {
    SCOPED_TRACE("OrientationRotDiff Constraint Test");
    const Quaternion<T> offset0 = boundedRandomRotation<T>(rng);
    const Quaternion<T> target0 = boundedRandomRotation<T>(rng);
    const Quaternion<T> offset1 = boundedRandomRotation<T>(rng);
    const Quaternion<T> target1 = boundedRandomRotation<T>(rng);
    std::vector<OrientationDataT<T>> cl{
        OrientationDataT<T>(offset0, target0, 2, kTestWeightValue),
        OrientationDataT<T>(offset1, target1, 1, kTestWeightValue)};
    errorFunction.setConstraints(std::move(cl));

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(0.06f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          rng.uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(0.06f, 5e-6), Eps<T>(1e-6f, 1e-7));
    }
  }
}
