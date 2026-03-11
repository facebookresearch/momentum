/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/mesh_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/center_of_mass_error_function.h>
#include <momentum/math/random.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/character_solver/error_function_helpers.h>

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct CenterOfMassErrorFunctionTest : Momentum_ErrorFunctionsTest<T> {};

TYPED_TEST_SUITE(CenterOfMassErrorFunctionTest, Types);

TYPED_TEST(CenterOfMassErrorFunctionTest, GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto& parameterTransform = character.parameterTransform;

  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;

  CenterOfMassErrorFunctionT<T> errorFunction(skeleton, parameterTransform);

  // Create a constraint using the first 4 joints
  CenterOfMassConstraintT<T> constraint;
  const size_t numJoints = std::min<size_t>(4, skeleton.joints.size());
  for (size_t j = 0; j < numJoints; ++j) {
    constraint.jointIndices.push_back(j);
    constraint.masses.push_back(T(1) + T(j) * T(0.5));
  }
  constraint.target = uniform<Vector3<T>>(-1, 1);
  constraint.weight = T(1.5);
  errorFunction.addConstraint(constraint);

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character,
      Eps<T>(5e-2f, 5e-4),
      Eps<T>(1e-4f, 1e-8),
      false,
      false);
}

TYPED_TEST(CenterOfMassErrorFunctionTest, ZeroError) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto& parameterTransform = character.parameterTransform;

  const ModelParametersT<T> modelParams =
      ModelParametersT<T>::Zero(parameterTransform.numAllModelParameters());
  const SkeletonStateT<T> skelState(parameterTransform.cast<T>().apply(modelParams), skeleton);

  CenterOfMassErrorFunctionT<T> errorFunction(skeleton, parameterTransform);

  // Compute actual CoM from current joint positions and use it as target
  const size_t numJoints = std::min<size_t>(4, skeleton.joints.size());
  CenterOfMassConstraintT<T> constraint;
  Eigen::Vector3<T> actualCoM = Eigen::Vector3<T>::Zero();
  T totalMass = T(0);
  for (size_t j = 0; j < numJoints; ++j) {
    constraint.jointIndices.push_back(j);
    const T mass = T(1) + T(j);
    constraint.masses.push_back(mass);
    actualCoM += mass * skelState.jointState[j].translation();
    totalMass += mass;
  }
  actualCoM /= totalMass;
  constraint.target = actualCoM;
  constraint.weight = T(1);
  errorFunction.addConstraint(constraint);

  const double error = errorFunction.getError(
      modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
  EXPECT_NEAR(error, 0.0, Eps<T>(1e-5f, 1e-10));
}

TYPED_TEST(CenterOfMassErrorFunctionTest, ConstraintManagement) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto& parameterTransform = character.parameterTransform;

  CenterOfMassErrorFunctionT<T> errorFunction(skeleton, parameterTransform);

  EXPECT_EQ(errorFunction.getNumConstraints(), 0u);

  // Add a constraint
  CenterOfMassConstraintT<T> constraint;
  constraint.jointIndices = {0, 1};
  constraint.masses = {T(1), T(2)};
  constraint.target = Eigen::Vector3<T>::Zero();
  constraint.weight = T(1);
  errorFunction.addConstraint(constraint);
  EXPECT_EQ(errorFunction.getNumConstraints(), 1u);

  // Add another
  errorFunction.addConstraint(constraint);
  EXPECT_EQ(errorFunction.getNumConstraints(), 2u);

  // Clear
  errorFunction.clearConstraints();
  EXPECT_EQ(errorFunction.getNumConstraints(), 0u);
}

TYPED_TEST(CenterOfMassErrorFunctionTest, MultipleConstraints) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto& parameterTransform = character.parameterTransform;

  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;

  CenterOfMassErrorFunctionT<T> errorFunction(skeleton, parameterTransform);

  // Add multiple constraints with different joint sets
  for (int c = 0; c < 3; ++c) {
    CenterOfMassConstraintT<T> constraint;
    const size_t numJoints = std::min<size_t>(3, skeleton.joints.size());
    for (size_t j = 0; j < numJoints; ++j) {
      constraint.jointIndices.push_back(j);
      constraint.masses.push_back(T(1) + T(c) * T(0.3));
    }
    constraint.target = uniform<Vector3<T>>(-2, 2);
    constraint.weight = T(0.5) + T(c) * T(0.5);
    errorFunction.addConstraint(constraint);
  }

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character,
      Eps<T>(5e-2f, 5e-4),
      Eps<T>(1e-4f, 1e-8),
      false,
      false);
}

// Verify that the Jacobian is correctly computed even when the function value is zero.
// A zero function value (f = 0) does not imply a zero derivative (df/dq = 0).
TYPED_TEST(CenterOfMassErrorFunctionTest, JacobianWithZeroFunctionValue) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto& parameterTransform = character.parameterTransform;
  const ParameterTransformT<T> transform = parameterTransform.cast<T>();

  // Use non-zero parameters to create a non-trivial joint configuration
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  parameters(0) = T(0.5);
  parameters(1) = T(-0.3);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  CenterOfMassErrorFunctionT<T> errorFunction(skeleton, parameterTransform);

  // Compute actual CoM and use it as target, so f = 0
  const size_t numJoints = std::min<size_t>(4, skeleton.joints.size());
  CenterOfMassConstraintT<T> constraint;
  Eigen::Vector3<T> actualCoM = Eigen::Vector3<T>::Zero();
  T totalMass = T(0);
  for (size_t j = 0; j < numJoints; ++j) {
    constraint.jointIndices.push_back(j);
    const T mass = T(1) + T(j);
    constraint.masses.push_back(mass);
    actualCoM += mass * state.jointState[j].translation();
    totalMass += mass;
  }
  actualCoM /= totalMass;
  constraint.target = actualCoM;
  constraint.weight = T(1);
  errorFunction.addConstraint(constraint);

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
  EXPECT_NEAR(error, 0.0, Eps<T>(1e-5f, 1e-10));
  EXPECT_NEAR(residual.norm(), 0.0, Eps<T>(1e-3f, 1e-5));

  // The Jacobian matrix should be non-zero: even though f = 0, df/dq != 0
  EXPECT_GT(jacobian.norm(), 0.0);
}
