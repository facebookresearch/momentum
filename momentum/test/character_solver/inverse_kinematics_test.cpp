/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/solver/fwd.h"
#include "momentum/solver/gauss_newton_solver.h"
#include "momentum/solver/gradient_descent_solver.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct Momentum_InverseKinematicsTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_InverseKinematicsTest, Types);

TEST(Momentum_InverseKinematics, Definitions) {
  EXPECT_TRUE(kParametersPerJoint == 7);
}

template <typename T, typename SolverT, typename SolverOptionsT>
void testSimpleTargets(const SolverOptionsT& options) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;
  const ParameterTransformT<T> castedCharacterParameterTransform = transform.cast<T>();
  VectorX<T> parameters = VectorX<T>::Zero(transform.numAllModelParameters());
  VectorX<T> optimizedParameters = parameters;

  // create skeleton solvable
  SkeletonSolverFunctionT<T> solverFunction(&skeleton, &castedCharacterParameterTransform);

  // create marker solvable
  auto errorFunction = std::make_shared<PositionErrorFunctionT<T>>(skeleton, transform);

  // add to solvable
  solverFunction.addErrorFunction(errorFunction);

  // create solver
  GaussNewtonSolverT<T> solver(options, &solverFunction);

  // generate rest state and constraints
  SkeletonStateT<T> state(castedCharacterParameterTransform.apply(parameters), skeleton);
  std::vector<PositionDataT<T>> constraints;
  constraints.push_back(
      PositionDataT<T>(
          Vector3<T>::UnitY(), state.jointState[2].transformation * Vector3<T>::UnitY(), 2, 1.0));

  {
    SCOPED_TRACE("Checking restpose");
    Vector3<T> diff =
        state.jointState[2].transformation * constraints[0].offset - Vector3<T>(0.0, 3.0, 0.0);
    EXPECT_LE(diff.norm(), Eps<T>(1e-7f, 1e-15));
  }

  {
    SCOPED_TRACE("Checking optimizing rest pose");
    errorFunction->setConstraints(constraints);
    const T error = solver.solve(optimizedParameters);
    state.set(castedCharacterParameterTransform.apply(optimizedParameters), skeleton);
    EXPECT_LE(error, Eps<T>(1e-7f, 1e-15));
    EXPECT_LE((parameters - optimizedParameters).norm(), Eps<T>(1e-7f, 1e-15));
    Vector3<T> diff =
        state.jointState[2].transformation * constraints[0].offset - Vector3<T>(0.0, 3.0, 0.0);
    EXPECT_LE(diff.norm(), Eps<T>(1e-7f, 1e-15));
  }

  {
    SCOPED_TRACE("Checking optimizing target pose");
    for (size_t i = 0; i < 10; i++) {
      constraints[0].target = Vector3<T>::Random() * 3.0;
      errorFunction->setConstraints(constraints);
      const T error = solver.solve(optimizedParameters);
      state.set(castedCharacterParameterTransform.apply(optimizedParameters), skeleton);
      EXPECT_LE(error, Eps<T>(5e-7f, 1e-8));
      EXPECT_GE((parameters - optimizedParameters).norm(), Eps<T>(1e-7f, 1e-15));
      Vector3<T> diff =
          state.jointState[2].transformation * constraints[0].offset - constraints[0].target;
      EXPECT_LE(diff.norm(), Eps<T>(5e-5f, 1e-5));
    }
  }
}

TYPED_TEST(Momentum_InverseKinematicsTest, SimpleTargets) {
  using T = typename TestFixture::Type;

  {
    GradientDescentSolverOptions options;
    options.maxIterations = 24;
    options.minIterations = 6;
    options.threshold = 1.0;
    options.learningRate = 1.0;
    testSimpleTargets<T, GradientDescentSolverT<T>>(options);
  }

  {
    GaussNewtonSolverOptions options;
    options.maxIterations = 6;
    options.minIterations = 6;
    options.threshold = 1.0;
    options.regularization = 1e-7;
    options.useBlockJtJ = true;
    testSimpleTargets<T, GaussNewtonSolverT<T>>(options);
  }
}
