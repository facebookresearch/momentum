/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/state_error_function.h"
#include "momentum/character_solver_simd/simd_state_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"

#include "momentum/math/random.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

TEST(Momentum_ErrorFunctions, SimdStateFunctionIsSame) {
  const bool verbose = false;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

  // Build a target skeleton state from a different parameter set so the residuals are non-trivial.
  const auto targetParameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);
  SkeletonState targetState(transform.apply(targetParameters), skeleton);

  StateErrorFunctionT<float> errorFunction(
      skeleton, transform, RotationErrorType::RotationMatrixDifference);
  errorFunction.setTargetState(targetState);

  SimdStateErrorFunction errorFunction_simd(skeleton, transform);
  errorFunction_simd.setTargetState(targetState);

  VALIDATE_IDENTICAL(
      float,
      errorFunction,
      errorFunction_simd,
      skeleton,
      transform,
      parameters,
      0.1f,
      0.1f,
      verbose);
  timeJacobian(character, errorFunction, parameters, "StateErrorFunction");
  timeJacobian(character, errorFunction_simd, parameters, "SimdStateErrorFunction");
}
