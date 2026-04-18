/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/joint_to_joint_distance_error_function.h"
#include "momentum/character_solver_simd/simd_joint_to_joint_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"

#include "momentum/math/random.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

TEST(Momentum_ErrorFunctions, SimdJointToJointDistanceFunctionIsSame) {
  const bool verbose = false;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdJointToJointDistanceConstraints cl_simd;
  const std::initializer_list<size_t> constraintCounts = {0ul, 1ul, 5ul, 17ul, 64ul};

  for (size_t nConstraints : constraintCounts) {
    if (verbose) {
      fmt::print("nConstraints: {}\n", nConstraints);
    }
    const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

    JointToJointDistanceErrorFunctionT<float> errorFunction(skeleton, transform);
    cl_simd.clearConstraints();
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      // Use distinct joints for joint1 and joint2 (test character has 3 joints).
      const size_t j1 = 1;
      const size_t j2 = 2;
      const auto off1 = uniform<Vector3f>(0, 1);
      const auto off2 = uniform<Vector3f>(0, 1);
      const auto target = uniform<float>(0.5f, 1.5f);
      const auto weight = uniform<float>(0, 1);
      errorFunction.addConstraint(j1, off1, j2, off2, target, weight);
      cl_simd.addConstraint(j1, off1, j2, off2, target, weight);
    }
    timeJacobian(character, errorFunction, parameters, "JointToJointDistanceErrorFunction");

    SimdJointToJointDistanceErrorFunction errorFunction_simd(skeleton, transform);
    errorFunction_simd.setConstraints(&cl_simd);

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
    timeJacobian(
        character, errorFunction_simd, parameters, "SimdJointToJointDistanceErrorFunction");
  }
}
