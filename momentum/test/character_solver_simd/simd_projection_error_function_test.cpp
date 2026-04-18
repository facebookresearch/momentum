/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/projection_error_function.h"
#include "momentum/character_solver_simd/simd_projection_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"

#include "momentum/math/random.h"
#include "momentum/simd/simd.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

TEST(Momentum_ErrorFunctions, SimdProjectionFunctionIsSame) {
  const bool verbose = false;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdProjectionConstraints cl_simd(&skeleton);
  const std::initializer_list<size_t> constraintCounts = {
      0ul,
      1ul,
      kAvxPacketSize - 1,
      kAvxPacketSize,
      kAvxPacketSize + 1,
      kSimdPacketSize - 1,
      kSimdPacketSize,
      kSimdPacketSize + 1,
      100ul};

  for (size_t nConstraints : constraintCounts) {
    if (verbose) {
      fmt::print("nConstraints: {}\n", nConstraints);
    }
    const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

    std::vector<ProjectionConstraintDataT<float>> cl;
    ProjectionErrorFunctionT<float> errorFunction(skeleton, transform);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      ProjectionConstraintDataT<float> c;
      c.parent = 2;
      c.offset = uniform<Vector3f>(0, 1);
      // Build a simple identity-like 3x4 projection (just shifts world point by camera offset)
      // with z biased far enough from 0 to avoid the near-clip skip.
      Eigen::Matrix<float, 3, 4> proj = Eigen::Matrix<float, 3, 4>::Zero();
      proj(0, 0) = 1.0f;
      proj(1, 1) = 1.0f;
      proj(2, 2) = 1.0f;
      proj(2, 3) = 5.0f; // Push z away from origin so the projected z is well above near clip.
      c.projection = proj;
      c.target = uniform<Vector2f>(-1, 1);
      c.weight = uniform<float>(0, 1);
      cl.push_back(c);
      errorFunction.addConstraint(c);
    }
    timeJacobian(character, errorFunction, parameters, "ProjectionErrorFunction");

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.projection, c.target, c.weight);
    }
    SimdProjectionErrorFunction errorFunction_simd(skeleton, transform);
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
    timeJacobian(character, errorFunction_simd, parameters, "SimdProjectionErrorFunction");
  }
}
