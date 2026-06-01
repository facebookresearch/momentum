/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "momentum/character_solver/distance_error_function.h"
#include "momentum/character_solver_simd/simd_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"

#include "momentum/math/random.h"
#include "momentum/simd/simd.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

TEST(Momentum_ErrorFunctions, SimdDistanceConstraintsRejectsJointIndexOutsideSkeleton) {
  const Character character = createTestCharacter(3);
  const Skeleton& skeleton = character.skeleton;
  SimdDistanceConstraints constraints(&skeleton);

  const size_t invalidJointIndex = skeleton.joints.size();
  ASSERT_LT(invalidJointIndex, SimdDistanceConstraints::kMaxJoints);

  const auto addInvalidConstraint = [&]() {
    constraints.addConstraint(invalidJointIndex, Vector3f::Zero(), Vector3f::UnitX(), 1.0f, 1.0f);
  };

  // MT_CHECK aborts via XR_CHECK in XR-logger builds, but throws std::runtime_error in the
  // non-XR backend.
#if defined(MOMENTUM_WITH_XR_LOGGER)
  EXPECT_DEATH_IF_SUPPORTED(
      addInvalidConstraint(), "jointIndex < static_cast<size_t>\\(numJoints\\)");
#else
  EXPECT_THROW(addInvalidConstraint(), std::runtime_error);
#endif
}

// Verify SIMD distance error function matches the scalar reference at multiple constraint counts
// that exercise both partial and full SIMD packets.
TEST(Momentum_ErrorFunctions, SimdDistanceFunctionIsSame) {
  Random<>::GetSingleton().setSeed(12345);
  const bool verbose = false;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdDistanceConstraints cl_simd(&skeleton);
  const std::initializer_list<size_t> constraintCounts = {
      0ul, 1ul, kSimdPacketSize - 1, kSimdPacketSize, kSimdPacketSize + 1, 100ul};

  for (size_t nConstraints : constraintCounts) {
    if (verbose) {
      fmt::print("nConstraints: {}\n", nConstraints);
    }
    const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

    std::vector<DistanceConstraintDataT<float>> cl;
    DistanceErrorFunctionT<float> errorFunction(skeleton, transform);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      DistanceConstraintDataT<float> c;
      c.parent = 2;
      c.offset = uniform<Vector3f>(0, 1);
      c.origin = uniform<Vector3f>(0, 1);
      // Bias target away from zero to avoid the singular-distance branch in both implementations.
      c.target = uniform<float>(0.5f, 1.5f);
      c.weight = uniform<float>(0, 1);
      cl.push_back(c);
      errorFunction.addConstraint(c);
    }
    timeJacobian(character, errorFunction, parameters, "DistanceErrorFunction");

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.origin, c.target, c.weight);
    }
    SimdDistanceErrorFunction errorFunction_simd(skeleton, transform);
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
    timeJacobian(character, errorFunction_simd, parameters, "SimdDistanceErrorFunction");
  }
}
