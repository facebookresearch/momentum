/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver_simd/simd_orientation_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"

#include "momentum/math/random.h"
#include "momentum/simd/simd.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

TEST(Momentum_ErrorFunctions, SimdOrientationFunctionIsSame) {
  const bool verbose = false;

  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdOrientationConstraints cl_simd(&skeleton);
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

    std::vector<OrientationDataT<float>> cl;
    OrientationErrorFunctionT<float> errorFunction(skeleton, transform);
    errorFunction.setWeight(OrientationErrorFunctionT<float>::kLegacyWeight);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      // Random unit quaternions for offset and target.
      const Vector4f offsetVec = uniform<Vector4f>(-1, 1).normalized();
      const Vector4f targetVec = uniform<Vector4f>(-1, 1).normalized();
      const Eigen::Quaternionf offset(offsetVec.w(), offsetVec.x(), offsetVec.y(), offsetVec.z());
      const Eigen::Quaternionf target(targetVec.w(), targetVec.x(), targetVec.y(), targetVec.z());
      cl.emplace_back(offset, target, /*pIndex=*/2, uniform<float>(0, 1));
    }
    errorFunction.setConstraints(cl);
    timeJacobian(character, errorFunction, parameters, "OrientationErrorFunction");

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.target, c.weight);
    }
    SimdOrientationErrorFunction errorFunction_simd(skeleton, transform);
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
    timeJacobian(character, errorFunction_simd, parameters, "SimdOrientationErrorFunction");
  }
}
