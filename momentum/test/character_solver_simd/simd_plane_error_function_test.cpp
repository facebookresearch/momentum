/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver_simd/simd_plane_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"

#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/simd/simd.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

namespace {

// Compares SimdPlaneErrorFunction against the scalar PlaneErrorFunction across
// the full constraint-count sweep. Parameterized over the half-plane flag so the same
// sweep covers both equality (point-on-plane) and inequality (point-above-plane)
// formulations and verifies the half-plane mask zeroes-out above-the-plane constraints
// identically to the scalar reference.
void runSimdPlaneEquivalenceSweep(bool above) {
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdPlaneConstraints cl_simd(&skeleton);
  const std::initializer_list<size_t> constraints = {
      0ul,
      1ul,
      kAvxPacketSize - 1,
      kAvxPacketSize,
      kAvxPacketSize + 1,
      kAvxPacketSize * 2,
      kSimdPacketSize - 1,
      kSimdPacketSize,
      kSimdPacketSize + 1,
      kSimdPacketSize * 2,
      100ul};
  for (size_t nConstraints : constraints) {
    const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

    std::vector<PlaneData> cl;
    PlaneErrorFunction errorFunction(skeleton, transform, above);
    errorFunction.setWeight(PlaneErrorFunction::kLegacyWeight);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      // Sample d in a range that straddles the joint origin so a healthy mix of
      // active/inactive constraints exists in half-plane mode.
      cl.emplace_back(
          uniform<Vector3f>(0, 1),
          uniform<Vector3f>(0, 1).normalized(),
          uniform<float>(-1, 1),
          2,
          uniform<float>(0, 1));
    }
    errorFunction.setConstraints(cl);

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.normal, c.d, c.weight);
    }
    SimdPlaneErrorFunction errorFunction_simd(skeleton, transform, above);
    errorFunction_simd.setConstraints(&cl_simd);

    // TODO: the result difference between SimdPlaneError and PlaneError is larger than that with
    // simdNormal. May need investigation.
    VALIDATE_IDENTICAL(
        float, errorFunction, errorFunction_simd, skeleton, transform, parameters, 0.1f, 5e-1f);
    timeJacobian(character, errorFunction_simd, parameters, "SimdPlaneErrorFunction");
  }
}

} // namespace

// TODO: Test for double once we have SIMD implementation for double
TEST(Momentum_ErrorFunctions, SimdPlaneFunctionIsSame) {
  runSimdPlaneEquivalenceSweep(/*above=*/false);
}

TEST(Momentum_ErrorFunctions, SimdPlaneFunctionHalfPlaneIsSame) {
  runSimdPlaneEquivalenceSweep(/*above=*/true);
}
