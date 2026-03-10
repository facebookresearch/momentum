/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/collision_error_function_stateless.h"
#include "momentum/character_solver/simd_collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"

#include "momentum/math/constants.h"

#include "momentum/common/log.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

// Verify that the SIMD version of the collision error function is the same as the non-SIMD version.
TYPED_TEST(Momentum_ErrorFunctionsTest, SimdCollisionErrorFunctionIsSame) {
  using T = typename TestFixture::Type;

  const size_t nJoints = 6;

  // create skeleton and reference values
  const Character character = createTestCharacter(nJoints);
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  CollisionErrorFunctionT<T> errf_base(character);
  CollisionErrorFunctionStatelessT<T> errf_stateless(character);
  // TODO: Fix this test for SIMD when MOMENTUM_ENABLE_SIMD=OFF
#ifdef MOMENTUM_ENABLE_SIMD
  SimdCollisionErrorFunctionT<T> errf_simd(character);
#endif

  const size_t nBendTests = 8;
  for (size_t iBendAmount = 4; iBendAmount < nBendTests; ++iBendAmount) {
    const T bendAmount = iBendAmount * (pi<T>() / nBendTests);

    auto mp = ModelParametersT<T>::Zero(transform.numAllModelParameters());
    for (auto jParam = 0; jParam < transform.numAllModelParameters(); ++jParam) {
      auto name = transform.name[jParam];
      if (name.find("root") == std::string::npos && name.find("_rx") != std::string::npos) {
        mp[jParam] = bendAmount;
      }
    }

    const JointParametersT<T> jp = transform.apply(mp);
    const SkeletonStateT<T> skelState(jp, skeleton);

    const double err_base = errf_base.getError(mp, skelState, MeshStateT<T>());
    const double err_stateless = errf_stateless.getError(mp, skelState, MeshStateT<T>());
#ifdef MOMENTUM_ENABLE_SIMD
    const double err_simd = errf_simd.getError(mp, skelState, MeshStateT<T>());
#endif
    ASSERT_NEAR(err_base, err_stateless, 0.0001 * err_base);
#ifdef MOMENTUM_ENABLE_SIMD
    ASSERT_NEAR(err_base, err_simd, 0.0001 * err_base);
#endif
    VALIDATE_IDENTICAL(T, errf_base, errf_stateless, skeleton, transform, mp.v);
#ifdef MOMENTUM_ENABLE_SIMD
    VALIDATE_IDENTICAL(T, errf_base, errf_simd, skeleton, transform, mp.v);
#endif
  }

  MT_LOGI("done.");
}

// Verify that the SIMD version of the collision error function is at least as fast as the
// multithreaded versions. Disabled by default because it's too slow to run in CI.
TEST(Momentum_ErrorFunctions, DISABLED_SimdCollisionErrorFunctionIsFaster) {
  const size_t nJoints = 50;

  // create skeleton and reference values
  const Character character = createTestCharacter(nJoints);
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdCollisionErrorFunction errf_simd(character);
  CollisionErrorFunction errf_base(character);

  const size_t nBendTests = 8;
  for (size_t iTest = 0; iTest < nBendTests; ++iTest) {
    // The higher the total bend, the more joints will actually be in contact,
    // and we'll converge to the limit of how fast the actual Jacobian computation is.
    const float totalBendAmount = iTest * 4.0f * pi() / nBendTests;
    const float bendAmountPerJoint = totalBendAmount / nJoints;
    MT_LOGI("Total bend: {} degrees.\n", totalBendAmount * 180 / pi());
    // SCOPED_TRACE(::testing::Message() << "Total bend: " << totalBendAmount);

    auto mp = ModelParameters::Zero(transform.numAllModelParameters());
    for (auto jParam = 0; jParam < transform.numAllModelParameters(); ++jParam) {
      auto name = transform.name[jParam];
      if (name.find("root") == std::string::npos && name.find("_rx") != std::string::npos) {
        mp[jParam] = bendAmountPerJoint;
      }
    }

    const JointParameters jp = transform.apply(mp);
    const SkeletonState skelState(jp, skeleton);

    const double err_base = errf_base.getError(mp, skelState, MeshState());
    const double err_simd = errf_simd.getError(mp, skelState, MeshState());
    ASSERT_NEAR(err_simd, err_base, 0.0001 * err_base);

    timeJacobian(character, errf_simd, mp, "Simd");
    timeJacobian(character, errf_base, mp, "Base");

    timeError(character, errf_simd, mp, "Simd");
    timeError(character, errf_base, mp, "Base");
  }
}
