/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character_solver/simd_collision_error_function.h"
#include "momentum/character_solver/simd_normal_error_function.h"
#include "momentum/character_solver/simd_plane_error_function.h"
#include "momentum/character_solver/simd_position_error_function.h"

#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/collision_error_function_stateless.h"
#include "momentum/character_solver/normal_error_function.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver/position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"

#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/math/utility.h"

#include "momentum/common/log.h"

#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

// TODO: Test for double once we have SIMD implementation for double
TEST(Momentum_ErrorFunctions, SimdNormalFunctionIsSame) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdNormalConstraints cl_simd(&skeleton);
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

    std::vector<NormalData> cl;
    NormalErrorFunction errorFunction(skeleton, transform);
    errorFunction.setWeight(NormalErrorFunction::kLegacyWeight);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      cl.emplace_back(
          uniform<Vector3f>(0, 1),
          uniform<Vector3f>(0, 1),
          uniform<Vector3f>(0, 1),
          2,
          uniform<float>(0, 1));
    }
    errorFunction.setConstraints(cl);
    timeJacobian(character, errorFunction, parameters, "NormalErrorFunction");

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.localPoint, c.localNormal, c.globalPoint, c.weight);
    }
    SimdNormalErrorFunction errorFunction_simd(skeleton, transform);
    errorFunction_simd.setConstraints(&cl_simd);

    VALIDATE_IDENTICAL(float, errorFunction, errorFunction_simd, skeleton, transform, parameters);
    timeJacobian(character, errorFunction_simd, parameters, "SimdNormalErrorFunction");

#ifdef MOMENTUM_ENABLE_AVX
    // Intel-specific version:
    SimdNormalErrorFunctionAVX errorFunction_simd_avx(skeleton, transform);
    errorFunction_simd_avx.setConstraints(&cl_simd);
    VALIDATE_IDENTICAL(
        float, errorFunction, errorFunction_simd_avx, skeleton, transform, parameters);
    timeJacobian(character, errorFunction_simd_avx, parameters, "SimdNormalErrorFunctionAVX");
#endif
  }

  MT_LOGI("done.");
}

// TODO: Test for double once we have SIMD implementation for double
TEST(Momentum_ErrorFunctions, SimdPositionFunctionIsSame) {
  const bool verbose = false;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  SimdPositionConstraints cl_simd(&skeleton);
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
    if (verbose) {
      fmt::print("nConstraints: {}\n", nConstraints);
    }
    const auto parameters = uniform<VectorXf>(transform.numAllModelParameters(), -0.5, 0.5);

    std::vector<PositionData> cl;
    PositionErrorFunction errorFunction(skeleton, transform);
    errorFunction.setWeight(PositionErrorFunction::kLegacyWeight);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      cl.emplace_back(uniform<Vector3f>(0, 1), uniform<Vector3f>(0, 1), 2, uniform<float>(0, 1));
    }
    errorFunction.setConstraints(cl);
    timeJacobian(character, errorFunction, parameters, "PositionErrorFunction");

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.target, c.weight);
    }
    SimdPositionErrorFunction errorFunction_simd(skeleton, transform);
    errorFunction_simd.setConstraints(&cl_simd);

    // TODO: the result difference between SimdPositionError and PositionError is larger than that
    // with simdNormal. May need investigation.
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
    timeJacobian(character, errorFunction_simd, parameters, "SimdPositionErrorFunction");

#ifdef MOMENTUM_ENABLE_AVX
    SimdPositionErrorFunctionAVX errorFunction_simd_avx(skeleton, transform);
    errorFunction_simd_avx.setConstraints(&cl_simd);
    VALIDATE_IDENTICAL(
        float, errorFunction, errorFunction_simd_avx, skeleton, transform, parameters, 0.1f, 0.1f);
    timeJacobian(character, errorFunction_simd_avx, parameters, "SimdPositionErrorFunctionAVX");
#endif
  }
}

// TODO: Test for double once we have SIMD implementation for double
TEST(Momentum_ErrorFunctions, SimdPlaneFunctionIsSame) {
  // create skeleton and reference values
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
    PlaneErrorFunction errorFunction(skeleton, transform);
    errorFunction.setWeight(PlaneErrorFunction::kLegacyWeight);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      cl.emplace_back(
          uniform<Vector3f>(0, 1),
          uniform<Vector3f>(0, 1).normalized(),
          uniform<float>(0, 1),
          2,
          uniform<float>(0, 1));
    }
    errorFunction.setConstraints(cl);

    cl_simd.clearConstraints();
    for (const auto& c : cl) {
      cl_simd.addConstraint(c.parent, c.offset, c.normal, c.d, c.weight);
    }
    SimdPlaneErrorFunction errorFunction_simd(skeleton, transform);
    errorFunction_simd.setConstraints(&cl_simd);

    // TODO: the result difference between SimdPlaneError and PlaneError is larger than that with
    // simdNormal. May need investigation.
    VALIDATE_IDENTICAL(
        float, errorFunction, errorFunction_simd, skeleton, transform, parameters, 0.1f, 5e-1f);
    timeJacobian(character, errorFunction_simd, parameters, "SimdPlaneErrorFunction");

#ifdef MOMENTUM_ENABLE_AVX
    SimdPlaneErrorFunctionAVX errorFunction_simd_avx(skeleton, transform);
    errorFunction_simd_avx.setConstraints(&cl_simd);
    VALIDATE_IDENTICAL(
        float, errorFunction, errorFunction_simd_avx, skeleton, transform, parameters, 0.1f, 5e-1f);
    timeJacobian(character, errorFunction_simd_avx, parameters, "SimdPlaneErrorFunctionAVX");
#endif
  }
}

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
