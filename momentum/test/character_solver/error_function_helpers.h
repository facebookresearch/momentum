/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "momentum/character/fwd.h"
#include "momentum/character/types.h"
#include "momentum/character_solver/fwd.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"

namespace momentum {

template <typename T>
struct Momentum_ErrorFunctionsTest : testing::Test {
  using Type = T;

  static constexpr uint32_t kTestSeed = 12345;

  void SetUp() override {
    Random<>::GetSingleton().setSeed(kTestSeed);
  }

  void TearDown() override {
    // Check that the seed hasn't been changed by other code during the test
    EXPECT_EQ(Random<>::GetSingleton().getSeed(), kTestSeed)
        << "Random seed was modified during test execution. Expected: " << kTestSeed
        << ", Actual: " << Random<>::GetSingleton().getSeed();
  }

  static constexpr T getEps() {
    return Eps<T>(1e-5f, 1e-9);
  }

  static constexpr T getNumThreshold() {
    return Eps<T>(1e-3f, 1e-3);
    // TODO: Decrease the tolerance more for double once the Joint class is templatized for the
    // scalar type
  }

  static constexpr T getJacThreshold() {
    return Eps<T>(1e-5f, 1e-6);
    // TODO: Decrease the tolerance more for double once the Joint class is templatized for the
    // scalar type
  }
};

#define TEST_GRADIENT_AND_JACOBIAN(T, ...) \
  testGradientAndJacobian<T>(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
void testGradientAndJacobian(
    const char* file,
    int line,
    SkeletonErrorFunctionT<T>* errorFunction,
    const ModelParametersT<T>& referenceParameters,
    const Character& character,
    const T& numThreshold = Momentum_ErrorFunctionsTest<T>::getNumThreshold(),
    const T& jacThreshold = Momentum_ErrorFunctionsTest<T>::getJacThreshold(),
    bool checkJacError = true,
    bool checkJacobian = true);

template <typename T>
constexpr T getErrorDiffThreshold() {
  return Eps<T>(1e-2f, 1e-2);
}

template <typename T>
constexpr T getGradDiffThreshold() {
  return Eps<T>(1e-3f, 1e-3);
}

#define VALIDATE_IDENTICAL(T, ...) validateIdentical<T>(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
void validateIdentical(
    const char* file,
    int line,
    SkeletonErrorFunctionT<T>& err1,
    SkeletonErrorFunctionT<T>& err2,
    const Skeleton& skeleton,
    const ParameterTransformT<T>& transform,
    const Eigen::VectorX<T>& parameters,
    const T& errorDiffThreshold = getErrorDiffThreshold<T>(),
    const T& gradDiffThreshold = getGradDiffThreshold<T>(),
    bool verbose = false);

void timeJacobian(
    const Character& character,
    SkeletonErrorFunction& errorFunction,
    const ModelParameters& modelParams,
    const char* type);

void timeError(
    const Character& character,
    SkeletonErrorFunction& errorFunction,
    const ModelParameters& modelParams,
    const char* type);

} // namespace momentum
