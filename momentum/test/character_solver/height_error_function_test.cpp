/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/height_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, HeightError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  SCOPED_TRACE("Height Error Function Test");

  // Test with blend shapes enabled (so we have shape and scale parameters to work with)
  const Character character = withTestBlendShapes(createTestCharacter());
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Create error function with target height (it automatically uses blend shape and scale
  // parameters)
  HeightErrorFunctionT<T> errorFunction(character, T(1.8), Eigen::Vector3<T>::UnitY(), 4);

  // Test 1: Validate gradients and Jacobians
  {
    SCOPED_TRACE("Test standard gradient/Jacobian validation");

    // Test with zero parameters
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-14),
        true,
        false);

    // Test with random parameters
    for (size_t i = 0; i < 5; i++) {
      ModelParametersT<T> parameters =
          0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(5e-2f, 1e-5),
          Eps<T>(1e-6f, 1e-14),
          true,
          false);
    }
  }

  // Test 2: Verify error does not depend on inactive parameters (pose parameters)
  {
    SCOPED_TRACE("Test that error does not depend on inactive parameters");

    // Get the set of active parameters (blend shapes + scale)
    const ParameterSet activeParams = transform.getBlendShapeParameters() |
        transform.getFaceExpressionParameters() | transform.getScalingParameters();

    // Create a base parameter set
    ModelParametersT<T> baseParameters =
        0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

    // Compute error with base parameters
    const SkeletonStateT<T> skelState(transform.apply(baseParameters), character.skeleton);
    const double baseError = errorFunction.getError(
        baseParameters, skelState, MeshStateT<T>(baseParameters, skelState, character));

    // Now perturb ONLY inactive parameters and verify error doesn't change
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> perturbedParameters = baseParameters;

      // Perturb only inactive parameters (pose parameters)
      for (Eigen::Index p = 0; p < perturbedParameters.size(); ++p) {
        if (!activeParams.test(p)) {
          perturbedParameters(p) = uniform<T>(-2.0, 2.0);
        }
      }

      const SkeletonStateT<T> perturbedSkelState(
          transform.apply(perturbedParameters), character.skeleton);
      const double perturbedError = errorFunction.getError(
          perturbedParameters,
          perturbedSkelState,
          MeshStateT<T>(perturbedParameters, perturbedSkelState, character));

      // Error should be identical (or very close) since we only changed inactive parameters
      EXPECT_NEAR(perturbedError, baseError, Eps<T>(1e-10f, 1e-15))
          << "Error changed when modifying inactive parameters. "
          << "Base error: " << baseError << ", Perturbed error: " << perturbedError;
    }
  }

  // Test 3: Verify gradients are zero for inactive parameters
  {
    SCOPED_TRACE("Test that gradients are zero for inactive parameters");

    // Get the set of active parameters (blend shapes + scale)
    const ParameterSet activeParams = transform.getBlendShapeParameters() |
        transform.getFaceExpressionParameters() | transform.getScalingParameters();

    ModelParametersT<T> parameters =
        0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

    const SkeletonStateT<T> skelState(transform.apply(parameters), character.skeleton);
    Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(transform.numAllModelParameters());

    errorFunction.getGradient(
        parameters, skelState, MeshStateT<T>(parameters, skelState, character), gradient);

    // Check that gradient is zero for all inactive parameters
    for (Eigen::Index p = 0; p < gradient.size(); ++p) {
      if (!activeParams.test(p)) {
        EXPECT_NEAR(gradient(p), T(0), Eps<T>(1e-10f, 1e-15))
            << "Gradient for inactive parameter " << p << "("
            << character.parameterTransform.name.at(p) << ") is non-zero: " << gradient(p);
      }
    }
  }
}
