/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <numeric>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/floor_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, FloorError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  SCOPED_TRACE("Floor Error Function Test");

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use a subset of the lower vertices as candidates. The test mesh has
  // 2 vertices per segment, 5 segments per bone, 3 bones = 30 vertices.
  // Use the first 10 vertices (covering the bottom bone region).
  std::vector<size_t> vertexIndices;
  for (size_t i = 0; i < 10; ++i) {
    vertexIndices.push_back(i);
  }

  // Use k=1 so only a single minimum vertex is selected, avoiding ties
  // at degenerate configurations where multiple vertices share the same
  // height (the hard argmin is not differentiable at tied heights).
  FloorErrorFunctionT<T> errorFunction(
      character, vertexIndices, T(0), Eigen::Vector3<T>::UnitY(), 1);

  // Test with random parameters (non-degenerate configurations where
  // a single vertex is clearly the minimum). Skip zero parameters since
  // the test mesh has symmetric vertex pairs at the same height, causing
  // the hard-argmin selection to be non-differentiable.
  for (size_t i = 0; i < 5; i++) {
    ModelParametersT<T> parameters =
        0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        parameters,
        character,
        Eps<T>(5e-2f, 1e-5),
        Eps<T>(1e-5f, 1e-14),
        true,
        false);
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FloorError_RespondsToAllParameters) {
  using T = typename TestFixture::Type;

  SCOPED_TRACE("Floor Error responds to pose parameters");

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  const size_t nVertices = character.mesh->vertices.size();
  std::vector<size_t> vertexIndices(nVertices);
  std::iota(vertexIndices.begin(), vertexIndices.end(), 0);

  FloorErrorFunctionT<T> errorFunction(
      character, vertexIndices, T(0), Eigen::Vector3<T>::UnitY(), 3);

  // Compute error at identity pose
  ModelParametersT<T> zeroParams = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  SkeletonStateT<T> skelState(transform.apply(zeroParams), character.skeleton);
  MeshStateT<T> meshState;
  meshState.update(zeroParams, skelState, character);
  const double zeroError = errorFunction.getError(zeroParams, skelState, meshState);

  // Perturb a pose parameter (root rotation about x) and verify error changes
  ModelParametersT<T> perturbedParams = zeroParams;
  perturbedParams(3) = T(0.5); // root_rx
  SkeletonStateT<T> perturbedState(transform.apply(perturbedParams), character.skeleton);
  MeshStateT<T> perturbedMesh;
  perturbedMesh.update(perturbedParams, perturbedState, character);
  const double perturbedError =
      errorFunction.getError(perturbedParams, perturbedState, perturbedMesh);

  EXPECT_NE(perturbedError, zeroError) << "Floor error should change when pose parameters change";

  // Verify gradient is non-zero for pose parameters
  Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(transform.numAllModelParameters());
  errorFunction.getGradient(perturbedParams, perturbedState, perturbedMesh, gradient);

  bool hasPoseGradient = false;
  for (Eigen::Index p = 3; p < 6; ++p) { // root rotation params
    if (std::abs(gradient(p)) > Eps<T>(1e-10f, 1e-15)) {
      hasPoseGradient = true;
      break;
    }
  }
  EXPECT_TRUE(hasPoseGradient) << "Gradient should be non-zero for pose parameters";
}
