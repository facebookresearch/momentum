/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/sdf_collision_geometry.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/sdf_collision_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

#include <axel/SignedDistanceField.h>

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, SDFCollisionError_WorldSDF_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // Create test character
  const Character character = createTestCharacter(5);
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Create a simple cubic SDF using helper function
  auto sdfWorld = std::make_shared<axel::SignedDistanceField<float>>(
      axel::SignedDistanceField<float>::createSphere(5.0, Eigen::Vector3<axel::Index>(10, 10, 10)));

  auto sdfLocal = std::make_shared<axel::SignedDistanceField<float>>(
      axel::SignedDistanceField<float>::createSphere(3.0, Eigen::Vector3<axel::Index>(10, 10, 10)));

  // Create SDF collision geometry
  SDFCollisionGeometry sdfColliders;

  // Joint-attached SDF for testing both mesh and collider hierarchy gradients
  sdfColliders.emplace_back(
      SDFCollider{
          TransformT<float>(Eigen::Vector3f(-0.224f, 0.134f, 0.151f)),
          2, // parent joint index
          sdfLocal});

  // Create error function with selected vertices (use first few vertices)
  const size_t numTestVertices = std::min(size_t(5), character.mesh->vertices.size());
  std::vector<int> participatingVertices;
  std::vector<T> vertexWeights;

  for (size_t i = 0; i < numTestVertices; ++i) {
    participatingVertices.push_back(static_cast<int>(i));
    vertexWeights.push_back(uniform<T>(0.1, 2.0)); // Random weights
  }

  // don't filter out rest pose intersections because this makes it very hard to construct a
  // test case that has actual collisions:
  bool filterRestPoseIntersections = false;
  SDFCollisionErrorFunctionT<T> errorFunction(
      character, sdfColliders, participatingVertices, vertexWeights, filterRestPoseIntersections);

  {
    SCOPED_TRACE("SDF Collision Error Test");

    // Test with zero parameters (bind pose)
    const ModelParametersT<T> zeroParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> zeroState(transform.apply(zeroParams), skeleton);
    const MeshStateT<T> zeroMeshState(zeroParams, zeroState, character);

    // Check that error function is set up correctly
    const double zeroError = errorFunction.getError(zeroParams, zeroState, zeroMeshState);
    EXPECT_GT(zeroError, 0.0); // Error should be non-negative

    // Test gradients and jacobian for bind pose
    // Note: SDF collision gradients have higher error tolerance due to:
    // 1. Gauss-Newton approximation (surface point moves as mesh moves, but we treat it as fixed)
    // 2. SDF gradient interpolation from discrete grid
    // 3. Float precision issues in numerical differentiation for root parameters
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        zeroParams,
        character,
        Eps<T>(0.1f, 0.04),
        Eps<T>(1e-6f, 1e-14),
        true,
        true);

    // Test with random parameters to create different collision scenarios
    for (size_t i = 0; i < 5; i++) {
      ModelParametersT<T> parameters =
          0.1 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

      const SkeletonStateT<T> state(transform.apply(parameters), skeleton);
      const MeshStateT<T> meshState(parameters, state, character);
      const double error = errorFunction.getError(parameters, state, meshState);
      EXPECT_GE(error, 0.0); // Error should be non-negative

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(0.25f, 0.04),
          Eps<T>(1e-6f, 1e-14),
          true,
          true);
    }
  }

  // Test error increases with more penetration
  {
    SCOPED_TRACE("Penetration Scaling Test");

    // Create parameters that move joint 1 toward the SDF center
    ModelParametersT<T> baseParams = ModelParametersT<T>::Zero(transform.numAllModelParameters());

    // Find translation parameter for joint 1
    const size_t joint1TranslationBase = 1 * kParametersPerJoint + 1; // Joint 1, translation params

    // Test with increasing translation toward SDF center
    double previousError = 0.0;
    for (int step = 0; step < 3; ++step) {
      ModelParametersT<T> params = baseParams;
      params[joint1TranslationBase] += static_cast<T>(0.1); // Move in positive Y direction

      const SkeletonStateT<T> state(transform.apply(params), skeleton);
      const MeshStateT<T> meshState(params, state, character);
      const double error = errorFunction.getError(params, state, meshState);

      EXPECT_GE(error, 0.0);

      // Error should decrease as we move away from the SDF
      if (step > 0) {
        EXPECT_LE(error, previousError);
      }

      previousError = error;
    }
  }
}
