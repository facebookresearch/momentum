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
#include "momentum/character_solver/vertex_sdf_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

#include <axel/SignedDistanceField.h>

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexSDFError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // Create test character
  const Character character = createTestCharacter(5);
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Create a sphere SDF attached to joint 2
  auto sdfLocal = std::make_shared<axel::SignedDistanceField<float>>(
      axel::SignedDistanceField<float>::createSphere(3.0, Eigen::Vector3<axel::Index>(10, 10, 10)));

  SDFCollider sdfCollider{
      TransformT<float>(Eigen::Vector3f(-0.224f, 0.134f, 0.151f)),
      2, // parent joint index
      sdfLocal};

  VertexSDFErrorFunctionT<T> errorFunction(character, sdfCollider);

  // Add constraints with various target distances
  const size_t numTestVertices = std::min(size_t(5), character.mesh->vertices.size());
  for (size_t i = 0; i < numTestVertices; ++i) {
    VertexSDFConstraintT<T> constraint;
    constraint.vertexIndex = static_cast<int>(i);
    constraint.weight = uniform<T>(0.1, 2.0);

    // Vary target distances: surface (0), outside (1), inside (-0.5)
    if (i % 3 == 0) {
      constraint.targetDistance = T(0);
    } else if (i % 3 == 1) {
      constraint.targetDistance = T(1);
    } else {
      constraint.targetDistance = T(-0.5);
    }

    errorFunction.addConstraint(constraint);
  }

  EXPECT_EQ(errorFunction.getNumConstraints(), numTestVertices);

  {
    SCOPED_TRACE("Vertex SDF Error Test - Zero Parameters");

    const ModelParametersT<T> zeroParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> zeroState(transform.apply(zeroParams), skeleton);
    const MeshStateT<T> zeroMeshState(zeroParams, zeroState, character);

    const double zeroError = errorFunction.getError(zeroParams, zeroState, zeroMeshState);
    EXPECT_GE(zeroError, 0.0);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        zeroParams,
        character,
        Eps<T>(0.1f, 0.04),
        Eps<T>(1e-6f, 1e-14),
        true,
        true);
  }

  {
    SCOPED_TRACE("Vertex SDF Error Test - Random Parameters");

    for (size_t i = 0; i < 5; i++) {
      ModelParametersT<T> parameters =
          0.1 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);

      const SkeletonStateT<T> state(transform.apply(parameters), skeleton);
      const MeshStateT<T> meshState(parameters, state, character);
      const double error = errorFunction.getError(parameters, state, meshState);
      EXPECT_GE(error, 0.0);

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          character,
          Eps<T>(0.3f, 0.04),
          Eps<T>(1e-6f, 1e-14),
          true,
          true);
    }
  }

  // Test that clearing constraints results in zero error
  {
    SCOPED_TRACE("Clear Constraints Test");

    errorFunction.clearConstraints();
    EXPECT_EQ(errorFunction.getNumConstraints(), 0);

    const ModelParametersT<T> zeroParams =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> zeroState(transform.apply(zeroParams), skeleton);
    const MeshStateT<T> zeroMeshState(zeroParams, zeroState, character);

    const double error = errorFunction.getError(zeroParams, zeroState, zeroMeshState);
    EXPECT_EQ(error, 0.0);
  }
}
