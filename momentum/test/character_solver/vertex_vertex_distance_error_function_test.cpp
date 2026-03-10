/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/vertex_vertex_distance_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexVertexDistanceError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  // create skeleton and reference values
  const size_t nConstraints = 10;

  // Test WITHOUT blend shapes:
  {
    SCOPED_TRACE("Without blend shapes");

    const Character character_orig = createTestCharacter();
    const Eigen::VectorX<T> refParams = 0.25 *
        uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
    const ModelParametersT<T> modelParams = refParams;
    const ParameterTransformT<T> transform = character_orig.parameterTransform.cast<T>();

    VertexVertexDistanceErrorFunctionT<T> errorFunction(character_orig);

    // Add some vertex-vertex distance constraints
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const int vertexIndex1 =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      int vertexIndex2 =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));

      // Ensure we don't constrain the same vertex to itself
      while (vertexIndex2 == vertexIndex1) {
        vertexIndex2 = uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      }

      const T weight = uniform<T>(0.1, 2.0);
      const T targetDistance = uniform<T>(0.1, 1.0);

      errorFunction.addConstraint(vertexIndex1, vertexIndex2, weight, targetDistance);
    }

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_orig,
        Eps<T>(5e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-14),
        true,
        false);
  }

  // Test WITH blend shapes:
  {
    SCOPED_TRACE("With blend shapes");

    const Character character_blend = withTestBlendShapes(createTestCharacter());
    const Eigen::VectorX<T> refParams = 0.25 *
        uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
    const ModelParametersT<T> modelParams = refParams;
    const ParameterTransformT<T> transform = character_blend.parameterTransform.cast<T>();

    VertexVertexDistanceErrorFunctionT<T> errorFunction(character_blend);

    // Add some vertex-vertex distance constraints
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const int vertexIndex1 =
          uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));
      int vertexIndex2 =
          uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));

      // Ensure we don't constrain the same vertex to itself
      while (vertexIndex2 == vertexIndex1) {
        vertexIndex2 = uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));
      }

      const T weight = uniform<T>(0.1, 2.0);
      const T targetDistance = uniform<T>(0.1, 1.0);

      errorFunction.addConstraint(vertexIndex1, vertexIndex2, weight, targetDistance);
    }

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_blend,
        Eps<T>(1e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-14),
        true,
        false);
  }

  // Test that error is zero when vertices are at target distance
  {
    SCOPED_TRACE("Zero error test");

    const Character character = createTestCharacter();
    const ModelParametersT<T> modelParams =
        ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters());
    const SkeletonStateT<T> skelState(
        character.parameterTransform.cast<T>().apply(modelParams), character.skeleton);

    VertexVertexDistanceErrorFunctionT<T> errorFunction(character);

    // Calculate actual distance between two vertices
    momentum::TransformationListT<T> ibp;
    for (const auto& js : character.inverseBindPose) {
      ibp.push_back(js.cast<T>());
    }
    const auto mesh = character.mesh->cast<T>();
    const auto& skin = *character.skinWeights;
    momentum::MeshT<T> posedMesh = character.mesh->cast<T>();
    applySSD(ibp, skin, mesh, skelState, posedMesh);

    const int vertexIndex1 = 0;
    const int vertexIndex2 = 1;
    const T actualDistance =
        (posedMesh.vertices[vertexIndex1] - posedMesh.vertices[vertexIndex2]).norm();

    // Add constraint with the actual distance as target
    errorFunction.addConstraint(vertexIndex1, vertexIndex2, T(1.0), actualDistance);

    const double error = errorFunction.getError(
        modelParams, skelState, MeshStateT<T>(modelParams, skelState, character));
    EXPECT_NEAR(error, 0.0, Eps<T>(1e-10f, 1e-15));
  }
}
