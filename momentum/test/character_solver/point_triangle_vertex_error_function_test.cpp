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
#include "momentum/character_solver/error_function_types.h"
#include "momentum/character_solver/point_triangle_vertex_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, PointTriangleVertexErrorFunction_Position) {
  using T = typename TestFixture::Type;

  // The Normal and SymmetricNormal tests are currently failing, for very confusing
  // reasons.  We get large nonzero values in the Jacobian for the _rigid_ parameters
  // when computing them _numerically_, which suggests something funny is going on
  // that makes them depend on e.g. a rigid translation of the entire model -- but
  // only for Normal/SymmetricNormal constrains.  This seems especially weird to me
  // for global translations, because the skinning code doesn't even use the translation
  // part of the matrix when skinning normals.
  //
  // For now I'm only using Plane constraints anyway, because letting the Triangle
  // in the PointTriangle constraint determine what the normal is seems like the right
  // move.  At some point we should revisit, either to disable support for Normal constraints
  // altogether or figure out where this very weird issue comes from.

  Character character = createTestCharacter();
  character = withTestBlendShapes(character);

  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  SCOPED_TRACE("constraintType: Position");

  PointTriangleVertexErrorFunctionT<T> errorFunction(character, VertexConstraintType::Position);

  // Create a simple triangle and a point above it
  {
    Eigen::Vector3i triangleIndices(0, 1, 2);
    Eigen::Vector3<T> triangleBaryCoords(0.3, 0.3, 0.4);
    T depth = 0.1;
    T weight = 1.0;

    // Add a constraint
    errorFunction.addConstraint(3, triangleIndices, triangleBaryCoords, depth, weight);
  }

  {
    Eigen::Vector3i triangleIndices(10, 12, 14);
    Eigen::Vector3<T> triangleBaryCoords(0.2, 0.4, 0.4);
    T depth = 0.2;
    T weight = 0.5;

    // Add a constraint
    errorFunction.addConstraint(3, triangleIndices, triangleBaryCoords, depth, weight);
  }

  // Set up model parameters and skeleton state
  // ModelParametersT<T> modelParams =
  // ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);

  SkeletonStateT<T> state(transform.apply(modelParams), skeleton);

  // Test error calculation
  double error =
      errorFunction.getError(modelParams, state, MeshStateT<T>(modelParams, state, character));
  EXPECT_GT(error, 0);

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character,
      Eps<T>(5e-2f, 5e-4),
      Eps<T>(1e-6f, 1e-13),
      true,
      true);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PointTriangleVertexErrorFunction_Plane) {
  using T = typename TestFixture::Type;

  Character character = createTestCharacter();
  character = withTestBlendShapes(character);

  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  SCOPED_TRACE("constraintType: Plane");

  PointTriangleVertexErrorFunctionT<T> errorFunction(character, VertexConstraintType::Plane);

  // Create a simple triangle and a point above it
  {
    Eigen::Vector3i triangleIndices(0, 1, 2);
    Eigen::Vector3<T> triangleBaryCoords(0.3, 0.3, 0.4);
    T depth = 0.1;
    T weight = 1.0;

    // Add a constraint
    errorFunction.addConstraint(3, triangleIndices, triangleBaryCoords, depth, weight);
  }

  {
    Eigen::Vector3i triangleIndices(10, 12, 14);
    Eigen::Vector3<T> triangleBaryCoords(0.2, 0.4, 0.4);
    T depth = 0.2;
    T weight = 0.5;

    // Add a constraint
    errorFunction.addConstraint(3, triangleIndices, triangleBaryCoords, depth, weight);
  }

  const ModelParametersT<T> modelParams =
      0.25 * uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1, 1);

  SkeletonStateT<T> state(transform.apply(modelParams), skeleton);

  // Test error calculation
  double error =
      errorFunction.getError(modelParams, state, MeshStateT<T>(modelParams, state, character));
  EXPECT_GT(error, 0);

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character,
      Eps<T>(5e-2f, 5e-4),
      Eps<T>(1e-6f, 1e-13),
      true,
      true);
}
