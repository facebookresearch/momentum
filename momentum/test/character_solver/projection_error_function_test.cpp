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
#include "momentum/character_solver/projection_error_function.h"
#include "momentum/math/random.h"
#include "momentum/math/utility.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, ProjectionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  ProjectionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform, 0.01);
  Eigen::Matrix4<T> projection = Eigen::Matrix4<T>::Identity();
  projection(2, 3) = 10;
  {
    SCOPED_TRACE("Projection Constraint Test");
    // Make a few projection constraints to ensure that at least one of them is active, since
    // projections are ignored behind the camera
    for (int i = 0; i < 5; ++i) {
      errorFunction.addConstraint(
          ProjectionConstraintDataT<T>{
              (projection + uniformAffine3<T>().matrix()).topRows(3),
              uniform<size_t>(0, 2),
              normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
              uniform<T>(0.1, 2.0),
              normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones())});
    }

    if constexpr (std::is_same_v<T, float>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          5e-2f);
    } else if constexpr (std::is_same_v<T, double>) {
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          character,
          5e-3,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    }

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), 0.0f, 1.0f);
      const momentum::SkeletonStateT<T> skelState(transform.apply(parameters), skeleton);
      ASSERT_GT(errorFunction.getError(parameters, skelState, MeshStateT<T>()), 0);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(1e-1f, 5e-3), Eps<T>(2e-6f, 1e-7));
    }
  }
}
