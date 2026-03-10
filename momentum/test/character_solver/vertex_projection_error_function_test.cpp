/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/vertex_projection_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexProjectionErrorFunction_WithoutBlendShapes) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  const size_t nConstraints = 10;

  const Character character_orig = createTestCharacter();
  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;
  const ModelParametersT<T> modelParamsTarget = refParams +
      0.05 * uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
  const Skeleton& skeleton = character_orig.skeleton;
  const ParameterTransformT<T> transform = character_orig.parameterTransform.cast<T>();
  const momentum::SkeletonStateT<T> skelState(transform.apply(modelParamsTarget), skeleton);
  momentum::TransformationListT<T> ibp;
  for (const auto& js : character_orig.inverseBindPose) {
    ibp.push_back(js.cast<T>());
  }
  const auto mesh = character_orig.mesh->cast<T>();
  const auto& skin = *character_orig.skinWeights;
  momentum::MeshT<T> targetMesh = character_orig.mesh->cast<T>();
  applySSD(ibp, skin, mesh, skelState, targetMesh);

  Eigen::Matrix<T, 3, 4> projection = Eigen::Matrix<T, 3, 4>::Identity();
  projection(0, 0) = 10.0;
  projection(1, 1) = 10.0;
  projection(2, 3) = 10.0;

  const T errorTol = Eps<T>(5e-2f, 1e-5);

  VertexProjectionErrorFunctionT<T> errorFunction(
      character_orig, character_orig.parameterTransform);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    const int index = uniform<int>(0, character_orig.mesh->vertices.size() - 1);
    const Eigen::Vector3<T> target = projection * targetMesh.vertices[index].homogeneous();
    const Eigen::Vector2<T> target2d = target.hnormalized() + uniform<Vector2<T>>(-1, 1) * 0.1;
    errorFunction.addConstraint(
        VertexProjectionDataT<T>(
            static_cast<size_t>(index), target2d, projection, uniform<T>(0, 1e-2)));
  }

  TEST_GRADIENT_AND_JACOBIAN(
      T, &errorFunction, modelParams, character_orig, errorTol, Eps<T>(1e-6f, 1e-14), true, false);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexProjectionErrorFunction_WithBlendShapes) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  const size_t nConstraints = 10;

  const Character character_blend = withTestBlendShapes(createTestCharacter());
  const Eigen::VectorX<T> refParams =
      0.25 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
  const ModelParametersT<T> modelParams = refParams;
  const ModelParametersT<T> modelParamsTarget = refParams +
      0.05 * uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
  const Skeleton& skeleton = character_blend.skeleton;
  const ParameterTransformT<T> transform = character_blend.parameterTransform.cast<T>();
  const momentum::SkeletonStateT<T> skelState(transform.apply(modelParamsTarget), skeleton);
  momentum::TransformationListT<T> ibp;
  for (const auto& js : character_blend.inverseBindPose) {
    ibp.push_back(js.cast<T>());
  }
  const auto mesh = character_blend.mesh->cast<T>();
  const auto& skin = *character_blend.skinWeights;
  momentum::MeshT<T> targetMesh = character_blend.mesh->cast<T>();
  applySSD(ibp, skin, mesh, skelState, targetMesh);

  Eigen::Matrix<T, 3, 4> projection = Eigen::Matrix<T, 3, 4>::Identity();
  projection(2, 3) = 10;

  // It's trickier to test Normal and SymmetricNormal constraints in the blend shape case because
  // the mesh normals are recomputed after blend shapes are applied (this is the only sensible
  // thing to do since the blend shapes can drastically change the shape) and thus the normals
  // depend on the blend shapes in a very complicated way that we aren't currently trying to
  // model.
  VertexProjectionErrorFunctionT<T> errorFunction(
      character_blend, character_blend.parameterTransform);
  for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
    const int index = uniform<int>(0, character_blend.mesh->vertices.size() - 1);
    const Eigen::Vector3<T> target = projection * targetMesh.vertices[index].homogeneous();
    const Eigen::Vector2<T> target2d = target.hnormalized();
    errorFunction.addConstraint(
        VertexProjectionDataT<T>(
            static_cast<size_t>(index), target2d, projection, uniform<T>(0, 1e-2)));
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
