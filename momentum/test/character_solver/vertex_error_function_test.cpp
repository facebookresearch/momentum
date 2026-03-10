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
#include "momentum/character/sdf_collision_geometry.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_types.h"
#include "momentum/character_solver/point_triangle_vertex_error_function.h"
#include "momentum/character_solver/vertex_normal_error_function.h"
#include "momentum/character_solver/vertex_plane_error_function.h"
#include "momentum/character_solver/vertex_position_error_function.h"
#include "momentum/character_solver/vertex_projection_error_function.h"
#include "momentum/character_solver/vertex_sdf_error_function.h"
#include "momentum/character_solver/vertex_vertex_distance_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

#include <axel/SignedDistanceField.h>

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

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexErrorFunctionSerial) {
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
    const ModelParametersT<T> modelParamsTarget = refParams +
        0.05 *
            uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
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

    // Position constraints
    {
      SCOPED_TRACE("Constraint type: Position");
      VertexPositionErrorFunctionT<T> errorFunction(
          character_orig, character_orig.parameterTransform);
      errorFunction.setMaxThreads(0);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-2)));
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

    // Plane constraints
    {
      SCOPED_TRACE("Constraint type: Plane");
      VertexPlaneErrorFunctionT<T> errorFunction(character_orig, character_orig.parameterTransform);
      errorFunction.setMaxThreads(0);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexPlaneDataT<T>(
                index,
                targetMesh.normals[index],
                targetMesh.vertices[index],
                false,
                uniform<T>(0, 1e-2)));
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

    // Normal constraints
    {
      SCOPED_TRACE("Constraint type: Normal");
      VertexNormalErrorFunctionT<T> errorFunction(
          character_orig, character_orig.parameterTransform, T(1), T(0));
      errorFunction.setMaxThreads(0);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexNormalDataT<T>(
                index, targetMesh.vertices[index], targetMesh.normals[index], uniform<T>(0, 1e-2)));
      }
      // TODO Normal constraints have a much higher epsilon than I'd prefer to see;
      // it would be good to dig into this.
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_orig,
          Eps<T>(5e-2f, 5e-2),
          Eps<T>(1e-6f, 1e-14),
          true,
          false);
    }

    // SymmetricNormal constraints
    {
      SCOPED_TRACE("Constraint type: SymmetricNormal");
      VertexNormalErrorFunctionT<T> errorFunction(
          character_orig, character_orig.parameterTransform, T(0.5), T(0.5));
      errorFunction.setMaxThreads(0);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexNormalDataT<T>(
                index, targetMesh.vertices[index], targetMesh.normals[index], uniform<T>(0, 1e-2)));
      }
      // TODO Normal constraints have a much higher epsilon than I'd prefer to see;
      // it would be good to dig into this.
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_orig,
          Eps<T>(5e-2f, 5e-2),
          Eps<T>(1e-6f, 1e-14),
          true,
          false);
    }
  }

  // Test WITH blend shapes:
  {
    SCOPED_TRACE("With blend shapes");

    const Character character_blend = withTestBlendShapes(createTestCharacter());
    const Eigen::VectorX<T> refParams = 0.25 *
        uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
    const ModelParametersT<T> modelParams = refParams;
    const ModelParametersT<T> modelParamsTarget = refParams +
        0.05 *
            uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
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

    // It's trickier to test Normal and SymmetricNormal constraints in the blend shape case because
    // the mesh normals are recomputed after blend shapes are applied (this is the only sensible
    // thing to do since the blend shapes can drastically change the shape) and thus the normals
    // depend on the blend shapes in a very complicated way that we aren't currently trying to
    // model.
    // Position constraints
    {
      SCOPED_TRACE("Constraint type: Position");
      VertexPositionErrorFunctionT<T> errorFunction(
          character_blend, character_blend.parameterTransform);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-2)));
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

    // Plane constraints
    {
      SCOPED_TRACE("Constraint type: Plane");
      VertexPlaneErrorFunctionT<T> errorFunction(
          character_blend, character_blend.parameterTransform);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        const size_t index =
            uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size() - 1));
        errorFunction.addConstraint(
            VertexPlaneDataT<T>(
                index,
                targetMesh.normals[index],
                targetMesh.vertices[index],
                false,
                uniform<T>(0, 1e-2)));
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
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexErrorFunctionParallel) {
  using T = typename TestFixture::Type;
  SCOPED_TRACE(fmt::format("ScalarType: {}", typeid(T).name()));

  // create skeleton and reference values

  const size_t nConstraints = 1000;

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

  // Position constraints
  {
    SCOPED_TRACE("Constraint type: Position");
    VertexPositionErrorFunctionT<T> errorFunction(
        character_orig, character_orig.parameterTransform);
    errorFunction.setMaxThreads(100000);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const size_t index =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      errorFunction.addConstraint(
          VertexPositionDataT<T>(index, targetMesh.vertices[index], uniform<T>(0, 1e-4)));
    }
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_orig,
        Eps<T>(5e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-15),
        true,
        false);
  }

  // Plane constraints
  {
    SCOPED_TRACE("Constraint type: Plane");
    VertexPlaneErrorFunctionT<T> errorFunction(character_orig, character_orig.parameterTransform);
    errorFunction.setMaxThreads(100000);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const size_t index =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      errorFunction.addConstraint(
          VertexPlaneDataT<T>(
              index,
              targetMesh.normals[index],
              targetMesh.vertices[index],
              false,
              uniform<T>(0, 1e-4)));
    }
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_orig,
        Eps<T>(5e-2f, 1e-5),
        Eps<T>(1e-6f, 1e-15),
        true,
        false);
  }

  // Normal constraints
  {
    SCOPED_TRACE("Constraint type: Normal");
    VertexNormalErrorFunctionT<T> errorFunction(
        character_orig, character_orig.parameterTransform, T(1), T(0));
    errorFunction.setMaxThreads(100000);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const size_t index =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      errorFunction.addConstraint(
          VertexNormalDataT<T>(
              index, targetMesh.vertices[index], targetMesh.normals[index], uniform<T>(0, 1e-4)));
    }
    // TODO Normal constraints have a much higher epsilon than I'd prefer to see;
    // it would be good to dig into this.
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_orig,
        Eps<T>(5e-2f, 5e-2),
        Eps<T>(1e-6f, 1e-15),
        true,
        false);
  }

  // SymmetricNormal constraints
  {
    SCOPED_TRACE("Constraint type: SymmetricNormal");
    VertexNormalErrorFunctionT<T> errorFunction(
        character_orig, character_orig.parameterTransform, T(0.5), T(0.5));
    errorFunction.setMaxThreads(100000);
    for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
      const size_t index =
          uniform<int>(0, static_cast<int>(character_orig.mesh->vertices.size() - 1));
      errorFunction.addConstraint(
          VertexNormalDataT<T>(
              index, targetMesh.vertices[index], targetMesh.normals[index], uniform<T>(0, 1e-4)));
    }
    // TODO Normal constraints have a much higher epsilon than I'd prefer to see;
    // it would be good to dig into this.
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        modelParams,
        character_orig,
        Eps<T>(5e-2f, 5e-2),
        Eps<T>(1e-6f, 1e-15),
        true,
        false);
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexProjectionErrorFunction) {
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
    const ModelParametersT<T> modelParamsTarget = refParams +
        0.05 *
            uniform<VectorX<T>>(character_orig.parameterTransform.numAllModelParameters(), -1, 1);
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
        T,
        &errorFunction,
        modelParams,
        character_orig,
        errorTol,
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
    const ModelParametersT<T> modelParamsTarget = refParams +
        0.05 *
            uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);
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
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PointTriangleVertexErrorFunction) {
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

  for (const auto& constraintType : {
           VertexConstraintType::Position,
           VertexConstraintType::Plane,
           // VertexConstraintType::Normal,
           // VertexConstraintType::SymmetricNormal
       }) {
    Character character = createTestCharacter();
    character = withTestBlendShapes(character);

    const Skeleton& skeleton = character.skeleton;
    const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

    SCOPED_TRACE("constraintType: " + std::string(toString(constraintType)));

    PointTriangleVertexErrorFunctionT<T> errorFunction(character, constraintType);

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
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexPositionErrorFunctionFaceParameters) {
  using T = typename TestFixture::Type;

  const size_t nConstraints = 10;

  // Face expression blend shapes only
  {
    const Character character_blend = withTestFaceExpressionBlendShapes(createTestCharacter());
    const ModelParametersT<T> modelParams = 0.25 *
        uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);

    // TODO: Add Plane, Normal and SymmetricNormal?
    {
      VertexPositionErrorFunctionT<T> errorFunction(
          character_blend, character_blend.parameterTransform);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            VertexPositionDataT<T>(
                static_cast<size_t>(
                    uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size()) - 1)),
                uniform<Vector3<T>>(0, 1),
                uniform<T>(0, 1)));
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_blend,
          Eps<T>(5e-2f, 1e-5),
          Eps<T>(1e-6f, 5e-16),
          true,
          false);
    }
  }

  // Face expression blend shapes plus shape blend shapes
  {
    const Character character_blend =
        withTestBlendShapes(withTestFaceExpressionBlendShapes(createTestCharacter()));
    const ModelParametersT<T> modelParams = 0.25 *
        uniform<VectorX<T>>(character_blend.parameterTransform.numAllModelParameters(), -1, 1);

    // TODO: Add Plane, Normal and SymmetricNormal?
    {
      VertexPositionErrorFunctionT<T> errorFunction(
          character_blend, character_blend.parameterTransform);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            VertexPositionDataT<T>(
                static_cast<size_t>(
                    uniform<int>(0, static_cast<int>(character_blend.mesh->vertices.size()) - 1)),
                uniform<Vector3<T>>(0, 1),
                uniform<T>(0, 1)));
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_blend,
          Eps<T>(1e-2f, 1e-5),
          Eps<T>(1e-6f, 5e-16),
          true,
          false);
    }
  }
}

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
