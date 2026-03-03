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
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skinned_locator_error_function.h"
#include "momentum/character_solver/skinned_locator_triangle_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

namespace {

Character getSkinnedLocatorTestCharacter() {
  Character character = withTestBlendShapes(createTestCharacter(4));

  std::vector<bool> activeLocators(character.skinnedLocators.size(), true);
  activeLocators[1] = false;
  const auto [transform, limits] = addSkinnedLocatorParameters(
      character.parameterTransform, character.parameterLimits, activeLocators);

  return {
      character.skeleton,
      transform,
      limits,
      character.locators,
      character.mesh.get(),
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      character.inverseBindPose,
      character.skinnedLocators};
}

} // namespace

TYPED_TEST(Momentum_ErrorFunctionsTest, SkinnedLocatorError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // Create a test character with skinned locators
  const Character character = getSkinnedLocatorTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  // Verify that the character has skinned locators
  ASSERT_GT(character.skinnedLocators.size(), 2);

  // Create the error function
  SkinnedLocatorErrorFunctionT<T> errorFunction(character);

  // Add constraints for some of the skinned locators
  const T kTestWeightValue = 4.5;
  for (int i = 0; i < std::min(3, static_cast<int>(character.skinnedLocators.size())); ++i) {
    errorFunction.addConstraint(
        i, kTestWeightValue, uniform<Vector3<T>>(-1, 1)); // Random target position
  }

  // Test with random parameters
  for (size_t i = 0; i < 10; i++) {
    ModelParametersT<T> parameters = uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
    TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(5e-2f, 5e-6));
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PointTriangleSkinnedLocatorError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // Create a test character with skinned locators
  const Character character = getSkinnedLocatorTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Verify that the character has skinned locators
  ASSERT_GT(character.skinnedLocators.size(), 0);

  // Create the error function with Plane constraint
  for (const auto vertexConstraintType :
       {VertexConstraintType::Plane, VertexConstraintType::Position}) {
    SkinnedLocatorTriangleErrorFunctionT<T> errorFunction(character, vertexConstraintType);

    // Add constraints for some of the skinned locators
    const T kTestWeightValue = 5.0;
    for (int i = 0; i < character.skinnedLocators.size(); ++i) {
      // Create a simple triangle constraint
      Eigen::Vector3i triangleIndices =
          character.mesh->faces[uniform<int>(0, character.mesh->faces.size() - 1)];
      auto triangleBaryCoords = uniform<Vector3<T>>(0, 1);
      triangleBaryCoords /= triangleBaryCoords.sum();
      T depth = 0.5;

      errorFunction.addConstraint(i, triangleIndices, triangleBaryCoords, depth, kTestWeightValue);
    }

    // Test with random parameters
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          0.25 * uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, character, Eps<T>(0.03, 1e-3), true, true);
    }
  }
}

// Test that sliding constraints correctly slide to nearest candidate triangle
TYPED_TEST(Momentum_ErrorFunctionsTest, SkinnedLocatorTriangle_SlidingConstraints) {
  using T = typename TestFixture::Type;

  // Create a test character with skinned locators
  const Character character = getSkinnedLocatorTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const Mesh& mesh = *character.mesh;

  ASSERT_GT(character.skinnedLocators.size(), 0);
  ASSERT_GT(mesh.faces.size(), 3); // Need at least a few triangles

  // Test with Position constraint type (similar results expected for Plane)
  SkinnedLocatorTriangleErrorFunctionT<T> errorFunction(character, VertexConstraintType::Position);

  // Pick three triangles for testing: initial triangle and two candidates
  const size_t initialTriIdx = 0;
  const size_t candidateTriIdx1 = 1;
  const size_t candidateTriIdx2 = 2;
  const size_t excludedTriIdx = 3; // Not in candidate list

  const Eigen::Vector3i initialTriIndices = mesh.faces[initialTriIdx];
  const Eigen::Vector3i candidateTri1Indices = mesh.faces[candidateTriIdx1];
  const Eigen::Vector3i candidateTri2Indices = mesh.faces[candidateTriIdx2];
  const Eigen::Vector3i excludedTriIndices = mesh.faces[excludedTriIdx];

  // Create constraint with candidate triangles
  SkinnedLocatorTriangleConstraintT<T> constraint;
  constraint.locatorIndex = 0;
  constraint.tgtTriangleIndices = initialTriIndices;
  constraint.tgtTriangleBaryCoords = Eigen::Vector3<T>(T(1.0 / 3.0), T(1.0 / 3.0), T(1.0 / 3.0));
  constraint.depth = T(0);
  constraint.weight = T(1);

  // Add candidate triangles (including the initial one)
  CandidateTriangle candidate0;
  candidate0.triangleIdx = initialTriIdx;
  candidate0.vertexIndices = initialTriIndices;
  constraint.candidateTriangles.push_back(candidate0);

  CandidateTriangle candidate1;
  candidate1.triangleIdx = candidateTriIdx1;
  candidate1.vertexIndices = candidateTri1Indices;
  constraint.candidateTriangles.push_back(candidate1);

  CandidateTriangle candidate2;
  candidate2.triangleIdx = candidateTriIdx2;
  candidate2.vertexIndices = candidateTri2Indices;
  constraint.candidateTriangles.push_back(candidate2);

  // Set the constraint
  std::vector<SkinnedLocatorTriangleConstraintT<T>> constraints = {constraint};
  errorFunction.setConstraints(constraints);

  // Create parameters and state first
  ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  SkeletonStateT<T> state(transform.apply(params), character.skeleton);

  // Set up mesh state using the MeshStateT constructor
  MeshStateT<T> meshState(params, state, character);

  // Helper to compute centroid of a triangle
  auto getTriangleCentroid = [&mesh](size_t triIdx) -> Eigen::Vector3<T> {
    const Eigen::Vector3i& face = mesh.faces[triIdx];
    return (mesh.vertices[face(0)].template cast<T>() + mesh.vertices[face(1)].template cast<T>() +
            mesh.vertices[face(2)].template cast<T>()) /
        T(3);
  };

  // Get centroids for testing
  const Eigen::Vector3<T> initialCentroid = getTriangleCentroid(initialTriIdx);
  const Eigen::Vector3<T> candidate1Centroid = getTriangleCentroid(candidateTriIdx1);
  const Eigen::Vector3<T> candidate2Centroid = getTriangleCentroid(candidateTriIdx2);
  const Eigen::Vector3<T> excludedCentroid = getTriangleCentroid(excludedTriIdx);

  // Get the skinned locator parameter index
  const SkinnedLocator& locator = character.skinnedLocators[0];
  int locatorParamIdx = -1;
  if (character.parameterTransform.skinnedLocatorParameters.size() > 0) {
    locatorParamIdx = character.parameterTransform.skinnedLocatorParameters[0];
  }
  ASSERT_GE(locatorParamIdx, 0); // Ensure the locator has parameters

  // Test 1: Position locator at initial triangle centroid - error should be near zero
  {
    Eigen::Vector3<T> offset = initialCentroid - locator.position.template cast<T>();
    params.v.template segment<3>(locatorParamIdx) = offset;
    double errorAtInitial = errorFunction.getError(params, state, meshState);
    // Error should be small (locator is at the target)
    EXPECT_LT(errorAtInitial, T(0.01)) << "Error at initial triangle should be near zero";
  }

  // Test 2: Position locator at candidate1 centroid - should slide to it with low error
  {
    Eigen::Vector3<T> offset = candidate1Centroid - locator.position.template cast<T>();
    params.v.template segment<3>(locatorParamIdx) = offset;
    double errorAtCandidate1 = errorFunction.getError(params, state, meshState);
    // Error should be small since it can slide to candidate1
    EXPECT_LT(errorAtCandidate1, T(0.01)) << "Error at candidate1 should be near zero (sliding)";
  }

  // Test 3: Position locator at candidate2 centroid - should slide to it with low error
  {
    Eigen::Vector3<T> offset = candidate2Centroid - locator.position.template cast<T>();
    params.v.template segment<3>(locatorParamIdx) = offset;
    double errorAtCandidate2 = errorFunction.getError(params, state, meshState);
    // Error should be small since it can slide to candidate2
    EXPECT_LT(errorAtCandidate2, T(0.01)) << "Error at candidate2 should be near zero (sliding)";
  }

  // Test 4: Position locator at excluded triangle centroid - should NOT slide there
  // Error should be larger since it can only slide to candidates
  {
    Eigen::Vector3<T> offset = excludedCentroid - locator.position.template cast<T>();
    params.v.template segment<3>(locatorParamIdx) = offset;
    double errorAtExcluded = errorFunction.getError(params, state, meshState);

    // Now create a constraint WITHOUT candidates (fixed to initial triangle)
    SkinnedLocatorTriangleConstraintT<T> fixedConstraint;
    fixedConstraint.locatorIndex = 0;
    fixedConstraint.tgtTriangleIndices = initialTriIndices;
    fixedConstraint.tgtTriangleBaryCoords =
        Eigen::Vector3<T>(T(1.0 / 3.0), T(1.0 / 3.0), T(1.0 / 3.0));
    fixedConstraint.depth = T(0);
    fixedConstraint.weight = T(1);
    // No candidate triangles - should behave as fixed constraint

    SkinnedLocatorTriangleErrorFunctionT<T> fixedErrorFunction(
        character, VertexConstraintType::Position);
    fixedErrorFunction.setConstraints({fixedConstraint});

    double fixedErrorAtExcluded = fixedErrorFunction.getError(params, state, meshState);

    // The sliding error should be less than or equal to the fixed error
    // because sliding can find a closer point among candidates
    // (unless excluded triangle happens to be the same as initial, which we avoid)

    // Both errors should be non-zero since the locator is at the excluded triangle
    // but the sliding constraint finds the closest candidate while fixed stays at initial
    EXPECT_GT(errorAtExcluded, T(0.0))
        << "Error at excluded triangle should be non-zero for sliding constraint";
    EXPECT_GT(fixedErrorAtExcluded, T(0.0))
        << "Error at excluded triangle should be non-zero for fixed constraint";
  }

  // Note: We intentionally don't test gradients/jacobians for sliding constraints
  // because the sliding introduces non-differentiable discontinuities when switching
  // between triangles. The Jacobian is computed as if the current triangle is fixed,
  // which is a common approximation for closest-point constraints, but it won't match
  // numerical gradients at switching boundaries.
}

TYPED_TEST(Momentum_ErrorFunctionsTest, TestSkinningErrorFunction) {
  using T = typename TestFixture::Type;

  // this unit tests checks the accuracy of our linear skinning constraint accuracy
  // against our simplified approximation that's way faster.
  // we expect our gradients to be within 10% of the true gradients of the mesh

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto mesh = character.mesh->cast<T>();
  const auto& skin = *character.skinWeights;
  VectorX<T> parameters = VectorX<T>::Zero(transform.numAllModelParameters());

  VectorX<T> gradient = VectorX<T>::Zero(transform.numAllModelParameters());

  // create constraints
  std::vector<PositionDataT<T>> cl;
  PositionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  SkeletonStateT<T> bindState(transform.apply(parameters), skeleton);
  SkeletonStateT<T> state(transform.apply(parameters), skeleton);
  MeshStateT<T> meshState;
  TransformationListT<T> bindpose;
  for (const auto& js : bindState.jointState) {
    bindpose.push_back(js.transform.inverse().toAffine3());
  }

  {
    SCOPED_TRACE("Skinning mesh constraint test");

    for (size_t vi = 0; vi < mesh.vertices.size(); vi++) {
      const Eigen::Vector3<T> target = mesh.vertices[vi];

      // add vertex to constraint list
      cl.clear();
      for (size_t si = 0; si < kMaxSkinJoints; si++) {
        if (skin.weight(vi, si) == 0.0) {
          continue;
        }
        const auto parent = skin.index(vi, si);
        cl.push_back(
            PositionDataT<T>(
                (target - bindState.jointState[parent].translation()),
                target,
                parent,
                skin.weight(vi, si)));
      }
      errorFunction.setConstraints(cl);

      std::vector<Vector3<T>> v = applySSD(bindpose, skin, mesh.vertices, bindState);

      // check position of skinning
      EXPECT_LE((v[vi] - target).norm(), Eps<T>(2e-7f, 2e-7));

      // check error
      gradient.setZero();
      T gradientError = errorFunction.getGradient(parameters, bindState, meshState, gradient);
      EXPECT_NEAR(gradientError, 0, Eps<T>(1e-15f, 1e-15));
      EXPECT_LE(gradient.norm(), Eps<T>(1e-7f, 1e-7));

      for (size_t i = 0; i < 10; i++) {
        parameters = uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
        state.set(transform.apply(parameters), skeleton);
        v = applySSD(bindpose, skin, mesh.vertices, state);

        cl.clear();
        for (size_t si = 0; si < kMaxSkinJoints; si++) {
          if (skin.weight(vi, si) == 0.0) {
            continue;
          }
          const auto parent = skin.index(vi, si);
          const Vector3<T> offset = state.jointState[parent].transform.inverse() * v[vi];
          cl.push_back(PositionDataT<T>(offset, target, parent, skin.weight(vi, si)));
        }
        errorFunction.setConstraints(cl);

        gradient.setZero();
        gradientError = errorFunction.getGradient(parameters, state, meshState, gradient);
        auto numError = (v[vi] - target).squaredNorm();
        EXPECT_NEAR(gradientError, numError, Eps<T>(1e-5f, 1e-5));

        // calculate numerical gradient
        constexpr T kStepSize = 1e-5;
        VectorX<T> numGradient = VectorX<T>::Zero(transform.numAllModelParameters());
        for (auto p = 0; p < transform.numAllModelParameters(); p++) {
          // perform higher-order finite differences for accuracy
          VectorX<T> params = parameters;
          params(p) = parameters(p) - kStepSize;
          state.set(transform.apply(params), skeleton);
          v = applySSD(bindpose, skin, mesh.vertices, state);
          const T h_1 = (v[vi] - target).squaredNorm();

          params(p) = parameters(p) + kStepSize;
          state.set(transform.apply(params), skeleton);
          v = applySSD(bindpose, skin, mesh.vertices, state);
          const T h1 = (v[vi] - target).squaredNorm();

          numGradient(p) = (h1 - h_1) / (2.0 * kStepSize);
        }

        // check the gradients are similar
        {
          SCOPED_TRACE("Checking Numerical Gradient");
          if ((numGradient + gradient).norm() != 0.0) {
            EXPECT_LE(
                (numGradient - gradient).norm() / (numGradient + gradient).norm(),
                Eps<T>(1e-1f, 1e-1));
          }
        }
      }
    }
  }
}
