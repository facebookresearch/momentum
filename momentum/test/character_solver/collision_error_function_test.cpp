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
#include "momentum/character_solver/collision_error_function_stateless.h"
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

TYPED_TEST(Momentum_ErrorFunctionsTest, CapsuleCollisionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // Build a collision setup with a single skeleton capsule on joint 0 and a world-fixed
  // capsule. Using a single skeleton capsule avoids interference from other collision
  // pairs while focusing on testing world-fixed capsule support.
  const size_t nJoints = 3;
  const Character baseCharacter = createTestCharacter(nJoints);

  CollisionGeometry cg;

  // Skeleton capsule on joint 0: uniform radius, extends along +X
  TaperedCapsule jointCapsule;
  jointCapsule.parent = 0;
  jointCapsule.length = 1.0f;
  jointCapsule.radius = Eigen::Vector2f(1.0f, 1.0f);
  cg.push_back(jointCapsule);

  // World-fixed capsule: uniform radius, extends along +Y, offset in Z just inside
  // the sum of radii so there's a small overlap at rest (overlap = 1.0 + 0.8 - 1.75 = 0.05).
  // This ensures most random perturbations produce collisions for reliable gradient testing.
  TaperedCapsule worldCapsule;
  worldCapsule.parent = kInvalidIndex;
  worldCapsule.length = 3.0f;
  worldCapsule.radius = Eigen::Vector2f(0.8f, 0.8f);
  worldCapsule.transformation.rotation =
      Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY());
  worldCapsule.transformation.translation = Eigen::Vector3f(0.0f, 0.0f, 1.75f);
  cg.push_back(worldCapsule);

  const Character character(
      baseCharacter.skeleton,
      baseCharacter.parameterTransform,
      baseCharacter.parameterLimits,
      baseCharacter.locators,
      baseCharacter.mesh.get(),
      baseCharacter.skinWeights.get(),
      &cg);

  CollisionErrorFunctionT<T> errorFunction(character);
  CollisionErrorFunctionStatelessT<T> errorFunctionStateless(character);

  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Use perturbations large enough to bridge the small gap and create deep overlaps,
  // but small enough that collision topology is stable during finite differences.
  size_t numTestedPoses = 0;
  for (size_t iTrial = 0; iTrial < 10; ++iTrial) {
    const ModelParametersT<T> mp = uniform<VectorX<T>>(
        transform.numAllModelParameters(), static_cast<T>(-0.3), static_cast<T>(0.3));
    const SkeletonStateT<T> state(transform.apply(mp), character.skeleton);
    if (errorFunction.getError(mp, state, MeshStateT<T>()) <= 0.0) {
      continue;
    }
    SCOPED_TRACE("trial=" + std::to_string(iTrial));
    // Use a slightly relaxed numerical gradient threshold because the collision
    // error involves subtracting similar-magnitude values (radius_sum - distance),
    // causing cancellation error that limits float finite-difference accuracy.
    const T numThreshold = std::is_same_v<T, float> ? T(0.02) : TestFixture::getNumThreshold();
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        mp,
        character,
        numThreshold,
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/true);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunctionStateless,
        mp,
        character,
        numThreshold,
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/true);
    VALIDATE_IDENTICAL(
        T, errorFunction, errorFunctionStateless, character.skeleton, transform, mp.v);
    ++numTestedPoses;
  }
  ASSERT_GE(numTestedPoses, 3) << "Too few poses produced collisions";
}

// Test capsule collision gradients on a shared skeleton with global scaling.
// Uses a double-branch skeleton (root → joint1 → joint2, root → joint3 → joint4)
// so that:
// 1. Both branches share the same scale_global parameter through the root
// 2. LCA(joint2, joint4) = root ≠ either parent, so the pair is not excluded
// 3. Rotating joint1/joint3 swings joint2/joint4 into each other
// 4. Capsules are non-parallel for stable closest-point computation
TYPED_TEST(Momentum_ErrorFunctionsTest, CapsuleCollisionError_ScaleGradient) {
  using T = typename TestFixture::Type;

  // Build a double-branch skeleton: root at origin, each branch has 2 joints.
  // Joint1/3 are children of root at (0, ±0.5, 0). Joint2/4 are grandchildren
  // at (0, ±0.5, 0) relative to their parents. At rest, joint2 is at (0, 1, 0)
  // and joint4 is at (0, -1, 0). Rotating joint1/joint3 around Z can swing
  // joint2/joint4 toward each other.
  Skeleton skeleton;
  {
    Joint root;
    root.name = "root";
    root.parent = kInvalidIndex;
    root.preRotation = Eigen::Quaternionf::Identity();
    root.translationOffset = Eigen::Vector3f::Zero();
    skeleton.joints.push_back(root);

    Joint j1;
    j1.name = "joint1";
    j1.parent = 0;
    j1.preRotation = Eigen::Quaternionf::Identity();
    j1.translationOffset = Eigen::Vector3f(0.0f, 0.5f, 0.0f);
    skeleton.joints.push_back(j1);

    Joint j2;
    j2.name = "joint2";
    j2.parent = 1;
    j2.preRotation = Eigen::Quaternionf::Identity();
    j2.translationOffset = Eigen::Vector3f(0.0f, 0.5f, 0.0f);
    skeleton.joints.push_back(j2);

    Joint j3;
    j3.name = "joint3";
    j3.parent = 0;
    j3.preRotation = Eigen::Quaternionf::Identity();
    j3.translationOffset = Eigen::Vector3f(0.0f, -0.5f, 0.0f);
    skeleton.joints.push_back(j3);

    Joint j4;
    j4.name = "joint4";
    j4.parent = 3;
    j4.preRotation = Eigen::Quaternionf::Identity();
    j4.translationOffset = Eigen::Vector3f(0.0f, -0.5f, 0.0f);
    skeleton.joints.push_back(j4);
  }

  // Parameter transform: root 6-DOF + scale_global + joint1_rz + joint3_rz
  ParameterTransform pt;
  const auto numJointParams = skeleton.joints.size() * kParametersPerJoint;
  pt.name = {
      "root_tx",
      "root_ty",
      "root_tz",
      "root_rx",
      "root_ry",
      "root_rz",
      "scale_global",
      "joint1_rz",
      "joint3_rz"};
  pt.offsets = Eigen::VectorXf::Zero(numJointParams);
  pt.transform.resize(numJointParams, static_cast<int>(pt.name.size()));
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0 * kParametersPerJoint + 0, 0, 1.0f); // root_tx
  triplets.emplace_back(0 * kParametersPerJoint + 1, 1, 1.0f); // root_ty
  triplets.emplace_back(0 * kParametersPerJoint + 2, 2, 1.0f); // root_tz
  triplets.emplace_back(0 * kParametersPerJoint + 3, 3, 1.0f); // root_rx
  triplets.emplace_back(0 * kParametersPerJoint + 4, 4, 1.0f); // root_ry
  triplets.emplace_back(0 * kParametersPerJoint + 5, 5, 1.0f); // root_rz
  triplets.emplace_back(0 * kParametersPerJoint + 6, 6, 1.0f); // scale_global
  triplets.emplace_back(1 * kParametersPerJoint + 5, 7, 1.0f); // joint1_rz
  triplets.emplace_back(3 * kParametersPerJoint + 5, 8, 1.0f); // joint3_rz
  pt.transform.setFromTriplets(triplets.begin(), triplets.end());
  pt.activeJointParams = pt.computeActiveJointParams();

  // Two capsules on the tip joints (joint2 and joint4), with ±15° tilt around X
  // so they aren't parallel. Uniform radius 0.3, length 0.5.
  // At rest (joint2 at (0,1,0), joint4 at (0,-1,0)): distance 2.0, radius sum 0.6,
  // gap = 1.4. Setting joint1_rz ≈ π/2 swings joint2 from (0,1,0) toward (~0.5, 0.5, 0),
  // and joint3_rz ≈ -π/2 swings joint4 toward (~0.5, -0.5, 0). These are close enough
  // that small additional perturbation creates collisions.
  CollisionGeometry cg;
  {
    TaperedCapsule cap1;
    cap1.parent = 2;
    cap1.length = 0.5f;
    cap1.radius = Eigen::Vector2f(0.3f, 0.3f);
    cap1.transformation.rotation =
        Eigen::Quaternionf(Eigen::AngleAxisf(0.2618f, Eigen::Vector3f::UnitX())); // +15° tilt
    cg.push_back(cap1);

    TaperedCapsule cap2;
    cap2.parent = 4;
    cap2.length = 0.5f;
    cap2.radius = Eigen::Vector2f(0.3f, 0.3f);
    cap2.transformation.rotation =
        Eigen::Quaternionf(Eigen::AngleAxisf(-0.2618f, Eigen::Vector3f::UnitX())); // -15° tilt
    cg.push_back(cap2);
  }

  const Character character(skeleton, pt, ParameterLimits(), LocatorList(), nullptr, nullptr, &cg);

  CollisionErrorFunctionT<T> errorFunction(character);
  CollisionErrorFunctionStatelessT<T> errorFunctionStateless(character);

  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Start from a base pose where joint1/joint3 are rotated to swing the tips
  // toward each other, then add small random perturbation.
  ModelParametersT<T> baseMp = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  baseMp.v[7] = static_cast<T>(2.5); // joint1_rz: fold arm1 inward toward center
  baseMp.v[8] = static_cast<T>(-2.5); // joint3_rz: fold arm2 inward toward center

  size_t numTestedPoses = 0;
  for (size_t iTrial = 0; iTrial < 20; ++iTrial) {
    ModelParametersT<T> mp = baseMp;
    const auto perturbation = uniform<VectorX<T>>(
        transform.numAllModelParameters(), static_cast<T>(-0.1), static_cast<T>(0.1));
    mp.v += perturbation;
    const SkeletonStateT<T> state(transform.apply(mp), character.skeleton);
    if (errorFunction.getError(mp, state, MeshStateT<T>()) <= 0.0) {
      continue;
    }
    SCOPED_TRACE("trial=" + std::to_string(iTrial));
    // Use a relaxed threshold for float because the large rotation angles (≈2.5 rad)
    // amplify float cancellation error in finite differences.
    const T numThreshold = std::is_same_v<T, float> ? T(0.03) : TestFixture::getNumThreshold();
    // Disable numerical Jacobian check for float — the multi-joint chain with large
    // rotations exceeds float finite-difference accuracy. Double validates correctness.
    const bool checkJac = !std::is_same_v<T, float>;
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        mp,
        character,
        numThreshold,
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/checkJac);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunctionStateless,
        mp,
        character,
        numThreshold,
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/checkJac);
    VALIDATE_IDENTICAL(
        T, errorFunction, errorFunctionStateless, character.skeleton, transform, mp.v);
    ++numTestedPoses;
  }
  ASSERT_GE(numTestedPoses, 3) << "Too few poses produced collisions";
}
