/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/plane_collision_error_function.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

namespace {

ParameterTransform makeRootYTransform(size_t numJoints) {
  const auto numJointParams = numJoints * kParametersPerJoint;
  ParameterTransform transform;
  transform.name = {"root_ty"};
  transform.offsets = Eigen::VectorXf::Zero(numJointParams);
  transform.transform.resize(numJointParams, 1);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0 * kParametersPerJoint + 1, 0, 1.0f);
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());
  transform.activeJointParams = transform.computeActiveJointParams();
  return transform;
}

Character createSinglePrimitiveCharacter(const CollisionPrimitive& primitive) {
  const Character baseCharacter = createTestCharacter(3);

  CollisionGeometry collisionGeometry;
  collisionGeometry.push_back(primitive);

  return {
      baseCharacter.skeleton,
      makeRootYTransform(baseCharacter.skeleton.joints.size()),
      ParameterLimits(),
      baseCharacter.locators,
      baseCharacter.mesh.get(),
      baseCharacter.skinWeights.get(),
      &collisionGeometry};
}

// Maps the first 7 model parameters one-to-one onto joint 0's [tx, ty, tz, rx, ry, rz, scale], so a
// test can exercise translation, rotation, and scale derivatives of the root joint.
ParameterTransform makeRootSevenDofTransform(size_t numJoints) {
  const auto numJointParams = numJoints * kParametersPerJoint;
  ParameterTransform transform;
  transform.name = {"tx", "ty", "tz", "rx", "ry", "rz", "scale"};
  transform.offsets = Eigen::VectorXf::Zero(numJointParams);
  transform.transform.resize(numJointParams, kParametersPerJoint);
  std::vector<Eigen::Triplet<float>> triplets;
  for (size_t i = 0; i < kParametersPerJoint; ++i) {
    triplets.emplace_back(0 * kParametersPerJoint + i, i, 1.0f);
  }
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());
  transform.activeJointParams = transform.computeActiveJointParams();
  return transform;
}

Character createSinglePrimitiveCharacterSevenDof(const CollisionPrimitive& primitive) {
  const Character baseCharacter = createTestCharacter(3);

  CollisionGeometry collisionGeometry;
  collisionGeometry.push_back(primitive);

  return {
      baseCharacter.skeleton,
      makeRootSevenDofTransform(baseCharacter.skeleton.joints.size()),
      ParameterLimits(),
      baseCharacter.locators,
      baseCharacter.mesh.get(),
      baseCharacter.skinWeights.get(),
      &collisionGeometry};
}

CollisionPrimitive createCapsulePrimitive() {
  TaperedCapsule capsule;
  capsule.parent = 0;
  capsule.length = 1.0f;
  capsule.radius = Eigen::Vector2f(0.4f, 0.4f);
  capsule.transformation.translation = Eigen::Vector3f(0.0f, 0.2f, 0.0f);
  return capsule;
}

CollisionPrimitive createEllipsoidPrimitive() {
  CollisionEllipsoid ellipsoid;
  ellipsoid.parent = 0;
  ellipsoid.radii = Eigen::Vector3f(0.2f, 0.5f, 0.3f);
  ellipsoid.transformation.translation = Eigen::Vector3f(0.0f, 0.2f, 0.0f);
  return ellipsoid;
}

CollisionPrimitive createBoxPrimitive() {
  CollisionBox box;
  box.parent = 0;
  box.halfExtents = Eigen::Vector3f(0.2f, 0.45f, 0.3f);
  box.transformation.translation = Eigen::Vector3f(0.0f, 0.2f, 0.0f);
  return box;
}

} // namespace

TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneCollisionError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  const std::vector<CollisionPrimitive> primitives = {
      createCapsulePrimitive(), createEllipsoidPrimitive(), createBoxPrimitive()};

  for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
    SCOPED_TRACE("primitiveIndex=" + std::to_string(primitiveIndex));
    const Character character = createSinglePrimitiveCharacter(primitives[primitiveIndex]);
    PlaneCollisionErrorFunctionT<T> errorFunction(character, Vector3<T>::UnitY(), T(0));
    const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
    const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
    const SkeletonStateT<T> state(transform.apply(params), character.skeleton);

    const double error = errorFunction.getError(params, state, MeshStateT<T>());
    EXPECT_GT(error, 0.0);
    EXPECT_EQ(errorFunction.getCollidingPrimitives().size(), 1);
    const auto debugInfo = errorFunction.getCollisionDebugInfo();
    ASSERT_EQ(debugInfo.size(), 1);
    EXPECT_EQ(debugInfo[0].primitiveIndex, 0);
    EXPECT_GT(debugInfo[0].overlap, 0.0);

    // The float forward-difference Jacobian loses precision (catastrophic cancellation on the
    // sqrt-based support radius) and needs a looser numerical tolerance than the default 1e-3;
    // double precision stays tight.
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        params,
        character,
        Eps<T>(5e-3f, 1e-3),
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/true);
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneCollisionError_NoErrorAbovePlane) {
  using T = typename TestFixture::Type;

  const Character character = createSinglePrimitiveCharacter(createEllipsoidPrimitive());
  PlaneCollisionErrorFunctionT<T> errorFunction(character, Vector3<T>::UnitY(), T(0));
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  params[0] = T(1);
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);

  EXPECT_EQ(errorFunction.getError(params, state, MeshStateT<T>()), 0.0);
  EXPECT_TRUE(errorFunction.getCollidingPrimitives().empty());

  MatrixX<T> jacobian =
      MatrixX<T>::Zero(errorFunction.getJacobianSize(), transform.numAllModelParameters());
  VectorX<T> residual = VectorX<T>::Zero(errorFunction.getJacobianSize());
  int usedRows = -1;
  EXPECT_EQ(
      errorFunction.getJacobian(params, state, MeshStateT<T>(), jacobian, residual, usedRows), 0.0);
  EXPECT_EQ(usedRows, 0);
}

// Exercises translation, rotation, and scale DOFs. The support radius of a centered primitive
// depends on the collider's world orientation, so an active rotation above the collider must be
// reflected in the analytic gradient/Jacobian. This guards against regressing that term.
TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneCollisionError_RotationAndScaleGradients) {
  using T = typename TestFixture::Type;

  const std::vector<CollisionPrimitive> primitives = {
      createCapsulePrimitive(), createEllipsoidPrimitive(), createBoxPrimitive()};

  for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
    SCOPED_TRACE("primitiveIndex=" + std::to_string(primitiveIndex));
    const Character character = createSinglePrimitiveCharacterSevenDof(primitives[primitiveIndex]);
    PlaneCollisionErrorFunctionT<T> errorFunction(character, Vector3<T>::UnitY(), T(0));
    const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

    // Push the root below the plane (ty) and add rotation + scale so the contact point and support
    // radius both vary with the parameters.
    ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
    params[1] = T(-0.3); // ty: penetrate the plane
    params[3] = T(0.4); // rx
    params[4] = T(0.3); // ry
    params[5] = T(0.2); // rz
    params[6] = T(0.15); // scale (log2)
    const SkeletonStateT<T> state(transform.apply(params), character.skeleton);

    ASSERT_GT(errorFunction.getError(params, state, MeshStateT<T>()), 0.0);

    // double validates the analytic derivatives tightly; float gets a looser numerical tolerance
    // because the forward-difference Jacobian's cancellation error (~eps/step) dominates for this
    // multi-DOF pose on the sqrt-based support radius.
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        params,
        character,
        Eps<T>(1e-2f, 1e-3),
        TestFixture::getJacThreshold(),
        /*checkJacError=*/true,
        /*checkJacobian=*/true);
  }
}
