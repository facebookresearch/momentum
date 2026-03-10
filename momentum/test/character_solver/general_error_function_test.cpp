/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for CenterOfMassErrorFunctionT edge cases:
// - getError path (no dfdv needed)
// - zero-weight constraint produces zero error
// - needsMesh() behavior

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/center_of_mass_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

// Test getError path for CenterOfMassErrorFunctionT
TEST(CenterOfMassErrorFunctionEdgeCaseTest, GetErrorPath) {
  const Character character = createTestCharacter(8);
  const auto& skel = character.skeleton;
  const auto& pt = character.parameterTransform;
  const auto numParams = pt.numAllModelParameters();

  CenterOfMassErrorFunctionT<float> errorFunction(skel, pt);

  CenterOfMassConstraintT<float> constraint;
  constraint.jointIndices = {0, 1, 2};
  constraint.masses = {1.0f, 2.0f, 1.5f};
  constraint.target = Eigen::Vector3f(0.5f, 1.0f, 0.0f);
  constraint.weight = 1.0f;
  errorFunction.addConstraint(constraint);

  const ModelParameters modelParams = ModelParameters::Zero(numParams);
  const SkeletonState skelState(pt.apply(modelParams), skel, true);
  const MeshState meshState;

  // getError should produce a non-negative value
  const double error = errorFunction.getError(modelParams, skelState, meshState);
  EXPECT_GE(error, 0.0);
}

// Test that a zero global weight produces zero error
TEST(CenterOfMassErrorFunctionEdgeCaseTest, ZeroGlobalWeight) {
  const Character character = createTestCharacter(8);
  const auto& skel = character.skeleton;
  const auto& pt = character.parameterTransform;
  const auto numParams = pt.numAllModelParameters();

  CenterOfMassErrorFunctionT<float> errorFunction(skel, pt);
  errorFunction.setWeight(0.0f);

  CenterOfMassConstraintT<float> constraint;
  constraint.jointIndices = {0, 1, 2};
  constraint.masses = {1.0f, 2.0f, 1.5f};
  constraint.target = Eigen::Vector3f(100.0f, 100.0f, 100.0f);
  constraint.weight = 1.0f;
  errorFunction.addConstraint(constraint);

  const ModelParameters modelParams = ModelParameters::Zero(numParams);
  const SkeletonState skelState(pt.apply(modelParams), skel, true);
  const MeshState meshState;

  const double error = errorFunction.getError(modelParams, skelState, meshState);
  EXPECT_NEAR(error, 0.0, 1e-10);

  Eigen::VectorXf gradient = Eigen::VectorXf::Zero(numParams);
  const double gradError = errorFunction.getGradient(modelParams, skelState, meshState, gradient);
  EXPECT_NEAR(gradError, 0.0, 1e-10);
  EXPECT_NEAR(gradient.squaredNorm(), 0.0f, 1e-10f);
}

// Test needsMesh() returns false for CenterOfMassErrorFunctionT (joint-only)
TEST(CenterOfMassErrorFunctionEdgeCaseTest, NeedsMeshReturnsFalseForJointOnly) {
  const Character character = createTestCharacter(8);
  const auto& skel = character.skeleton;
  const auto& pt = character.parameterTransform;

  CenterOfMassErrorFunctionT<float> errorFunction(skel, pt);

  // CoM uses only joint positions, no mesh vertices
  CenterOfMassConstraintT<float> constraint;
  constraint.jointIndices = {0, 1};
  constraint.masses = {1.0f, 1.0f};
  constraint.target = Eigen::Vector3f::Zero();
  constraint.weight = 1.0f;
  errorFunction.addConstraint(constraint);

  EXPECT_FALSE(errorFunction.needsMesh());
}

// Test that getJacobian produces correct-sized output
TEST(CenterOfMassErrorFunctionEdgeCaseTest, JacobianSize) {
  const Character character = createTestCharacter(8);
  const auto& skel = character.skeleton;
  const auto& pt = character.parameterTransform;
  const auto numParams = pt.numAllModelParameters();

  CenterOfMassErrorFunctionT<float> errorFunction(skel, pt);

  CenterOfMassConstraintT<float> c1;
  c1.jointIndices = {0, 1};
  c1.masses = {1.0f, 1.0f};
  c1.target = Eigen::Vector3f::Zero();
  c1.weight = 1.0f;
  errorFunction.addConstraint(c1);

  CenterOfMassConstraintT<float> c2;
  c2.jointIndices = {2, 3};
  c2.masses = {1.0f, 1.0f};
  c2.target = Eigen::Vector3f::Ones();
  c2.weight = 0.5f;
  errorFunction.addConstraint(c2);

  // FuncDim=3 for CoM, 2 constraints => JacobianSize = 6
  EXPECT_EQ(errorFunction.getJacobianSize(), 6u);

  const ModelParameters modelParams = ModelParameters::Zero(numParams);
  const SkeletonState skelState(pt.apply(modelParams), skel, true);
  const MeshState meshState;

  Eigen::MatrixXf jacobian = Eigen::MatrixXf::Zero(6, numParams);
  Eigen::VectorXf residual = Eigen::VectorXf::Zero(6);
  int usedRows = 0;

  errorFunction.getJacobian(modelParams, skelState, meshState, jacobian, residual, usedRows);
  EXPECT_EQ(usedRows, 6);
}
