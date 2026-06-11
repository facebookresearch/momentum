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
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/math/generalized_loss.h"
#include "momentum/test/character/character_helpers.h"

#include <cfloat>
#include <cmath>

using namespace momentum;

namespace {

SkeletonStated makeState(const Character& character, const ModelParametersd& params) {
  const ParameterTransformT<double> transform = character.parameterTransform.cast<double>();
  return {transform.apply(params), character.skeleton};
}

void expectCauchyScalesSingleLimit(
    const char* name,
    const Character& character,
    const ParameterLimit& limit,
    const ModelParametersd& params,
    const SkeletonStated& state) {
  SCOPED_TRACE(name);

  LimitErrorFunctiond l2Fn(character.skeleton, character.parameterTransform, {limit});
  LimitErrorFunctiond cauchyFn(
      character.skeleton,
      character.parameterTransform,
      {limit},
      GeneralizedLossT<double>::kCauchy,
      1.0);

  const MeshStated meshState{};
  const double l2Error = l2Fn.getError(params, state, meshState);
  const double cauchyError = cauchyFn.getError(params, state, meshState);
  ASSERT_GT(l2Error, 1e-12);
  ASSERT_GT(cauchyError, 0.0);
  EXPECT_LT(cauchyError, l2Error);

  VectorXd l2Gradient = VectorXd::Zero(params.size());
  VectorXd cauchyGradient = VectorXd::Zero(params.size());
  EXPECT_NEAR(l2Fn.getGradient(params, state, meshState, l2Gradient), l2Error, 1e-9);
  EXPECT_NEAR(cauchyFn.getGradient(params, state, meshState, cauchyGradient), cauchyError, 1e-9);

  const auto rows = static_cast<Eigen::Index>(l2Fn.getJacobianSize());
  ASSERT_EQ(cauchyFn.getJacobianSize(), static_cast<size_t>(rows));
  MatrixXd l2Jacobian = MatrixXd::Zero(rows, params.size());
  MatrixXd cauchyJacobian = MatrixXd::Zero(rows, params.size());
  VectorXd l2Residual = VectorXd::Zero(rows);
  VectorXd cauchyResidual = VectorXd::Zero(rows);
  int l2Rows = 0;
  int cauchyRows = 0;
  EXPECT_NEAR(
      l2Fn.getJacobian(params, state, meshState, l2Jacobian, l2Residual, l2Rows), l2Error, 1e-9);
  EXPECT_NEAR(
      cauchyFn.getJacobian(params, state, meshState, cauchyJacobian, cauchyResidual, cauchyRows),
      cauchyError,
      1e-9);
  ASSERT_EQ(l2Rows, rows);
  ASSERT_EQ(cauchyRows, rows);

  Eigen::Index maxGradientIndex = 0;
  const double maxGradient = l2Gradient.cwiseAbs().maxCoeff(&maxGradientIndex);
  ASSERT_GT(maxGradient, 1e-12);

  const double gradientScale = cauchyGradient[maxGradientIndex] / l2Gradient[maxGradientIndex];
  ASSERT_GT(gradientScale, 0.0);
  ASSERT_LT(gradientScale, 1.0);
  EXPECT_TRUE(cauchyGradient.isApprox(l2Gradient * gradientScale, 1e-8))
      << name << " gradient is not uniformly scaled by the loss derivative";

  const double jacobianScale = std::sqrt(gradientScale);
  EXPECT_TRUE(cauchyJacobian.isApprox(l2Jacobian * jacobianScale, 1e-8))
      << name << " Jacobian is not uniformly scaled by sqrt(loss derivative)";
  EXPECT_TRUE(cauchyResidual.isApprox(l2Residual * jacobianScale, 1e-8))
      << name << " residual is not uniformly scaled by sqrt(loss derivative)";

  // For Cauchy with c = 1, loss'(s) = 1 / (s + 2). Invert the observed gradient
  // scale to recover the raw squared residual and independently check the loss value.
  const double sqrError = 1.0 / gradientScale - 2.0;
  ASSERT_GT(sqrError, 0.0);
  const double expectedErrorRatio = std::log(1.0 + 0.5 * sqrError) / sqrError;
  EXPECT_NEAR(cauchyError / l2Error, expectedErrorRatio, 1e-8);
}

void expectL2ScaleC(
    const Character& character,
    const ParameterLimit& limit,
    const ModelParametersd& params,
    const SkeletonStated& state) {
  SCOPED_TRACE("L2 scale c");

  LimitErrorFunctiond l2Fn(character.skeleton, character.parameterTransform, {limit});
  LimitErrorFunctiond l2ScaledFn(
      character.skeleton,
      character.parameterTransform,
      {limit},
      GeneralizedLossT<double>::kL2,
      2.0);

  const MeshStated meshState{};
  const double l2Error = l2Fn.getError(params, state, meshState);
  const double l2ScaledError = l2ScaledFn.getError(params, state, meshState);
  ASSERT_GT(l2Error, 1e-12);
  EXPECT_NEAR(l2ScaledError, l2Error / 4.0, 1e-9);

  VectorXd l2Gradient = VectorXd::Zero(params.size());
  VectorXd l2ScaledGradient = VectorXd::Zero(params.size());
  l2Fn.getGradient(params, state, meshState, l2Gradient);
  l2ScaledFn.getGradient(params, state, meshState, l2ScaledGradient);
  EXPECT_TRUE(l2ScaledGradient.isApprox(l2Gradient / 4.0, 1e-9));

  const auto rows = static_cast<Eigen::Index>(l2Fn.getJacobianSize());
  MatrixXd l2Jacobian = MatrixXd::Zero(rows, params.size());
  MatrixXd l2ScaledJacobian = MatrixXd::Zero(rows, params.size());
  VectorXd l2Residual = VectorXd::Zero(rows);
  VectorXd l2ScaledResidual = VectorXd::Zero(rows);
  int l2Rows = 0;
  int l2ScaledRows = 0;
  l2Fn.getJacobian(params, state, meshState, l2Jacobian, l2Residual, l2Rows);
  l2ScaledFn.getJacobian(
      params, state, meshState, l2ScaledJacobian, l2ScaledResidual, l2ScaledRows);
  ASSERT_EQ(l2Rows, rows);
  ASSERT_EQ(l2ScaledRows, rows);
  EXPECT_TRUE(l2ScaledJacobian.isApprox(l2Jacobian / 2.0, 1e-9));
  EXPECT_TRUE(l2ScaledResidual.isApprox(l2Residual / 2.0, 1e-9));
}

} // namespace

TEST(LimitErrorFunctionLossTest, GeneralizedLossScalesEachLimitType) {
  const Character character = createTestCharacter();
  const ParameterTransformT<double> transform = character.parameterTransform.cast<double>();
  const auto makeParams = [&]() {
    return ModelParametersd::Zero(transform.numAllModelParameters());
  };

  {
    ParameterLimit minMax{};
    minMax.type = MinMax;
    minMax.weight = 1.0f;
    minMax.data.minMax.limits = Vector2f(-0.1f, 0.1f);
    minMax.data.minMax.parameterIndex = 0;

    ModelParametersd params = makeParams();
    params[minMax.data.minMax.parameterIndex] = 1.0; // 0.9 past the upper bound
    const SkeletonStated state = makeState(character, params);

    expectCauchyScalesSingleLimit("MinMax", character, minMax, params, state);
    expectL2ScaleC(character, minMax, params, state);
  }

  {
    ParameterLimit minMaxJoint{};
    minMaxJoint.type = MinMaxJoint;
    minMaxJoint.weight = 1.0f;
    minMaxJoint.data.minMaxJoint.limits = Vector2f(-0.1f, 0.1f);
    minMaxJoint.data.minMaxJoint.jointIndex = 2;
    minMaxJoint.data.minMaxJoint.jointParameter = 5;

    ModelParametersd params = makeParams();
    SkeletonStated state = makeState(character, params);
    const size_t jointParameterIndex =
        minMaxJoint.data.minMaxJoint.jointIndex * kParametersPerJoint +
        minMaxJoint.data.minMaxJoint.jointParameter;
    state.jointParameters[jointParameterIndex] = 1.0;

    expectCauchyScalesSingleLimit("MinMaxJoint", character, minMaxJoint, params, state);
  }

  {
    ParameterLimit linear{};
    linear.type = Linear;
    linear.weight = 1.5f;
    linear.data.linear.referenceIndex = 0;
    linear.data.linear.targetIndex = 5;
    linear.data.linear.scale = 0.25f;
    linear.data.linear.offset = 0.25f;
    linear.data.linear.rangeMin = -FLT_MAX;
    linear.data.linear.rangeMax = FLT_MAX;

    ModelParametersd params = makeParams();
    params[linear.data.linear.referenceIndex] = -1.0;
    params[linear.data.linear.targetIndex] = 2.0;
    const SkeletonStated state = makeState(character, params);

    expectCauchyScalesSingleLimit("Linear", character, linear, params, state);
  }

  {
    ParameterLimit linearJoint{};
    linearJoint.type = LinearJoint;
    linearJoint.weight = 0.75f;
    linearJoint.data.linearJoint.referenceJointIndex = 0;
    linearJoint.data.linearJoint.referenceJointParameter = 2;
    linearJoint.data.linearJoint.targetJointIndex = 1;
    linearJoint.data.linearJoint.targetJointParameter = 5;
    linearJoint.data.linearJoint.scale = 1.0f;
    linearJoint.data.linearJoint.offset = 0.0f;
    linearJoint.data.linearJoint.rangeMin = -FLT_MAX;
    linearJoint.data.linearJoint.rangeMax = FLT_MAX;

    ModelParametersd params = makeParams();
    SkeletonStated state = makeState(character, params);
    const size_t referenceParameterIndex =
        linearJoint.data.linearJoint.referenceJointIndex * kParametersPerJoint +
        linearJoint.data.linearJoint.referenceJointParameter;
    const size_t targetParameterIndex =
        linearJoint.data.linearJoint.targetJointIndex * kParametersPerJoint +
        linearJoint.data.linearJoint.targetJointParameter;
    state.jointParameters[referenceParameterIndex] = 0.0;
    state.jointParameters[targetParameterIndex] = 1.0;

    expectCauchyScalesSingleLimit("LinearJoint", character, linearJoint, params, state);
  }

  {
    ParameterLimit halfPlane{};
    halfPlane.type = HalfPlane;
    halfPlane.weight = 1.0f;
    halfPlane.data.halfPlane.param1 = 0;
    halfPlane.data.halfPlane.param2 = 2;
    halfPlane.data.halfPlane.normal = Eigen::Vector2f(1.0f, -1.0f).normalized();
    halfPlane.data.halfPlane.offset = 0.5f;

    ModelParametersd params = makeParams();
    params[halfPlane.data.halfPlane.param1] = 0.0;
    params[halfPlane.data.halfPlane.param2] = 1.0;
    const SkeletonStated state = makeState(character, params);

    expectCauchyScalesSingleLimit("HalfPlane", character, halfPlane, params, state);
  }

  {
    ParameterLimit ellipsoid{};
    ellipsoid.type = Ellipsoid;
    ellipsoid.weight = 1.0f;
    ellipsoid.data.ellipsoid.parent = 2;
    ellipsoid.data.ellipsoid.ellipsoidParent = 0;
    ellipsoid.data.ellipsoid.offset = Eigen::Vector3f(0.5f, 0.0f, 0.0f);
    Eigen::Affine3f ellipsoidTransform = Eigen::Affine3f::Identity();
    ellipsoidTransform.linear()(0, 0) = 0.2f;
    ellipsoidTransform.linear()(1, 1) = 0.1f;
    ellipsoidTransform.linear()(2, 2) = 0.15f;
    ellipsoid.data.ellipsoid.ellipsoid = ellipsoidTransform;
    ellipsoid.data.ellipsoid.ellipsoidInv = ellipsoidTransform.inverse();

    const ModelParametersd params = makeParams();
    const SkeletonStated state = makeState(character, params);

    expectCauchyScalesSingleLimit("Ellipsoid", character, ellipsoid, params, state);
  }
}
