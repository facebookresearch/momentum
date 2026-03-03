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
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, LimitError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  LimitErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);

  // TODO: None of these work right at the moment due to precision issues with numerical gradients,
  // need to fix code to use double
  {
    SCOPED_TRACE("Limit MinMax Test");
    ParameterLimit limit;
    limit.type = MinMax;
    limit.weight = 1.0;
    limit.data.minMax.limits = Vector2f(-0.1, 0.1);
    limit.data.minMax.parameterIndex = 0;
    errorFunction.setLimits({limit});
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(1e-2f, 1e-11));
    }
  }

  {
    SCOPED_TRACE("Limit MinMax Joint Test");
    ParameterLimit limit;
    limit.type = MinMaxJoint;
    limit.weight = 1.0;
    limit.data.minMaxJoint.limits = Vector2f(-0.1, 0.1);
    limit.data.minMaxJoint.jointIndex = 2;
    limit.data.minMaxJoint.jointParameter = 5;
    errorFunction.setLimits({limit});
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(5e-3f, 1e-10));
    }
  }

  {
    SCOPED_TRACE("Limit LinearTest");
    ParameterLimit limit;
    limit.type = Linear;
    limit.weight = 1.5;
    limit.data.linear.referenceIndex = 0;
    limit.data.linear.targetIndex = 5;
    limit.data.linear.scale = 0.25;
    limit.data.linear.offset = 0.25;
    errorFunction.setLimits({limit});
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(1e-3f, 5e-12));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(2e-2f, 1e-10));
    }
  }

  {
    SCOPED_TRACE("Limit PiecewiseLinearTest");
    ParameterLimits limits;

    ParameterLimit limit;
    limit.type = LimitType::Linear;
    limit.data.linear.referenceIndex = 0;
    limit.data.linear.targetIndex = 5;
    limit.weight = 0.5f;

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = -1.0f;
      cur.data.linear.offset = 3.0f;
      cur.data.linear.rangeMin = -FLT_MAX;
      cur.data.linear.rangeMax = -3.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = 1.0f;
      cur.data.linear.offset = -3.0f;
      cur.data.linear.rangeMin = -3.0f;
      cur.data.linear.rangeMax = FLT_MAX;
      limits.push_back(cur);
    }

    errorFunction.setLimits(limits);

    // Verify that gradients are ok on either side of the first-derivative discontinuity:
    ModelParametersT<T> parametersBefore = VectorX<T>::Zero(transform.numAllModelParameters());
    parametersBefore(limit.data.linear.targetIndex) = -3.01f;
    const SkeletonStateT<T> skelStateBefore(
        character.parameterTransform.cast<T>().apply(parametersBefore), character.skeleton);
    Eigen::VectorX<T> gradBefore = VectorX<T>::Zero(transform.numAllModelParameters());
    errorFunction.getGradient(parametersBefore, skelStateBefore, MeshStateT<T>(), gradBefore);
    TEST_GRADIENT_AND_JACOBIAN(
        T, &errorFunction, parametersBefore, character, Eps<T>(1e-3f, 1e-10));

    ModelParametersT<T> parametersAfter = VectorX<T>::Zero(transform.numAllModelParameters());
    parametersAfter(limit.data.linear.targetIndex) = -2.99f;
    const SkeletonStateT<T> skelStateAfter(
        character.parameterTransform.cast<T>().apply(parametersAfter), character.skeleton);
    Eigen::VectorX<T> gradAfter = VectorX<T>::Zero(transform.numAllModelParameters());
    errorFunction.getGradient(parametersAfter, skelStateAfter, MeshStateT<T>(), gradAfter);
    TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parametersAfter, character, Eps<T>(1e-3f, 1e-10));

    // Gradient won't be the same at the point 3.0:
    ModelParametersT<T> parametersMid = VectorX<T>::Zero(transform.numAllModelParameters());
    parametersMid(limit.data.linear.targetIndex) = -3.0f;
    const SkeletonStateT<T> skelStateMid(
        character.parameterTransform.cast<T>().apply(parametersMid), character.skeleton);

    // Verify that the error is C0 continuous:
    const float errorBefore =
        errorFunction.getError(parametersBefore, skelStateBefore, MeshStateT<T>());
    const float errorMid = errorFunction.getError(parametersMid, skelStateMid, MeshStateT<T>());
    const float errorAfter =
        errorFunction.getError(parametersAfter, skelStateAfter, MeshStateT<T>());

    ASSERT_NEAR(errorBefore, errorMid, 0.03f);
    ASSERT_NEAR(errorMid, errorAfter, 0.03f);
  }

  {
    SCOPED_TRACE("LimitJoint PiecewiseLinearTest");
    ParameterLimits limits;

    ParameterLimit limit;
    limit.type = LimitType::LinearJoint;
    limit.data.linearJoint.referenceJointIndex = 0;
    limit.data.linearJoint.referenceJointParameter = 2;
    limit.data.linearJoint.targetJointIndex = 1;
    limit.data.linearJoint.targetJointParameter = 5;
    limit.weight = 0.75f;

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = 1.0f;
      cur.data.linearJoint.offset = -4.0f;
      cur.data.linearJoint.rangeMin = -FLT_MAX;
      cur.data.linearJoint.rangeMax = 0.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = -1.0f;
      cur.data.linearJoint.offset = -4.0f;
      cur.data.linearJoint.rangeMin = 0.0f;
      cur.data.linearJoint.rangeMax = 2.0;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = 1.0f;
      cur.data.linearJoint.offset = 0.0f;
      cur.data.linearJoint.rangeMin = 2.0f;
      cur.data.linearJoint.rangeMax = FLT_MAX;
      limits.push_back(cur);
    }

    errorFunction.setLimits(limits);

    auto rz_param = character.parameterTransform.getParameterIdByName("shared_rz");
    ASSERT_NE(rz_param, momentum::kInvalidIndex);

    for (const auto& testPos : {-4.0, 0.0, 4.0}) {
      // Verify that gradients are ok on either side of the first-derivative discontinuity:
      ModelParametersT<T> parametersBefore = VectorX<T>::Zero(transform.numAllModelParameters());
      parametersBefore(rz_param) = testPos - 0.001f;
      const SkeletonStateT<T> skelStateBefore(
          character.parameterTransform.cast<T>().apply(parametersBefore), character.skeleton);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parametersBefore, character, Eps<T>(5e-2f, 1e-10));

      ModelParametersT<T> parametersAfter = VectorX<T>::Zero(transform.numAllModelParameters());
      parametersAfter(rz_param) = testPos + 0.001f;
      const SkeletonStateT<T> skelStateAfter(
          character.parameterTransform.cast<T>().apply(parametersAfter), character.skeleton);
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parametersAfter, character, Eps<T>(5e-2f, 1e-10));

      // Verify that the error is C0 continuous:
      const float errorBefore =
          errorFunction.getError(parametersBefore, skelStateBefore, MeshStateT<T>());
      const float errorAfter =
          errorFunction.getError(parametersAfter, skelStateAfter, MeshStateT<T>());
      ASSERT_NEAR(errorAfter, errorBefore, 0.03f);

      // Make sure the parameter actually varies:
      const auto targetJointParameter =
          limit.data.linearJoint.targetJointIndex * kParametersPerJoint +
          limit.data.linearJoint.targetJointParameter;
      EXPECT_GT(
          std::abs(
              skelStateBefore.jointParameters[targetJointParameter] -
              skelStateAfter.jointParameters[targetJointParameter]),
          0.0005f);
    }
  }

  {
    SCOPED_TRACE("Limit HalfPlaneTest");

    ParameterLimit limit;
    limit.type = HalfPlane;
    limit.weight = 1.0;
    limit.data.halfPlane.param1 = 0;
    limit.data.halfPlane.param2 = 2;
    limit.data.halfPlane.normal = Eigen::Vector2f(1, -1).normalized();
    limit.data.halfPlane.offset = 0.5f;
    errorFunction.setLimits({limit});
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        character,
        Eps<T>(5e-3f, 5e-12));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters =
          uniform<VectorX<T>>(transform.numAllModelParameters(), -1, 1);
      TEST_GRADIENT_AND_JACOBIAN(T, &errorFunction, parameters, character, Eps<T>(2e-2f, 1e-10));
    }
  }

  //{
  //    SCOPED_TRACE("Limit Ellipsoid Test");
  //    limit.type = ELLIPSOID;
  //    limit.weight = 1.0f;
  //    limit.data.ellipsoid.parent = 2;
  //    limit.data.ellipsoid.ellipsoidParent = 0;
  //    limit.data.ellipsoid.offset = Vector3f(0, -1, 0);
  //    limit.data.ellipsoid.ellipsoid = Affine3f::Identity();
  //    limit.data.ellipsoid.ellipsoid.translation() = Vector3f(0.5f, 0.5f, 0.5f);
  //    limit.data.ellipsoid.ellipsoid.linear() = (Quaternionf(Eigen::AngleAxisf(0.1f,
  //    Vector3f::UnitZ())) *
  //                                            Quaternionf(Eigen::AngleAxisf(0.2f,
  //                                            Vector3f::UnitY())) *
  //                                            Quaternionf(Eigen::AngleAxisf(0.3f,
  //                                            Vector3f::UnitX()))).toRotationMatrix() *
  //                                            Scaling(2.0f, 1.5f, 0.5f);
  //    limit.data.ellipsoid.ellipsoidInv = limit.data.ellipsoid.ellipsoid.inverse();

  //    errorFunction.setLimits(lm);
  //    parameters.setZero();
  //    testGradientAndJacobian(&errorFunction, parameters, skeleton, transform);
  //    for (size_t i = 0; i < 3; i++)
  //    {
  //        parameters = VectorXd::Random(transform.numAllModelParameters());
  //        testGradientAndJacobian(&errorFunction, parameters, skeleton, transform);
  //    }
  //}
}
