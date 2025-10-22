/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_util.h"

#include <gtest/gtest.h>
#include <momentum/character/character.h>
#include <momentum/character/mesh_state.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/model_parameters_error_function.h>
#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/diff_ik/fully_differentiable_distance_error_function.h>
#include <momentum/diff_ik/fully_differentiable_motion_error_function.h>
#include <momentum/diff_ik/fully_differentiable_orientation_error_function.h>
#include <momentum/diff_ik/fully_differentiable_pose_prior_error_function.h>
#include <momentum/diff_ik/fully_differentiable_position_error_function.h>
#include <momentum/diff_ik/fully_differentiable_projection_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_state_error_function.h>
#include <momentum/diff_ik/union_error_function.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/math/mppca.h>
#include <momentum/math/random.h>
#include <momentum/test/character/character_helpers.h>

#include <array>
#include <random>

using namespace momentum;

namespace {

// TODO these are copied from BodyTrackingTest
// Can we put them in some shared location?
// useRestPose here is because some constraints have discontinuous
// derivatives around the rest pose.
template <typename T>
void testConstraintDerivs(
    const Character& character,
    SkeletonErrorFunctionT<T>& errorFunction,
    const bool useRestPose,
    const Eigen::Vector3d& bodyCenter_world = Eigen::Vector3d::Zero()) {
  const ParameterTransformT<T> paramTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  const T w_init = errorFunction.getWeight();
  const T TEST_WEIGHT = 3.5f;

  const auto nParam = paramTransform.numAllModelParameters();

  for (size_t iTest = 0; iTest < 10; ++iTest) {
    errorFunction.setWeight(TEST_WEIGHT);

    std::mt19937 rng;
    std::seed_seq seq{iTest + 25678, iTest + 52312, iTest + 123876};
    rng.seed(seq);

    ModelParametersT<T> curModelParam =
        ModelParametersT<T>::Zero(paramTransform.numAllModelParameters());
    if (iTest == 0 && useRestPose) {
      curModelParam.v.setConstant(0);
    } else {
      curModelParam = randomBodyParameters(character.parameterTransform, rng).cast<T>();
    }

    // Apply the global transform
    const std::array<const char*, 3> rigidParamNames = {"root_tx", "root_ty", "root_tz"};
    for (size_t i = 0; i < 3; ++i) {
      auto paramItr =
          std::find(paramTransform.name.begin(), paramTransform.name.end(), rigidParamNames[i]);
      ASSERT_NE(paramItr, paramTransform.name.end());
      const auto paramIndex = std::distance(paramTransform.name.begin(), paramItr);
      curModelParam[paramIndex] += T(0.1 * bodyCenter_world[i]);
    }

    JointParametersT<T> curJointParams = paramTransform.apply(curModelParam);
    SkeletonStateT<T> skelState(curJointParams, skeleton);
    MeshStateT<T> meshState;
    if (errorFunction.needsMesh()) {
      meshState.update(curModelParam, skelState, character);
    }

    const double curError = errorFunction.getError(curModelParam, skelState, meshState);

    const auto nResidTerms = errorFunction.getJacobianSize();
    Eigen::VectorX<T> residualCur = Eigen::VectorX<T>::Zero(nResidTerms);
    ASSERT_EQ(nResidTerms, residualCur.rows());
    Eigen::MatrixX<T> jacobianCur = Eigen::MatrixX<T>::Zero(nResidTerms, nParam);
    ASSERT_EQ(nResidTerms, jacobianCur.rows());
    ASSERT_EQ(paramTransform.numAllModelParameters(), jacobianCur.cols());
    int usedRows = 0;
    auto error3 = errorFunction.getJacobian(
        curModelParam, skelState, meshState, jacobianCur, residualCur, usedRows);
    EXPECT_LE(relativeError(curError, (double)error3), 1e-4);

    // make sure usedRows is correct:
    for (Eigen::Index i = usedRows; i < (Eigen::Index)nResidTerms; ++i) {
      ASSERT_TRUE(jacobianCur.row(i).isZero());
      ASSERT_TRUE(residualCur(i) == 0);
    }

    const double residError = residualCur.squaredNorm();
    EXPECT_LE(relativeError(curError, residError), 1e-4);

    // TODO test the sparsity pattern.
    for (size_t iParam = 0; iParam < paramTransform.numAllModelParameters(); ++iParam) {
      const T eps = 0.0001;

      const Eigen::VectorX<T> modelParam_p =
          curModelParam.v + eps * Eigen::VectorX<T>::Unit(nParam, iParam);
      const Eigen::VectorX<T> modelParam_m =
          curModelParam.v - eps * Eigen::VectorX<T>::Unit(nParam, iParam);

      const SkeletonStateT<T> skelState_p(paramTransform.apply(modelParam_p), skeleton);
      const SkeletonStateT<T> skelState_m(paramTransform.apply(modelParam_m), skeleton);

      MeshStateT<T> meshState_p;
      MeshStateT<T> meshState_m;
      if (errorFunction.needsMesh()) {
        meshState_p.update(modelParam_p, skelState_p, character);
        meshState_m.update(modelParam_m, skelState_m, character);
      }

      Eigen::VectorX<T> residual_p = Eigen::VectorX<T>::Zero(nResidTerms);
      Eigen::MatrixX<T> jacobian_p = Eigen::MatrixX<T>::Zero(nResidTerms, nParam);
      errorFunction.getJacobian(
          modelParam_p, skelState_p, meshState_p, jacobian_p, residual_p, usedRows);

      Eigen::VectorX<T> residual_m = Eigen::VectorX<T>::Zero(nResidTerms);
      Eigen::MatrixX<T> jacobian_m = Eigen::MatrixX<T>::Zero(nResidTerms, nParam);
      errorFunction.getJacobian(
          modelParam_m, skelState_m, meshState_m, jacobian_m, residual_m, usedRows);

      for (size_t k = 0; k < nResidTerms; ++k) {
        const double jacEst = ((double)residual_p(k) - (double)residual_m(k)) / (2.0 * (double)eps);
        const double jacComp = jacobianCur(k, iParam);

        if (jacEst == 0 && jacComp == 0) {
          continue;
        }

        EXPECT_LE(relativeError(jacEst, jacComp), 0.012);
      }
    }

    Eigen::VectorX<T> gradient(nParam);
    gradient.setZero();
    auto error2 = errorFunction.getGradient(curModelParam, skelState, meshState, gradient);
    EXPECT_LE(relativeError(curError, (double)error2), 1e-4);

    const Eigen::VectorXd grad2 =
        2.0 * jacobianCur.template cast<double>().transpose() * residualCur.template cast<double>();

    // Validate the gradient:
    for (size_t iParam = 0; iParam < paramTransform.numAllModelParameters(); ++iParam) {
      const T eps = 1e-3f;

      // This is a much more accurate estimate of the gradient (grad = J^T*r) so
      // we can use a much smaller eps.
      const double grad2Val = grad2(iParam);
      const double gradExact = gradient(iParam);
      EXPECT_LE(relativeError(grad2Val, gradExact), 1e-4);

      const Eigen::VectorX<T> modelParam_p =
          curModelParam.v + eps * Eigen::VectorX<T>::Unit(nParam, iParam);
      const Eigen::VectorX<T> modelParam_m =
          curModelParam.v - eps * Eigen::VectorX<T>::Unit(nParam, iParam);

      const SkeletonStateT<T> skelState_p(paramTransform.apply(modelParam_p), skeleton);
      const SkeletonStateT<T> skelState_m(paramTransform.apply(modelParam_m), skeleton);

      MeshStateT<T> meshState_p;
      MeshStateT<T> meshState_m;
      if (errorFunction.needsMesh()) {
        meshState_p.update(modelParam_p, skelState_p, character);
        meshState_m.update(modelParam_m, skelState_m, character);
      }

      const double error_p = errorFunction.getError(modelParam_p, skelState_p, meshState_p);
      const double error_m = errorFunction.getError(modelParam_m, skelState_m, meshState_m);

      const double gradEst = (error_p - error_m) / (2.0 * eps);

      if (gradEst == 0 && gradExact == 0) {
        continue;
      }

      // Have to use a big epsilon here because of floating point issues.
      EXPECT_LE(relativeError(gradEst, gradExact), 0.1);
    }

    // Verify that the weight is handled as expected:
    {
      const T s = 2.0f;
      errorFunction.setWeight(s * TEST_WEIGHT);
      const double scaledError = errorFunction.getError(curModelParam, skelState, meshState);
      EXPECT_LE(relativeError(s * curError, scaledError), 1e-4);
      errorFunction.setWeight(TEST_WEIGHT);
    }
  }

  errorFunction.setWeight(w_init);
}

template <typename T>
void testInputDerivs(
    const Character& character,
    FullyDifferentiableSkeletonErrorFunctionT<T>& errorFunction,
    T maxRelativeError = 2e-2) {
  const ParameterTransformT<T> parameterTransform = character.parameterTransform.cast<T>();

  SCOPED_TRACE("testInputDerivs");

  const ModelParametersT<T> curModelParam =
      Eigen::VectorX<T>::Random(parameterTransform.numAllModelParameters());
  const JointParametersT<T> curJointParam = parameterTransform.apply(curModelParam);
  SkeletonStateT<T> skelState(curJointParam, character.skeleton);
  MeshStateT<T> meshState;

  auto getGradient = [&]() {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());
    dynamic_cast<SkeletonErrorFunctionT<T>&>(errorFunction)
        .getGradient(curModelParam, skelState, meshState, result);
    return result;
  };

  const T eps = 1e-4f;
  const Eigen::VectorX<T> grad_init = getGradient();
  for (const auto& inputName : errorFunction.inputs()) {
    SCOPED_TRACE(inputName);
    const Eigen::VectorX<T> inputVal_init = errorFunction.getInput(inputName);
    const auto inputSize = inputVal_init.size();
    ASSERT_EQ(inputSize, errorFunction.getInputSize(inputName));

    {
      // Test setInput:
      Eigen::VectorX<T> testInput = inputVal_init + Eigen::VectorX<T>::Ones(inputSize);
      errorFunction.setInput(inputName, testInput);
      const Eigen::VectorX<T> inputVal_cur = errorFunction.getInput(inputName);
      EXPECT_GT((inputVal_cur - inputVal_init).norm(), 0.25f);
      errorFunction.setInput(inputName, inputVal_init);
    }

    Eigen::MatrixX<T> dGrad_dInput_exact(parameterTransform.numAllModelParameters(), inputSize);
    for (size_t kParam = 0; kParam < parameterTransform.numAllModelParameters(); ++kParam) {
      Eigen::VectorX<T> e_k =
          Eigen::VectorX<T>::Unit(parameterTransform.numAllModelParameters(), kParam);
      dGrad_dInput_exact.row(kParam) =
          errorFunction.d_gradient_d_input_dot(inputName, curModelParam, skelState, e_k);
    }

    Eigen::MatrixX<T> dGrad_dInput_approx(parameterTransform.numAllModelParameters(), inputSize);
    for (Eigen::Index jInput = 0; jInput < inputSize; ++jInput) {
      Eigen::VectorX<T> inputVal_plus = inputVal_init;
      inputVal_plus(jInput) += eps;
      errorFunction.setInput(inputName, inputVal_plus);
      const Eigen::VectorX<T> grad_plus = getGradient();
      errorFunction.setInput(inputName, inputVal_init);

      dGrad_dInput_approx.col(jInput) = (grad_plus - grad_init) / eps;
    }

    for (size_t kParam = 0; kParam < parameterTransform.numAllModelParameters(); ++kParam) {
      SCOPED_TRACE(kParam);
      for (Eigen::Index jInput = 0; jInput < inputSize; ++jInput) {
        SCOPED_TRACE(jInput);
        const T v_approx = dGrad_dInput_approx(kParam, jInput);
        const T v_exact = dGrad_dInput_exact(kParam, jInput);
        EXPECT_LT(relativeError<T>(v_approx, v_exact), maxRelativeError)
            << "Error function        : " << typeid(errorFunction).name() << "\n"
            << "input                 : " << inputName << "\n"
            << "jInput                : " << jInput << "\n"
            << "kParam                : " << kParam << "\n"
            << "finite difference     : " << v_approx << "\n"
            << "d_gradient_d_input_dot: " << v_exact << "\n";
      }
    }
  }
}

TEST(ErrorFunction, FullyDifferentiablePositionErrorFunction) {
  SCOPED_TRACE("FullyDifferentiablePositionErrorFunction");

  const Character character = createTestCharacter();

  std::mt19937 rng;
  std::uniform_real_distribution<float> unif;

  FullyDifferentiablePositionErrorFunction errf(character.skeleton, character.parameterTransform);
  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    errf.addConstraint(
        PositionConstraint(randomVec(rng, 3), randomVec(rng, 3), i, 2.5f + unif(rng)));
  }

  testConstraintDerivs<float>(character, errf, true);
  testInputDerivs<float>(character, errf);
}

TEST(ErrorFunction, FullyDifferentiableOrientationErrorFunction) {
  SCOPED_TRACE("FullyDifferentiableOrientationErrorFunction");

  const Character character = createTestCharacter();

  std::mt19937 rng;
  std::uniform_real_distribution<float> unif;

  // There is actually a UnitRandom function on Eigen::Quaternion but it doesn't
  // allow you to provide a random generator or seed or anything, so I don't know
  // how to guarantee it is deterministic.  This is a lot less principled but
  // should work fine for generating noise to use in test cases.
  auto makeRandomQuat = [](std::mt19937& rng) {
    return Eigen::Quaternionf(Eigen::Vector4f(randomVec(rng, 4))).normalized();
  };

  FullyDifferentiableOrientationErrorFunction errf(
      character.skeleton, character.parameterTransform);
  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    errf.addConstraint(
        OrientationConstraint(makeRandomQuat(rng), makeRandomQuat(rng), i, 2.5 + unif(rng)));
  }

  testConstraintDerivs<float>(character, errf, true);
  testInputDerivs<float>(character, errf);
}

TEST(ErrorFunction, UnionErrorFunction) {
  SCOPED_TRACE("UnionErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  std::mt19937 rng;
  std::uniform_real_distribution<float> unif;

  auto errf_1 = std::make_shared<FullyDifferentiablePositionErrorFunction>(
      skeleton, character.parameterTransform);
  errf_1->setWeight(3);
  auto errf_2 = std::make_shared<FullyDifferentiablePositionErrorFunction>(
      skeleton, character.parameterTransform);
  // empty error function just to make sure it doesn't crash:
  auto errf_3 = std::make_shared<FullyDifferentiablePositionErrorFunction>(
      skeleton, character.parameterTransform);
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    errf_1->addConstraint(
        PositionConstraint(randomVec(rng, 3), randomVec(rng, 3), i, 2.5f + unif(rng)));
    errf_2->addConstraint(
        PositionConstraint(randomVec(rng, 3), randomVec(rng, 3), i, 2.5f + unif(rng)));
  }

  UnionErrorFunction errf(skeleton, character.parameterTransform, {errf_1, errf_2, errf_3});
  testConstraintDerivs<float>(character, errf, true);
  testInputDerivs<float>(character, errf);

  // Make sure it's actually the sum
  ModelParameters curModelParams =
      Eigen::VectorXf::Zero(character.parameterTransform.numAllModelParameters());
  JointParameters curJointParams = character.parameterTransform.apply(curModelParams);
  SkeletonState skelState(curJointParams, skeleton);
  const float w = 3;
  errf.setWeight(w);
  MeshState emptyMeshState{};
  ASSERT_NEAR(
      errf.getError(curModelParams, skelState, emptyMeshState),
      w *
          (errf_1->getError(curModelParams, skelState, emptyMeshState) +
           errf_2->getError(curModelParams, skelState, emptyMeshState) +
           errf_3->getError(curModelParams, skelState, emptyMeshState)),
      1e-4);
}

TEST(ErrorFunction, ModelParametersErrorFunction) {
  SCOPED_TRACE("ModelParametersErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  std::mt19937 rng;
  std::uniform_real_distribution<double> unif;

  FullyDifferentiableMotionErrorFunctionT<double> errf(skeleton, character.parameterTransform);
  ModelParametersT<double> targetParams(character.parameterTransform.numAllModelParameters());
  Eigen::VectorXd targetWeights(character.parameterTransform.numAllModelParameters());

  for (size_t i = 0; i < character.parameterTransform.numAllModelParameters(); ++i) {
    targetParams(i) = unif(rng);
    targetWeights(i) = std::abs(unif(rng));
  }

  errf.setTargetParameters(targetParams, targetWeights);

  testConstraintDerivs<double>(character, errf, true);
  testInputDerivs<double>(character, errf);
}

TEST(ErrorFunction, StateErrorFunction) {
  SCOPED_TRACE("StateErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  std::mt19937 rng;
  std::uniform_real_distribution<double> unif;

  auto fillVec = [&rng, &unif](Eigen::Ref<Eigen::VectorXd> vec) {
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
      vec(i) = unif(rng);
    }
  };

  ModelParametersT<double> targetParams(character.parameterTransform.numAllModelParameters());
  Eigen::VectorXd positionWeights(skeleton.joints.size());
  Eigen::VectorXd orientationWeights(skeleton.joints.size());

  fillVec(targetParams.v);
  fillVec(positionWeights);
  fillVec(orientationWeights);

  const SkeletonStated skelState(
      character.parameterTransform.cast<double>().apply(targetParams), skeleton);
  FullyDifferentiableStateErrorFunctionT<double> stateError(skeleton, character.parameterTransform);
  stateError.setTargetState(skelState);
  stateError.setTargetWeights(positionWeights, orientationWeights);

  testConstraintDerivs<double>(character, stateError, true);
  testInputDerivs<double>(character, stateError);
}

namespace {

template <typename T>
Eigen::MatrixX<T> randomMatrix(Eigen::Index nRows, Eigen::Index nCols, std::mt19937& rng) {
  std::normal_distribution<T> norm;

  Eigen::MatrixX<T> result(nRows, nCols);
  for (Eigen::Index i = 0; i < nRows; ++i) {
    for (Eigen::Index j = 0; j < nCols; ++j) {
      result(i, j) = norm(rng);
    }
  }
  return result;
}

} // namespace

TEST(ErrorFunction, PosePriorErrorFunction) {
  SCOPED_TRACE("PosePriorErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  std::vector<std::string> paramNames = {"joint1_rx", "shared_rz", "joint2_rx"};

  std::mt19937 rng;

  // Input dimensionality:
  const int d = paramNames.size();

  // Number of mixtures:
  const int p = 2;

  Eigen::VectorXd pi = randomMatrix<double>(p, 1, rng).array().abs();
  pi /= pi.sum();
  Eigen::MatrixXd mmu = randomMatrix<double>(p, d, rng);
  Eigen::VectorXd sigma = randomMatrix<double>(p, 1, rng).array().abs();

  std::vector<Eigen::MatrixXd> W(p);
  const int q = 2;
  for (int i = 0; i < p; ++i) {
    W[i] = randomMatrix<double>(d, q, rng);
  }

  FullyDifferentiablePosePriorErrorFunctionT<double> posePriorError(
      skeleton, character.parameterTransform, paramNames);
  posePriorError.setPosePrior(pi, mmu, W, sigma);

  auto mppca = std::make_shared<Mppcad>();
  mppca->set(pi, mmu, W, sigma.array().square());
  mppca->names = paramNames;

  PosePriorErrorFunctionT<double> basicPosePriorError(
      skeleton, character.parameterTransform, mppca);
  basicPosePriorError.setPosePrior(mppca);
  testConstraintDerivs<double>(character, basicPosePriorError, true);

  testConstraintDerivs<double>(character, posePriorError, true);
  testInputDerivs<double>(character, posePriorError);
}

TEST(ErrorFunction, ProjectionErrorFunction) {
  SCOPED_TRACE("ProjectionErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  using T = double;

  Random<> rng(12345);

  FullyDifferentiableProjectionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, -FLT_MAX);
  // Make a few projection constraints to ensure that at least one of them is active, since
  // projections are ignored behind the camera

  Eigen::Matrix4<T> projection = Eigen::Matrix4<T>::Identity();
  projection(2, 3) = 10;
  for (int i = 0; i < 5; ++i) {
    errorFunction.addConstraint(ProjectionConstraintDataT<T>{
        (projection + rng.uniformAffine3<T>().matrix()).topRows(3),
        rng.uniform<size_t>(size_t(0), size_t(2)),
        rng.normal<Vector3<T>>(Vector3<T>::Zero(), Vector3<T>::Ones()),
        rng.uniform<T>(0.1, 2.0),
        rng.normal<Vector2<T>>(Vector2<T>::Zero(), Vector2<T>::Ones())});
  }

  testConstraintDerivs<double>(character, errorFunction, true);
  testInputDerivs<double>(character, errorFunction);
}

TEST(ErrorFunction, DistanceErrorFunction) {
  SCOPED_TRACE("DistanceErrorFunction");

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;

  using T = double;

  Random<> rng(12345);

  FullyDifferentiableDistanceErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("Distance Constraint Test");
    DistanceConstraintDataT<T> constraintData;
    constraintData.parent = 1;
    constraintData.offset = Vector3<T>::Random();
    constraintData.origin = Vector3<T>::Random();
    constraintData.target = 2.3f;
    constraintData.weight = TEST_WEIGHT_VALUE;
    errorFunction.addConstraint(constraintData);
  }

  testConstraintDerivs<double>(character, errorFunction, true);
  testInputDerivs<double>(character, errorFunction);
}

} // anonymous namespace
