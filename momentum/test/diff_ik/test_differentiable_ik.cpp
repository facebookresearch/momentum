/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_util.h"

#include <ceres/jet.h>
#include <gtest/gtest.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/gauss_newton_solver_qr.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/model_parameters_error_function.h>
#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/common/log.h>
#include <momentum/diff_ik/fully_differentiable_body_ik.h>
#include <momentum/diff_ik/fully_differentiable_orientation_error_function.h>
#include <momentum/diff_ik/fully_differentiable_position_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/mppca_io.h>
#include <momentum/test/character/character_helpers.h>
#include <cstdlib>
#include <random>
#include <sstream>

using namespace momentum;

// #define VERBOSE 1

TEST(DifferentiableIK, Basic) {
  using T = double;

  const Character character = createTestCharacter();
  const auto& skeleton = character.skeleton;
  const auto parameterTransform = character.parameterTransform.cast<T>();
  const auto& parameterLimits = character.parameterLimits;

  std::mt19937 rng;

  // drop half the parameters to make sure we treat un-solved parameters properly.
  const ParameterSet activeParams = [&]() {
    ParameterSet result;
    std::vector<size_t> selectedParams(parameterTransform.numAllModelParameters());
    std::iota(std::begin(selectedParams), std::end(selectedParams), 0);
    std::shuffle(selectedParams.begin(), selectedParams.end(), rng);
    selectedParams.resize(std::min<size_t>(10, parameterTransform.numAllModelParameters() / 2));
    for (const auto pi : selectedParams) {
      result.set(pi);
    }
    return result;
  }();

  std::stringstream ss;
  ss << "Parameters:\n";
  for (size_t iParam = 0; iParam < parameterTransform.numAllModelParameters(); ++iParam) {
    ss << "  " << iParam << ": " << parameterTransform.name[iParam];
    if (!activeParams.test(iParam)) {
      ss << " DISABLED.";
    }
    ss << "\n";
  }
  MT_LOGI("{}", ss.str());

  GaussNewtonSolverQROptions solverOptions;
  solverOptions.minIterations = 4;
  solverOptions.maxIterations = 40;
  solverOptions.threshold = 10.f;
  solverOptions.regularization = 1e-5f;

  ModelParameters targetParams = randomBodyParameters(character.parameterTransform, rng);
  const SkeletonState skelState_target(character.parameterTransform.apply(targetParams), skeleton);

  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions;

  SkeletonSolverFunctionT<T> solverFunction(&skeleton, &parameterTransform);

  const size_t nErrFun = 3;
  for (size_t j = 0; j < nErrFun; ++j) {
    auto positionError = std::make_shared<FullyDifferentiablePositionErrorFunctionT<T>>(
        skeleton, character.parameterTransform);
    for (size_t iJoint = 0; iJoint < skelState_target.jointState.size(); ++iJoint) {
      positionError->addConstraint(
          PositionConstraintT<T>(
              Eigen::Vector3<T>::Zero(),
              skelState_target.jointState[iJoint].translation().cast<T>() +
                  4.0 * randomVec(rng, 3).cast<T>(),
              iJoint,
              1.f));
    }
    positionError->setWeight(j + 1);
    errorFunctions.push_back(positionError);
    solverFunction.addErrorFunction(positionError);
  }

  {
    auto rotationError = std::make_shared<FullyDifferentiableOrientationErrorFunctionT<T>>(
        skeleton, character.parameterTransform);
    for (size_t iJoint = 0; iJoint < skelState_target.jointState.size(); ++iJoint) {
      rotationError->addConstraint(
          OrientationConstraintT<T>(
              Eigen::Quaternion<T>::Identity(),
              skelState_target.jointState[iJoint].rotation().cast<T>(),
              iJoint,
              1.0));
    }
    rotationError->setWeight(3);
    errorFunctions.push_back(rotationError);
    solverFunction.addErrorFunction(rotationError);
  }

  {
    auto limitError = std::make_shared<LimitErrorFunctionT<T>>(
        skeleton, character.parameterTransform, parameterLimits);
    limitError->setWeight(0.1f);
    errorFunctions.push_back(limitError);
    solverFunction.addErrorFunction(limitError);
  }

  GaussNewtonSolverQRT<T> solver(solverOptions, &solverFunction);
  solver.setEnabledParameters(activeParams);
  ModelParametersT<T> parameters_opt = targetParams.cast<T>();
  solver.solve(parameters_opt.v);

  const Eigen::VectorX<T> testVec =
      randomVec(rng, parameterTransform.numAllModelParameters()).cast<T>();

  const std::vector<ErrorFunctionDerivativesT<T>> ikDerivatives = d_modelParams_d_inputs<T>(
      skeleton, parameterTransform, activeParams, parameters_opt, solverFunction, testVec);

  const auto& errFunctions = solverFunction.getErrorFunctions();
  const T eps = 1e-3;
  for (size_t iErrFun = 0; iErrFun < errFunctions.size(); ++iErrFun) {
    auto& errFun = *errFunctions[iErrFun];
    SCOPED_TRACE(typeid(errFun).name());
    const auto& errFunDerivs = ikDerivatives[iErrFun];

    // Check the basic weight derivative, which is present for any error
    // function:
    {
      SCOPED_TRACE("Global weight");
      const auto curWeight = errFun.getWeight();
      errFun.setWeight(curWeight + eps);
      // Need to start in the same local minimum.
      ModelParametersT<T> parameters_cur = parameters_opt;
      solver.solve(parameters_cur.v);
      errFun.setWeight(curWeight);

      const T diff_est = testVec.dot((parameters_cur.v - parameters_opt.v) / eps);

      EXPECT_LE(relativeError(diff_est, errFunDerivs.gradWeight), 5e-2f);
      MT_LOGD(
          "iErrFun: {}; global weight; diff_finite_difference: {}; diff_exact: {}",
          iErrFun,
          diff_est,
          errFunDerivs.gradWeight);
    }

    // Check the derivatives wrt the inputs:
    auto* differentiableErr = dynamic_cast<FullyDifferentiableSkeletonErrorFunctionT<T>*>(&errFun);
    if (differentiableErr != nullptr) {
      for (const auto& inputName : differentiableErr->inputs()) {
        SCOPED_TRACE(inputName);
        const auto& grad_input_itr = ikDerivatives[iErrFun].gradInputs.find(inputName);
        ASSERT_NE(grad_input_itr, ikDerivatives[iErrFun].gradInputs.end());

        const Eigen::VectorX<T> initialValue = differentiableErr->getInput(inputName);
        ASSERT_EQ(grad_input_itr->second.size(), initialValue.size());

        for (int k = 0; k < initialValue.size(); ++k) {
          differentiableErr->setInput(
              inputName, initialValue + eps * Eigen::VectorX<T>::Unit(initialValue.size(), k));
          ModelParametersT<T> parameters_cur = parameters_opt;
          solver.solve(parameters_cur.v);
          differentiableErr->setInput(inputName, initialValue);

          const T diff_est = ((parameters_cur.v - parameters_opt.v) / eps).dot(testVec);
          if (diff_est == 0 && grad_input_itr->second[k] == 0) {
            continue;
          }

#if defined(__APPLE__) // TODO: Fix for mac that requires a larger epsilon
          EXPECT_LE(relativeError(diff_est, grad_input_itr->second[k]), 5e-0f);
#else
          EXPECT_LE(relativeError(diff_est, grad_input_itr->second[k]), 1e-1f);
#endif
          MT_LOGD(
              "iErrFun: {}; {}; diff_finite_difference: {}; diff_exact: {}",
              iErrFun,
              inputName,
              diff_est,
              grad_input_itr->second[k]);
        }
      }
    }
  }
}
