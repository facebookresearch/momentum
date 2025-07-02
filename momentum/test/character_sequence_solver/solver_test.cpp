/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"
#include "momentum/character_sequence_solver/multipose_solver.h"
#include "momentum/character_sequence_solver/multipose_solver_function.h"
#include "momentum/character_sequence_solver/sequence_solver.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_solver/gauss_newton_solver_qr.h"
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/trust_region_qr.h"
#include "momentum/math/fmt_eigen.h"
#include "momentum/solver/gauss_newton_solver.h"
#include "momentum/solver/subset_gauss_newton_solver.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/solver/solver_test_helpers.h"

#include <gtest/gtest.h>

#include <chrono>

using namespace momentum;

using Types = testing::Types<float, double>;

namespace {

std::vector<int> toIntVector(const ParameterSet& params, const Character& character) {
  std::vector<int> result(character.parameterTransform.numAllModelParameters());
  for (size_t i = 0; i < result.size(); ++i) {
    if (params.test(i)) {
      result[i] = 1;
    }
  }

  return result;
}

template <typename T>
struct MultiPoseTestProblem {
  MultiPoseTestProblem(const Character& character, size_t nFrames, ParameterSet universalParams_in);

  std::vector<std::shared_ptr<PositionErrorFunctionT<T>>> positionErrors;
  std::vector<std::shared_ptr<OrientationErrorFunctionT<T>>> orientErrors;
  std::vector<Eigen::VectorX<T>> targetParams;
  ParameterSet universalParams;
};

template <typename T>
MultiPoseTestProblem<T>::MultiPoseTestProblem(
    const Character& character,
    size_t nFrames,
    ParameterSet universalParams_in)
    : universalParams(universalParams_in) {
  const ParameterTransformT<T> parameterTransform = character.parameterTransform.cast<T>();
  Eigen::VectorX<T> randomParams_base =
      Eigen::VectorX<T>::Random(parameterTransform.numAllModelParameters());

  positionErrors.resize(nFrames);
  orientErrors.resize(nFrames);
  targetParams.resize(nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    Eigen::VectorX<T> randomParams_cur =
        Eigen::VectorX<T>::Random(parameterTransform.numAllModelParameters());
    for (int i = 0; i < parameterTransform.numAllModelParameters(); ++i) {
      if (universalParams.test(i)) {
        randomParams_cur[i] = randomParams_base[i];
      }
    }

    targetParams[iFrame] = randomParams_cur;

    SkeletonStateT<T> skelState_cur(parameterTransform.apply(randomParams_cur), character.skeleton);
    positionErrors[iFrame] = std::make_unique<PositionErrorFunctionT<T>>(character);
    orientErrors[iFrame] = std::make_unique<OrientationErrorFunctionT<T>>(character);
    for (size_t iJoint = 0; iJoint < skelState_cur.jointState.size(); ++iJoint) {
      positionErrors[iFrame]->addConstraint(PositionDataT<T>(
          Eigen::Vector3<T>::Zero(),
          skelState_cur.jointState[iJoint].translation().template cast<T>(),
          iJoint,
          1.0));
      orientErrors[iFrame]->addConstraint(OrientationDataT<T>(
          Eigen::Quaternion<T>::Identity(),
          skelState_cur.jointState[iJoint].rotation().template cast<T>(),
          iJoint,
          1.0));
    }
  }
}

} // namespace

template <typename T>
struct MultiposeSolverTest : public testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(MultiposeSolverTest, Types);

TYPED_TEST(MultiposeSolverTest, CompareGaussNewton) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  ParameterSet enabledParams;
  enabledParams.set();

  const size_t nFrames = 2;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  MultiposeSolverFunctionT<T> solverFunction(
      &character.skeleton,
      &castedCharacterParameterTransform,
      toIntVector(problem.universalParams, character),
      nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    solverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame].get());
    solverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame].get());
  }

  const Eigen::VectorX<T> parametersInit = solverFunction.getJoinedParameterVector();

  GaussNewtonSolverT<T> solver_gn(test::defaultSolverOptions(), &solverFunction);
  const T err_gn = test::checkAndTimeSolver<T>(solverFunction, solver_gn, parametersInit);
  MultiposeSolverT<T> solver_mp(test::defaultSolverOptions(), &solverFunction);
  const T err_mp = test::checkAndTimeSolver<T>(solverFunction, solver_mp, parametersInit);

  // Multipose solver should do at least as well as Gauss-Newton.
  EXPECT_LE(err_mp, (T(1.001) * err_gn + T(0.001)));
}

template <typename T>
struct SequenceSolverTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(SequenceSolverTest, Types);

// Sequence and multipose problems should be identical if there are no smoothness constraints:
TYPED_TEST(SequenceSolverTest, CompareSequenceMultiPose) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet enabledParams;
  enabledParams.set();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 3;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      &character.skeleton, &castedCharacterParameterTransform, problem.universalParams, nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  MultiposeSolverFunctionT<T> multiposeSolverFunction(
      &character.skeleton,
      &castedCharacterParameterTransform,
      toIntVector(problem.universalParams, character),
      nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    multiposeSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame].get());
    multiposeSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame].get());
  }

  EXPECT_EQ(sequenceSolverFunction.getNumParameters(), multiposeSolverFunction.getNumParameters());
  const Eigen::VectorX<T> parameters_init = multiposeSolverFunction.getJoinedParameterVector();

  GaussNewtonSolverOptions solverOptions;
  solverOptions.minIterations = 4;
  solverOptions.maxIterations = 40;
  solverOptions.threshold = 1000.f;

  GaussNewtonSolverT<T> solver_mp(solverOptions, &multiposeSolverFunction);
  solver_mp.setEnabledParameters(enabledParams);
  Eigen::VectorX<T> parameters_mp = parameters_init;
  solver_mp.solve(parameters_mp);

  GaussNewtonSolverT<T> solver_sq(solverOptions, &sequenceSolverFunction);
  solver_sq.setEnabledParameters(enabledParams);
  Eigen::VectorX<T> parameters_sq = parameters_init;
  solver_sq.solve(parameters_sq);

  // Since the problems are identical, the converged solutions should be close:
  const T err_mp = multiposeSolverFunction.getError(parameters_mp);
  const T err_sq = sequenceSolverFunction.getError(parameters_sq);
  EXPECT_LE(err_sq, err_mp);

  // Parameter vectors should be interchangeable too; the only differences here should
  // be due to floating point accumulation differences:
  const T err1 = multiposeSolverFunction.getError(parameters_mp);
  const T err2 = sequenceSolverFunction.getError(parameters_mp);
  EXPECT_NEAR(err1, err2, Eps<T>(1e-7f, 1e-14));
}

TYPED_TEST(SequenceSolverTest, CompareGaussNewton) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet enabledParams;
  enabledParams.set();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 5;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      &character.skeleton, &castedCharacterParameterTransform, problem.universalParams, nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    sequenceSolverFunction.addSequenceErrorFunction(iFrame, smoothnessError);
  }

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunction.getJoinedParameterVector();
  GaussNewtonSolverT<T> solver_gn(test::defaultSolverOptions(), &sequenceSolverFunction);
  const T err_gn = test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_gn, parametersInit);
  SequenceSolverT<T> solver_sq(test::defaultSolverOptions(), &sequenceSolverFunction);
  const T err_sq = test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_sq, parametersInit);

  // Multipose solver should do at least as well as Gauss-Newton.
  EXPECT_LE(err_sq, (T(1.001) * err_gn + T(0.001)));
}

TYPED_TEST(SequenceSolverTest, CompareMultithreaded) {
  using T = typename TestFixture::Type;

  // Compare a single-threaded solve against a multi-threaded solve.
  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet enabledParams;
  enabledParams.set();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 5;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      &character.skeleton, &castedCharacterParameterTransform, problem.universalParams, nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    sequenceSolverFunction.addSequenceErrorFunction(iFrame, smoothnessError);
  }

  auto solverOptionsSingle = SequenceSolverOptions(test::defaultSolverOptions());
  solverOptionsSingle.multithreaded = 0;

  auto solverOptionsMulti = SequenceSolverOptions(test::defaultSolverOptions());
  solverOptionsMulti.multithreaded = 1;

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunction.getJoinedParameterVector();

  SequenceSolverT<T> solver_st(solverOptionsSingle, &sequenceSolverFunction);
  const T err_single =
      test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_st, parametersInit);
  SequenceSolverT<T> solver_mt(solverOptionsMulti, &sequenceSolverFunction);
  const T err_multi =
      test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_mt, parametersInit);
  GaussNewtonSolverT<T> solver_gn(test::defaultSolverOptions(), &sequenceSolverFunction);
  const T err_gn = test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_gn, parametersInit);

  // Multipose solver should do at least as well as Gauss-Newton.
  EXPECT_NEAR(err_single, err_multi, Eps<T>(1e-7f, 1e-15));
  EXPECT_LE(err_single, err_gn + Eps<T>(5e-7f, 1e-14));
}
