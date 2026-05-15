/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/acceleration_sequence_error_function.h"
#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"
#include "momentum/character_sequence_solver/multipose_solver.h"
#include "momentum/character_sequence_solver/multipose_solver_function.h"
#include "momentum/character_sequence_solver/sequence_cholesky_solver.h"
#include "momentum/character_sequence_solver/sequence_solver.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_sequence_solver/state_sequence_error_function.h"
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
#include <cmath>

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
Eigen::VectorX<T> deterministicParameters(Eigen::Index nParams, size_t seed) {
  Eigen::VectorX<T> result(nParams);
  for (Eigen::Index i = 0; i < nParams; ++i) {
    result(i) = static_cast<T>(0.25 * std::sin(0.37 * static_cast<double>(i + 1 + 17 * seed)));
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
      deterministicParameters<T>(parameterTransform.numAllModelParameters(), 0);

  positionErrors.resize(nFrames);
  orientErrors.resize(nFrames);
  targetParams.resize(nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    Eigen::VectorX<T> randomParams_cur =
        deterministicParameters<T>(parameterTransform.numAllModelParameters(), iFrame + 1);
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
      positionErrors[iFrame]->addConstraint(
          PositionDataT<T>(
              Eigen::Vector3<T>::Zero(),
              skelState_cur.jointState[iJoint].translation().template cast<T>(),
              iJoint,
              1.0));
      orientErrors[iFrame]->addConstraint(
          OrientationDataT<T>(
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
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
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

  const size_t nFrames = 3;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
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
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    sequenceSolverFunction.addSequenceErrorFunction(iFrame, smoothnessError);
  }

  auto solverOptionsSingle = SequenceSolverOptions(test::defaultSolverOptions());
  solverOptionsSingle.multithreaded = false;

  auto solverOptionsMulti = SequenceSolverOptions(test::defaultSolverOptions());
  solverOptionsMulti.multithreaded = true;

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

TYPED_TEST(SequenceSolverTest, CompareCholesky) {
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

  SequenceSolverFunctionT<T> sequenceSolverFunctionQr(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  SequenceSolverFunctionT<T> sequenceSolverFunctionCholesky(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunctionQr.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunctionQr.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
    sequenceSolverFunctionCholesky.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunctionCholesky.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    sequenceSolverFunctionQr.addSequenceErrorFunction(iFrame, smoothnessError);
    sequenceSolverFunctionCholesky.addSequenceErrorFunction(iFrame, smoothnessError);
  }

  auto solverOptionsQr = SequenceSolverOptions();
  solverOptionsQr.minIterations = 0;
  solverOptionsQr.maxIterations = 1;
  solverOptionsQr.threshold = 0.0f;
  solverOptionsQr.regularization = 0.37f;
  solverOptionsQr.multithreaded = false;

  auto solverOptionsCholesky = SequenceCholeskySolverOptions();
  solverOptionsCholesky.minIterations = solverOptionsQr.minIterations;
  solverOptionsCholesky.maxIterations = solverOptionsQr.maxIterations;
  solverOptionsCholesky.threshold = solverOptionsQr.threshold;
  solverOptionsCholesky.regularization = solverOptionsQr.regularization;
  solverOptionsCholesky.multithreaded = false;
  solverOptionsCholesky.chunkSize = 1;

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunctionQr.getJoinedParameterVector();

  SequenceSolverT<T> solverQr(solverOptionsQr, &sequenceSolverFunctionQr);
  Eigen::VectorX<T> parametersQr = parametersInit;
  solverQr.solve(parametersQr);
  const T errQr = sequenceSolverFunctionQr.getError(parametersQr);

  SequenceCholeskySolverT<T> solverCholesky(solverOptionsCholesky, &sequenceSolverFunctionCholesky);
  Eigen::VectorX<T> parametersCholesky = parametersInit;
  solverCholesky.solve(parametersCholesky);
  const T errCholesky = sequenceSolverFunctionCholesky.getError(parametersCholesky);

  EXPECT_NEAR(errQr, errCholesky, Eps<T>(5e-5f, 1e-8));
  EXPECT_NEAR((parametersQr - parametersCholesky).norm(), T(0), Eps<T>(1e-4f, 1e-8));
}

TYPED_TEST(SequenceSolverTest, CompareCholeskyScalarLdltWithoutUniversalParameters) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  const ParameterSet universalParams;
  const size_t nFrames = 4;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  const auto populateSolverFunction = [&](SequenceSolverFunctionT<T>& result) {
    for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      result.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
      result.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
    }

    auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
    for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
      result.addSequenceErrorFunction(iFrame, smoothnessError);
    }
  };

  SequenceSolverFunctionT<T> sequenceSolverFunctionBlock(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  populateSolverFunction(sequenceSolverFunctionBlock);
  SequenceSolverFunctionT<T> sequenceSolverFunctionScalar(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  populateSolverFunction(sequenceSolverFunctionScalar);

  auto solverOptionsBlock = SequenceCholeskySolverOptions();
  solverOptionsBlock.minIterations = 0;
  solverOptionsBlock.maxIterations = 1;
  solverOptionsBlock.threshold = 0.0f;
  solverOptionsBlock.regularization = 0.37f;
  solverOptionsBlock.multithreaded = false;
  solverOptionsBlock.chunkSize = 1;

  auto solverOptionsScalar = solverOptionsBlock;
  solverOptionsScalar.useBlockLdlt = false;

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunctionBlock.getJoinedParameterVector();

  SequenceCholeskySolverT<T> solverBlock(solverOptionsBlock, &sequenceSolverFunctionBlock);
  Eigen::VectorX<T> parametersBlock = parametersInit;
  solverBlock.solve(parametersBlock);
  const T errBlock = sequenceSolverFunctionBlock.getError(parametersBlock);

  SequenceCholeskySolverT<T> solverScalar(solverOptionsScalar, &sequenceSolverFunctionScalar);
  Eigen::VectorX<T> parametersScalar = parametersInit;
  solverScalar.solve(parametersScalar);
  const T errScalar = sequenceSolverFunctionScalar.getError(parametersScalar);

  EXPECT_NEAR(errBlock, errScalar, Eps<T>(1e-5f, 1e-8));
  EXPECT_NEAR((parametersBlock - parametersScalar).norm(), T(0), Eps<T>(1e-5f, 1e-8));
}

TYPED_TEST(SequenceSolverTest, CompareCholeskyMultithreaded) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 4;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  const auto populateSolverFunction = [&](SequenceSolverFunctionT<T>& result) {
    for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      result.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
      result.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
    }

    auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctionT<T>>(character);
    for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
      result.addSequenceErrorFunction(iFrame, smoothnessError);
    }
  };

  SequenceSolverFunctionT<T> sequenceSolverFunctionQr(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  populateSolverFunction(sequenceSolverFunctionQr);
  SequenceSolverFunctionT<T> sequenceSolverFunctionSerial(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  populateSolverFunction(sequenceSolverFunctionSerial);

  auto solverOptionsQr = SequenceSolverOptions();
  solverOptionsQr.minIterations = 0;
  solverOptionsQr.maxIterations = 1;
  solverOptionsQr.threshold = 0.0f;
  solverOptionsQr.regularization = 0.37f;
  solverOptionsQr.multithreaded = false;

  auto solverOptionsSerial = SequenceCholeskySolverOptions();
  solverOptionsSerial.minIterations = 0;
  solverOptionsSerial.maxIterations = 1;
  solverOptionsSerial.threshold = 0.0f;
  solverOptionsSerial.regularization = 0.37f;
  solverOptionsSerial.multithreaded = false;
  solverOptionsSerial.chunkSize = 1;

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunctionQr.getJoinedParameterVector();

  SequenceSolverT<T> solverQr(solverOptionsQr, &sequenceSolverFunctionQr);
  Eigen::VectorX<T> parametersQr = parametersInit;
  solverQr.solve(parametersQr);
  const T errQr = sequenceSolverFunctionQr.getError(parametersQr);

  SequenceCholeskySolverT<T> solverSerial(solverOptionsSerial, &sequenceSolverFunctionSerial);
  Eigen::VectorX<T> parametersSerial = parametersInit;
  solverSerial.solve(parametersSerial);
  const T errSerial = sequenceSolverFunctionSerial.getError(parametersSerial);

  EXPECT_NEAR(errQr, errSerial, Eps<T>(5e-5f, 1e-8));
  EXPECT_NEAR((parametersQr - parametersSerial).norm(), T(0), Eps<T>(1e-4f, 5e-8));

  for (const size_t chunkSize : {size_t(1), size_t(2), size_t(3), size_t(8)}) {
    for (const bool useDoublePrecisionNormalEquations : {false, true}) {
      SequenceSolverFunctionT<T> sequenceSolverFunctionMulti(
          character, castedCharacterParameterTransform, problem.universalParams, nFrames);
      populateSolverFunction(sequenceSolverFunctionMulti);
      auto solverOptionsMulti = solverOptionsSerial;
      solverOptionsMulti.multithreaded = true;
      solverOptionsMulti.chunkSize = chunkSize;
      solverOptionsMulti.useDoublePrecisionNormalEquations = useDoublePrecisionNormalEquations;

      SequenceCholeskySolverT<T> solverMulti(solverOptionsMulti, &sequenceSolverFunctionMulti);
      Eigen::VectorX<T> parametersMulti = parametersInit;
      solverMulti.solve(parametersMulti);
      const T errMulti = sequenceSolverFunctionMulti.getError(parametersMulti);

      EXPECT_NEAR(errQr, errMulti, Eps<T>(5e-5f, 1e-8));
      EXPECT_NEAR((parametersQr - parametersMulti).norm(), T(0), Eps<T>(1e-4f, 5e-8));
      EXPECT_NEAR(errSerial, errMulti, Eps<T>(1e-5f, 1e-8));
      EXPECT_NEAR((parametersSerial - parametersMulti).norm(), T(0), Eps<T>(1e-5f, 1e-8));
    }
  }
}

TYPED_TEST(SequenceSolverTest, CompareCholeskyMixedSequenceBoundaryCases) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 5;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  const auto populateSolverFunction = [&](SequenceSolverFunctionT<T>& result) {
    for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      result.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
      result.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
    }

    auto stateError = std::make_shared<StateSequenceErrorFunctionT<T>>(character);
    for (size_t iFrame = 0; iFrame < nFrames - 1; ++iFrame) {
      result.addSequenceErrorFunction(iFrame, stateError);
    }

    auto accelerationError = std::make_shared<AccelerationSequenceErrorFunctionT<T>>(character);
    for (size_t iFrame = 0; iFrame < nFrames - 2; ++iFrame) {
      result.addSequenceErrorFunction(iFrame, accelerationError);
    }
  };

  SequenceSolverFunctionT<T> sequenceSolverFunctionSerial(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  populateSolverFunction(sequenceSolverFunctionSerial);

  auto solverOptionsSerial = SequenceCholeskySolverOptions();
  solverOptionsSerial.minIterations = 0;
  solverOptionsSerial.maxIterations = 1;
  solverOptionsSerial.threshold = 0.0f;
  solverOptionsSerial.regularization = 0.37f;
  solverOptionsSerial.multithreaded = false;
  solverOptionsSerial.chunkSize = 1;

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunctionSerial.getJoinedParameterVector();

  SequenceCholeskySolverT<T> solverSerial(solverOptionsSerial, &sequenceSolverFunctionSerial);
  Eigen::VectorX<T> parametersSerial = parametersInit;
  solverSerial.solve(parametersSerial);
  const T errSerial = sequenceSolverFunctionSerial.getError(parametersSerial);

  for (const size_t chunkSize : {size_t(1), size_t(2), size_t(3), size_t(8)}) {
    for (const bool useDoublePrecisionNormalEquations : {false, true}) {
      SequenceSolverFunctionT<T> sequenceSolverFunctionMulti(
          character, castedCharacterParameterTransform, problem.universalParams, nFrames);
      populateSolverFunction(sequenceSolverFunctionMulti);
      auto solverOptionsMulti = solverOptionsSerial;
      solverOptionsMulti.multithreaded = true;
      solverOptionsMulti.chunkSize = chunkSize;
      solverOptionsMulti.useDoublePrecisionNormalEquations = useDoublePrecisionNormalEquations;

      SequenceCholeskySolverT<T> solverMulti(solverOptionsMulti, &sequenceSolverFunctionMulti);
      Eigen::VectorX<T> parametersMulti = parametersInit;
      solverMulti.solve(parametersMulti);
      const T errMulti = sequenceSolverFunctionMulti.getError(parametersMulti);

      EXPECT_NEAR(errSerial, errMulti, Eps<T>(5e-5f, 1e-8));
      EXPECT_NEAR((parametersSerial - parametersMulti).norm(), T(0), Eps<T>(5e-5f, 1e-8));
    }
  }
}

TYPED_TEST(SequenceSolverTest, CompareCholeskyOnlyUniversalParameters) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 4;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunctionCholesky(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);
  momentum::ParameterTransformT<T> parameterTransform = character.parameterTransform.cast<T>();
  SkeletonSolverFunctionT<T> basicSolverFunction(character, parameterTransform);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunctionCholesky.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunctionCholesky.addErrorFunction(iFrame, problem.orientErrors[iFrame]);

    basicSolverFunction.addErrorFunction(problem.positionErrors[iFrame]);
    basicSolverFunction.addErrorFunction(problem.orientErrors[iFrame]);
  }

  auto solverOptionsCholesky = SequenceCholeskySolverOptions(test::defaultSolverOptions());
  solverOptionsCholesky.multithreaded = true;
  solverOptionsCholesky.chunkSize = 2;
  solverOptionsCholesky.useDoublePrecisionNormalEquations = true;

  SequenceCholeskySolverT<T> solverCholesky(solverOptionsCholesky, &sequenceSolverFunctionCholesky);
  solverCholesky.setEnabledParameters(universalParams);
  const Eigen::VectorX<T> parametersInit =
      sequenceSolverFunctionCholesky.getJoinedParameterVector();
  test::checkAndTimeSolver<T>(
      sequenceSolverFunctionCholesky, solverCholesky, parametersInit, universalParams);
  const auto modelParamsSequence = sequenceSolverFunctionCholesky.getUniversalParameters();

  GaussNewtonSolverQRT<T> solverGn(test::defaultSolverOptions(), &basicSolverFunction);
  solverGn.setEnabledParameters(universalParams);
  momentum::ModelParametersT<T> basicModelParams =
      momentum::ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters());
  solverGn.solve(basicModelParams.v);

  ASSERT_LT((modelParamsSequence.v - basicModelParams.v).norm(), 1e-4);
}

TYPED_TEST(SequenceSolverTest, MixedFrameCountErrorFunctions) {
  using T = typename TestFixture::Type;

  // Test combining error functions with different frame counts.
  // StateSequenceErrorFunction uses 2 frames, AccelerationSequenceErrorFunction uses 3 frames.
  // This tests that the solver correctly passes the right number of frames to each error function.
  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet enabledParams;
  enabledParams.set();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  // Need at least 4 frames to have both:
  // - StateSequenceErrorFunction at frames 0,1,2 (2-frame windows)
  // - AccelerationSequenceErrorFunction at frames 0,1 (3-frame windows)
  const size_t nFrames = 5;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      character, castedCharacterParameterTransform, problem.universalParams, nFrames);

  // Add per-frame position errors
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);
  }

  // Add 2-frame sequence error function (StateSequenceErrorFunction)
  auto stateError = std::make_shared<StateSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    sequenceSolverFunction.addSequenceErrorFunction(iFrame, stateError);
  }

  // Add 3-frame sequence error function (AccelerationSequenceErrorFunction)
  // This is the key part of the test: mixing 2-frame and 3-frame error functions
  auto accelerationError = std::make_shared<AccelerationSequenceErrorFunctionT<T>>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 2); ++iFrame) {
    sequenceSolverFunction.addSequenceErrorFunction(iFrame, accelerationError);
  }

  const Eigen::VectorX<T> parametersInit = sequenceSolverFunction.getJoinedParameterVector();

  // This should not crash - the bug was that passing a 3-frame skelStates span to a 2-frame
  // error function would fail the MT_CHECK(skelStates.size() == 2) assertion.
  SequenceSolverT<T> solver_sq(test::defaultSolverOptions(), &sequenceSolverFunction);
  const T err_sq = test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_sq, parametersInit);

  // Also test with GaussNewton to verify correctness
  GaussNewtonSolverT<T> solver_gn(test::defaultSolverOptions(), &sequenceSolverFunction);
  const T err_gn = test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_gn, parametersInit);

  // SequenceSolver should do at least as well as Gauss-Newton.
  EXPECT_LE(err_sq, (T(1.001) * err_gn + T(0.001)));
}

TYPED_TEST(SequenceSolverTest, OnlySharedParams) {
  using T = typename TestFixture::Type;

  // Solving a multi-frame system where only the shared parameters are enabled should produce the
  // same result as solving a single frame system with all the error functions from all the frames,
  // since in the first case we'll have the same parameters at every frame and hence all the error
  // functions will be summed.
  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();

  ParameterSet universalParams;
  universalParams.set(2);
  universalParams.set(4);
  universalParams.set(7);

  const size_t nFrames = 5;
  MultiPoseTestProblem<T> problem(character, nFrames, universalParams);

  SequenceSolverFunctionT<T> sequenceSolverFunction(
      character, castedCharacterParameterTransform, universalParams, nFrames);
  momentum::ParameterTransformT<T> parameterTransform = character.parameterTransform.cast<T>();
  SkeletonSolverFunctionT<T> basicSolverFunction(character, parameterTransform);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sequenceSolverFunction.addErrorFunction(iFrame, problem.positionErrors[iFrame]);
    sequenceSolverFunction.addErrorFunction(iFrame, problem.orientErrors[iFrame]);

    basicSolverFunction.addErrorFunction(problem.positionErrors[iFrame]);
    basicSolverFunction.addErrorFunction(problem.orientErrors[iFrame]);
  }

  SequenceSolverT<T> solver_mp(test::defaultSolverOptions(), &sequenceSolverFunction);
  solver_mp.setEnabledParameters(universalParams);
  const Eigen::VectorX<T> parametersInit = sequenceSolverFunction.getJoinedParameterVector();
  test::checkAndTimeSolver<T>(sequenceSolverFunction, solver_mp, parametersInit, universalParams);
  const auto modelParamsSequence = sequenceSolverFunction.getUniversalParameters();

  GaussNewtonSolverQRT<T> solver_gn(test::defaultSolverOptions(), &basicSolverFunction);
  solver_gn.setEnabledParameters(universalParams);
  momentum::ModelParametersT<T> basicModelParams =
      momentum::ModelParametersT<T>::Zero(character.parameterTransform.numAllModelParameters());
  solver_gn.solve(basicModelParams.v);

  ASSERT_LT((modelParamsSequence.v - basicModelParams.v).norm(), 1e-4);
}
