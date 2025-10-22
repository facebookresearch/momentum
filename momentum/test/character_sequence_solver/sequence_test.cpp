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
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_solver.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_sequence_solver/state_sequence_error_function.h"
#include "momentum/character_sequence_solver/vertex_sequence_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

namespace {

template <typename T2, typename T>
std::vector<ModelParametersT<T2>> castModelParameters(
    const std::vector<ModelParametersT<T>>& params) {
  std::vector<ModelParametersT<T2>> result;
  for (const auto& p : params) {
    result.push_back(p.template cast<T2>());
  }
  return result;
}

template <typename T>
void testGradientAndJacobian(
    SequenceErrorFunctionT<T>& errorFunction,
    const std::vector<ModelParametersT<T>>& referenceParameters,
    const Character& character,
    const float numThreshold = 1e-3f,
    const float jacThreshold = 1e-6f,
    bool checkJacobian = true) {
  const ParameterTransformd transform = character.parameterTransform.cast<double>();
  const auto np = transform.numAllModelParameters();

  const auto w_orig = errorFunction.getWeight();
  errorFunction.setWeight(3.0f);

  const auto nFrames = errorFunction.numFrames();

  ASSERT_EQ(nFrames, referenceParameters.size());

  auto parametersToSkelAndMeshStates = [&character, transform, nFrames, &errorFunction](
                                           const std::vector<ModelParametersd>& modelParams)
      -> std::tuple<std::vector<SkeletonStateT<T>>, std::vector<MeshStateT<T>>> {
    std::vector<SkeletonStateT<T>> skelStates(nFrames);
    for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      const SkeletonStated skelStated(transform.apply(modelParams[iFrame]), character.skeleton);
      skelStates[iFrame].set(skelStated);
    }

    std::vector<MeshStateT<T>> meshStates(nFrames);
    if (errorFunction.needsMesh()) {
      for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
        meshStates[iFrame].update(modelParams[iFrame], skelStates[iFrame], character);
      }
    }

    return {skelStates, meshStates};
  };

  const auto [referenceSkelStates, referenceMeshStates] =
      parametersToSkelAndMeshStates(castModelParameters<double>(referenceParameters));

  // test getError and getGradient produce the same value
  // Double precision state here limits how much errors accumulate up the
  // kinematic chain.

  Eigen::VectorX<T> anaGradient = Eigen::VectorX<T>::Zero(nFrames * np);
  const size_t jacobianSize = errorFunction.getJacobianSize();
  Eigen::MatrixX<T> jacobian = Eigen::MatrixX<T>::Zero(jacobianSize, np * nFrames);
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(jacobianSize);

  const double functionError =
      errorFunction.getError(referenceParameters, referenceSkelStates, referenceMeshStates);
  const double gradientError = errorFunction.getGradient(
      referenceParameters, referenceSkelStates, referenceMeshStates, anaGradient);
  int rows = 0;
  const double jacobianError = errorFunction.getJacobian(
      referenceParameters, referenceSkelStates, referenceMeshStates, jacobian, residual, rows);

  const Eigen::VectorX<T> jacGradient = 2.0f * jacobian.transpose() * residual;
  const T jacError = residual.dot(residual);

  {
    SCOPED_TRACE("Checking Error Value");
    if ((gradientError + functionError) != 0.0f) {
      EXPECT_LE(std::abs(gradientError - functionError) / (gradientError + functionError), 1e-6f);
    } else {
      EXPECT_NEAR(gradientError, functionError, 1e-7f);
    }
    if ((jacobianError + functionError) != 0.0f) {
      EXPECT_LE(std::abs(jacobianError - functionError) / (jacobianError + functionError), 1e-6f);
    } else {
      EXPECT_NEAR(jacobianError, functionError, 1e-7f);
    }
    if ((jacError + functionError) != 0.0f) {
      EXPECT_LE(std::abs(jacError - functionError) / (jacError + functionError), 1e-4f);
    } else {
      EXPECT_NEAR(jacError, functionError, 1e-4f);
    }
  }

  auto evalError = [&errorFunction, &parametersToSkelAndMeshStates](
                       const std::vector<ModelParametersd>& modelParameters) {
    auto [skelStates, meshStates] = parametersToSkelAndMeshStates(modelParameters);
    return errorFunction.getError(castModelParameters<T>(modelParameters), skelStates, meshStates);
  };

  // calculate numerical gradient
  constexpr double kStepSize = 1e-5;
  Eigen::VectorX<T> numGradient = Eigen::VectorX<T>::Zero(nFrames * np);
  for (size_t p = 0; p < nFrames * np; p++) {
    // perform higher-order finite differences for accuracy
    std::vector<ModelParametersd> parameters = castModelParameters<double>(referenceParameters);
    const auto iFrame = p / np;
    const auto iParam = p % np;

    parameters[iFrame](iParam) = referenceParameters[iFrame](iParam) - kStepSize;
    const double h_1 = evalError(parameters);
    parameters[iFrame](iParam) = referenceParameters[iFrame](iParam) + kStepSize;
    const double h1 = evalError(parameters);
    numGradient(p) = static_cast<T>((h1 - h_1) / (2.0 * kStepSize));
  }

  if (checkJacobian) {
    SCOPED_TRACE("Checking Numerical Jacobian");
    Eigen::MatrixX<T> numJacobian = Eigen::MatrixX<T>::Zero(jacobianSize, nFrames * np);
    for (size_t p = 0; p < nFrames * np; p++) {
      const auto iFrame = p / np;
      const auto iParam = p % np;

      std::vector<ModelParametersd> parameters = castModelParameters<double>(referenceParameters);
      parameters[iFrame](iParam) = referenceParameters[iFrame](iParam) + kStepSize;
      const auto [state, meshState] = parametersToSkelAndMeshStates(parameters);

      Eigen::MatrixX<T> jacobianPlus = Eigen::MatrixX<T>::Zero(jacobianSize, nFrames * np);
      Eigen::VectorX<T> residualPlus = Eigen::VectorX<T>::Zero(jacobianSize);
      int usedRows = 0;
      errorFunction.getJacobian(
          castModelParameters<T>(parameters),
          state,
          meshState,
          jacobianPlus,
          residualPlus,
          usedRows);
      numJacobian.block(0, p, usedRows, 1) = (residualPlus - residual) / kStepSize;
    }

    EXPECT_LE(
        (numJacobian - jacobian).norm() / ((numJacobian + jacobian).norm() + FLT_EPSILON),
        numThreshold);
  }

  // check the gradients are similar
  {
    SCOPED_TRACE("Checking Numerical Gradient");
    if ((numGradient + anaGradient).norm() != 0.0f) {
      EXPECT_LE(
          (numGradient - anaGradient).norm() / ((numGradient + anaGradient).norm() + FLT_EPSILON),
          numThreshold);
    }
  }

  // check the gradients by comparing against the jacobian gradient
  {
    SCOPED_TRACE("Checking Jacobian Gradient");
    if ((jacGradient + anaGradient).norm() != 0.0f) {
      EXPECT_LE(
          (jacGradient - anaGradient).norm() / ((jacGradient + anaGradient).norm() + FLT_EPSILON),
          jacThreshold);
    }
  }

  // check the global weight value:
  if (functionError != 0) {
    SCOPED_TRACE("Checking Global Weight");
    const auto s = 2.0f;
    const auto w_new = s * errorFunction.getWeight();
    errorFunction.setWeight(w_new);
    const double functionError_scaled =
        errorFunction.getError(referenceParameters, referenceSkelStates, referenceMeshStates);

    EXPECT_LE(
        std::abs(functionError_scaled - s * functionError) /
            (s * functionError + functionError_scaled),
        1e-4f);
  }

  errorFunction.setWeight(w_orig);
}

} // namespace

static std::vector<ModelParametersd> zeroModelParameters(const Character& c, size_t nFrames) {
  return std::vector<ModelParametersd>(
      nFrames, Eigen::VectorXd::Zero(c.parameterTransform.numAllModelParameters()));
}

static std::vector<ModelParametersd> randomModelParameters(const Character& c, size_t nFrames) {
  std::vector<ModelParametersd> result;
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    result.push_back(VectorXd::Random(c.parameterTransform.numAllModelParameters()) * 0.25f);
  }
  return result;
}

TEST(Momentum_SequenceErrorFunctions, ModelParametersSequenceError_GradientsAndJacobians) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  // create constraints
  ModelParametersSequenceErrorFunctiond errorFunction(character);
  {
    SCOPED_TRACE("Motion Test");
    SkeletonState reference(transform.bindPose(), skeleton);
    VectorXd weights = VectorXd::Ones(transform.numAllModelParameters());
    weights(0) = 4.0f;
    weights(1) = 5.0f;
    weights(2) = 0;
    errorFunction.setTargetWeights(weights);
    testGradientAndJacobian<double>(errorFunction, zeroModelParameters(character, 2), character);
    for (size_t i = 0; i < 10; i++) {
      auto parameters = randomModelParameters(character, 2);
      testGradientAndJacobian<double>(errorFunction, parameters, character, 2e-3f);
    }
  }
}

TEST(Momentum_SequenceErrorFunctions, StateSequenceError_GradientsAndJacobians) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  // create constraints
  StateSequenceErrorFunctiond errorFunction(character);
  {
    SCOPED_TRACE("Motion Test");
    SkeletonState reference(transform.bindPose(), skeleton);

    VectorXd posWeights = VectorXd::Ones(skeleton.joints.size());
    VectorXd rotWeights = VectorXd::Ones(skeleton.joints.size());
    posWeights(0) = 4.0f;
    posWeights(1) = 5.0f;
    posWeights(2) = 0;

    rotWeights(0) = 2.0f;
    rotWeights(1) = 0.0f;
    rotWeights(2) = 3;
    errorFunction.setTargetWeights(posWeights, rotWeights);
    testGradientAndJacobian<double>(errorFunction, zeroModelParameters(character, 2), character);
    for (size_t i = 0; i < 10; i++) {
      auto parameters = randomModelParameters(character, 2);
      testGradientAndJacobian<double>(errorFunction, parameters, character, 2e-3f, 1e-6f, true);
    }
  }
}

TEST(Momentum_SequenceErrorFunctions, StateSequenceError_WithOffsets) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& transform = character.parameterTransform;

  // create constraints
  StateSequenceErrorFunctiond errorFunction(character);
  {
    Random<> gen(1234);
    SCOPED_TRACE("Motion Test");
    SkeletonState reference(transform.bindPose(), skeleton);

    VectorXd posWeights = VectorXd::Ones(skeleton.joints.size());
    VectorXd rotWeights = VectorXd::Ones(skeleton.joints.size());

    std::vector<Transformd> offsets(skeleton.joints.size());
    for (size_t i = 0; i < skeleton.joints.size(); ++i) {
      offsets[i].translation = gen.normal<Eigen::Vector3d>(0, 2.0);
      offsets[i].rotation = gen.uniformQuaternion<double>();
      offsets[i].scale = 1.0 + gen.normal<double>(0, 0.02);
    }
    errorFunction.setTargetState(offsets);

    posWeights(0) = 2.0f;
    posWeights(1) = 1.0f;
    posWeights(2) = 3;

    rotWeights(0) = 2.0f;
    rotWeights(1) = 2.0f;
    rotWeights(2) = 0;
    errorFunction.setTargetWeights(posWeights, rotWeights);
    testGradientAndJacobian<double>(errorFunction, zeroModelParameters(character, 2), character);
    for (size_t i = 0; i < 10; i++) {
      auto parameters = randomModelParameters(character, 2);
      testGradientAndJacobian<double>(errorFunction, parameters, character, 2e-3f, 1e-6f, true);
    }
  }
}

TEST(Momentum_SequenceErrorFunctions, VertexSequenceError_GradientsAndJacobians) {
  // create skeleton and reference values
  const Character character = createTestCharacter();

  // create constraints
  VertexSequenceErrorFunctiond errorFunction(character);
  {
    SCOPED_TRACE("Motion Test");

    // Add some vertex velocity constraints
    errorFunction.addConstraint(0, 1.0, Eigen::Vector3d(0.1, 0.0, 0.0)); // vertex 0 moving in x
    errorFunction.addConstraint(1, 2.0, Eigen::Vector3d(0.0, 0.1, 0.0)); // vertex 1 moving in y
    errorFunction.addConstraint(2, 0.5, Eigen::Vector3d(0.0, 0.0, 0.1)); // vertex 2 moving in z

    testGradientAndJacobian<double>(errorFunction, zeroModelParameters(character, 2), character);

    for (size_t i = 0; i < 10; i++) {
      auto parameters = randomModelParameters(character, 2);
      testGradientAndJacobian<double>(errorFunction, parameters, character, 2e-3f, 1e-6f, true);
    }
  }

  {
    SCOPED_TRACE("Zero Velocity Test");
    // Test with zero target velocities
    errorFunction.clearConstraints();
    errorFunction.addConstraint(0, 1.0, Eigen::Vector3d::Zero());
    errorFunction.addConstraint(1, 1.0, Eigen::Vector3d::Zero());

    testGradientAndJacobian<double>(errorFunction, zeroModelParameters(character, 2), character);

    for (size_t i = 0; i < 5; i++) {
      auto parameters = randomModelParameters(character, 2);
      testGradientAndJacobian<double>(errorFunction, parameters, character, 2e-3f, 1e-6f, true);
    }
  }
}

namespace {

template <typename T>
void testGradientAndJacobian(
    const Character& character,
    SequenceSolverFunctionT<T>& solverFunction,
    const std::vector<ModelParametersT<T>>& referenceParameters,
    const float numThreshold = 1e-3f,
    const float jacThreshold = 1e-6f,
    bool checkJacobian = true) {
  ASSERT_EQ(solverFunction.getNumFrames(), referenceParameters.size());
  for (size_t i = 0; i < solverFunction.getNumFrames(); ++i) {
    solverFunction.setFrameParameters(i, referenceParameters[i]);
  }

  const Eigen::VectorX<T> allParameters = solverFunction.getJoinedParameterVector();

  // Verify that we can round-trip through the model parameters:
  {
    solverFunction.setJoinedParameterVector(Eigen::VectorX<T>::Random(allParameters.size()));
    solverFunction.setJoinedParameterVector(allParameters);
    const ParameterSet univ = solverFunction.getUniversalParameterSet();
    for (size_t iFrame = 0; iFrame < referenceParameters.size(); ++iFrame) {
      ModelParametersT<T> modelParams_i = solverFunction.getFrameParameters(iFrame);
      for (auto k = 0; k < character.parameterTransform.numAllModelParameters(); ++k) {
        if (univ.test(k)) {
          ASSERT_NEAR(referenceParameters[0](k), modelParams_i(k), 1e-4);
        } else {
          ASSERT_NEAR(referenceParameters[iFrame](k), modelParams_i(k), 1e-4);
        }
      }
    }
  }

  solverFunction.getError(allParameters);

  Eigen::VectorX<T> anaGradient;
  solverFunction.getGradient(allParameters, anaGradient);

  Eigen::VectorX<T> numGradient = Eigen::VectorX<T>::Zero(allParameters.size());

  constexpr double kStepSize = 1e-5;
  for (Eigen::Index i = 0; i < allParameters.size(); ++i) {
    Eigen::VectorX<T> parametersPlus = allParameters;
    parametersPlus(i) += kStepSize;

    Eigen::VectorX<T> parametersMinus = allParameters;
    parametersMinus(i) -= kStepSize;
    numGradient(i) =
        (solverFunction.getError(parametersPlus) - solverFunction.getError(parametersMinus)) /
        (T(2) * kStepSize);
  }

  {
    SCOPED_TRACE("Checking Numerical Gradient");
    if ((numGradient + anaGradient).norm() != 0.0f) {
      EXPECT_LE(
          (numGradient - anaGradient).norm() / ((numGradient + anaGradient).norm() + FLT_EPSILON),
          numThreshold);
    }
  }

  // check the gradients by comparing against the jacobian gradient
  Eigen::VectorX<T> residual;
  Eigen::MatrixX<T> jacobian;
  size_t actualRows = 0;
  solverFunction.getJacobian(allParameters, jacobian, residual, actualRows);

  Eigen::VectorX<T> jacGradient =
      T(2) * jacobian.topRows(actualRows).transpose() * residual.head(actualRows);

  {
    SCOPED_TRACE("Checking Jacobian Gradient");
    if ((jacGradient + anaGradient).norm() != 0.0f) {
      EXPECT_LE(
          (jacGradient - anaGradient).norm() / ((jacGradient + anaGradient).norm() + FLT_EPSILON),
          jacThreshold);
    }
  }

  // Check the Jacobian if requested:
  if (checkJacobian) {
    SCOPED_TRACE("Checking Numerical Jacobian");
    Eigen::MatrixX<T> numJacobian = Eigen::MatrixX<T>::Zero(jacobian.rows(), jacobian.cols());
    for (Eigen::Index p = 0; p < allParameters.size(); p++) {
      Eigen::VectorX<T> parametersPlus = allParameters;
      parametersPlus(p) += kStepSize;

      Eigen::MatrixX<T> jacobianPlus;
      Eigen::VectorX<T> residualPlus;
      size_t usedRowsPlus = 0;
      solverFunction.getJacobian(parametersPlus, jacobianPlus, residualPlus, usedRowsPlus);
      ASSERT_EQ(usedRowsPlus, actualRows);
      numJacobian.col(p) = (residualPlus - residual) / kStepSize;
    }

    EXPECT_LE(
        (numJacobian - jacobian).norm() / ((numJacobian + jacobian).norm() + FLT_EPSILON),
        numThreshold);
  }
}

} // namespace

TEST(Momentum_SequenceSolver, SequenceSolverFunction_GradientsAndJacobians) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const ParameterTransform& transform = character.parameterTransform;

  const size_t nFrames = 3;

  const ParameterTransformd pt = transform.cast<double>();
  SequenceSolverFunctiond solverFunction(character, pt, transform.getScalingParameters(), nFrames);

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    auto positionError = std::make_shared<PositionErrorFunctiond>(character);
    positionError->addConstraint(
        PositionDataT<double>(Vector3d::UnitY(), Vector3d::UnitY(), 2, 2.4));
    solverFunction.addErrorFunction(iFrame, positionError);
  }

  auto smoothnessError = std::make_shared<ModelParametersSequenceErrorFunctiond>(character);
  for (size_t iFrame = 0; iFrame < (nFrames - 1); ++iFrame) {
    solverFunction.addSequenceErrorFunction(iFrame, smoothnessError);
  }

  testGradientAndJacobian(character, solverFunction, zeroModelParameters(character, nFrames));

  auto parameters = randomModelParameters(character, nFrames);
  testGradientAndJacobian(character, solverFunction, parameters);

  {
    // Make sure disabling individual parameters works.
    ParameterSet p;
    p.set();
    p.reset(3);
    solverFunction.setEnabledParameters(p);
    testGradientAndJacobian(character, solverFunction, parameters);
  }
}

TEST(Momentum_SequenceSolver, SequenceSolver_EnabledParameters) {
  // create skeleton and reference values
  const Character character = createTestCharacter();
  const size_t nFrames = 3;

  const ParameterTransformd pt = character.parameterTransform.cast<double>();
  ParameterSet universalParams;
  for (auto iParam = 6; iParam < pt.numAllModelParameters(); ++iParam) {
    universalParams.set(iParam);
  }

  SequenceSolverFunctiond solverFunction(character, pt, universalParams, nFrames);

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    auto positionError = std::make_shared<PositionErrorFunctiond>(character);
    positionError->addConstraint(
        PositionDataT<double>(Vector3d::UnitY(), Vector3d::UnitY(), 2, 2.4));
    solverFunction.addErrorFunction(iFrame, positionError);
  }

  SequenceSolverd solver(SequenceSolverOptions(), &solverFunction);
  ParameterSet variables;
  variables.set();

  for (size_t iParam = 5; iParam < 8; ++iParam) {
    variables.reset(iParam);
    solver.setEnabledParameters(variables);
    VectorXd dofs = solverFunction.getJoinedParameterVector();
    solver.solve(dofs);
  }
}
