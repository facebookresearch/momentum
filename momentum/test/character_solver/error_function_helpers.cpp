/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/character_solver/error_function_helpers.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/log.h"
#include "momentum/math/fmt_eigen.h"
#include "momentum/simd/simd.h"

namespace momentum {

using ::testing::DoubleNear;

template <typename T>
void testGradientAndJacobian(
    const char* file,
    int line,
    SkeletonErrorFunctionT<T>* errorFunction,
    const ModelParametersT<T>& referenceParameters,
    const Character& character,
    const T& numThreshold,
    const T& jacThreshold,
    bool checkJacError,
    bool checkJacobian) {
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Create a scoped trace with file and line information
  SCOPED_TRACE(::testing::Message() << "Called from file: " << file << ", line: " << line);

  const T w_orig = errorFunction->getWeight();
  errorFunction->setWeight(3.0);

  // test getError and getGradient produce the same value
  // Double precision state here limits how much errors accumulate up the
  // kinematic chain.
  SkeletonStateT<T> state(transform.apply(referenceParameters), character.skeleton);
  MeshStateT<T> meshState{};
  MeshStateT<T> referenceMeshState{};
  if (errorFunction->needsMesh()) {
    meshState.update(referenceParameters, state, character);
    referenceMeshState.update(referenceParameters, state, character);
  }
  const SkeletonStateT<T> referenceState(state);
  Eigen::VectorX<T> anaGradient = Eigen::VectorX<T>::Zero(transform.numAllModelParameters());
  const size_t jacobianSize = errorFunction->getJacobianSize();
  const size_t paddedJacobianSize = jacobianSize + 8 - (jacobianSize % 8);
  Eigen::MatrixX<T> jacobian =
      Eigen::MatrixX<T>::Zero(paddedJacobianSize, transform.numAllModelParameters());
  Eigen::VectorX<T> residual = Eigen::VectorX<T>::Zero(paddedJacobianSize);

  const T functionError =
      errorFunction->getError(referenceParameters, referenceState, referenceMeshState);
  const T gradientError = errorFunction->getGradient(
      referenceParameters, referenceState, referenceMeshState, anaGradient);
  int rows = 0;
  const T jacobianError = errorFunction->getJacobian(
      referenceParameters,
      referenceState,
      meshState,
      jacobian.topRows(jacobianSize),
      residual.topRows(jacobianSize),
      rows);
  EXPECT_LE(rows, jacobianSize);

  const Eigen::VectorX<T> jacGradient = 2.0 * jacobian.transpose() * residual;
  const T jacError = residual.dot(residual);

  {
    SCOPED_TRACE("Checking Error Value");
    if ((gradientError + functionError) != 0.0) {
      EXPECT_LE(std::abs(gradientError - functionError) / (gradientError + functionError), 1e-6);
    } else {
      EXPECT_NEAR(gradientError, functionError, 1e-7);
    }

    if ((jacobianError + functionError) != 0.0) {
      EXPECT_LE(std::abs(jacobianError - functionError) / (jacobianError + functionError), 1e-6);
    } else {
      EXPECT_NEAR(jacobianError, functionError, 1e-7);
    }

    if (checkJacError) {
      if ((jacError + functionError) != 0.0) {
        EXPECT_LE(std::abs(jacError - functionError) / (jacError + functionError), 5e-4);
      } else {
        EXPECT_NEAR(jacError, functionError, 1e-4);
      }
    }
  }

  // calculate numerical gradient
  constexpr T kStepSize = 1e-5;
  Eigen::VectorX<T> numGradient = Eigen::VectorX<T>::Zero(transform.numAllModelParameters());
  for (auto p = 0; p < transform.numAllModelParameters(); p++) {
    // perform higher-order finite differences for accuracy
    ModelParametersT<T> parameters = referenceParameters;
    // parameters(p) = static_cast<float>(referenceParameters(p) - 2.0f * kStepSize);
    // state.set(transform.apply(parameters), skeleton);
    // const T h_2 = errorFunction->getError(parameters, state);
    parameters(p) = referenceParameters(p) - kStepSize;
    state.set(transform.apply(parameters), character.skeleton);
    if (errorFunction->needsMesh()) {
      meshState.update(parameters, state, character);
    }
    const T h_1 = errorFunction->getError(parameters, state, meshState);
    parameters(p) = referenceParameters(p) + kStepSize;
    state.set(transform.apply(parameters), character.skeleton);
    if (errorFunction->needsMesh()) {
      meshState.update(parameters, state, character);
    }
    const T h1 = errorFunction->getError(parameters, state, meshState);
    // parameters(p) = static_cast<float>(referenceParameters(p) + 2.0f * kStepSize);
    // state.set(transform.apply(parameters), skeleton);
    // const T h2 = errorFunction->getError(parameters, state);
    //        numGradient(p) = static_cast<float>((h_2 - 8.0 * h_1 + 8.0 * h1 - h2) / (12.0 *
    //        kStepSize));
    numGradient(p) = (h1 - h_1) / (2.0 * kStepSize);
  }

  if (checkJacobian) {
    SCOPED_TRACE("Checking Numerical Jacobian");
    Eigen::MatrixX<T> numJacobian(paddedJacobianSize, transform.numAllModelParameters());
    for (auto k = 0; k < transform.numAllModelParameters(); ++k) {
      ModelParametersT<T> parameters = referenceParameters.template cast<T>();
      parameters(k) = referenceParameters(k) + kStepSize;
      state.set(transform.apply(parameters), character.skeleton);
      if (errorFunction->needsMesh()) {
        meshState.update(parameters, state, character);
      }

      Eigen::MatrixX<T> jacobianPlus =
          Eigen::MatrixX<T>::Zero(paddedJacobianSize, transform.numAllModelParameters());
      Eigen::MatrixX<T> residualPlus = Eigen::VectorX<T>::Zero(paddedJacobianSize);
      int usedRows = 0;
      errorFunction->getJacobian(
          parameters, state, meshState, jacobianPlus, residualPlus, usedRows);
      numJacobian.col(k) = (residualPlus - residual) / kStepSize;
    }

    EXPECT_LE(
        (numJacobian - jacobian).norm() /
            ((numJacobian + jacobian).norm() + Momentum_ErrorFunctionsTest<T>::getEps()),
        numThreshold);
  }

  // check the gradients are similar
  {
    SCOPED_TRACE("Checking Numerical Gradient");
    if ((numGradient + anaGradient).norm() != 0.0) {
      const T gradientMismatch = (numGradient - anaGradient).norm() /
          ((numGradient + anaGradient).norm() + Momentum_ErrorFunctionsTest<T>::getEps());

      if (gradientMismatch > numThreshold) {
        MT_LOGI(
            "Numerical gradient mismatch detected: {} > {} (threshold)\n"
            "Numerical gradient norm: {}, Analytical gradient norm: {}\n"
            "Difference norm: {}",
            gradientMismatch,
            numThreshold,
            numGradient.norm(),
            anaGradient.norm(),
            (numGradient - anaGradient).norm());

        // Dump mismatched gradient components:
        for (int iParam = 0; iParam < numGradient.size(); ++iParam) {
          const auto diff = std::abs(numGradient(iParam) - anaGradient(iParam));
          if (diff > 1e-3) {
            MT_LOGI(
                "Mismatched gradient component: {} ({})",
                iParam,
                character.parameterTransform.name.at(iParam));
            MT_LOGI("  numGradient: {}", numGradient(iParam));
            MT_LOGI("  anaGradient: {}", anaGradient(iParam));
            MT_LOGI("  diff: {}", diff);
          }
        }
      }

      EXPECT_LE(gradientMismatch, numThreshold);
    }
  }

  // check the gradients by comparing against the jacobian gradient
  {
    SCOPED_TRACE("Checking Numerical Gradient");
    if ((jacGradient + anaGradient).norm() != 0.0) {
      const T gradientMismatch = (jacGradient - anaGradient).norm() /
          ((jacGradient + anaGradient).norm() + Momentum_ErrorFunctionsTest<T>::getEps());

      if (gradientMismatch > jacThreshold) {
        MT_LOGI(
            "Jacobian gradient mismatch detected: {} > {} (threshold)\n"
            "Jacobian gradient norm: {}, Analytical gradient norm: {}\n"
            "Difference norm: {}",
            gradientMismatch,
            jacThreshold,
            jacGradient.norm(),
            anaGradient.norm(),
            (jacGradient - anaGradient).norm());
      }

      EXPECT_LE(gradientMismatch, jacThreshold);
    }
  }

  // check the global weight value:
  if (functionError != 0) {
    SCOPED_TRACE("Checking Global Weight");
    const T s = 2.0;
    const T w_new = s * errorFunction->getWeight();
    errorFunction->setWeight(w_new);
    const T functionError_scaled =
        errorFunction->getError(referenceParameters, referenceState, referenceMeshState);

    EXPECT_LE(
        std::abs(functionError_scaled - s * functionError) /
            (s * functionError + functionError_scaled),
        1e-4f);
  }

  errorFunction->setWeight(w_orig);
}

template <typename T>
void validateIdentical(
    const char* file,
    int line,
    SkeletonErrorFunctionT<T>& err1,
    SkeletonErrorFunctionT<T>& err2,
    const Skeleton& skeleton,
    const ParameterTransformT<T>& transform,
    const Eigen::VectorX<T>& parameters,
    const T& errorDiffThreshold,
    const T& gradDiffThreshold,
    bool verbose) {
  // Create a scoped trace with file and line information
  SCOPED_TRACE(::testing::Message() << "Called from file: " << file << ", line: " << line);

  SkeletonStateT<T> state(transform.apply(parameters), skeleton);

  // Create empty mesh state for test functions
  MeshStateT<T> emptyMeshState{};

  {
    auto e1 = err1.getError(parameters, state, emptyMeshState);
    auto e2 = err2.getError(parameters, state, emptyMeshState);
    if (verbose) {
      fmt::print("getError(): {} {}\n", e1, e2);
    }
    EXPECT_THAT(e1, DoubleNear(e2, errorDiffThreshold));
  }

  {
    VectorX<T> grad1 = VectorX<T>::Zero(transform.numAllModelParameters());
    VectorX<T> grad2 = VectorX<T>::Zero(transform.numAllModelParameters());
    auto e1 = err1.getGradient(parameters, state, emptyMeshState, grad1);
    auto e2 = err2.getGradient(parameters, state, emptyMeshState, grad2);

    const T diff = (grad1 - grad2).template lpNorm<Eigen::Infinity>();
    if (verbose) {
      fmt::print("getGradient(): {} {}\n", e1, e2);
      fmt::print("grad1: {}\n", grad1.transpose());
      fmt::print("grad2: {}\n", grad2.transpose());
    }

    // TODO It seems like e2 can be nonzero even when e1 is exactly zero, which seems
    // likely to be a bug.
    EXPECT_THAT(e1, DoubleNear(e2, errorDiffThreshold));
    EXPECT_LT(diff, gradDiffThreshold);
  }

  {
    const auto n = transform.numAllModelParameters();
    const size_t m1 = err1.getJacobianSize();
    const size_t m2 = err2.getJacobianSize();
    EXPECT_LE(m1, m2); // SIMD pads out to a multiple of kSimdPacketSize.

    Eigen::MatrixX<T> j1 = Eigen::MatrixX<T>::Zero(m1, n);
    Eigen::MatrixX<T> j2 = Eigen::MatrixX<T>::Zero(m2, n);
    Eigen::VectorX<T> r1 = Eigen::VectorX<T>::Zero(m1);
    Eigen::VectorX<T> r2 = Eigen::VectorX<T>::Zero(m2);

    int rows1 = 0;
    auto e1 = err1.getJacobian(parameters, state, emptyMeshState, j1, r1, rows1);
    int rows2 = 0;
    auto e2 = err2.getJacobian(parameters, state, emptyMeshState, j2, r2, rows2);

    // Can't trust the Jacobian ordering here, so only look at JtJ and Jt*r.
    const Eigen::MatrixX<T> JtJ1 = j1.transpose() * j1;
    const Eigen::MatrixX<T> JtJ2 = j2.transpose() * j2;

    const Eigen::VectorX<T> g1 = j1.transpose() * r1;
    const Eigen::VectorX<T> g2 = j2.transpose() * r2;

    if (verbose) {
      fmt::print("getJacobian(): {} {}\n", e1, e2);
      fmt::print("JtJ1: \n{}\n", JtJ1);
      fmt::print("JtJ2: \n{}\n", JtJ2);
      fmt::print("Jt1*r: {}\n", g1.transpose());
      fmt::print("Jt2*r: {}\n", g2.transpose());
    }

    EXPECT_THAT(e1, DoubleNear(e2, errorDiffThreshold));

    const float j_diff = (JtJ1 - JtJ2).template lpNorm<Eigen::Infinity>();
    EXPECT_LE(j_diff, gradDiffThreshold);

    const float g_diff = (g1 - g2).template lpNorm<Eigen::Infinity>();
    EXPECT_LE(g_diff, gradDiffThreshold);
  }
}

void timeJacobian(
    const Character& character,
    SkeletonErrorFunction& errorFunction,
    const ModelParameters& modelParams,
    const char* type) {
  const size_t jacobianSize = errorFunction.getJacobianSize();
  const size_t paddedJacobianSize =
      jacobianSize + kSimdPacketSize - (jacobianSize % kSimdPacketSize);
  Eigen::MatrixXf jacobian = Eigen::MatrixXf::Zero(
      paddedJacobianSize, character.parameterTransform.numAllModelParameters());
  Eigen::VectorXf residual = Eigen::VectorXf::Zero(paddedJacobianSize);
  SkeletonState skelState(character.parameterTransform.apply(modelParams), character.skeleton);

  const size_t nTests = 5000;
  const auto startTime = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < nTests; ++i) {
    int usedRows = 0;
    jacobian.setZero();
    residual.setZero();
    // Create empty mesh state for test functions
    MeshState emptyMeshState{};
    errorFunction.getJacobian(modelParams, skelState, emptyMeshState, jacobian, residual, usedRows);
  }
  const auto endTime = std::chrono::high_resolution_clock::now();

  MT_LOGI(
      "Time to compute Jacobian ({}): {}us",
      type,
      (std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() /
       nTests));
}

void timeError(
    const Character& character,
    SkeletonErrorFunction& errorFunction,
    const ModelParameters& modelParams,
    const char* type) {
  SkeletonState skelState(character.parameterTransform.apply(modelParams), character.skeleton);

  const size_t nTests = 5000;
  const auto startTime = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < nTests; ++i) {
    // Create empty mesh state for test functions
    MeshState emptyMeshState{};
    errorFunction.getError(modelParams, skelState, emptyMeshState);
  }
  const auto endTime = std::chrono::high_resolution_clock::now();

  MT_LOGI(
      "Time to compute getError ({}): {}us",
      type,
      (std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() /
       nTests));
}

template struct Momentum_ErrorFunctionsTest<float>;
template struct Momentum_ErrorFunctionsTest<double>;

template void testGradientAndJacobian<float>(
    const char* file,
    int line,
    SkeletonErrorFunction* errorFunction,
    const ModelParameters& referenceParameters,
    const Character& character,
    const float& numThreshold,
    const float& jacThreshold,
    bool checkJacError,
    bool checkJacobian);

template void testGradientAndJacobian<double>(
    const char* file,
    int line,
    SkeletonErrorFunctiond* errorFunction,
    const ModelParametersd& referenceParameters,
    const Character& character,
    const double& numThreshold,
    const double& jacThreshold,
    bool checkJacError,
    bool checkJacobian);

template void validateIdentical<float>(
    const char* file,
    int line,
    SkeletonErrorFunction& err1,
    SkeletonErrorFunction& err2,
    const Skeleton& skeleton,
    const ParameterTransform& transform,
    const Eigen::VectorXf& parameters,
    const float& errorDiffThreshold,
    const float& gradDiffThreshold,
    bool verbose);

template void validateIdentical<double>(
    const char* file,
    int line,
    SkeletonErrorFunctiond& err1,
    SkeletonErrorFunctiond& err2,
    const Skeleton& skeleton,
    const ParameterTransformd& transform,
    const Eigen::VectorXd& parameters,
    const double& errorDiffThreshold,
    const double& gradDiffThreshold,
    bool verbose);

} // namespace momentum
