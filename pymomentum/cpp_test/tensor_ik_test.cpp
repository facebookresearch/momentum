/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/diff_ik/fully_differentiable_orientation_error_function.h>
#include <momentum/diff_ik/fully_differentiable_position_error_function.h>
#include <momentum/solver/gauss_newton_solver.h>
#include <momentum/test/character/character_helpers.h>

#include <pymomentum/tensor_ik/tensor_error_function.h>
#include <pymomentum/tensor_ik/tensor_ik.h>
#include <pymomentum/tensor_ik/tensor_limit_error_function.h>
#include <pymomentum/tensor_ik/tensor_marker_error_function.h>
#include <pymomentum/tensor_utility/tensor_utility.h>

#include <numeric>
#include <random>

struct PositionErrorFunctionTensors {
  at::Tensor parents;
  at::Tensor offsets;
  at::Tensor weights;
  at::Tensor targets;
};

template <typename T>
PositionErrorFunctionTensors toPositionTensors(
    const std::vector<momentum::PositionConstraintT<T>>& constraints) {
  const int nCons = (int)constraints.size();

  PositionErrorFunctionTensors result;

  result.parents = at::zeros({1, nCons}, at::kInt);
  result.offsets = at::zeros({1, nCons, 3}, at::kFloat);
  result.weights = at::zeros({1, nCons}, at::kFloat);
  result.targets = at::zeros({1, nCons, 3}, at::kFloat);

  for (int iCons = 0; iCons < nCons; ++iCons) {
    pymomentum::toEigenMap<int>(result.parents)(iCons) =
        constraints[iCons].parent;
    pymomentum::toEigenMap<float>(result.offsets).segment<3>(3 * iCons) =
        constraints[iCons].offset;
    pymomentum::toEigenMap<float>(result.weights)(iCons) =
        constraints[iCons].weight;
    pymomentum::toEigenMap<float>(result.targets).segment<3>(3 * iCons) =
        constraints[iCons].target;
  }

  return result;
}

at::Tensor concatBatch(std::vector<at::Tensor> tensors) {
  if (tensors.empty()) {
    return at::Tensor();
  }

  const auto sizes = tensors[0].sizes();
  std::vector<int64_t> sizes_batch = {(int64_t)tensors.size()};
  std::copy(
      std::begin(sizes), std::end(sizes), std::back_inserter(sizes_batch));

  at::Tensor result = at::zeros(sizes_batch, tensors[0].scalar_type());

  for (int iBatch = 0; iBatch < static_cast<int>(tensors.size()); ++iBatch) {
    assert(result.scalar_type() == tensors[iBatch].scalar_type());
    // ASSERT_EQ(sizes_orig, std::vector<long>{tensors[iBatch].sizes()});
    result.select(0, iBatch) = tensors[iBatch];
  }

  return result;
}

template <typename T>
void addPositionConstraints(
    momentum::FullyDifferentiablePositionErrorFunctionT<T>& func,
    const momentum::Character& character,
    const Eigen::VectorX<T>& modelParameters,
    std::mt19937& rng) {
  const momentum::SkeletonState state(
      character.parameterTransform.apply(
          modelParameters.template cast<float>()),
      character.skeleton);

  auto randomVec3 = [&]() {
    std::normal_distribution<> norm(0, 1);
    return Eigen::Vector3f(norm(rng), norm(rng), norm(rng));
  };

  std::uniform_real_distribution<> unif(0, 1);

  for (int iJoint = 0;
       iJoint < static_cast<int>(character.skeleton.joints.size());
       ++iJoint) {
    // three constraints per joint; if it isn't overdetermined, the solver
    // derivatives don't work as well.
    for (int jCons = 0; jCons < 3; ++jCons) {
      const auto parent = iJoint;
      const Eigen::Vector3<T> offset = 3.0f * randomVec3().template cast<T>();
      const Eigen::Vector3<T> target =
          state.jointState[jCons].transformation.template cast<T>() * offset +
          randomVec3().template cast<T>();
      const float weight = 1.0f + unif(rng);
      func.addConstraint(
          momentum::PositionConstraintT<T>(offset, target, parent, weight));
    }
  }
}

template <typename T>
struct IKProblem {
  std::shared_ptr<momentum::FullyDifferentiablePositionErrorFunctionT<T>>
      markerError;
  std::shared_ptr<momentum::LimitErrorFunction> limitError;
  std::shared_ptr<momentum::FullyDifferentiableOrientationErrorFunctionT<T>>
      orientationError;
  std::shared_ptr<momentum::SkeletonSolverFunction> solverFunction;

  Eigen::VectorX<T> modelParameters_init;
  Eigen::VectorX<T> modelParameters_final;
};

template <typename T>
at::Tensor toTensor(const Eigen::VectorX<T>& vec) {
  auto result = at::zeros({vec.size()}, at::kFloat);
  pymomentum::toEigenMap<T>(result) = vec;
  return result;
}

template <typename T>
Eigen::VectorX<T> extractErrorFunctionWeights(
    const momentum::SkeletonSolverFunction& solverFunction) {
  const auto& errorFunctions = solverFunction.getErrorFunctions();
  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(errorFunctions.size());
  for (int i = 0; i < static_cast<int>(errorFunctions.size()); ++i) {
    result(i) = errorFunctions[i]->getWeight();
  }
  return result;
}

TEST(TensorIK, TensorIK) {
  const momentum::Character character = momentum::createTestCharacter();

  typedef float T;

  momentum::GaussNewtonSolverOptions solverOptions;
  solverOptions.minIterations = 4;
  solverOptions.maxIterations = 40;
  solverOptions.threshold = 10.f;
  solverOptions.regularization = 1e-3f;

  // Construct a very basic IK problem:
  const int nBatch = 3;
  std::vector<IKProblem<T>> ikProblems(nBatch);
  std::vector<Eigen::VectorX<T>> ikSolutions;

  std::uniform_real_distribution<> unif(0, 1);

  std::mt19937 rng;

  momentum::ParameterSet activeParams;
  activeParams.set();

  Eigen::VectorX<T> dLoss_dModelParams(
      character.parameterTransform.numAllModelParameters());
  for (int i = 0; i < dLoss_dModelParams.size(); ++i) {
    dLoss_dModelParams(i) = unif(rng);
  }

  for (int iBatch = 0; iBatch < nBatch; ++iBatch) {
    ikProblems[iBatch].modelParameters_init = Eigen::VectorX<T>::Zero(
        character.parameterTransform.numAllModelParameters());

    ikProblems[iBatch].solverFunction =
        std::make_unique<momentum::SkeletonSolverFunction>(
            &character.skeleton, &character.parameterTransform);

    ikProblems[iBatch].markerError = std::make_shared<
        momentum::FullyDifferentiablePositionErrorFunctionT<T>>(
        character.skeleton, character.parameterTransform);
    ikProblems[iBatch].markerError->setWeight(2.0f + unif(rng));
    addPositionConstraints(
        *ikProblems[iBatch].markerError,
        character,
        ikProblems[iBatch].modelParameters_init,
        rng);
    // std::cout << "GAUSS-NEWTON\n========================================\n";
    // pymomentum::printConstraintList(ikProblems[iBatch].markerError->getConstraints(),
    // std::cout);
    ikProblems[iBatch].solverFunction->addErrorFunction(
        ikProblems[iBatch].markerError);

    ikProblems[iBatch].limitError =
        std::make_shared<momentum::LimitErrorFunction>(
            character.skeleton,
            character.parameterTransform,
            character.parameterLimits);
    ikProblems[iBatch].solverFunction->addErrorFunction(
        ikProblems[iBatch].limitError);

    ikProblems[iBatch].orientationError = std::make_shared<
        momentum::FullyDifferentiableOrientationErrorFunctionT<T>>(
        character.skeleton, character.parameterTransform);
    ikProblems[iBatch].solverFunction->addErrorFunction(
        ikProblems[iBatch].orientationError);

    momentum::GaussNewtonSolver solver(
        solverOptions, ikProblems[iBatch].solverFunction.get());
    solver.setEnabledParameters(activeParams);

    ikProblems[iBatch].modelParameters_final =
        ikProblems[iBatch].modelParameters_init;
    solver.solve(ikProblems[iBatch].modelParameters_final);
    // std::cout << "Optimized parameters (GaussNewtonSolver): "<<
    // ikProblems[iBatch].modelParameters_final.transpose() << "\n";

    const auto positionTensors =
        toPositionTensors<T>(ikProblems[iBatch].markerError->getConstraints());
    std::vector<std::unique_ptr<pymomentum::TensorErrorFunction<T>>>
        tensorErrorFunctions;
    tensorErrorFunctions.push_back(pymomentum::createPositionErrorFunction<T>(
        1,
        0,
        positionTensors.parents,
        positionTensors.offsets,
        positionTensors.weights,
        positionTensors.targets));
    tensorErrorFunctions.push_back(
        pymomentum::createLimitErrorFunction<T>(1, 0));
    tensorErrorFunctions.push_back(
        pymomentum::createOrientationErrorFunction<T>(
            1, 0, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()));

    auto tensorErrorFunctionWeights = toTensor(
        extractErrorFunctionWeights<T>(*ikProblems[iBatch].solverFunction));

    size_t numActiveErrorFunctions = tensorErrorFunctionWeights.sizes()[0];
    std::vector<int> weightsMap(numActiveErrorFunctions);
    std::iota(weightsMap.begin(), weightsMap.end(), 0);

    const std::vector<const momentum::Character*> characters(
        nBatch, &character);
    at::Tensor modelParameters_final_tensor = solveTensorIKProblem(
        characters,
        activeParams,
        toTensor(ikProblems[iBatch].modelParameters_init).unsqueeze(0),
        tensorErrorFunctions,
        tensorErrorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap,
        pymomentum::SolverOptions());
    const Eigen::VectorX<T> modelParameters_final_extract =
        pymomentum::toEigenMap<T>(modelParameters_final_tensor);

    auto dumpStats = [&](const char* name,
                         Eigen::Ref<const Eigen::VectorX<T>> modelParams) {
      std::cout << "Eval " << name << ".\n";
      std::cout << "   modelParams: " << modelParams.transpose() << "\n";
      std::cout << "   error: "
                << ikProblems[iBatch].solverFunction->getError(modelParams)
                << "\n";
      Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(modelParams.size());
      ikProblems[iBatch].solverFunction->getGradient(modelParams, gradient);
      std::cout << "   gradient: " << gradient.transpose() << "\n";
    };

    dumpStats("Non-tensor version", ikProblems[iBatch].modelParameters_final);
    dumpStats("Tensor version", modelParameters_final_extract);

    ASSERT_EQ(
        character.parameterTransform.numAllModelParameters(),
        modelParameters_final_extract.size());
    for (int iParam = 0; iParam <
         static_cast<int>(character.parameterTransform.numAllModelParameters());
         ++iParam) {
      ASSERT_NEAR(
          ikProblems[iBatch].modelParameters_final(iParam),
          modelParameters_final_extract(iParam),
          5e-2f);
    }

    auto derivatives = d_solveTensorIKProblem(
        characters,
        activeParams,
        modelParameters_final_tensor,
        toTensor(dLoss_dModelParams).unsqueeze(0),
        tensorErrorFunctions,
        tensorErrorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap);
  }
}
