/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_ik.h"

#include "pymomentum/tensor_ik/tensor_error_function.h"
#include "pymomentum/tensor_ik/tensor_ik_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <dispenso/parallel_for.h> // @manual
#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/model_parameters_sequence_error_function.h>
#include <momentum/character_sequence_solver/sequence_solver.h>
#include <momentum/character_sequence_solver/sequence_solver_function.h>
#include <momentum/character_solver/gauss_newton_solver_qr.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/character_solver/trust_region_qr.h>
#include <momentum/diff_ik/fully_differentiable_body_ik.h>
#include <momentum/solver/gauss_newton_solver.h>
#include <momentum/solver/subset_gauss_newton_solver.h>

#include <atomic>

namespace pymomentum {

using namespace pymomentum::detail;

namespace {
// How many times of solving IK problems from all solve_ik().
std::atomic<size_t> nTotalSolveIK{0};
// How many total iterations when solving IK problems.
std::atomic<size_t> nTotalSolveIKIter{0};
// How many IK gradients are non zero. Here the gradients are not the
// gradients of the solveTensorIKProblem()-like function, but the
// gradients of the IK problem. Gradients are close to zero means the
// IK solver converges.
std::atomic<size_t> nNonZeroGradient{0};
std::atomic<size_t> nTotalGradient{0};
std::atomic<size_t> nGradientPrintouts{0};
constexpr float gradientRMSEThreshold = 0.01;
const size_t MAX_GRADIENT_PRINTOUTS = 10;
} // namespace

std::pair<size_t, size_t> getSolveIKStatistics() {
  return {nTotalSolveIK, nTotalSolveIKIter};
}

void resetSolveIKStatistics() {
  nTotalSolveIK = 0;
  nTotalSolveIKIter = 0;
}

std::pair<size_t, size_t> getGradientStatistics() {
  return {nNonZeroGradient, nTotalGradient};
}

void resetGradientStatistics() {
  nNonZeroGradient = 0;
  nTotalGradient = 0;
}

namespace {

class ThreadSafeExceptionWrapper {
 public:
  ThreadSafeExceptionWrapper() = default;
  ~ThreadSafeExceptionWrapper() = default;

  void set(std::exception_ptr ptr) {
    std::unique_lock<std::mutex> lock(_mutex);
    _ptr = ptr;
  }

  void maybeThrow() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_ptr) {
      std::rethrow_exception(_ptr);
    }
  }

 private:
  std::mutex _mutex;
  std::exception_ptr _ptr;
};

} // namespace

template <typename T>
at::Tensor solveTensorIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& solverOptions) {
  std::tie(modelParams_init, errorFunctionWeights) = checkIKInputs<T>(
      characters,
      modelParams_init,
      errorFunctionWeights,
      numActiveErrorFunctions,
      "solveTensorIKProblem()");
  const auto nBatch = modelParams_init.size(0);

  momentum::SolverOptions momentumSolverOptions;
  momentumSolverOptions.minIterations = solverOptions.minIter;
  momentumSolverOptions.maxIterations = solverOptions.maxIter;
  momentumSolverOptions.threshold = solverOptions.threshold;
  // momentumSolverOptions.regularization = solverOptions.levmar_lambda;
  // momentumSolverOptions.doLineSearch = solverOptions.lineSearch;
  momentumSolverOptions.verbose = solverOptions.verbose;

  const auto nParams = characters.front()->parameterTransform.numAllModelParameters();

  at::Tensor modelParameters_final = at::zeros({nBatch, (int)nParams}, toScalarType<T>());

  ThreadSafeExceptionWrapper exception;

  std::atomic<uint32_t> numNaNs = 0;
  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    try {
      const auto& character = *characters[iBatch];
      const momentum::ParameterTransformT<T> parameterTransform =
          character.parameterTransform.cast<T>();

      at::Tensor modelParameters_init_cur = modelParams_init.select(0, iBatch);
      at::Tensor modelParameters_final_cur = modelParameters_final.select(0, iBatch);

      const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> errorFunctions_cur =
          buildMomentumErrorFunctions(
              characters, errorFunctions, errorFunctionWeights, weightsMap, iBatch);
      momentum::SkeletonSolverFunctionT<T> solverFunction =
          buildSolverFunction(*characters[iBatch], parameterTransform, errorFunctions_cur);

      std::unique_ptr<momentum::SolverT<T>> solver;
      if (solverOptions.linearSolverType == LinearSolverType::Cholesky) {
        auto derivedSolverOptions = momentum::SubsetGaussNewtonSolverOptions(momentumSolverOptions);
        derivedSolverOptions.regularization = solverOptions.levmar_lambda;
        derivedSolverOptions.doLineSearch = solverOptions.lineSearch;
        solver = std::make_unique<momentum::SubsetGaussNewtonSolverT<T>>(
            derivedSolverOptions, &solverFunction);
      } else if (solverOptions.linearSolverType == LinearSolverType::TrustRegionQR) {
        solver =
            std::make_unique<momentum::TrustRegionQRT<T>>(momentumSolverOptions, &solverFunction);
      } else {
        auto derivedSolverOptions = momentum::GaussNewtonSolverQROptions(momentumSolverOptions);
        derivedSolverOptions.regularization = solverOptions.levmar_lambda;
        derivedSolverOptions.doLineSearch = solverOptions.lineSearch;
        solver = std::make_unique<momentum::GaussNewtonSolverQRT<T>>(
            derivedSolverOptions, &solverFunction);
      }
      solver->setEnabledParameters(activeParams);

      Eigen::VectorX<T> parameters_opt = toEigenMap<T>(modelParameters_init_cur);
      solver->solve(parameters_opt);

      // Record iteration count.
      const std::vector<double>& iterHistory = solver->getErrorHistory();
      nTotalSolveIKIter += iterHistory.size();

      if (parameters_opt.array().isNaN().any() || parameters_opt.array().isInf().any()) {
        ++numNaNs;
        toEigenMap<T>(modelParameters_final_cur) = toEigenMap<T>(modelParameters_init_cur);
      } else {
        toEigenMap<T>(modelParameters_final_cur) = parameters_opt;
      }
    } catch (...) {
      exception.set(std::current_exception());
    }
  });
  nTotalSolveIK += nBatch;

  if (numNaNs > 0) {
    std::cerr << "WARNING: Detected" << static_cast<uint32_t>(numNaNs)
              << "NAN/INF values in outputs from solve_ik.  Reverting to initial parameters.";
  }

  exception.maybeThrow();

  return modelParameters_final;
}

template <typename T>
std::tuple<at::Tensor, torch::autograd::variable_list> d_solveTensorIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_final,
    at::Tensor d_loss_dModelParameters,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap) {
  bool squeezeErrorFunctionWeights = false;
  std::tie(modelParams_final, errorFunctionWeights) = checkIKInputs<T>(
      characters,
      modelParams_final,
      errorFunctionWeights,
      numActiveErrorFunctions,
      "solveTensorIKProblem()",
      &squeezeErrorFunctionWeights);

  d_loss_dModelParameters =
      d_loss_dModelParameters.contiguous().to(at::DeviceType::CPU, toScalarType<T>());

  const auto nBatch = modelParams_final.size(0);

  if (errorFunctionWeights.ndimension() == 1) {
    errorFunctionWeights = errorFunctionWeights.expand({nBatch, -1});
    squeezeErrorFunctionWeights = true;
  }

  std::vector<ErrorFunctionInput<T>> grad_inputs =
      buildErrorFunctionInputs(errorFunctions, weightsMap);

  at::Tensor grad_errorFunctionWeights = at::zeros(errorFunctionWeights.sizes(), toScalarType<T>());

  std::vector<T> gradient_rmse(nBatch);

  ThreadSafeExceptionWrapper exception;

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    try {
      const auto& character = *characters[iBatch];
      const momentum::ParameterTransformT<T> parameterTransform =
          character.parameterTransform.cast<T>();

      const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> errorFunctions_cur =
          buildMomentumErrorFunctions(
              characters, errorFunctions, errorFunctionWeights, weightsMap, iBatch);
      momentum::SkeletonSolverFunctionT<T> solverFunction =
          buildSolverFunction(*characters[iBatch], parameterTransform, errorFunctions_cur);

      at::Tensor modelParameters_cur = modelParams_final.select(0, iBatch);
      at::Tensor dLoss_dModelParameters_cur = d_loss_dModelParameters.select(0, iBatch);
      std::vector<momentum::ErrorFunctionDerivativesT<T>> errorFunctionDerivs =
          momentum::d_modelParams_d_inputs<T>(
              character.skeleton,
              parameterTransform,
              activeParams,
              toEigenMap<T>(modelParameters_cur),
              solverFunction,
              toEigenMap<T>(dLoss_dModelParameters_cur),
              &gradient_rmse[iBatch]);
      assert(errorFunctionDerivs.size() == errorFunctions.size());

      // Don't fill in the gradients in this case, as they are unreliable.
      if (gradient_rmse[iBatch] > gradientRMSEThreshold) {
        return;
      }

      at::Tensor grad_errorFunctionWeights_cur = grad_errorFunctionWeights.select(0, iBatch);
      for (size_t iErr = 0; iErr < errorFunctions.size(); ++iErr) {
        if (weightsMap[iErr] < 0) {
          continue;
        }

        toEigenMap<T>(grad_errorFunctionWeights_cur)(weightsMap[iErr]) =
            errorFunctionDerivs[iErr].gradWeight;
      }

      // For each differentiable input,
      for (size_t iGlobalInput = 0; iGlobalInput < grad_inputs.size(); ++iGlobalInput) {
        if (grad_inputs[iGlobalInput].dLoss_dInput.empty()) {
          continue;
        }

        const auto iErrorFunction = grad_inputs[iGlobalInput].iErrorFunction;
        const auto jInput = grad_inputs[iGlobalInput].jInput;
        const auto& errf = errorFunctions[iErrorFunction];
        const auto& input = errf->tensorInputs()[jInput];

        auto itr = errorFunctionDerivs[iErrorFunction].gradInputs.find(input.inputName);

        if (itr == errorFunctionDerivs[iErrorFunction].gradInputs.end()) {
          std::ostringstream availableDerivatives;
          bool firstDeriv = true;
          for (const auto& [name, _] : errorFunctionDerivs[iErrorFunction].gradInputs) {
            if (firstDeriv) {
              firstDeriv = false;
            } else {
              availableDerivatives << ", ";
            }
            availableDerivatives << name;
          }
          MT_THROW(
              "Input {} in error function {} is supposed to be differentiable but no derivatives detected; available derivatives are: ({})",
              input.inputName,
              errf->name(),
              availableDerivatives.str());
        }

        grad_inputs[iGlobalInput].dLoss_dInput[iBatch] = std::move(itr->second);
      }
    } catch (...) {
      exception.set(std::current_exception());
    }
  });

  exception.maybeThrow();

  // Decide if we need to de-batch each gradient (by summing along the
  // batch dimension):
  if (squeezeErrorFunctionWeights) {
    grad_errorFunctionWeights = grad_errorFunctionWeights.sum(0);
  }

  const size_t nExceeded =
      std::count_if(std::begin(gradient_rmse), std::end(gradient_rmse), [](const T& value) {
        return value > gradientRMSEThreshold;
      });
  const T maxValue = *std::max_element(std::begin(gradient_rmse), std::end(gradient_rmse));
  if (nExceeded > 0) {
    if (nGradientPrintouts < MAX_GRADIENT_PRINTOUTS) {
      // I'd like to use py::print here, but for some reason Pytorch
      // insta-crashes if I do that, I guess because we're inside the autograd
      // evaluation.
      std::cerr << "WARNING: in backward pass of solve_ik(), " << nExceeded << "/" << nBatch
                << " gradients exceeded the threshold " << gradientRMSEThreshold
                << " (max: " << maxValue << ").\n";
      ++nGradientPrintouts;
      if (nGradientPrintouts >= MAX_GRADIENT_PRINTOUTS) {
        std::cerr
            << "Suppressing further warnings about gradient thresholds.  Use get_gradient_statistics() for more details.\n";
      }
    }
  }

  nNonZeroGradient += nExceeded;
  nTotalGradient += gradient_rmse.size();

  // Map the input derivatives back to tensors:
  return {grad_errorFunctionWeights, toTensors(errorFunctions, grad_inputs)};
}

template <typename T>
at::Tensor solveTensorSequenceIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    const momentum::ParameterSet& sharedParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& solverOptions) {
  std::tie(modelParams_init, errorFunctionWeights) = checkSequenceIKInputs<T>(
      characters,
      modelParams_init,
      errorFunctionWeights,
      numActiveErrorFunctions,
      "solveTensorIKProblem()");
  const auto nBatch = modelParams_init.size(0);
  const auto nFrames = modelParams_init.size(1);

  momentum::SequenceSolverOptions momentumSolverOptions;
  momentumSolverOptions.minIterations = solverOptions.minIter;
  momentumSolverOptions.maxIterations = solverOptions.maxIter;
  momentumSolverOptions.threshold = solverOptions.threshold;
  momentumSolverOptions.regularization = solverOptions.levmar_lambda;
  momentumSolverOptions.doLineSearch = solverOptions.lineSearch;
  momentumSolverOptions.multithreaded = true;
  momentumSolverOptions.verbose = solverOptions.verbose;

  const auto nParams = characters.front()->parameterTransform.numAllModelParameters();

  at::Tensor modelParameters_final = at::zeros({nBatch, nFrames, (int)nParams}, toScalarType<T>());

  ThreadSafeExceptionWrapper exception;

  std::atomic<uint32_t> numNaNs = 0;
  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    try {
      const auto& character = *characters[iBatch];
      const momentum::ParameterTransformT<T> parameterTransform =
          character.parameterTransform.cast<T>();

      at::Tensor modelParameters_init_cur = modelParams_init.select(0, iBatch);
      at::Tensor modelParameters_final_cur = modelParameters_final.select(0, iBatch);

      std::unique_ptr<momentum::SequenceSolverFunctionT<T>> solverFunction =
          buildSequenceSolverFunction(
              *characters[iBatch],
              parameterTransform,
              modelParameters_init_cur,
              sharedParams,
              errorFunctions,
              errorFunctionWeights.select(0, iBatch),
              weightsMap,
              iBatch);

      if (solverOptions.sequenceSmoothingWeight > 0.f) {
        auto modelParamsSequenceSmoothingErrFunc =
            std::make_shared<momentum::ModelParametersSequenceErrorFunctionT<T>>(character);
        modelParamsSequenceSmoothingErrFunc->setWeight(solverOptions.sequenceSmoothingWeight);
        solverFunction->addSequenceErrorFunction(
            momentum::kAllFrames, modelParamsSequenceSmoothingErrFunc);
      }

      momentum::SequenceSolverT<T> solver(momentumSolverOptions, solverFunction.get());
      solver.setEnabledParameters(activeParams);

      Eigen::VectorX<T> parameters_opt = solverFunction->getJoinedParameterVector();
      solver.solve(parameters_opt);

      // Nan in result, back out the solve:
      if (parameters_opt.array().isNaN().any() || parameters_opt.array().isInf().any()) {
        ++numNaNs;
        toEigenMap<T>(modelParameters_final_cur) = toEigenMap<T>(modelParameters_init_cur);
        return;
      }

      dispenso::parallel_for((size_t)0, nFrames, [&](size_t iFrame) {
        at::Tensor modelParameters_final_frame = modelParameters_final_cur.select(0, iFrame);
        toEigenMap<T>(modelParameters_final_frame) = solverFunction->getFrameParameters(iFrame).v;
      });
    } catch (...) {
      exception.set(std::current_exception());
    }
  });

  if (numNaNs > 0) {
    std::cerr << "WARNING: Detected" << static_cast<uint32_t>(numNaNs)
              << "NAN/INF values in outputs from solve_ik.  Reverting to initial parameters.";
  }

  exception.maybeThrow();

  return modelParameters_final;
}

template at::Tensor solveTensorIKProblem<float>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& options);
template at::Tensor solveTensorIKProblem<double>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& options);

template std::tuple<at::Tensor, torch::autograd::variable_list> d_solveTensorIKProblem<float>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_final,
    at::Tensor d_loss_dModelParameters,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);
template std::tuple<at::Tensor, torch::autograd::variable_list> d_solveTensorIKProblem<double>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_final,
    at::Tensor d_loss_dModelParameters,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template at::Tensor solveTensorSequenceIKProblem<float>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    const momentum::ParameterSet& sharedParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& solverOptions);
template at::Tensor solveTensorSequenceIKProblem<double>(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    const momentum::ParameterSet& sharedParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& solverOptions);

} // namespace pymomentum
