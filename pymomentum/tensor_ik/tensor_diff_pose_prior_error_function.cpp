/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_diff_pose_prior_error_function.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/fully_differentiable_motion_error_function.h>
#include <momentum/diff_ik/fully_differentiable_pose_prior_error_function.h>

namespace pymomentum {

using momentum::FullyDifferentiablePosePriorErrorFunction;
using momentum::FullyDifferentiablePosePriorErrorFunctionT;

namespace {

template <typename T>
class TensorDiffPosePriorErrorFunction : public TensorErrorFunction<T> {
 public:
  explicit TensorDiffPosePriorErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor pi,
      at::Tensor mu,
      at::Tensor W,
      at::Tensor sigma,
      at::Tensor parameterIndices);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t /* unused iBatch */,
      size_t /* unused nFrames */) const override;
};

const int NMIXTURES_IDX = -1;
const int NPARAMETERS_IDX = -2;
const int NCOMPONENTS_IDX = -3;

template <typename T>
TensorDiffPosePriorErrorFunction<T>::TensorDiffPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices)
    : TensorErrorFunction<T>(
          "PosePrior",
          "pose_prior",
          batchSize,
          nFrames,
          {{FullyDifferentiablePosePriorErrorFunction::kPi,
            pi,
            {NMIXTURES_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiablePosePriorErrorFunction::kMu,
            mu,
            {NMIXTURES_IDX, NPARAMETERS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiablePosePriorErrorFunction::kW,
            W,
            {NMIXTURES_IDX, NCOMPONENTS_IDX, NPARAMETERS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiablePosePriorErrorFunction::kSigma,
            sigma,
            {NMIXTURES_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiablePosePriorErrorFunction::kParameterIndices,
            parameterIndices,
            {NPARAMETERS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {"pose_prior",
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {{NMIXTURES_IDX, "nMixtures"},
           {NPARAMETERS_IDX, "nParameters"},
           {NCOMPONENTS_IDX, "nPrincipalComponents"}}) {}

std::vector<std::string> parameterIndicesToNames(
    Eigen::Ref<const Eigen::VectorXi> parameterIndices,
    const momentum::ParameterTransform& paramTransform) {
  std::vector<std::string> paramNames;
  paramNames.reserve(parameterIndices.size());
  for (int i = 0; i < parameterIndices.size(); ++i) {
    const auto pi = parameterIndices[i];
    if (pi < 0 || pi >= paramTransform.name.size()) {
      paramNames.emplace_back();
    } else {
      paramNames.push_back(paramTransform.name[pi]);
    }
  }
  return paramNames;
}

template <typename T>
std::vector<Eigen::MatrixX<T>> tensor_W_to_matrix_list(
    Eigen::Ref<const Eigen::VectorX<T>> W,
    int nMixtures,
    int nParameters,
    int nComponents) {
  std::vector<Eigen::MatrixX<T>> result;
  assert(W.size() == nMixtures * nParameters * nComponents);
  result.reserve(nMixtures);
  for (int iMix = 0; iMix < nMixtures; ++iMix) {
    result.emplace_back(nParameters, nComponents);
    for (int jComponent = 0; jComponent < nComponents; ++jComponent) {
      result.back().col(jComponent) =
          W.segment(iMix * nComponents * nParameters + jComponent * nParameters, nParameters);
    }
  }

  return result;
}

template <typename T>
Eigen::MatrixX<T>
tensor_mu_to_matrix(Eigen::Ref<const Eigen::VectorX<T>> mu, int nMixtures, int nParameters) {
  Eigen::MatrixX<T> muMat(nMixtures, nParameters);
  assert(mu.size() == nMixtures * nParameters);

  for (int iMix = 0; iMix < nMixtures; ++iMix) {
    muMat.row(iMix) = mu.segment(iMix * nParameters, nParameters);
  }
  return muMat;
}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorDiffPosePriorErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  const auto pi = this->getTensorInput(FullyDifferentiablePosePriorErrorFunction::kPi)
                      .template toEigenMap<T>(iBatch, jFrame);
  const auto mu = this->getTensorInput(FullyDifferentiablePosePriorErrorFunction::kMu)
                      .template toEigenMap<T>(iBatch, jFrame);
  const auto W = this->getTensorInput(FullyDifferentiablePosePriorErrorFunction::kW)
                     .template toEigenMap<T>(iBatch, jFrame);
  const auto sigma = this->getTensorInput(FullyDifferentiablePosePriorErrorFunction::kSigma)
                         .template toEigenMap<T>(iBatch, jFrame);
  const std::vector<std::string> parameterNames = parameterIndicesToNames(
      this->getTensorInput(FullyDifferentiablePosePriorErrorFunction::kParameterIndices)
          .template toEigenMap<int>(iBatch, jFrame),
      character.parameterTransform);

  const auto nMixtures = this->sharedSize(NMIXTURES_IDX);
  const auto nParameters = this->sharedSize(NPARAMETERS_IDX);
  const auto nComponents = this->sharedSize(NCOMPONENTS_IDX);

  auto result = std::make_unique<momentum::FullyDifferentiablePosePriorErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform, parameterNames);
  if (nMixtures != 0 && nParameters != 0) {
    result->setPosePrior(
        pi,
        tensor_mu_to_matrix<T>(mu, nMixtures, nParameters),
        tensor_W_to_matrix_list<T>(W, nMixtures, nParameters, nComponents),
        sigma);
  }

  return result;
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createDiffPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices) {
  return std::make_unique<TensorDiffPosePriorErrorFunction<T>>(
      batchSize, nFrames, pi, mu, W, sigma, parameterIndices);
}

template std::unique_ptr<TensorErrorFunction<float>> createDiffPosePriorErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices);

template std::unique_ptr<TensorErrorFunction<double>> createDiffPosePriorErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices);

} // namespace pymomentum
