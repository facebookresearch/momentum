/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tensor_vertex_projection_error_function.h"

#include "tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/vertex_projection_error_function.h>

namespace pymomentum {

namespace {

const static int NCONS_IDX = -1;
static constexpr const char* kVerticesName = "vertices";
static constexpr const char* kWeightsName = "weights";
static constexpr const char* kTargetPositionsName = "target_positions";
static constexpr const char* kProjectionsName = "projections";

template <typename T>
class TensorVertexProjectionErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorVertexProjectionErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor vertexIndex,
      at::Tensor weights,
      at::Tensor target_positions,
      at::Tensor projections);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t jFrame) const override;
};

template <typename T>
TensorVertexProjectionErrorFunction<T>::TensorVertexProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor projections)
    : TensorErrorFunction<T>(
          "VertexProjection",
          "vertex_projection_cons",
          batchSize,
          nFrames,
          std::initializer_list<TensorInput>{
              {kVerticesName,
               vertexIndex,
               {NCONS_IDX},
               TensorType::TYPE_INT,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::REQUIRED},
              {kWeightsName,
               weights,
               {NCONS_IDX},
               TensorType::TYPE_FLOAT,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::OPTIONAL},
              {kTargetPositionsName,
               target_positions,
               {NCONS_IDX, 2},
               TensorType::TYPE_FLOAT,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::REQUIRED},
              {kProjectionsName,
               projections,
               {NCONS_IDX, 3, 4},
               TensorType::TYPE_FLOAT,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::REQUIRED}},
          {{NCONS_IDX, "nConstraints"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorVertexProjectionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto result =
      std::make_unique<momentum::VertexProjectionErrorFunctionT<T>>(character);

  const auto weights =
      this->getTensorInput(kWeightsName).template toEigenMap<T>(iBatch, jFrame);
  const auto vertices = this->getTensorInput(kVerticesName)
                            .template toEigenMap<int>(iBatch, jFrame);
  const auto target_positions = this->getTensorInput(kTargetPositionsName)
                                    .template toEigenMap<T>(iBatch, jFrame);
  const auto projections = this->getTensorInput(kProjectionsName)
                               .template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index i = 0; i < nCons; ++i) {
    result->addConstraint(
        extractScalar<int>(vertices, i),
        extractScalar<T>(weights, i, T(1)),
        extractVector<T, 2>(target_positions, i),
        extractMatrix<T, 3, 4>(projections, i));
  }

  return result;
}

template class TensorVertexProjectionErrorFunction<float>;
template class TensorVertexProjectionErrorFunction<double>;

} // namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createVertexProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor projections) {
  return std::make_unique<TensorVertexProjectionErrorFunction<T>>(
      batchSize, nFrames, vertexIndex, weights, target_positions, projections);
}

template std::unique_ptr<TensorErrorFunction<float>>
createVertexProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor projections);

template std::unique_ptr<TensorErrorFunction<double>>
createVertexProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor projections);

} // namespace pymomentum
