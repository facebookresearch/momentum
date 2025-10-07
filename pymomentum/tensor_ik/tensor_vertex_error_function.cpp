/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_vertex_error_function.h"

#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/vertex_error_function.h>

namespace pymomentum {

namespace {

const static int NCONS_IDX = -1;
static constexpr const char* kVerticesName = "vertices";
static constexpr const char* kWeightsName = "weights";
static constexpr const char* kTargetPositionsName = "target_positions";
static constexpr const char* kTargetNormalsName = "target_normals";
static constexpr const char* kTypeName = "type";

template <typename T>
class TensorVertexErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorVertexErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor vertexIndex,
      at::Tensor weights,
      at::Tensor target_positions,
      at::Tensor target_normals,
      momentum::VertexConstraintType constraintType);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t jFrame) const override;

  const momentum::VertexConstraintType _constraintType;
};

template <typename T>
TensorVertexErrorFunction<T>::TensorVertexErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor target_normals,
    momentum::VertexConstraintType constraintType)
    : TensorErrorFunction<T>(
          "Vertex",
          "vertex_cons",
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
               {NCONS_IDX, 3},
               TensorType::TYPE_FLOAT,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::REQUIRED},
              {kTargetNormalsName,
               target_normals,
               {NCONS_IDX, 3},
               TensorType::TYPE_FLOAT,
               TensorInput::NON_DIFFERENTIABLE,
               constraintType == momentum::VertexConstraintType::Position ? TensorInput::OPTIONAL
                                                                          : TensorInput::REQUIRED},
              {kTypeName,
               at::Tensor(),
               {},
               TensorType::TYPE_SENTINEL,
               TensorInput::NON_DIFFERENTIABLE,
               TensorInput::OPTIONAL}},
          {{NCONS_IDX, "nConstraints"}}),
      _constraintType(constraintType) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorVertexErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto result = std::make_unique<momentum::VertexErrorFunctionT<T>>(character, _constraintType);

  const auto weights = this->getTensorInput(kWeightsName).template toEigenMap<T>(iBatch, jFrame);
  const auto vertices =
      this->getTensorInput(kVerticesName).template toEigenMap<int>(iBatch, jFrame);
  const auto target_positions =
      this->getTensorInput(kTargetPositionsName).template toEigenMap<T>(iBatch, jFrame);
  const auto target_normals =
      this->getTensorInput(kTargetNormalsName).template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index i = 0; i < nCons; ++i) {
    result->addConstraint(
        extractScalar<int>(vertices, i),
        extractScalar<T>(weights, i, T(1)),
        extractVector<T, 3>(target_positions, i),
        extractVector<T, 3>(target_normals, i, Eigen::Vector3<T>::Zero()));
  }

  return result;
}

template class TensorVertexErrorFunction<float>;
template class TensorVertexErrorFunction<double>;

} // namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createVertexErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor target_normals,
    momentum::VertexConstraintType constraintType) {
  return std::make_unique<TensorVertexErrorFunction<T>>(
      batchSize, nFrames, vertexIndex, weights, target_positions, target_normals, constraintType);
}

template std::unique_ptr<TensorErrorFunction<float>> createVertexErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor target_normals,
    momentum::VertexConstraintType constraintType);

template std::unique_ptr<TensorErrorFunction<double>> createVertexErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor target_normals,
    momentum::VertexConstraintType constraintType);

} // namespace pymomentum
