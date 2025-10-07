/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_limit_error_function.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/limit_error_function.h>

namespace pymomentum {

namespace {

template <typename T>
class TensorLimitErrorFunction : public TensorErrorFunction<T> {
 public:
  explicit TensorLimitErrorFunction(size_t batchSize, size_t nFrames);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t /* unused iBatch */,
      size_t /* unused nFrames */) const override;
};

template <typename T>
TensorLimitErrorFunction<T>::TensorLimitErrorFunction(size_t batchSize, size_t nFrames)
    : TensorErrorFunction<T>("Limit", "limit", batchSize, nFrames, {}, {}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorLimitErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t /* unused iBatch*/,
    size_t /* unused jFrame*/) const {
  return std::make_unique<momentum::LimitErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform, character.parameterLimits);
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createLimitErrorFunction(size_t batchSize, size_t nFrames) {
  return std::make_unique<TensorLimitErrorFunction<T>>(batchSize, nFrames);
}

template std::unique_ptr<TensorErrorFunction<float>> createLimitErrorFunction(size_t, size_t);
template std::unique_ptr<TensorErrorFunction<double>> createLimitErrorFunction(size_t, size_t);

} // namespace pymomentum
