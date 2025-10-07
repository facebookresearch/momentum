/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_collision_error_function.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/collision_error_function.h>

namespace pymomentum {

namespace {

template <typename T>
class TensorCollisionErrorFunction : public TensorErrorFunction<T> {
 public:
  explicit TensorCollisionErrorFunction(size_t batchSize, size_t nFrames);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t /* unused iBatch */,
      size_t /* unused jFrame */) const override;
};

template <typename T>
TensorCollisionErrorFunction<T>::TensorCollisionErrorFunction(size_t batchSize, size_t nFrames)
    : TensorErrorFunction<T>("Collision", "collision", batchSize, nFrames, {}, {}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorCollisionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t /* unused iBatch*/,
    size_t /* unused jFrame*/) const {
  if (character.collision) {
    return std::make_shared<momentum::CollisionErrorFunctionT<T>>(character);
  } else {
    // Dummy error function; since CollisionErrorFunction takes a reference to
    // the actual collision geometry it's not safe to create the concrete err
    // func.
    return std::make_shared<momentum::SkeletonErrorFunctionT<T>>(
        character.skeleton, character.parameterTransform);
  }
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createCollisionErrorFunction(
    size_t batchSize,
    size_t nFrames) {
  return std::make_unique<TensorCollisionErrorFunction<T>>(batchSize, nFrames);
}

template std::unique_ptr<TensorErrorFunction<float>> createCollisionErrorFunction(size_t, size_t);
template std::unique_ptr<TensorErrorFunction<double>> createCollisionErrorFunction(size_t, size_t);

} // namespace pymomentum
