/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape_base.h>

#include <momentum/common/checks.h>

namespace momentum {

template <typename T>
VectorX<T> BlendShapeBase::computeDeltas(const BlendWeightsT<T>& blendWeights) const {
  MT_CHECK(
      blendWeights.size() <= shapeVectors_.cols(),
      "{} vs {}",
      blendWeights.size(),
      shapeVectors_.cols());
  return shapeVectors_.leftCols(blendWeights.size()).template cast<T>() * blendWeights.v;
}

template <typename T>
void BlendShapeBase::applyDeltas(
    const BlendWeightsT<T>& blendWeights,
    std::vector<Eigen::Vector3<T>>& result) const {
  const VectorX<T> deltas = computeDeltas(blendWeights);
  Eigen::Map<Eigen::VectorX<T>> baseShape(&result[0][0], result.size() * 3);
  baseShape += deltas;
}

BlendShapeBase::BlendShapeBase(const size_t modelSize, const size_t numShapes)
    : shapeVectors_(MatrixXf::Zero(modelSize * 3, numShapes)) {}

void BlendShapeBase::setShapeVector(const size_t index, gsl::span<const Vector3f> shape) {
  MT_CHECK(modelSize() == shape.size(), "{} is not {}", modelSize(), shape.size());
  MT_CHECK(
      gsl::narrow<Eigen::Index>(shape.size()) * 3 == shapeVectors_.rows(),
      "{} is not {}",
      shape.size() * 3,
      shapeVectors_.rows());
  MT_CHECK(
      index < static_cast<size_t>(shapeVectors_.cols()), "{} vs {}", index, shapeVectors_.cols());

  shapeVectors_.col(index) = Map<const VectorXf>(&shape[0][0], shape.size() * 3);
}

template VectorX<float> BlendShapeBase::computeDeltas<float>(
    const BlendWeightsT<float>& parameters) const;
template VectorX<double> BlendShapeBase::computeDeltas<double>(
    const BlendWeightsT<double>& parameters) const;

template void BlendShapeBase::applyDeltas(
    const BlendWeightsT<float>& blendWeights,
    std::vector<Eigen::Vector3<float>>& result) const;
template void BlendShapeBase::applyDeltas(
    const BlendWeightsT<double>& blendWeights,
    std::vector<Eigen::Vector3<double>>& result) const;

} // namespace momentum
