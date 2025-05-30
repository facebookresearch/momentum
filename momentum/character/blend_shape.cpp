/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/blend_shape.h"

#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
std::vector<Eigen::Vector3<T>> BlendShape::computeShape(
    const BlendWeightsT<T>& coefficients) const {
  std::vector<Eigen::Vector3<T>> result;
  computeShape(coefficients, result);
  return result;
}

template <typename T>
void BlendShape::computeShape(
    const BlendWeightsT<T>& blendWeights,
    std::vector<Eigen::Vector3<T>>& output) const {
  MT_CHECK(
      gsl::narrow<Eigen::Index>(baseShape_.size()) * 3 == shapeVectors_.rows(),
      "{} is not {}",
      baseShape_.size() * 3,
      shapeVectors_.rows());

  output.resize(baseShape_.size());
  Eigen::Map<Eigen::VectorX<T>> outputVec(&output[0][0], output.size() * 3);
  const Eigen::Map<const Eigen::VectorXf> baseVec(&baseShape_[0][0], baseShape_.size() * 3);

  outputVec = baseVec.template cast<T>() + computeDeltas<T>(blendWeights);
}

VectorXf BlendShape::estimateCoefficients(
    gsl::span<const Vector3f> vertices,
    const float regularization,
    const VectorXf& weights) const {
  MT_CHECK(
      gsl::narrow<Eigen::Index>(baseShape_.size()) * 3 == shapeVectors_.rows(),
      "{} is not {}",
      baseShape_.size() * 3,
      shapeVectors_.rows());
  MT_CHECK(
      vertices.size() == baseShape_.size(), "{} is not {}", vertices.size(), baseShape_.size());

  const Map<const VectorXf> baseVec(&baseShape_[0][0], baseShape_.size() * 3);
  const Map<const VectorXf> vertexVec(&vertices[0][0], vertices.size() * 3);

  // if we're not using weights, use default pipeline
  if (weights.size() != gsl::narrow<Eigen::Index>(vertices.size())) {
    // check if factorization is valid
    if (!factorizationValid_) {
      factorization_.compute(shapeVectors_, Eigen::ComputeThinU | Eigen::ComputeThinV);
      factorizationValid_ = true;
    }

    // estimate the shapeVectors for the given vertices
    return factorization_.solve(vertexVec - baseVec);
  } else {
    // if we have weights, we definitely have to compute a new factorization that incorporates the
    // weights
    VectorXf wts(weights.size() * 3);
    Eigen::Map<VectorXf, 0, Eigen::Stride<1, 3>>(&wts(0), weights.size()) = weights;
    Eigen::Map<VectorXf, 0, Eigen::Stride<1, 3>>(&wts(1), weights.size()) = weights;
    Eigen::Map<VectorXf, 0, Eigen::Stride<1, 3>>(&wts(2), weights.size()) = weights;

    // calculate weighted coefficient matrix and rhs
    const auto rows = shapeVectors_.rows();
    const auto cols = shapeVectors_.cols();
    MatrixXf A(rows + cols, cols);
    for (int i = 0; i < wts.size(); i++) {
      A.row(i) = shapeVectors_.row(i) * wts(i);
    }
    A.block(rows, 0, cols, cols) = MatrixXf::Identity(cols, cols) * regularization;

    VectorXf b = VectorXf::Zero(rows + cols);
    b.topRows(rows) = (vertexVec - baseVec).cwiseProduct(wts);

    // solve normal equations to get results
    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
  }
}

BlendShape::BlendShape(gsl::span<const Vector3f> baseShape, const size_t numShapes)
    : BlendShapeBase(baseShape.size(), numShapes),
      baseShape_(baseShape.begin(), baseShape.end()),
      factorizationValid_(false) {}

void BlendShape::setShapeVector(const size_t index, gsl::span<const Vector3f> shape) {
  BlendShapeBase::setShapeVector(index, shape);

  // mark as not up to date
  factorizationValid_ = false;
}

bool BlendShape::isApprox(const BlendShape& blendShape) const {
  if (baseShape_.size() != blendShape.getBaseShape().size()) {
    return false;
  }
  for (size_t i = 0; i < baseShape_.size(); i++) {
    if (!baseShape_[i].isApprox(blendShape.getBaseShape()[i])) {
      return false;
    }
  }

  return shapeVectors_.isApprox(blendShape.getShapeVectors()) &&
      (factorizationValid_ == blendShape.getFactorizationValid());
}

template std::vector<Eigen::Vector3<float>> BlendShape::computeShape<float>(
    const BlendWeightsT<float>& parameters) const;
template std::vector<Eigen::Vector3<double>> BlendShape::computeShape<double>(
    const BlendWeightsT<double>& parameters) const;

template void BlendShape::computeShape<float>(
    const BlendWeightsT<float>& parameters,
    std::vector<Eigen::Vector3<float>>& output) const;
template void BlendShape::computeShape<double>(
    const BlendWeightsT<double>& parameters,
    std::vector<Eigen::Vector3<double>>& output) const;

} // namespace momentum
