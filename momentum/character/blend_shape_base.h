/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/types.h>

namespace momentum {

/// Skinning class that manages blend shape vectors/offsets
///
/// Used to model facial expressions and potentially other deformations
/// such as pose-dependent shape changes
struct BlendShapeBase {
 public:
  BlendShapeBase() {}

  /// @param modelSize Number of vertices in the model
  /// @param numShapes Number of blend shapes
  BlendShapeBase(size_t modelSize, size_t numShapes);

  /// @param shapeVectors Matrix where each column is a shape vector
  void setShapeVectors(const MatrixXf& shapeVectors) {
    shapeVectors_ = shapeVectors;
  }

  const MatrixXf& getShapeVectors() const {
    return shapeVectors_;
  }

  /// Calculates weighted combination of shape vectors
  ///
  /// @tparam T Scalar type (float or double)
  /// @param blendWeights Weights for each shape vector
  /// @return Combined vertex offsets
  template <typename T>
  VectorX<T> computeDeltas(const BlendWeightsT<T>& blendWeights) const;

  /// Adds weighted shape vectors to existing vertices
  ///
  /// @tparam T Scalar type (float or double)
  /// @param blendWeights Weights for each shape vector
  /// @param result [in,out] Vertices to modify
  template <typename T>
  void applyDeltas(const BlendWeightsT<T>& blendWeights, std::vector<Eigen::Vector3<T>>& result)
      const;

  /// @param index Index of the shape vector to set
  /// @param shapeVector Vector of vertex offsets
  void setShapeVector(size_t index, gsl::span<const Vector3f> shapeVector);

  Eigen::Index shapeSize() const {
    return shapeVectors_.cols();
  }

  /// Returns number of vertices (rows/3)
  size_t modelSize() const {
    return shapeVectors_.rows() / 3;
  }

 protected:
  MatrixXf shapeVectors_;
};

} // namespace momentum
