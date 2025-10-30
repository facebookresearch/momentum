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
  BlendShapeBase() = default;

  /// @param modelSize Number of vertices in the model
  /// @param numShapes Number of blend shapes
  BlendShapeBase(size_t modelSize, size_t numShapes);

  /// @param shapeVectors Matrix where each column is a shape vector
  void setShapeVectors(const MatrixXf& shapeVectors) {
    shapeVectors_ = shapeVectors;
  }

  [[nodiscard]] const MatrixXf& getShapeVectors() const {
    return shapeVectors_;
  }

  /// Calculates weighted combination of shape vectors
  ///
  /// @tparam T Scalar type (float or double)
  /// @param blendWeights Weights for each shape vector
  /// @return Combined vertex offsets
  template <typename T>
  [[nodiscard]] VectorX<T> computeDeltas(const BlendWeightsT<T>& blendWeights) const;

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
  void setShapeVector(size_t index, std::span<const Vector3f> shapeVector);

  [[nodiscard]] Eigen::Index shapeSize() const {
    return shapeVectors_.cols();
  }

  /// Returns number of vertices (rows/3)
  [[nodiscard]] size_t modelSize() const {
    return shapeVectors_.rows() / 3;
  }

 protected:
  MatrixXf shapeVectors_;
};

} // namespace momentum
