/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape_base.h>

namespace momentum {

/// Skinning class that combines blend shape vectors with a base shape
///
/// Extends BlendShapeBase by adding a base shape (e.g., PCA mean) and
/// functionality for computing deformed shapes and estimating blend coefficients.
/// Primarily used to model identity-dependent shape deformations.
struct BlendShape : public BlendShapeBase {
 public:
  BlendShape() : factorizationValid_(false) {}

  /// @param baseShape Base shape vertices
  /// @param numShapes Number of blend shapes
  BlendShape(std::span<const Vector3f> baseShape, size_t numShapes);

  void setBaseShape(std::span<const Vector3f> baseShape) {
    baseShape_.assign(baseShape.begin(), baseShape.end());
  }

  [[nodiscard]] const std::vector<Vector3f>& getBaseShape() const {
    return baseShape_;
  }

  /// Whether SVD factorization is up-to-date
  [[nodiscard]] bool getFactorizationValid() const {
    return factorizationValid_;
  }

  /// Applies blend weights to create a deformed shape
  ///
  /// Adds weighted shape vectors to the base shape
  ///
  /// @tparam T Scalar type (float or double)
  /// @param coefficients Weights for each shape vector
  /// @return Resulting deformed shape
  template <typename T>
  [[nodiscard]] std::vector<Eigen::Vector3<T>> computeShape(
      const BlendWeightsT<T>& coefficients) const;

  /// Output parameter version of computeShape
  ///
  /// @tparam T Scalar type (float or double)
  /// @param coefficients Weights for each shape vector
  /// @param result [out] Resulting deformed shape
  template <typename T>
  void computeShape(const BlendWeightsT<T>& coefficients, std::vector<Eigen::Vector3<T>>& result)
      const;

  /// Solves for blend weights that best approximate a target shape
  ///
  /// Uses SVD to find optimal coefficients with optional regularization
  /// and per-vertex weighting
  ///
  /// @param vertices Target shape to approximate
  /// @param regularization Higher values produce smaller coefficients
  /// @param weights Optional per-vertex importance weights
  /// @return Estimated blend shape coefficients
  [[nodiscard]] VectorXf estimateCoefficients(
      std::span<const Vector3f> vertices,
      float regularization = 1.0f,
      const VectorXf& weights = VectorXf()) const;

  /// Overrides base method to also invalidate factorization
  ///
  /// @param index Index of the shape vector to set
  /// @param shapeVector Vector of vertex offsets
  void setShapeVector(size_t index, std::span<const Vector3f> shapeVector);

  /// Compares all components of two blend shapes
  ///
  /// @param blendShape Other blend shape to compare with
  [[nodiscard]] bool isApprox(const BlendShape& blendShape) const;

 private:
  std::vector<Vector3f> baseShape_;
  mutable Eigen::JacobiSVD<MatrixXf> factorization_;
  mutable bool factorizationValid_;
};

} // namespace momentum
