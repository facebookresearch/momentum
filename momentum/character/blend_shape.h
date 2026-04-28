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
  /// @param shapeNames Names of the blend shapes (will be automatically generated if empty or not
  /// the right size)
  BlendShape(
      momentum::span<const Vector3f> baseShape,
      size_t numShapes,
      momentum::span<const std::string> shapeNames = {});

  void setBaseShape(momentum::span<const Vector3f> baseShape) {
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

  /// Output parameter version of computeShape, for callers that want to reuse a
  /// preallocated buffer.
  ///
  /// @tparam T Scalar type (float or double)
  /// @param coefficients Weights for each shape vector
  /// @param output [out] Resulting deformed shape; resized to the base shape's vertex count.
  template <typename T>
  void computeShape(const BlendWeightsT<T>& coefficients, std::vector<Eigen::Vector3<T>>& output)
      const;

  /// Solves for blend weights that best approximate a target shape.
  ///
  /// Uses a thin SVD of the shape vectors when no per-vertex weights are
  /// supplied (the factorization is cached and reused across calls). When
  /// weights are supplied, builds a weighted normal-equation system with
  /// Tikhonov regularization on each call.
  ///
  /// @pre `vertices.size() == getBaseShape().size()`.
  /// @param vertices Target shape to approximate, in the same vertex order as the base shape.
  /// @param regularization Tikhonov regularization weight; higher values bias coefficients toward
  /// zero.
  ///   Only used when per-vertex `weights` are supplied.
  /// @param weights Optional per-vertex importance weights. Ignored unless its size equals
  ///   `vertices.size()`; when ignored, the cached unweighted SVD path is used.
  /// @return Estimated blend shape coefficients (length = number of shape vectors).
  [[nodiscard]] VectorXf estimateCoefficients(
      momentum::span<const Vector3f> vertices,
      float regularization = 1.0f,
      const VectorXf& weights = VectorXf()) const;

  /// Overrides base method to also invalidate the cached SVD factorization used by
  /// `estimateCoefficients`.
  ///
  /// @param index Index of the shape vector to set
  /// @param shapeVector Vector of vertex offsets
  /// @param name Optional name for the shape vector; if empty, the name is auto-generated
  ///   as `"shape_<index>"`.
  void setShapeVector(
      size_t index,
      momentum::span<const Vector3f> shapeVector,
      std::string_view name = "");

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
