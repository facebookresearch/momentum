/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "axel/BoundingBox.h"
#include "axel/common/Types.h"

namespace axel {

/**
 * A 3D signed distance field implementation that stores distance values in a uniform grid.
 * Provides efficient trilinear interpolation for smooth distance queries.
 *
 * The SDF uses a regular 3D grid where each cell stores the signed distance to the nearest surface.
 * Negative values indicate inside the surface, positive values indicate outside.
 */
template <typename ScalarType>
class SignedDistanceField {
 public:
  using Scalar = ScalarType;
  using Vector3 = Eigen::Vector3<Scalar>;
  using BoundingBoxType = BoundingBox<Scalar>;

  /// Default "very far" distance value used for uninitialized voxels
  static constexpr Scalar kVeryFarDistance = std::numeric_limits<Scalar>::max();

  /**
   * Constructs an empty SDF with the given dimensions and bounds.
   * All voxels are initialized to the specified initial value.
   *
   * @param bounds The 3D bounding box that defines the spatial extent of the SDF
   * @param resolution Grid resolution in each dimension (nx, ny, nz)
   * @param initialValue Initial distance value for all voxels (default: kVeryFarDistance)
   */
  SignedDistanceField(
      const BoundingBoxType& bounds,
      const Eigen::Vector3<Index>& resolution,
      Scalar initialValue = kVeryFarDistance);

  /**
   * Constructs an SDF with the given dimensions, bounds, and initial data.
   *
   * @param bounds The 3D bounding box that defines the spatial extent of the SDF
   * @param resolution Grid resolution in each dimension (nx, ny, nz)
   * @param data Initial distance values. Must have size resolution.x() * resolution.y() *
   * resolution.z()
   */
  SignedDistanceField(
      const BoundingBoxType& bounds,
      const Eigen::Vector3<Index>& resolution,
      std::vector<Scalar> data);

  /**
   * Get the distance value at a specific grid cell using integer indices.
   * No bounds checking is performed for performance.
   *
   * @param i Grid index in x dimension
   * @param j Grid index in y dimension
   * @param k Grid index in z dimension
   * @return The signed distance value at grid cell (i,j,k)
   */
  [[nodiscard]] Scalar at(Index i, Index j, Index k) const;

  /**
   * Set the distance value at a specific grid cell using integer indices.
   * No bounds checking is performed for performance.
   *
   * @param i Grid index in x dimension
   * @param j Grid index in y dimension
   * @param k Grid index in z dimension
   * @param value The signed distance value to store
   */
  void set(Index i, Index j, Index k, Scalar value);

  /**
   * Sample the SDF at a continuous 3D position using trilinear interpolation.
   * If the query point is outside the SDF bounds, returns the nearest boundary value.
   *
   * @param position 3D world-space position to query
   * @return Interpolated signed distance value
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] InputScalar sample(const Eigen::Vector3<InputScalar>& position) const;

  /**
   * Sample the SDF gradient at a continuous 3D position using analytical gradients
   * from trilinear interpolation. The gradient points in the direction of increasing distance.
   *
   * @param position 3D world-space position to query
   * @return Gradient vector at the given position
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] Eigen::Vector3<InputScalar> gradient(
      const Eigen::Vector3<InputScalar>& position) const;

  /**
   * Sample both the SDF value and gradient at a continuous 3D position using
   * trilinear interpolation. More efficient than calling sample() and gradient()
   * separately as it uses the same 8 corner values for both computations.
   *
   * @param position 3D world-space position to query
   * @return Pair of (value, gradient) at the given position
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] std::pair<InputScalar, Eigen::Vector3<InputScalar>> sampleWithGradient(
      const Eigen::Vector3<InputScalar>& position) const;

  /**
   * Convert a 3D world-space position to continuous grid coordinates.
   *
   * @param position 3D world-space position
   * @return Continuous grid coordinates (may be fractional)
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] Eigen::Vector3<InputScalar> worldToGrid(
      const Eigen::Vector3<InputScalar>& position) const;

  /**
   * Convert continuous grid coordinates to 3D world-space position.
   *
   * @param gridPos Continuous grid coordinates
   * @return 3D world-space position
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] Eigen::Vector3<InputScalar> gridToWorld(
      const Eigen::Vector3<InputScalar>& gridPos) const;

  /**
   * Get the world-space position of a grid cell center given discrete indices.
   * This is equivalent to gridToWorld(Vector3(i, j, k)).
   *
   * @param i Grid index in x dimension
   * @param j Grid index in y dimension
   * @param k Grid index in z dimension
   * @return 3D world-space position of the grid cell center
   */
  [[nodiscard]] Vector3 gridLocation(Index i, Index j, Index k) const;

  /**
   * Check if the given grid coordinates are within bounds.
   *
   * @param i Grid index in x dimension
   * @param j Grid index in y dimension
   * @param k Grid index in z dimension
   * @return True if indices are within valid range
   */
  [[nodiscard]] bool isValidIndex(Index i, Index j, Index k) const;

  /**
   * Get the resolution of the SDF grid.
   *
   * @return Grid resolution as (nx, ny, nz)
   */
  [[nodiscard]] const Eigen::Vector3<Index>& resolution() const;

  /**
   * Get the bounding box of the SDF.
   *
   * @return The 3D bounding box
   */
  [[nodiscard]] const BoundingBoxType& bounds() const;

  /**
   * Get the voxel size in each dimension.
   *
   * @return Voxel size as (dx, dy, dz)
   */
  [[nodiscard]] Vector3 voxelSize() const;

  /**
   * Get the total number of voxels in the SDF.
   *
   * @return Total number of grid cells
   */
  [[nodiscard]] Size totalVoxels() const;

  /**
   * Get read-only access to the underlying data array.
   * Data is stored in row-major order: data[k * nx * ny + j * nx + i]
   *
   * @return Const reference to the data vector
   */
  [[nodiscard]] const std::vector<Scalar>& data() const;

  /**
   * Get mutable access to the underlying data array.
   * Data is stored in row-major order: data[k * nx * ny + j * nx + i]
   *
   * @return Mutable reference to the data vector
   */
  [[nodiscard]] std::vector<Scalar>& data();

  /**
   * Fill the entire SDF with a constant value.
   *
   * @param value The value to fill with
   */
  void fill(Scalar value);

  /**
   * Clear the SDF and reset it to zero values.
   */
  void clear();

 private:
  /**
   * Convert 3D grid indices to a linear array index.
   *
   * @param i Grid index in x dimension
   * @param j Grid index in y dimension
   * @param k Grid index in z dimension
   * @return Linear array index
   */
  [[nodiscard]] Size linearIndex(Index i, Index j, Index k) const;

  /**
   * Clamp grid coordinates to valid bounds.
   *
   * @param gridPos Input grid coordinates
   * @return Clamped grid coordinates
   */
  template <typename InputScalar = Scalar>
  [[nodiscard]] Eigen::Vector3<InputScalar> clampToGrid(
      const Eigen::Vector3<InputScalar>& gridPos) const;

  BoundingBoxType bounds_;
  Eigen::Vector3<Index> resolution_;
  Vector3 voxelSize_;
  std::vector<Scalar> data_;
};

using SignedDistanceFieldf = SignedDistanceField<float>;
using SignedDistanceFieldd = SignedDistanceField<double>;

extern template class SignedDistanceField<float>;
extern template class SignedDistanceField<double>;

} // namespace axel
