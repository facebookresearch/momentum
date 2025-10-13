/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/SignedDistanceField.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "axel/common/Constants.h"

namespace axel {

template <typename ScalarType>
SignedDistanceField<ScalarType>::SignedDistanceField(
    const BoundingBoxType& bounds,
    const Eigen::Vector3<Index>& resolution,
    Scalar initialValue)
    : bounds_(bounds),
      resolution_(resolution),
      voxelSize_(
          (bounds_.max() - bounds_.min()).cwiseQuotient(resolution_.template cast<Scalar>())),
      data_(static_cast<Size>(resolution_.x()) * resolution_.y() * resolution_.z(), initialValue) {
  assert(resolution_.x() > 0 && resolution_.y() > 0 && resolution_.z() > 0);
}

template <typename ScalarType>
SignedDistanceField<ScalarType>::SignedDistanceField(
    const BoundingBoxType& bounds,
    const Eigen::Vector3<Index>& resolution,
    std::vector<Scalar> data)
    : bounds_(bounds),
      resolution_(resolution),
      voxelSize_(
          (bounds_.max() - bounds_.min()).cwiseQuotient(resolution_.template cast<Scalar>())),
      data_(std::move(data)) {
  assert(resolution_.x() > 0 && resolution_.y() > 0 && resolution_.z() > 0);
  assert(data_.size() == static_cast<Size>(resolution_.x()) * resolution_.y() * resolution_.z());
}

template <typename ScalarType>
ScalarType SignedDistanceField<ScalarType>::at(Index i, Index j, Index k) const {
  return data_[linearIndex(i, j, k)];
}

template <typename ScalarType>
void SignedDistanceField<ScalarType>::set(Index i, Index j, Index k, Scalar value) {
  data_[linearIndex(i, j, k)] = value;
}

template <typename ScalarType>
ScalarType SignedDistanceField<ScalarType>::sample(const Vector3& position) const {
  const Vector3 gridPos = worldToGrid(position);

  // Clamp to valid grid bounds
  const Vector3 clampedGridPos = clampToGrid(gridPos);

  // Get the integer grid coordinates of the lower corner
  const auto i0 = static_cast<Index>(std::floor(clampedGridPos.x()));
  const auto j0 = static_cast<Index>(std::floor(clampedGridPos.y()));
  const auto k0 = static_cast<Index>(std::floor(clampedGridPos.z()));

  // Get the integer grid coordinates of the upper corner
  const Index i1 = std::min(i0 + 1, resolution_.x() - 1);
  const Index j1 = std::min(j0 + 1, resolution_.y() - 1);
  const Index k1 = std::min(k0 + 1, resolution_.z() - 1);

  // Calculate interpolation weights
  const Scalar fx = clampedGridPos.x() - static_cast<Scalar>(i0);
  const Scalar fy = clampedGridPos.y() - static_cast<Scalar>(j0);
  const Scalar fz = clampedGridPos.z() - static_cast<Scalar>(k0);

  // Get the 8 corner values
  const Scalar c000 = at(i0, j0, k0);
  const Scalar c001 = at(i0, j0, k1);
  const Scalar c010 = at(i0, j1, k0);
  const Scalar c011 = at(i0, j1, k1);
  const Scalar c100 = at(i1, j0, k0);
  const Scalar c101 = at(i1, j0, k1);
  const Scalar c110 = at(i1, j1, k0);
  const Scalar c111 = at(i1, j1, k1);

  // Trilinear interpolation for value
  const Scalar c00 = c000 * (Scalar{1} - fx) + c100 * fx;
  const Scalar c01 = c001 * (Scalar{1} - fx) + c101 * fx;
  const Scalar c10 = c010 * (Scalar{1} - fx) + c110 * fx;
  const Scalar c11 = c011 * (Scalar{1} - fx) + c111 * fx;

  const Scalar c0 = c00 * (Scalar{1} - fy) + c10 * fy;
  const Scalar c1 = c01 * (Scalar{1} - fy) + c11 * fy;

  Scalar value = c0 * (Scalar{1} - fz) + c1 * fz;

  // Calculate offset from original query point to clamped point
  if (clampedGridPos != gridPos) {
    const Vector3 clampedWorldPos = gridToWorld(clampedGridPos);
    value += (position - clampedWorldPos).norm();
  }

  return value;
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3 SignedDistanceField<ScalarType>::gradient(
    const Vector3& position) const {
  return sampleWithGradient(position).second;
}

template <typename ScalarType>
std::pair<ScalarType, typename SignedDistanceField<ScalarType>::Vector3>
SignedDistanceField<ScalarType>::sampleWithGradient(const Vector3& position) const {
  const Vector3 gridPos = worldToGrid(position);

  // Clamp to valid grid bounds
  const Vector3 clampedGridPos = clampToGrid(gridPos);

  // Get the integer grid coordinates of the lower corner
  const auto i0 = static_cast<Index>(std::floor(clampedGridPos.x()));
  const auto j0 = static_cast<Index>(std::floor(clampedGridPos.y()));
  const auto k0 = static_cast<Index>(std::floor(clampedGridPos.z()));

  // Get the integer grid coordinates of the upper corner
  const Index i1 = std::min(i0 + 1, resolution_.x() - 1);
  const Index j1 = std::min(j0 + 1, resolution_.y() - 1);
  const Index k1 = std::min(k0 + 1, resolution_.z() - 1);

  // Calculate interpolation weights
  const Scalar fx = clampedGridPos.x() - static_cast<Scalar>(i0);
  const Scalar fy = clampedGridPos.y() - static_cast<Scalar>(j0);
  const Scalar fz = clampedGridPos.z() - static_cast<Scalar>(k0);

  // Get the 8 corner values
  const Scalar c000 = at(i0, j0, k0);
  const Scalar c001 = at(i0, j0, k1);
  const Scalar c010 = at(i0, j1, k0);
  const Scalar c011 = at(i0, j1, k1);
  const Scalar c100 = at(i1, j0, k0);
  const Scalar c101 = at(i1, j0, k1);
  const Scalar c110 = at(i1, j1, k0);
  const Scalar c111 = at(i1, j1, k1);

  // Trilinear interpolation for value
  const Scalar c00 = c000 * (Scalar{1} - fx) + c100 * fx;
  const Scalar c01 = c001 * (Scalar{1} - fx) + c101 * fx;
  const Scalar c10 = c010 * (Scalar{1} - fx) + c110 * fx;
  const Scalar c11 = c011 * (Scalar{1} - fx) + c111 * fx;

  const Scalar c0 = c00 * (Scalar{1} - fy) + c10 * fy;
  const Scalar c1 = c01 * (Scalar{1} - fy) + c11 * fy;

  Scalar value = c0 * (Scalar{1} - fz) + c1 * fz;

  // Calculate offset from original query point to clamped point
  if (clampedGridPos != gridPos) {
    const Vector3 clampedWorldPos = gridToWorld(clampedGridPos);
    const Vector3 offsetVector = position - clampedWorldPos;
    const auto offsetDistance = offsetVector.norm();

    // If point was outside bounds, add offset distance and use offset gradient
    if (offsetDistance > Scalar{0}) {
      return {value + offsetDistance, -(offsetVector / offsetDistance)};
    }
  }

  // Standard analytical gradient computation for points inside the grid bounds
  const Scalar dfdx_term = (c100 - c000) * (Scalar{1} - fy) * (Scalar{1} - fz) +
      (c101 - c001) * (Scalar{1} - fy) * fz + (c110 - c010) * fy * (Scalar{1} - fz) +
      (c111 - c011) * fy * fz;

  const Scalar dfdy_term = (c010 - c000) * (Scalar{1} - fx) * (Scalar{1} - fz) +
      (c011 - c001) * (Scalar{1} - fx) * fz + (c110 - c100) * fx * (Scalar{1} - fz) +
      (c111 - c101) * fx * fz;

  const Scalar dfdz_term = (c001 - c000) * (Scalar{1} - fx) * (Scalar{1} - fy) +
      (c011 - c010) * (Scalar{1} - fx) * fy + (c101 - c100) * fx * (Scalar{1} - fy) +
      (c111 - c110) * fx * fy;

  // Transform grid space gradients to world space
  const Vector3 gradient(
      dfdx_term / voxelSize_.x(), dfdy_term / voxelSize_.y(), dfdz_term / voxelSize_.z());

  return std::make_pair(value, gradient);
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3 SignedDistanceField<ScalarType>::worldToGrid(
    const Vector3& position) const {
  const Vector3 localPos = position - bounds_.min();
  return localPos.cwiseQuotient(voxelSize_);
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3 SignedDistanceField<ScalarType>::gridToWorld(
    const Vector3& gridPos) const {
  return bounds_.min() + gridPos.cwiseProduct(voxelSize_);
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3
SignedDistanceField<ScalarType>::gridLocation(Index i, Index j, Index k) const {
  return gridToWorld(
      Vector3(static_cast<Scalar>(i), static_cast<Scalar>(j), static_cast<Scalar>(k)));
}

template <typename ScalarType>
bool SignedDistanceField<ScalarType>::isValidIndex(Index i, Index j, Index k) const {
  return i >= 0 && i < resolution_.x() && j >= 0 && j < resolution_.y() && k >= 0 &&
      k < resolution_.z();
}

template <typename ScalarType>
const Eigen::Vector3<Index>& SignedDistanceField<ScalarType>::resolution() const {
  return resolution_;
}

template <typename ScalarType>
const typename SignedDistanceField<ScalarType>::BoundingBoxType&
SignedDistanceField<ScalarType>::bounds() const {
  return bounds_;
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3 SignedDistanceField<ScalarType>::voxelSize()
    const {
  return voxelSize_;
}

template <typename ScalarType>
Size SignedDistanceField<ScalarType>::totalVoxels() const {
  return static_cast<Size>(resolution_.x()) * resolution_.y() * resolution_.z();
}

template <typename ScalarType>
const std::vector<ScalarType>& SignedDistanceField<ScalarType>::data() const {
  return data_;
}

template <typename ScalarType>
std::vector<ScalarType>& SignedDistanceField<ScalarType>::data() {
  return data_;
}

template <typename ScalarType>
void SignedDistanceField<ScalarType>::fill(Scalar value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <typename ScalarType>
void SignedDistanceField<ScalarType>::clear() {
  fill(Scalar{0});
}

template <typename ScalarType>
Size SignedDistanceField<ScalarType>::linearIndex(Index i, Index j, Index k) const {
  return static_cast<Size>(k) * resolution_.x() * resolution_.y() +
      static_cast<Size>(j) * resolution_.x() + static_cast<Size>(i);
}

template <typename ScalarType>
typename SignedDistanceField<ScalarType>::Vector3 SignedDistanceField<ScalarType>::clampToGrid(
    const Vector3& gridPos) const {
  Vector3 clamped;
  clamped.x() = std::clamp(gridPos.x(), Scalar{0}, static_cast<Scalar>(resolution_.x() - 1));
  clamped.y() = std::clamp(gridPos.y(), Scalar{0}, static_cast<Scalar>(resolution_.y() - 1));
  clamped.z() = std::clamp(gridPos.z(), Scalar{0}, static_cast<Scalar>(resolution_.z() - 1));
  return clamped;
}

// Explicit instantiation
template class SignedDistanceField<float>;
template class SignedDistanceField<double>;

} // namespace axel
