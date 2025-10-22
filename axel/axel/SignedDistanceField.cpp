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
template <typename InputScalar>
InputScalar SignedDistanceField<ScalarType>::sample(
    const Eigen::Vector3<InputScalar>& position) const {
  const Eigen::Vector3<InputScalar> gridPos = worldToGrid(position);

  // Clamp to valid grid bounds
  const Eigen::Vector3<InputScalar> clampedGridPos = clampToGrid(gridPos);

  // Get the integer grid coordinates of the lower corner
  const auto i0 = static_cast<Index>(std::floor(clampedGridPos.x()));
  const auto j0 = static_cast<Index>(std::floor(clampedGridPos.y()));
  const auto k0 = static_cast<Index>(std::floor(clampedGridPos.z()));

  // Get the integer grid coordinates of the upper corner
  const Index i1 = std::min(i0 + 1, resolution_.x() - 1);
  const Index j1 = std::min(j0 + 1, resolution_.y() - 1);
  const Index k1 = std::min(k0 + 1, resolution_.z() - 1);

  // Calculate interpolation weights using InputScalar precision
  const InputScalar fx =
      static_cast<InputScalar>(clampedGridPos.x()) - static_cast<InputScalar>(i0);
  const InputScalar fy =
      static_cast<InputScalar>(clampedGridPos.y()) - static_cast<InputScalar>(j0);
  const InputScalar fz =
      static_cast<InputScalar>(clampedGridPos.z()) - static_cast<InputScalar>(k0);

  // Get the 8 corner values and convert to InputScalar precision for interpolation
  const auto c000 = static_cast<InputScalar>(at(i0, j0, k0));
  const auto c001 = static_cast<InputScalar>(at(i0, j0, k1));
  const auto c010 = static_cast<InputScalar>(at(i0, j1, k0));
  const auto c011 = static_cast<InputScalar>(at(i0, j1, k1));
  const auto c100 = static_cast<InputScalar>(at(i1, j0, k0));
  const auto c101 = static_cast<InputScalar>(at(i1, j0, k1));
  const auto c110 = static_cast<InputScalar>(at(i1, j1, k0));
  const auto c111 = static_cast<InputScalar>(at(i1, j1, k1));

  // Trilinear interpolation for value using InputScalar precision
  const InputScalar c00 = c000 * (InputScalar{1} - fx) + c100 * fx;
  const InputScalar c01 = c001 * (InputScalar{1} - fx) + c101 * fx;
  const InputScalar c10 = c010 * (InputScalar{1} - fx) + c110 * fx;
  const InputScalar c11 = c011 * (InputScalar{1} - fx) + c111 * fx;

  const InputScalar c0 = c00 * (InputScalar{1} - fy) + c10 * fy;
  const InputScalar c1 = c01 * (InputScalar{1} - fy) + c11 * fy;

  InputScalar value = c0 * (InputScalar{1} - fz) + c1 * fz;

  // Calculate offset from original query point to clamped point
  const Eigen::Vector3<InputScalar> originalGridPos = worldToGrid<InputScalar>(position);
  if (clampedGridPos != originalGridPos) {
    const Eigen::Vector3<InputScalar> clampedWorldPos = gridToWorld<InputScalar>(clampedGridPos);
    value += (position - clampedWorldPos).norm();
  }

  return value;
}

template <typename ScalarType>
template <typename InputScalar>
Eigen::Vector3<InputScalar> SignedDistanceField<ScalarType>::gradient(
    const Eigen::Vector3<InputScalar>& position) const {
  return sampleWithGradient(position).second;
}

template <typename ScalarType>
template <typename InputScalar>
std::pair<InputScalar, Eigen::Vector3<InputScalar>>
SignedDistanceField<ScalarType>::sampleWithGradient(
    const Eigen::Vector3<InputScalar>& position) const {
  const Eigen::Vector3<InputScalar> gridPos = worldToGrid(position);

  // Clamp to valid grid bounds
  const Eigen::Vector3<InputScalar> clampedGridPos = clampToGrid(gridPos);

  // Get the integer grid coordinates of the lower corner
  const auto i0 = static_cast<Index>(std::floor(clampedGridPos.x()));
  const auto j0 = static_cast<Index>(std::floor(clampedGridPos.y()));
  const auto k0 = static_cast<Index>(std::floor(clampedGridPos.z()));

  // Get the integer grid coordinates of the upper corner
  const Index i1 = std::min(i0 + 1, resolution_.x() - 1);
  const Index j1 = std::min(j0 + 1, resolution_.y() - 1);
  const Index k1 = std::min(k0 + 1, resolution_.z() - 1);

  // Calculate interpolation weights using InputScalar precision
  const InputScalar fx =
      static_cast<InputScalar>(clampedGridPos.x()) - static_cast<InputScalar>(i0);
  const InputScalar fy =
      static_cast<InputScalar>(clampedGridPos.y()) - static_cast<InputScalar>(j0);
  const InputScalar fz =
      static_cast<InputScalar>(clampedGridPos.z()) - static_cast<InputScalar>(k0);

  // Get the 8 corner values and convert to InputScalar precision for interpolation
  const auto c000 = static_cast<InputScalar>(at(i0, j0, k0));
  const auto c001 = static_cast<InputScalar>(at(i0, j0, k1));
  const auto c010 = static_cast<InputScalar>(at(i0, j1, k0));
  const auto c011 = static_cast<InputScalar>(at(i0, j1, k1));
  const auto c100 = static_cast<InputScalar>(at(i1, j0, k0));
  const auto c101 = static_cast<InputScalar>(at(i1, j0, k1));
  const auto c110 = static_cast<InputScalar>(at(i1, j1, k0));
  const auto c111 = static_cast<InputScalar>(at(i1, j1, k1));

  // Trilinear interpolation for value using InputScalar precision
  const InputScalar c00 = c000 * (InputScalar{1} - fx) + c100 * fx;
  const InputScalar c01 = c001 * (InputScalar{1} - fx) + c101 * fx;
  const InputScalar c10 = c010 * (InputScalar{1} - fx) + c110 * fx;
  const InputScalar c11 = c011 * (InputScalar{1} - fx) + c111 * fx;

  const InputScalar c0 = c00 * (InputScalar{1} - fy) + c10 * fy;
  const InputScalar c1 = c01 * (InputScalar{1} - fy) + c11 * fy;

  InputScalar value = c0 * (InputScalar{1} - fz) + c1 * fz;

  // Calculate offset from original query point to clamped point
  const Eigen::Vector3<InputScalar> originalGridPos = worldToGrid(position);
  if (clampedGridPos != originalGridPos) {
    const Eigen::Vector3<InputScalar> clampedWorldPos = gridToWorld(clampedGridPos);
    const Eigen::Vector3<InputScalar> offsetVector = position - clampedWorldPos;
    const InputScalar offsetDistance = offsetVector.norm();

    // If point was outside bounds, add offset distance and use offset gradient
    if (offsetDistance > InputScalar{0}) {
      return {value + offsetDistance, -(offsetVector / offsetDistance)};
    }
  }

  // Standard analytical gradient computation for points inside the grid bounds using InputScalar
  // precision
  const InputScalar dfdx_term = (c100 - c000) * (InputScalar{1} - fy) * (InputScalar{1} - fz) +
      (c101 - c001) * (InputScalar{1} - fy) * fz + (c110 - c010) * fy * (InputScalar{1} - fz) +
      (c111 - c011) * fy * fz;

  const InputScalar dfdy_term = (c010 - c000) * (InputScalar{1} - fx) * (InputScalar{1} - fz) +
      (c011 - c001) * (InputScalar{1} - fx) * fz + (c110 - c100) * fx * (InputScalar{1} - fz) +
      (c111 - c101) * fx * fz;

  const InputScalar dfdz_term = (c001 - c000) * (InputScalar{1} - fx) * (InputScalar{1} - fy) +
      (c011 - c010) * (InputScalar{1} - fx) * fy + (c101 - c100) * fx * (InputScalar{1} - fy) +
      (c111 - c110) * fx * fy;

  // Transform grid space gradients to world space using InputScalar precision
  const Eigen::Vector3<InputScalar> gradient(
      dfdx_term / static_cast<InputScalar>(voxelSize_.x()),
      dfdy_term / static_cast<InputScalar>(voxelSize_.y()),
      dfdz_term / static_cast<InputScalar>(voxelSize_.z()));

  return std::make_pair(value, gradient);
}

template <typename ScalarType>
template <typename InputScalar>
Eigen::Vector3<InputScalar> SignedDistanceField<ScalarType>::worldToGrid(
    const Eigen::Vector3<InputScalar>& position) const {
  const auto localPos = position - bounds_.min().template cast<InputScalar>();
  return localPos.cwiseQuotient(voxelSize_.template cast<InputScalar>());
}

template <typename ScalarType>
template <typename InputScalar>
typename Eigen::Vector3<InputScalar> SignedDistanceField<ScalarType>::gridToWorld(
    const Eigen::Vector3<InputScalar>& gridPos) const {
  return bounds_.min().template cast<InputScalar>() +
      gridPos.cwiseProduct(voxelSize_.template cast<InputScalar>());
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
template <typename InputScalar>
Eigen::Vector3<InputScalar> SignedDistanceField<ScalarType>::clampToGrid(
    const Eigen::Vector3<InputScalar>& gridPos) const {
  return Eigen::Vector3<InputScalar>(
      std::clamp(gridPos.x(), InputScalar{0}, static_cast<InputScalar>(resolution_.x() - 1)),
      std::clamp(gridPos.y(), InputScalar{0}, static_cast<InputScalar>(resolution_.y() - 1)),
      std::clamp(gridPos.z(), InputScalar{0}, static_cast<InputScalar>(resolution_.z() - 1)));
}

// Explicit instantiation for classes
template class SignedDistanceField<float>;
template class SignedDistanceField<double>;

// Explicit instantiation for SignedDistanceField templated methods
template float SignedDistanceField<float>::sample<float>(const Eigen::Vector3<float>&) const;
template double SignedDistanceField<float>::sample<double>(const Eigen::Vector3<double>&) const;
template float SignedDistanceField<double>::sample<float>(const Eigen::Vector3<float>&) const;
template double SignedDistanceField<double>::sample<double>(const Eigen::Vector3<double>&) const;

template Eigen::Vector3<float> SignedDistanceField<float>::gradient<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<float>::gradient<double>(
    const Eigen::Vector3<double>&) const;
template Eigen::Vector3<float> SignedDistanceField<double>::gradient<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<double>::gradient<double>(
    const Eigen::Vector3<double>&) const;

template std::pair<float, Eigen::Vector3<float>>
SignedDistanceField<float>::sampleWithGradient<float>(const Eigen::Vector3<float>&) const;
template std::pair<double, Eigen::Vector3<double>>
SignedDistanceField<float>::sampleWithGradient<double>(const Eigen::Vector3<double>&) const;
template std::pair<float, Eigen::Vector3<float>>
SignedDistanceField<double>::sampleWithGradient<float>(const Eigen::Vector3<float>&) const;
template std::pair<double, Eigen::Vector3<double>>
SignedDistanceField<double>::sampleWithGradient<double>(const Eigen::Vector3<double>&) const;

template Eigen::Vector3<float> SignedDistanceField<float>::worldToGrid<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<float>::worldToGrid<double>(
    const Eigen::Vector3<double>&) const;
template Eigen::Vector3<float> SignedDistanceField<double>::worldToGrid<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<double>::worldToGrid<double>(
    const Eigen::Vector3<double>&) const;

template Eigen::Vector3<float> SignedDistanceField<float>::gridToWorld<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<float>::gridToWorld<double>(
    const Eigen::Vector3<double>&) const;
template Eigen::Vector3<float> SignedDistanceField<double>::gridToWorld<float>(
    const Eigen::Vector3<float>&) const;
template Eigen::Vector3<double> SignedDistanceField<double>::gridToWorld<double>(
    const Eigen::Vector3<double>&) const;

} // namespace axel
