/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/BoundingBox.h"

#include <algorithm>

namespace axel {

template <typename ScalarType>
BoundingBox<ScalarType>::BoundingBox(
    const Eigen::Vector3<ScalarType>& min,
    const Eigen::Vector3<ScalarType>& max,
    const Index id)
    : aabb{Eigen::AlignedBox<Scalar, 3>{min, max}}, id{id} {}

template <typename ScalarType>
BoundingBox<ScalarType>::BoundingBox(
    const Eigen::Vector3<ScalarType>& p,
    const ScalarType thickness) {
  const Eigen::Vector3<ScalarType> offset{thickness, thickness, thickness};
  aabb = Eigen::AlignedBox<Scalar, 3>{p - offset, p + offset};
}

template <typename ScalarType>
const Eigen::Vector3<ScalarType>& BoundingBox<ScalarType>::min() const {
  return aabb.min();
}

template <typename ScalarType>
const Eigen::Vector3<ScalarType>& BoundingBox<ScalarType>::max() const {
  return aabb.max();
}

template <typename ScalarType>
Eigen::Vector3<ScalarType> BoundingBox<ScalarType>::center() const {
  return aabb.center();
}

template <typename ScalarType>
ScalarType BoundingBox<ScalarType>::squaredVolume() const {
  return aabb.diagonal().squaredNorm();
}

// Reference:
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
template <typename ScalarType>
bool BoundingBox<ScalarType>::intersects(
    const Eigen::Vector3<ScalarType>& origin,
    const Eigen::Vector3<ScalarType>& direction) const {
  const Eigen::Vector3<ScalarType>& min = aabb.min();
  const Eigen::Vector3<ScalarType>& max = aabb.max();

  ScalarType tmin = (min.x() - origin.x()) / direction.x();
  ScalarType tmax = (max.x() - origin.x()) / direction.x();

  if (tmin > tmax) {
    std::swap(tmin, tmax);
  }

  ScalarType tymin = (min.y() - origin.y()) / direction.y();
  ScalarType tymax = (max.y() - origin.y()) / direction.y();

  if (tymin > tymax) {
    std::swap(tymin, tymax);
  }

  if ((tmin > tymax) || (tymin > tmax)) {
    return false;
  }

  if (tymin > tmin) {
    tmin = tymin;
  }

  if (tymax < tmax) {
    tmax = tymax;
  }

  ScalarType tzmin = (min.z() - origin.z()) / direction.z();
  ScalarType tzmax = (max.z() - origin.z()) / direction.z();

  if (tzmin > tzmax) {
    std::swap(tzmin, tzmax);
  }

  if ((tmin > tzmax) || (tzmin > tmax)) {
    return false;
  }

  if (tzmin > tmin) {
    tmin = tzmin;
  }

  if (tzmax < tmax) {
    tmax = tzmax;
  }

  if (tmin < ScalarType{0} && tmax < ScalarType{0}) {
    return false;
  }
  return true;
}

// https://tavianator.com/2022/ray_box_boundary.html
template <typename ScalarType>
bool BoundingBox<ScalarType>::intersectsBranchless(
    const Eigen::Vector3<ScalarType>& origin,
    const Eigen::Vector3<ScalarType>& dirInv) const {
  const Eigen::Vector3<ScalarType>& bmin = aabb.min();
  const Eigen::Vector3<ScalarType>& bmax = aabb.max();

  ScalarType tmin = ScalarType{0};
  ScalarType tmax = std::numeric_limits<ScalarType>::max();
  for (int32_t i = 0; i < 3; ++i) {
    const ScalarType t1 = (bmin[i] - origin[i]) * dirInv[i];
    const ScalarType t2 = (bmax[i] - origin[i]) * dirInv[i];

    // These mins and maxs rely on standard NaN comparison to work for all edge-cases.
    tmin = std::min(std::max(tmin, t1), std::max(tmin, t2));
    tmax = std::max(std::min(tmax, t1), std::min(tmax, t2));
  }

  return tmin <= tmax;
}

template <typename ScalarType>
bool BoundingBox<ScalarType>::contains(const Eigen::Vector3<Scalar>& point) const {
  return aabb.contains(point);
}

template <typename ScalarType>
bool BoundingBox<ScalarType>::contains(const BoundingBox& other) const {
  return aabb.contains(other.aabb);
}

template <typename ScalarType>
void BoundingBox<ScalarType>::extend(const Eigen::Vector3<ScalarType>& p) {
  aabb.extend(p);
}

template <typename ScalarType>
void BoundingBox<ScalarType>::extend(const BoundingBox& b) {
  aabb.extend(b.aabb);
}

template <typename ScalarType>
Index BoundingBox<ScalarType>::maxDimension() const {
  const Eigen::Vector3<ScalarType> sizes = aabb.sizes();
  return sizes.x() > sizes.y() ? (sizes.x() > sizes.z() ? 0 : 2) : (sizes.y() > sizes.z() ? 1 : 2);
}

template <typename ScalarType>
bool BoundingBox<ScalarType>::intersects(const BoundingBox& box) const {
  return aabb.intersects(box.aabb);
}

template struct BoundingBox<float>;
template struct BoundingBox<double>;

} // namespace axel
