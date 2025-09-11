/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/common/memory.h>
#include <momentum/math/constants.h>
#include <momentum/math/transform.h>
#include <momentum/math/utility.h>

namespace momentum {

/// Tapered capsule collision geometry for character collision detection.
///
/// Defined by a transformation, length, and two radii at the endpoints, creating
/// a capsule with potentially different radii at each end.
template <typename S>
struct TaperedCapsuleT {
  using Scalar = S;

  /// Transformation defining the orientation and starting point relative to the parent coordinate
  /// system.
  TransformT<S> transformation;

  /// Radii at the two endpoints of the capsule.
  Vector2<S> radius;

  /// Parent joint to which the geometry is attached.
  size_t parent;

  /// Length of the collision geometry along the x-axis.
  S length;

  TaperedCapsuleT()
      : transformation(TransformT<S>()),
        radius(Vector2<S>::Zero()),
        parent(kInvalidIndex),
        length(S(0)) {
    // Empty
  }

  /// Checks if the current capsule is approximately equal to another.
  [[nodiscard]] bool isApprox(const TaperedCapsuleT& other, const S& tol = Eps<S>(1e-4f, 1e-10))
      const {
    if (!transformation.isApprox(other.transformation, tol)) {
      return false;
    }

    if (!radius.isApprox(other.radius)) {
      return false;
    }

    if (parent != other.parent) {
      return false;
    }

    if (!::momentum::isApprox(length, other.length)) {
      return false;
    }

    return true;
  }
};

/// Collection of tapered capsules representing a character's collision geometry.
template <typename S>
using CollisionGeometryT = std::vector<TaperedCapsuleT<S>>;

using CollisionGeometry = CollisionGeometryT<float>;
using CollisionGeometryd = CollisionGeometryT<double>;

MOMENTUM_DEFINE_POINTERS(CollisionGeometry)
MOMENTUM_DEFINE_POINTERS(CollisionGeometryd)

} // namespace momentum
