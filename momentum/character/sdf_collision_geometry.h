/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/common/memory.h>
#include <momentum/math/constants.h>
#include <momentum/math/transform.h>
#include <momentum/math/utility.h>

#include <axel/SignedDistanceField.h>

#include <memory>

namespace momentum {

/// SDF collision geometry for character collision detection.
///
/// Defined by a transformation, parent joint, and a shared pointer to a SignedDistanceField.
/// The SDF data is shared to avoid expensive copies of large SDF volumes.
template <typename S>
struct SDFColliderT {
  using Scalar = S;
  using SDFType = axel::SignedDistanceField<S>;

  /// Transformation defining the orientation and position relative to the parent coordinate
  /// system.
  TransformT<S> transformation;

  /// Parent joint to which the geometry is attached.  Use kInvalidIndex to indicate it is
  /// fixed in world space.
  size_t parent = kInvalidIndex;

  /// Shared pointer to the SignedDistanceField data.
  /// Using shared_ptr to avoid expensive copies of potentially large SDF volumes.
  std::shared_ptr<const SDFType> sdf;

  SDFColliderT() : transformation(TransformT<S>()), parent(kInvalidIndex), sdf(nullptr) {
    // Empty
  }

  /// Constructor with SDF initialization.
  SDFColliderT(
      const TransformT<S>& transform,
      size_t parentJoint,
      std::shared_ptr<const SDFType> sdfPtr)
      : transformation(transform), parent(parentJoint), sdf(std::move(sdfPtr)) {
    // Empty
  }

  /// Checks if the current SDF collider is approximately equal to another.
  [[nodiscard]] bool isApprox(const SDFColliderT& other, const S& tol = Eps<S>(1e-4f, 1e-10))
      const {
    if (!transformation.isApprox(other.transformation, tol)) {
      return false;
    }

    if (parent != other.parent) {
      return false;
    }

    // Compare SDF pointers - they should point to the same SDF instance
    if (sdf != other.sdf) {
      return false;
    }

    return true;
  }

  /// Checks if the SDF collider is valid (has a valid SDF pointer).
  [[nodiscard]] bool isValid() const {
    return sdf != nullptr;
  }
};

/// Collection of SDF colliders representing a character's collision geometry.
template <typename S>
using SDFCollisionGeometryT = std::vector<SDFColliderT<S>>;

using SDFCollisionGeometry = SDFCollisionGeometryT<float>;
using SDFCollisionGeometryd = SDFCollisionGeometryT<double>;

MOMENTUM_DEFINE_POINTERS(SDFCollisionGeometry)
MOMENTUM_DEFINE_POINTERS(SDFCollisionGeometryd)

} // namespace momentum
