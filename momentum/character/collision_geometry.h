/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/common/checks.h>
#include <momentum/common/memory.h>
#include <momentum/math/constants.h>
#include <momentum/math/transform.h>
#include <momentum/math/utility.h>

#include <cstdint>

namespace momentum {

/// Collision primitive kinds supported by character collision geometry.
enum class CollisionPrimitiveType : uint8_t {
  TaperedCapsule,
  Ellipsoid,
  Box,
};

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

/// Ellipsoid collision geometry for character collision detection.
///
/// Defined by a transformation and three local-space radii along the primitive's axes.
template <typename S>
struct CollisionEllipsoidT {
  using Scalar = S;

  /// Transformation defining the ellipsoid center and orientation relative to the parent.
  TransformT<S> transformation;

  /// Radii along the local x, y, and z axes.
  Vector3<S> radii;

  /// Parent joint to which the geometry is attached.
  size_t parent;

  CollisionEllipsoidT()
      : transformation(TransformT<S>()), radii(Vector3<S>::Zero()), parent(kInvalidIndex) {
    // Empty
  }

  /// Checks if the current ellipsoid is approximately equal to another.
  [[nodiscard]] bool isApprox(const CollisionEllipsoidT& other, const S& tol = Eps<S>(1e-4f, 1e-10))
      const {
    if (!transformation.isApprox(other.transformation, tol)) {
      return false;
    }

    if (!radii.isApprox(other.radii, tol)) {
      return false;
    }

    return parent == other.parent;
  }
};

/// Box collision geometry for character collision detection.
///
/// Defined by a transformation and three local-space half extents along the primitive's axes.
template <typename S>
struct CollisionBoxT {
  using Scalar = S;

  /// Transformation defining the box center and orientation relative to the parent.
  TransformT<S> transformation;

  /// Half extents along the local x, y, and z axes.
  Vector3<S> halfExtents;

  /// Parent joint to which the geometry is attached.
  size_t parent;

  CollisionBoxT()
      : transformation(TransformT<S>()), halfExtents(Vector3<S>::Zero()), parent(kInvalidIndex) {
    // Empty
  }

  /// Checks if the current box is approximately equal to another.
  [[nodiscard]] bool isApprox(const CollisionBoxT& other, const S& tol = Eps<S>(1e-4f, 1e-10))
      const {
    if (!transformation.isApprox(other.transformation, tol)) {
      return false;
    }

    if (!halfExtents.isApprox(other.halfExtents, tol)) {
      return false;
    }

    return parent == other.parent;
  }
};

/// Collision geometry primitive.
///
/// The tapered capsule fields are kept directly on the primitive so existing code that
/// constructs capsule-only collision geometry can continue to initialize `radius`, `length`,
/// `parent`, and `transformation` on elements of `CollisionGeometry`.
template <typename S>
struct CollisionPrimitiveT {
  using Scalar = S;

  /// Primitive kind.
  CollisionPrimitiveType type;

  /// Transformation defining the primitive pose relative to the parent coordinate system.
  TransformT<S> transformation;

  /// Radii at the two endpoints of a tapered capsule.
  Vector2<S> radius;

  /// Radii along the local axes of an ellipsoid.
  Vector3<S> ellipsoidRadii;

  /// Half extents along the local axes of a box.
  Vector3<S> boxHalfExtents;

  /// Parent joint to which the geometry is attached.
  size_t parent;

  /// Length of a tapered capsule along the x-axis.
  S length;

  CollisionPrimitiveT()
      : type(CollisionPrimitiveType::TaperedCapsule),
        transformation(TransformT<S>()),
        radius(Vector2<S>::Zero()),
        ellipsoidRadii(Vector3<S>::Zero()),
        boxHalfExtents(Vector3<S>::Zero()),
        parent(kInvalidIndex),
        length(S(0)) {
    // Empty
  }

  /// Creates a collision primitive from a tapered capsule.
  /* implicit */ CollisionPrimitiveT(const TaperedCapsuleT<S>& capsule)
      : type(CollisionPrimitiveType::TaperedCapsule),
        transformation(capsule.transformation),
        radius(capsule.radius),
        ellipsoidRadii(Vector3<S>::Zero()),
        boxHalfExtents(Vector3<S>::Zero()),
        parent(capsule.parent),
        length(capsule.length) {}

  /// Creates a collision primitive from an ellipsoid.
  /* implicit */ CollisionPrimitiveT(const CollisionEllipsoidT<S>& ellipsoid)
      : type(CollisionPrimitiveType::Ellipsoid),
        transformation(ellipsoid.transformation),
        radius(Vector2<S>::Zero()),
        ellipsoidRadii(ellipsoid.radii),
        boxHalfExtents(Vector3<S>::Zero()),
        parent(ellipsoid.parent),
        length(S(0)) {}

  /// Creates a collision primitive from a box.
  /* implicit */ CollisionPrimitiveT(const CollisionBoxT<S>& box)
      : type(CollisionPrimitiveType::Box),
        transformation(box.transformation),
        radius(Vector2<S>::Zero()),
        ellipsoidRadii(Vector3<S>::Zero()),
        boxHalfExtents(box.halfExtents),
        parent(box.parent),
        length(S(0)) {}

  /// Assigns tapered capsule data to this primitive.
  CollisionPrimitiveT& operator=(const TaperedCapsuleT<S>& capsule) {
    type = CollisionPrimitiveType::TaperedCapsule;
    transformation = capsule.transformation;
    radius = capsule.radius;
    ellipsoidRadii.setZero();
    boxHalfExtents.setZero();
    parent = capsule.parent;
    length = capsule.length;
    return *this;
  }

  /// Assigns ellipsoid data to this primitive.
  CollisionPrimitiveT& operator=(const CollisionEllipsoidT<S>& ellipsoid) {
    type = CollisionPrimitiveType::Ellipsoid;
    transformation = ellipsoid.transformation;
    radius.setZero();
    ellipsoidRadii = ellipsoid.radii;
    boxHalfExtents.setZero();
    parent = ellipsoid.parent;
    length = S(0);
    return *this;
  }

  /// Assigns box data to this primitive.
  CollisionPrimitiveT& operator=(const CollisionBoxT<S>& box) {
    type = CollisionPrimitiveType::Box;
    transformation = box.transformation;
    radius.setZero();
    ellipsoidRadii.setZero();
    boxHalfExtents = box.halfExtents;
    parent = box.parent;
    length = S(0);
    return *this;
  }

  /// Returns true when this primitive stores tapered capsule data.
  [[nodiscard]] bool isTaperedCapsule() const {
    return type == CollisionPrimitiveType::TaperedCapsule;
  }

  /// Returns true when this primitive stores ellipsoid data.
  [[nodiscard]] bool isEllipsoid() const {
    return type == CollisionPrimitiveType::Ellipsoid;
  }

  /// Returns true when this primitive stores box data.
  [[nodiscard]] bool isBox() const {
    return type == CollisionPrimitiveType::Box;
  }

  /// Converts this primitive to a tapered capsule.
  ///
  /// Requires `isTaperedCapsule()` so unhandled primitive types fail at the call site
  /// instead of returning unrelated default-initialized fields.
  [[nodiscard]] TaperedCapsuleT<S> toTaperedCapsule() const {
    MT_CHECK(
        isTaperedCapsule(),
        "Collision primitive type {} does not store tapered capsule data",
        static_cast<int>(type));

    TaperedCapsuleT<S> capsule;
    capsule.transformation = transformation;
    capsule.radius = radius;
    capsule.parent = parent;
    capsule.length = length;
    return capsule;
  }

  /// Converts this primitive to an ellipsoid.
  ///
  /// Requires `isEllipsoid()` so unhandled primitive types fail at the call site
  /// instead of returning unrelated default-initialized fields.
  [[nodiscard]] CollisionEllipsoidT<S> toEllipsoid() const {
    MT_CHECK(
        isEllipsoid(),
        "Collision primitive type {} does not store ellipsoid data",
        static_cast<int>(type));

    CollisionEllipsoidT<S> ellipsoid;
    ellipsoid.transformation = transformation;
    ellipsoid.radii = ellipsoidRadii;
    ellipsoid.parent = parent;
    return ellipsoid;
  }

  /// Converts this primitive to a box.
  ///
  /// Requires `isBox()` so unhandled primitive types fail at the call site
  /// instead of returning unrelated stale data.
  [[nodiscard]] CollisionBoxT<S> toBox() const {
    MT_CHECK(
        isBox(), "Collision primitive type {} does not store box data", static_cast<int>(type));
    CollisionBoxT<S> box;
    box.transformation = transformation;
    box.halfExtents = boxHalfExtents;
    box.parent = parent;
    return box;
  }

  /// Checks if the current primitive is approximately equal to another.
  [[nodiscard]] bool isApprox(const CollisionPrimitiveT& other, const S& tol = Eps<S>(1e-4f, 1e-10))
      const {
    if (type != other.type || parent != other.parent) {
      return false;
    }

    if (!transformation.isApprox(other.transformation, tol)) {
      return false;
    }

    if (type == CollisionPrimitiveType::TaperedCapsule) {
      return radius.isApprox(other.radius, tol) && ::momentum::isApprox(length, other.length, tol);
    }

    if (type == CollisionPrimitiveType::Ellipsoid) {
      return ellipsoidRadii.isApprox(other.ellipsoidRadii, tol);
    }

    if (type == CollisionPrimitiveType::Box) {
      return boxHalfExtents.isApprox(other.boxHalfExtents, tol);
    }

    return false;
  }
};

/// Collection of primitives representing a character's collision geometry.
template <typename S>
using CollisionGeometryT = std::vector<CollisionPrimitiveT<S>>;

using CollisionGeometry = CollisionGeometryT<float>;
using CollisionGeometryd = CollisionGeometryT<double>;

MOMENTUM_DEFINE_POINTERS(CollisionGeometry)
MOMENTUM_DEFINE_POINTERS(CollisionGeometryd)

} // namespace momentum
