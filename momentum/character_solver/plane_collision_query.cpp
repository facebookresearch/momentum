/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/plane_collision_query.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <cmath>

namespace momentum {

namespace {

const CollisionGeometry& checkedCollisionGeometry(const Character& character) {
  MT_THROW_IF(
      !character.collision,
      "Attempting to create plane collision query with a character that has no collision geometries");
  return *character.collision;
}

} // namespace

template <typename T>
PlaneCollisionQueryT<T>::PlaneCollisionQueryT(
    const CollisionGeometry& cg,
    const Vector3<T>& planeNormal,
    T planeOffset)
    : collisionGeometry_(cg), planeNormal_(Vector3<T>::Zero()), planeOffset_(T(0)) {
  setPlane(planeNormal, planeOffset);
  updatePrimitiveList();
}

template <typename T>
PlaneCollisionQueryT<T>::PlaneCollisionQueryT(
    const Character& character,
    const Vector3<T>& planeNormal,
    T planeOffset)
    : PlaneCollisionQueryT(checkedCollisionGeometry(character), planeNormal, planeOffset) {
  const SkeletonStateT<T> state(
      character.parameterTransform.zero().template cast<T>(), character.skeleton);
  update(state, T(0));
}

template <typename T>
void PlaneCollisionQueryT<T>::setPlane(const Vector3<T>& planeNormal, T planeOffset) {
  MT_CHECK(planeNormal.allFinite(), "Plane collision normal must contain only finite values");
  MT_CHECK(std::isfinite(planeOffset), "Plane collision offset must be finite");
  const T norm = planeNormal.norm();
  MT_CHECK(
      norm > Eps<T>(1e-8f, 1e-17), "Plane collision normal must be non-zero, got norm {}", norm);
  planeNormal_ = planeNormal / norm;
  planeOffset_ = planeOffset / norm;
}

template <typename T>
T PlaneCollisionQueryT<T>::primitiveParentSpaceBound(const size_t primitiveIndex) const {
  const auto& primitive = collisionGeometry_[primitiveIndex];
  const TransformT<T> primitiveTransform = primitive.transformation.cast<T>();
  const Vector3<T>& origin = primitiveTransform.translation;

  if (primitive.type == CollisionPrimitiveType::TaperedCapsule) {
    const Vector3<T> endpoint =
        origin + primitiveTransform.getAxisX() * primitiveTransform.scale * T(primitive.length);
    const T radius = primitive.radius.cast<T>().cwiseAbs().maxCoeff();
    return std::max(origin.norm(), endpoint.norm()) + radius;
  }

  if (primitive.type == CollisionPrimitiveType::Ellipsoid) {
    return origin.norm() + primitive.ellipsoidRadii.cast<T>().cwiseAbs().maxCoeff();
  }

  if (primitive.type == CollisionPrimitiveType::Box) {
    return origin.norm() + primitive.boxHalfExtents.cast<T>().cwiseAbs().norm();
  }

  MT_THROW(
      "PlaneCollisionQuery: unsupported collision primitive type {}",
      static_cast<int>(primitive.type));
}

template <typename T>
void PlaneCollisionQueryT<T>::updatePrimitiveList() {
  primitiveIndices_.clear();
  primitiveIndices_.reserve(collisionGeometry_.size());
  primitiveParentSpaceBounds_.assign(collisionGeometry_.size(), T(0));
  for (size_t i = 0; i < collisionGeometry_.size(); ++i) {
    if (collisionGeometry_[i].parent != kInvalidIndex) {
      primitiveIndices_.push_back(i);
      primitiveParentSpaceBounds_[i] = primitiveParentSpaceBound(i);
    }
  }
  candidatePrimitiveIndices_.clear();
  activeParentCollisions_.clear();
}

template <typename T>
bool PlaneCollisionQueryT<T>::mayPrimitiveContactPlane(
    const SkeletonStateT<T>& state,
    const size_t primitiveIndex,
    const T contactMargin) const {
  const auto& primitive = collisionGeometry_[primitiveIndex];
  const TransformT<T>& parentTransform = state.jointState[primitive.parent].transform;
  const T parentSignedDistance = planeNormal_.dot(parentTransform.translation) - planeOffset_;
  const T conservativeRadius =
      std::abs(parentTransform.scale) * primitiveParentSpaceBounds_[primitiveIndex];
  return parentSignedDistance - conservativeRadius <= contactMargin;
}

template <typename T>
void PlaneCollisionQueryT<T>::validateContactMargin(const T contactMargin) const {
  MT_THROW_IF(
      !std::isfinite(contactMargin) || contactMargin < T(0),
      "Plane collision contact margin must be finite and non-negative, got {}",
      contactMargin);
}

template <typename T>
void PlaneCollisionQueryT<T>::validateJointStateParents(
    const std::span<const JointStateT<T>> states) const {
  for (const auto primitiveIndex : primitiveIndices_) {
    const size_t parent = collisionGeometry_[primitiveIndex].parent;
    MT_THROW_IF(
        parent >= states.size(),
        "Plane collision primitive {} parent joint {} is outside the skeleton state",
        primitiveIndex,
        parent);
  }
}

template <typename T>
void PlaneCollisionQueryT<T>::update(const SkeletonStateT<T>& state, const T contactMargin) {
  validateContactMargin(contactMargin);

  MT_PROFILE_EVENT("PlaneCollision: updateCandidates");
  collisionState_.resize(collisionGeometry_.size());
  candidatePrimitiveIndices_.clear();
  candidatePrimitiveIndices_.reserve(primitiveIndices_.size());

  for (const auto primitiveIndex : primitiveIndices_) {
    if (!mayPrimitiveContactPlane(state, primitiveIndex, contactMargin)) {
      continue;
    }
    collisionState_.updatePrimitive(state, collisionGeometry_, primitiveIndex);
    candidatePrimitiveIndices_.push_back(primitiveIndex);
  }
  activeParentCollisions_.clear();
}

template <typename T>
void PlaneCollisionQueryT<T>::update(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  validateJointStateParents(states);
  SkeletonStateT<T> state;
  state.jointState.assign(states.begin(), states.end());
  update(state, contactMargin);
}

template <typename T>
bool PlaneCollisionQueryT<T>::checkCollision(size_t primitiveIndex, Result& result) const {
  const auto type = collisionState_.type[primitiveIndex];
  result.primitiveIndex = primitiveIndex;
  result.parentJoint = collisionGeometry_[primitiveIndex].parent;

  // `result.position` is the deepest surface point toward the plane (the contact point), offset
  // from the primitive center/endpoint by the support radius along `-planeNormal_`. Differentiating
  // at the contact point, rather than the center, lets the joint-chain walk capture orientation and
  // scale dependence through that world-space support point.
  if (type == CollisionPrimitiveType::TaperedCapsule) {
    const Vector3<T> position0 = collisionState_.origin[primitiveIndex];
    const Vector3<T> position1 =
        collisionState_.origin[primitiveIndex] + collisionState_.direction[primitiveIndex];
    const T signedDistance0 = planeNormal_.dot(position0) - planeOffset_;
    const T signedDistance1 = planeNormal_.dot(position1) - planeOffset_;
    const T radius0 = collisionState_.radius[primitiveIndex][0];
    const T radius1 = collisionState_.radius[primitiveIndex][1];
    const T surfaceDistance0 = signedDistance0 - radius0;
    const T surfaceDistance1 = signedDistance1 - radius1;

    if (surfaceDistance0 <= surfaceDistance1) {
      result.signedDistance = signedDistance0;
      result.radius = radius0;
      result.position = position0 - radius0 * planeNormal_;
    } else {
      result.signedDistance = signedDistance1;
      result.radius = radius1;
      result.position = position1 - radius1 * planeNormal_;
    }
  } else if (isCenteredPrimitive(type)) {
    const Vector3<T>& center = collisionState_.origin[primitiveIndex];
    const Quaternion<T>& orientation = collisionState_.orientation[primitiveIndex];
    result.signedDistance = planeNormal_.dot(center) - planeOffset_;
    result.radius = centeredPrimitiveRadiusAlongDirection(
        type,
        orientation,
        collisionState_.ellipsoidRadii[primitiveIndex],
        collisionState_.boxHalfExtents[primitiveIndex],
        planeNormal_);

    const Vector3<T> localNormal = orientation.inverse() * planeNormal_;
    Vector3<T> supportOffset;
    if (type == CollisionPrimitiveType::Ellipsoid) {
      const Vector3<T>& radii = collisionState_.ellipsoidRadii[primitiveIndex];
      const T denom = result.radius > Eps<T>(1e-8f, 1e-16) ? result.radius : T(1);
      supportOffset = orientation * (radii.cwiseProduct(radii).cwiseProduct(localNormal)) / denom;
    } else {
      const Vector3<T>& halfExtents = collisionState_.boxHalfExtents[primitiveIndex];
      const Vector3<T> localSign =
          localNormal.unaryExpr([](const T value) { return value >= T(0) ? T(1) : T(-1); });
      supportOffset = orientation * halfExtents.cwiseAbs().cwiseProduct(localSign);
    }
    result.position = center - supportOffset;
  } else {
    MT_THROW(
        "PlaneCollisionQuery: unsupported collision primitive type {}", static_cast<int>(type));
  }

  result.overlap = result.radius - result.signedDistance;
  return result.overlap > T(0);
}

template <typename T>
void PlaneCollisionQueryT<T>::updateActiveParentCollisions() {
  activeParentCollisions_.clear();
  activeParentCollisions_.reserve(candidatePrimitiveIndices_.size());

  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    Result result;
    if (!checkCollision(primitiveIndex, result)) {
      continue;
    }

    auto existing = std::ranges::find_if(activeParentCollisions_, [&](const Result& collision) {
      return collision.parentJoint == result.parentJoint;
    });
    if (existing == activeParentCollisions_.end()) {
      activeParentCollisions_.push_back(result);
      continue;
    }

    if (result.overlap > existing->overlap) {
      *existing = result;
    }
  }
}

template <typename T>
std::vector<size_t> PlaneCollisionQueryT<T>::getCollidingPrimitives() const {
  std::vector<size_t> collidingPrimitives;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    Result result;
    if (checkCollision(primitiveIndex, result)) {
      collidingPrimitives.push_back(primitiveIndex);
    }
  }
  return collidingPrimitives;
}

template <typename T>
std::vector<PlaneCollisionDebugEntryT<T>> PlaneCollisionQueryT<T>::getCollisionDebugInfo() const {
  std::vector<PlaneCollisionDebugEntryT<T>> entries;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    Result result;
    if (checkCollision(primitiveIndex, result)) {
      entries.push_back(
          {result.primitiveIndex,
           result.parentJoint,
           result.overlap,
           result.signedDistance,
           result.radius,
           result.position});
    }
  }
  return entries;
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionQueryT<T>::getContactPoints(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  update(states, contactMargin);

  std::vector<Vector3<T>> points;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    Result result;
    static_cast<void>(checkCollision(primitiveIndex, result));
    if (result.surfaceDistance() <= contactMargin) {
      points.push_back(result.position);
    }
  }
  return points;
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionQueryT<T>::getContactPointsByParent(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  const std::vector<PlaneCollisionContactPointT<T>> contacts =
      getContactPointsByParentWithDetails(states, contactMargin);
  std::vector<Vector3<T>> points;
  points.reserve(contacts.size());
  for (const auto& contact : contacts) {
    points.push_back(contact.position);
  }
  return points;
}

template <typename T>
std::vector<PlaneCollisionContactPointT<T>>
PlaneCollisionQueryT<T>::getContactPointsByParentWithDetails(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  update(states, contactMargin);

  std::vector<PlaneCollisionContactPointT<T>> parentContacts;
  parentContacts.reserve(candidatePrimitiveIndices_.size());
  parentContactIndices_.assign(states.size(), kInvalidIndex);
  parentSurfaceDistances_.assign(states.size(), T(0));
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    Result result;
    static_cast<void>(checkCollision(primitiveIndex, result));
    const T surfaceDistance = result.surfaceDistance();
    if (surfaceDistance > contactMargin) {
      continue;
    }

    const size_t parent = result.parentJoint;
    const size_t contactIndex = parentContactIndices_[parent];
    if (contactIndex == kInvalidIndex) {
      parentContactIndices_[parent] = parentContacts.size();
      parentSurfaceDistances_[parent] = surfaceDistance;
      parentContacts.push_back({parent, surfaceDistance, result.position});
      continue;
    }

    if (surfaceDistance < parentSurfaceDistances_[parent]) {
      parentSurfaceDistances_[parent] = surfaceDistance;
      parentContacts[contactIndex] = {parent, surfaceDistance, result.position};
    }
  }
  return parentContacts;
}

template class PlaneCollisionQueryT<float>;
template class PlaneCollisionQueryT<double>;

} // namespace momentum
