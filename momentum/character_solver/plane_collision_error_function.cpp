/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/plane_collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace momentum {

template <typename T>
PlaneCollisionErrorFunctionT<T>::PlaneCollisionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& cg,
    const Vector3<T>& planeNormal,
    T planeOffset)
    : SkeletonErrorFunctionT<T>(skel, pt),
      collisionGeometry_(cg),
      planeNormal_(Vector3<T>::UnitY()),
      planeOffset_(T(0)) {
  setPlane(planeNormal, planeOffset);
  updatePrimitiveList();

  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  updateCandidateCollisionState(state, T(0));
}

template <typename T>
PlaneCollisionErrorFunctionT<T>::PlaneCollisionErrorFunctionT(
    const Character& character,
    const Vector3<T>& planeNormal,
    T planeOffset)
    : PlaneCollisionErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create plane collision error function with a character that "
                    "has no collision geometries"),
          planeNormal,
          planeOffset) {
  // Do nothing
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::setPlane(const Vector3<T>& planeNormal, T planeOffset) {
  const T norm = planeNormal.norm();
  MT_CHECK(
      norm > Eps<T>(1e-8f, 1e-17), "Plane collision normal must be non-zero, got norm {}", norm);
  planeNormal_ = planeNormal / norm;
  planeOffset_ = planeOffset;
}

template <typename T>
T PlaneCollisionErrorFunctionT<T>::primitiveParentSpaceBound(const size_t primitiveIndex) const {
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
      "PlaneCollisionErrorFunction: unsupported collision primitive type {}",
      static_cast<int>(primitive.type));
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::updatePrimitiveList() {
  primitiveIndices_.clear();
  primitiveIndices_.reserve(collisionGeometry_.size());
  primitiveParentSpaceBounds_.assign(collisionGeometry_.size(), T(0));
  for (size_t i = 0; i < collisionGeometry_.size(); ++i) {
    // This error function constrains character-attached primitives against a world plane.
    if (collisionGeometry_[i].parent != kInvalidIndex) {
      primitiveIndices_.push_back(i);
      primitiveParentSpaceBounds_[i] = primitiveParentSpaceBound(i);
    }
  }
  candidatePrimitiveIndices_.clear();
  activeParentCollisions_.clear();
}

template <typename T>
bool PlaneCollisionErrorFunctionT<T>::mayPrimitiveContactPlane(
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
void PlaneCollisionErrorFunctionT<T>::updateCandidateCollisionState(
    const SkeletonStateT<T>& state,
    const T contactMargin) {
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
bool PlaneCollisionErrorFunctionT<T>::checkCollision(
    size_t primitiveIndex,
    PlaneCollisionResult& result) const {
  const auto type = collisionState_.type[primitiveIndex];

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
        "PlaneCollisionErrorFunction: unsupported collision primitive type {}",
        static_cast<int>(type));
  }

  result.overlap = result.radius - result.signedDistance;
  return result.overlap > T(0);
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::updateActiveParentCollisions() {
  activeParentCollisions_.clear();
  activeParentCollisions_.reserve(candidatePrimitiveIndices_.size());

  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    PlaneCollisionResult result;
    if (!checkCollision(primitiveIndex, result)) {
      continue;
    }

    const size_t parent = collisionGeometry_[primitiveIndex].parent;
    auto existing =
        std::ranges::find_if(activeParentCollisions_, [&](const ActivePlaneCollision& collision) {
          return collisionGeometry_[collision.primitiveIndex].parent == parent;
        });
    if (existing == activeParentCollisions_.end()) {
      activeParentCollisions_.push_back({primitiveIndex, result});
      continue;
    }

    if (result.overlap > existing->result.overlap) {
      *existing = {primitiveIndex, result};
    }
  }
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::accumulateGradientAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T weight,
    size_t startJoint,
    Ref<VectorX<T>> gradient,
    T scaleCorrection) const {
  accumulateChainDerivatives<T>(
      state,
      this->skeleton_,
      this->activeJointParams_,
      position,
      direction,
      weight,
      startJoint,
      kInvalidIndex,
      scaleCorrection,
      [&](T value, size_t jointParamIndex) {
        gradient_jointParams_to_modelParams(
            value, jointParamIndex, this->parameterTransform_, gradient);
      });
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::accumulateJacobianAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T factor,
    size_t startJoint,
    Ref<MatrixX<T>> jacobian,
    int row,
    T scaleCorrection) const {
  accumulateChainDerivatives<T>(
      state,
      this->skeleton_,
      this->activeJointParams_,
      position,
      direction,
      factor,
      startJoint,
      kInvalidIndex,
      scaleCorrection,
      [&](T value, size_t jointParamIndex) {
        jacobian_jointParams_to_modelParams(
            value, jointParamIndex, row, this->parameterTransform_, jacobian);
      });
}

template <typename T>
std::vector<size_t> PlaneCollisionErrorFunctionT<T>::getCollidingPrimitives() const {
  std::vector<size_t> collidingPrimitives;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    PlaneCollisionResult result;
    if (checkCollision(primitiveIndex, result)) {
      collidingPrimitives.push_back(primitiveIndex);
    }
  }
  return collidingPrimitives;
}

template <typename T>
std::vector<PlaneCollisionDebugEntryT<T>> PlaneCollisionErrorFunctionT<T>::getCollisionDebugInfo()
    const {
  std::vector<PlaneCollisionDebugEntryT<T>> entries;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    PlaneCollisionResult result;
    if (checkCollision(primitiveIndex, result)) {
      entries.push_back(
          {primitiveIndex,
           collisionGeometry_[primitiveIndex].parent,
           result.overlap,
           result.signedDistance,
           result.radius,
           result.position});
    }
  }
  return entries;
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::prepareContactPointQuery(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  MT_THROW_IF(
      !std::isfinite(contactMargin) || contactMargin < T(0),
      "Plane collision contact margin must be finite and non-negative, got {}",
      contactMargin);
  for (const auto primitiveIndex : primitiveIndices_) {
    const size_t parent = collisionGeometry_[primitiveIndex].parent;
    MT_THROW_IF(
        parent >= states.size(),
        "Plane collision primitive {} parent joint {} is outside the skeleton state",
        primitiveIndex,
        parent);
  }

  SkeletonStateT<T> state;
  state.jointState.assign(states.begin(), states.end());
  updateCandidateCollisionState(state, contactMargin);
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionErrorFunctionT<T>::getContactPoints(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  prepareContactPointQuery(states, contactMargin);

  std::vector<Vector3<T>> points;
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    PlaneCollisionResult result;
    static_cast<void>(checkCollision(primitiveIndex, result));
    const T surfaceDistance = result.signedDistance - result.radius;
    if (surfaceDistance <= contactMargin) {
      points.push_back(result.position);
    }
  }
  return points;
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionErrorFunctionT<T>::getContactPointsByParent(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  prepareContactPointQuery(states, contactMargin);

  struct ParentContact {
    size_t parent = kInvalidIndex;
    T surfaceDistance = T(0);
    Vector3<T> position = Vector3<T>::Zero();
  };

  std::vector<ParentContact> parentContacts;
  parentContacts.reserve(candidatePrimitiveIndices_.size());
  for (const auto primitiveIndex : candidatePrimitiveIndices_) {
    PlaneCollisionResult result;
    static_cast<void>(checkCollision(primitiveIndex, result));
    const T surfaceDistance = result.signedDistance - result.radius;
    if (surfaceDistance > contactMargin) {
      continue;
    }

    const size_t parent = collisionGeometry_[primitiveIndex].parent;
    auto existing = std::ranges::find_if(parentContacts, [parent](const ParentContact& contact) {
      return contact.parent == parent;
    });
    if (existing == parentContacts.end()) {
      parentContacts.push_back({parent, surfaceDistance, result.position});
      continue;
    }

    if (surfaceDistance < existing->surfaceDistance) {
      *existing = {parent, surfaceDistance, result.position};
    }
  }

  std::vector<Vector3<T>> points;
  points.reserve(parentContacts.size());
  for (const ParentContact& contact : parentContacts) {
    points.push_back(contact.position);
  }
  return points;
}

template <typename T>
double PlaneCollisionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  updateCandidateCollisionState(state, T(0));
  updateActiveParentCollisions();

  double error = 0;
  for (const ActivePlaneCollision& collision : activeParentCollisions_) {
    error += sqr(collision.result.overlap) * kCollisionWeight * this->weight_;
  }

  MT_LOGD(
      "PlaneCollisionError: {} active parents / {} candidates / {} primitives, error={:.8f}, weight={:.4f}",
      activeParentCollisions_.size(),
      candidatePrimitiveIndices_.size(),
      primitiveIndices_.size(),
      error,
      this->weight_);

  return error;
}

template <typename T>
double PlaneCollisionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  updateCandidateCollisionState(state, T(0));
  updateActiveParentCollisions();

  double error = 0;
  for (const ActivePlaneCollision& collision : activeParentCollisions_) {
    const PlaneCollisionResult& result = collision.result;
    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const T wgt = -T(2) * kCollisionWeight * this->weight_ * result.overlap;
    accumulateGradientAlongChain(
        state,
        result.position,
        planeNormal_,
        wgt,
        collisionGeometry_[collision.primitiveIndex].parent,
        gradient,
        /*scaleCorrection=*/T(0));
  }

  return error;
}

template <typename T>
double PlaneCollisionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(getJacobianSize()),
      "Jacobian rows mismatch: Actual {}, Expected {}",
      jacobian.rows(),
      gsl::narrow<Eigen::Index>(getJacobianSize()));
  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(getJacobianSize()),
      "Residual rows mismatch: Actual {}, Expected {}",
      residual.rows(),
      gsl::narrow<Eigen::Index>(getJacobianSize()));

  if (state.jointState.empty()) {
    usedRows = 0;
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  updateCandidateCollisionState(state, T(0));
  updateActiveParentCollisions();

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);
  int pos = 0;
  double error = 0;

  for (const ActivePlaneCollision& collision : activeParentCollisions_) {
    const PlaneCollisionResult& result = collision.result;
    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const int row = pos++;
    accumulateJacobianAlongChain(
        state,
        result.position,
        planeNormal_,
        -wgt,
        collisionGeometry_[collision.primitiveIndex].parent,
        jacobian,
        row,
        /*scaleCorrection=*/T(0));
    residual(row) = result.overlap * wgt;
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t PlaneCollisionErrorFunctionT<T>::getJacobianSize() const {
  return primitiveIndices_.size();
}

template class PlaneCollisionErrorFunctionT<float>;
template class PlaneCollisionErrorFunctionT<double>;

} // namespace momentum
