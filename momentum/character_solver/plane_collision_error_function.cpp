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
  updateCollisionState(state);
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
void PlaneCollisionErrorFunctionT<T>::updatePrimitiveList() {
  primitiveIndices_.clear();
  primitiveIndices_.reserve(collisionGeometry_.size());
  for (size_t i = 0; i < collisionGeometry_.size(); ++i) {
    // This error function constrains character-attached primitives against a world plane.
    if (collisionGeometry_[i].parent != kInvalidIndex) {
      primitiveIndices_.push_back(i);
    }
  }
}

template <typename T>
void PlaneCollisionErrorFunctionT<T>::updateCollisionState(const SkeletonStateT<T>& state) {
  MT_PROFILE_EVENT("PlaneCollision: updateState");
  collisionState_.update(state, collisionGeometry_);
}

template <typename T>
bool PlaneCollisionErrorFunctionT<T>::checkCollision(
    size_t primitiveIndex,
    PlaneCollisionResult& result) const {
  const auto type = collisionState_.type[primitiveIndex];

  // `result.position` is the deepest surface point toward the plane (the contact point), offset
  // from the primitive center/endpoint by the support radius along `-planeNormal_`. Differentiating
  // at the contact point — rather than the center — is what makes the analytic gradient correct:
  // the support radius depends on the primitive's world orientation, and folding the support offset
  // into `result.position` lets the joint-chain walk capture that orientation (and scale)
  // dependence automatically (so no separate scale/rotation correction term is needed).
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

    // Support offset: the vector from the center to the +planeNormal_ support point, satisfying
    // `planeNormal_.dot(supportOffset) == radius`. The contact point is `center - supportOffset`.
    const Vector3<T> localNormal = orientation.inverse() * planeNormal_;
    Vector3<T> supportOffset;
    if (type == CollisionPrimitiveType::Ellipsoid) {
      const Vector3<T>& radii = collisionState_.ellipsoidRadii[primitiveIndex];
      const T denom = result.radius > Eps<T>(1e-8f, 1e-16) ? result.radius : T(1);
      supportOffset = orientation * (radii.cwiseProduct(radii).cwiseProduct(localNormal)) / denom;
    } else { // Box
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
  for (const auto primitiveIndex : primitiveIndices_) {
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
  for (const auto primitiveIndex : primitiveIndices_) {
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

  updateCollisionState(state);

  double error = 0;
  size_t collidingCount = 0;
  for (const auto primitiveIndex : primitiveIndices_) {
    PlaneCollisionResult result;
    if (!checkCollision(primitiveIndex, result)) {
      continue;
    }

    ++collidingCount;
    error += sqr(result.overlap) * kCollisionWeight * this->weight_;
  }

  MT_LOGD(
      "PlaneCollisionError: {} colliding / {} primitives, error={:.8f}, weight={:.4f}",
      collidingCount,
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

  updateCollisionState(state);

  double error = 0;
  for (const auto primitiveIndex : primitiveIndices_) {
    PlaneCollisionResult result;
    if (!checkCollision(primitiveIndex, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const T wgt = -T(2) * kCollisionWeight * this->weight_ * result.overlap;
    accumulateGradientAlongChain(
        state,
        result.position,
        planeNormal_,
        wgt,
        collisionGeometry_[primitiveIndex].parent,
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

  updateCollisionState(state);

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);
  int pos = 0;
  double error = 0;

  for (const auto primitiveIndex : primitiveIndices_) {
    PlaneCollisionResult result;
    if (!checkCollision(primitiveIndex, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const int row = pos++;
    accumulateJacobianAlongChain(
        state,
        result.position,
        planeNormal_,
        -wgt,
        collisionGeometry_[primitiveIndex].parent,
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
