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
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

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
    : SkeletonErrorFunctionT<T>(skel, pt), collisionQuery_(cg, planeNormal, planeOffset) {
  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionQuery_.update(state, T(0));
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
  return collisionQuery_.getCollidingPrimitives();
}

template <typename T>
std::vector<PlaneCollisionDebugEntryT<T>> PlaneCollisionErrorFunctionT<T>::getCollisionDebugInfo()
    const {
  return collisionQuery_.getCollisionDebugInfo();
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionErrorFunctionT<T>::getContactPoints(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  return collisionQuery_.getContactPoints(states, contactMargin);
}

template <typename T>
std::vector<Vector3<T>> PlaneCollisionErrorFunctionT<T>::getContactPointsByParent(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  return collisionQuery_.getContactPointsByParent(states, contactMargin);
}

template <typename T>
std::vector<PlaneCollisionContactPointT<T>>
PlaneCollisionErrorFunctionT<T>::getContactPointsByParentWithDetails(
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  return collisionQuery_.getContactPointsByParentWithDetails(states, contactMargin);
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

  collisionQuery_.update(state, T(0));
  collisionQuery_.updateActiveParentCollisions();

  double error = 0;
  for (const auto& collision : collisionQuery_.activeParentCollisions()) {
    error += sqr(collision.overlap) * kCollisionWeight * this->weight_;
  }

  MT_LOGD(
      "PlaneCollisionError: {} active parents / {} candidates / {} primitives, error={:.8f}, weight={:.4f}",
      collisionQuery_.activeParentCollisions().size(),
      collisionQuery_.candidatePrimitiveCount(),
      collisionQuery_.primitiveCount(),
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

  collisionQuery_.update(state, T(0));
  collisionQuery_.updateActiveParentCollisions();

  double error = 0;
  for (const auto& collision : collisionQuery_.activeParentCollisions()) {
    error += sqr(collision.overlap) * kCollisionWeight * this->weight_;

    const T wgt = -T(2) * kCollisionWeight * this->weight_ * collision.overlap;
    accumulateGradientAlongChain(
        state,
        collision.position,
        collisionQuery_.getPlaneNormal(),
        wgt,
        collision.parentJoint,
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

  collisionQuery_.update(state, T(0));
  collisionQuery_.updateActiveParentCollisions();

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);
  int pos = 0;
  double error = 0;

  for (const auto& collision : collisionQuery_.activeParentCollisions()) {
    error += sqr(collision.overlap) * kCollisionWeight * this->weight_;

    const int row = pos++;
    accumulateJacobianAlongChain(
        state,
        collision.position,
        collisionQuery_.getPlaneNormal(),
        -wgt,
        collision.parentJoint,
        jacobian,
        row,
        /*scaleCorrection=*/T(0));
    residual(row) = collision.overlap * wgt;
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t PlaneCollisionErrorFunctionT<T>::getJacobianSize() const {
  return collisionQuery_.primitiveCount();
}

template class PlaneCollisionErrorFunctionT<float>;
template class PlaneCollisionErrorFunctionT<double>;

} // namespace momentum
