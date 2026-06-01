/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
CollisionErrorFunctionT<T>::CollisionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& cg,
    bool filterRestPoseOverlaps)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry_(cg) {
  updateCollisionPairs(filterRestPoseOverlaps);
}

template <typename T>
CollisionErrorFunctionT<T>::CollisionErrorFunctionT(
    const Character& character,
    bool filterRestPoseOverlaps)
    : CollisionErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries"),
          filterRestPoseOverlaps) {
  // Do nothing
}

template <typename T>
void CollisionErrorFunctionT<T>::updateCollisionPairs(bool filterRestPoseOverlaps) {
  validPairs_.clear();

  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionState_.update(state, collisionGeometry_);

  const auto n = collisionGeometry_.size();

  aabbs_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    aabbs_[i].id = gsl::narrow_cast<axel::Index>(i);
    updateAabb(aabbs_[i], collisionState_, i);
  }

  if (n == 0) {
    return;
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (isValidCollisionPair(
              collisionState_, collisionGeometry_, this->skeleton_, i, j, filterRestPoseOverlaps)) {
        const size_t lca = this->skeleton_.commonAncestor(
            collisionGeometry_[i].parent, collisionGeometry_[j].parent);
        validPairs_.push_back({i, j, lca});
      }
    }
  }
}

template <typename T>
void CollisionErrorFunctionT<T>::computeBroadPhase(const SkeletonStateT<T>& state) {
  {
    MT_PROFILE_EVENT("Collision: updateState");
    collisionState_.update(state, collisionGeometry_);
  }

  for (size_t i = 0; i < aabbs_.size(); ++i) {
    updateAabb(aabbs_[i], collisionState_, i);
  }
}

template <typename T>
bool CollisionErrorFunctionT<T>::checkCollision(
    const CollisionGeometryStateT<T>& colState,
    size_t indexA,
    size_t indexB,
    CollisionOverlapResultT<T>& result) const {
  if (!aabbs_[indexA].intersects(aabbs_[indexB])) {
    return false;
  }
  return overlaps(colState, collisionGeometry_, indexA, indexB, result);
}

template <typename T>
void CollisionErrorFunctionT<T>::accumulateGradientAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T weight,
    size_t startJoint,
    size_t stopJoint,
    Ref<VectorX<T>> gradient,
    T scaleCorrection) const {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex && jointIndex != stopJoint) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        const T val = direction.dot(jointState.getTranslationDerivative(d)) * weight;
        gradient_jointParams_to_modelParams(
            val, paramIndex + d, this->parameterTransform_, gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * weight;
        gradient_jointParams_to_modelParams(
            val, paramIndex + 3 + d, this->parameterTransform_, gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      const T val = (direction.dot(jointState.getScaleDerivative(posd)) + scaleCorrection) * weight;
      gradient_jointParams_to_modelParams(val, paramIndex + 6, this->parameterTransform_, gradient);
    }

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
void CollisionErrorFunctionT<T>::accumulateJacobianAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T factor,
    size_t startJoint,
    size_t stopJoint,
    Ref<MatrixX<T>> jacobian,
    int row,
    T scaleCorrection) const {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex && jointIndex != stopJoint) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        const T val = direction.dot(jointState.getTranslationDerivative(d)) * factor;
        jacobian_jointParams_to_modelParams(
            val, paramIndex + d, row, this->parameterTransform_, jacobian);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * factor;
        jacobian_jointParams_to_modelParams(
            val, paramIndex + 3 + d, row, this->parameterTransform_, jacobian);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      const T val = (direction.dot(jointState.getScaleDerivative(posd)) + scaleCorrection) * factor;
      jacobian_jointParams_to_modelParams(
          val, paramIndex + 6, row, this->parameterTransform_, jacobian);
    }

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
std::vector<Vector2i> CollisionErrorFunctionT<T>::getCollisionPairs() const {
  std::vector<Vector2i> collidingPairs;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (checkCollision(collisionState_, pair.indexA, pair.indexB, result)) {
      collidingPairs.emplace_back(static_cast<int>(pair.indexA), static_cast<int>(pair.indexB));
    }
  }
  return collidingPairs;
}

template <typename T>
double CollisionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  double error = 0;
  size_t collidingCount = 0;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, result)) {
      continue;
    }
    collidingCount++;
    error += sqr(result.overlap) * kCollisionWeight * this->weight_;
  }

  MT_LOGD(
      "CollisionError: {} colliding / {} valid pairs, error={:.8f}, weight={:.4f}",
      collidingCount,
      validPairs_.size(),
      error,
      this->weight_);

  return error;
}

template <typename T>
double CollisionErrorFunctionT<T>::getGradient(
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

  computeBroadPhase(state);

  double error = 0;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> direction = result.positionA - result.positionB;
    const T wgt = -T(2) * kCollisionWeight * this->weight_ * (result.overlap / result.distance);

    accumulateGradientAlongChain(
        state,
        result.positionA,
        direction,
        wgt,
        collisionGeometry_[pair.indexA].parent,
        pair.commonAncestor,
        gradient,
        -result.distance * result.radiusA * ln2<T>());
    accumulateGradientAlongChain(
        state,
        result.positionB,
        direction,
        -wgt,
        collisionGeometry_[pair.indexB].parent,
        pair.commonAncestor,
        gradient,
        result.distance * result.radiusB * ln2<T>());

    // Scale derivatives above the common ancestor: translation and rotation derivatives cancel
    // above the LCA (both capsule endpoints move identically), but scale derivatives do not
    // because the two endpoints are at different distances from the scaled joint, and scaling
    // also changes capsule radii.
    {
      const T netScaleVal =
          (direction.squaredNorm() - result.distance * (result.radiusA + result.radiusB)) *
          ln2<T>() * wgt;
      size_t ancestorIndex = pair.commonAncestor;
      while (ancestorIndex != kInvalidIndex) {
        const size_t paramIndex = ancestorIndex * kParametersPerJoint;
        if (this->activeJointParams_[paramIndex + 6]) {
          gradient_jointParams_to_modelParams(
              netScaleVal, paramIndex + 6, this->parameterTransform_, gradient);
        }
        ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
      }
    }
  }

  return error;
}

template <typename T>
double CollisionErrorFunctionT<T>::getJacobian(
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
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);
  int pos = 0;
  double error = 0;

  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> direction = result.positionA - result.positionB;
    const T fac = wgt / result.distance;
    const int row = pos++;

    accumulateJacobianAlongChain(
        state,
        result.positionA,
        direction,
        -fac,
        collisionGeometry_[pair.indexA].parent,
        pair.commonAncestor,
        jacobian,
        row,
        -result.distance * result.radiusA * ln2<T>());
    accumulateJacobianAlongChain(
        state,
        result.positionB,
        direction,
        fac,
        collisionGeometry_[pair.indexB].parent,
        pair.commonAncestor,
        jacobian,
        row,
        result.distance * result.radiusB * ln2<T>());

    // Scale derivatives above the common ancestor
    {
      const T netScaleVal = -fac * ln2<T>() * direction.squaredNorm() +
          wgt * (result.radiusA + result.radiusB) * ln2<T>();
      size_t ancestorIndex = pair.commonAncestor;
      while (ancestorIndex != kInvalidIndex) {
        const size_t paramIndex = ancestorIndex * kParametersPerJoint;
        if (this->activeJointParams_[paramIndex + 6]) {
          jacobian_jointParams_to_modelParams(
              netScaleVal, paramIndex + 6, row, this->parameterTransform_, jacobian);
        }
        ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
      }
    }

    residual(row) = result.overlap * wgt;
  }

  usedRows = pos;
  return error;
}

template <typename T>
std::vector<CollisionDebugEntryT<T>> CollisionErrorFunctionT<T>::getCollisionDebugInfo() const {
  std::vector<CollisionDebugEntryT<T>> entries;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (checkCollision(collisionState_, pair.indexA, pair.indexB, result)) {
      entries.push_back(
          {pair.indexA,
           pair.indexB,
           collisionGeometry_[pair.indexA].parent,
           collisionGeometry_[pair.indexB].parent,
           result.overlap,
           result.distance});
    }
  }
  return entries;
}

template <typename T>
size_t CollisionErrorFunctionT<T>::getJacobianSize() const {
  return validPairs_.size();
}

template class CollisionErrorFunctionT<float>;
template class CollisionErrorFunctionT<double>;

} // namespace momentum
