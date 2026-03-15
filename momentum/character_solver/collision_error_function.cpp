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
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
CollisionErrorFunctionT<T>::CollisionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& cg)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry_(cg) {
  updateCollisionPairs();
}

template <typename T>
CollisionErrorFunctionT<T>::CollisionErrorFunctionT(const Character& character)
    : CollisionErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries")) {
  // Do nothing
}

template <typename T>
void CollisionErrorFunctionT<T>::updateCollisionPairs() {
  validPairs_.clear();

  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionState_.update(state, collisionGeometry_);

  const auto n = collisionGeometry_.size();

  aabbs_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    aabbs_[i].id = gsl::narrow_cast<axel::Index>(i);
    updateAabb(
        aabbs_[i],
        collisionState_.origin[i],
        collisionState_.direction[i],
        collisionState_.radius[i]);
  }

  if (n == 0) {
    return;
  }

  T distance;
  Vector2<T> cp;
  T overlap;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (overlaps(
              collisionState_.origin[i],
              collisionState_.direction[i],
              collisionState_.radius[i],
              collisionState_.delta[i],
              collisionState_.origin[j],
              collisionState_.direction[j],
              collisionState_.radius[j],
              collisionState_.delta[j],
              distance,
              cp,
              overlap)) {
        continue;
      }

      size_t p0 = collisionGeometry_[i].parent;
      size_t p1 = collisionGeometry_[j].parent;
      size_t count = 0;
      while (p0 != p1) {
        if (p0 > p1) {
          p0 = this->skeleton_.joints[p0].parent;
        } else {
          p1 = this->skeleton_.joints[p1].parent;
        }
        count++;
      }

      if (count > 1) {
        validPairs_.push_back({i, j, p0});
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
    updateAabb(
        aabbs_[i],
        collisionState_.origin[i],
        collisionState_.direction[i],
        collisionState_.radius[i]);
  }
}

template <typename T>
bool CollisionErrorFunctionT<T>::checkCollision(
    const CollisionGeometryStateT<T>& colState,
    size_t indexA,
    size_t indexB,
    T& distance,
    Vector2<T>& cp,
    T& overlap) const {
  if (!aabbs_[indexA].intersects(aabbs_[indexB])) {
    return false;
  }
  return overlaps(
      colState.origin[indexA],
      colState.direction[indexA],
      colState.radius[indexA],
      colState.delta[indexA],
      colState.origin[indexB],
      colState.direction[indexB],
      colState.radius[indexB],
      colState.delta[indexB],
      distance,
      cp,
      overlap);
}

template <typename T>
void CollisionErrorFunctionT<T>::accumulateGradientAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T weight,
    size_t startJoint,
    size_t stopJoint,
    Ref<VectorX<T>> gradient) const {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex && jointIndex != stopJoint) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        const T val = direction.dot(jointState.getTranslationDerivative(d)) * weight;
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * weight;
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      const T val = direction.dot(jointState.getScaleDerivative(posd)) * weight;
      for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
            val * this->parameterTransform_.transform.valuePtr()[index];
      }
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
    int row) const {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex && jointIndex != stopJoint) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        const T val = direction.dot(jointState.getTranslationDerivative(d)) * factor;
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * factor;
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      const T val = direction.dot(jointState.getScaleDerivative(posd)) * factor;
      for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
            val * this->parameterTransform_.transform.valuePtr()[index];
      }
    }

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
std::vector<Vector2i> CollisionErrorFunctionT<T>::getCollisionPairs() const {
  std::vector<Vector2i> collidingPairs;
  for (const auto& pair : validPairs_) {
    T distance;
    Vector2<T> cp;
    T overlap;
    if (checkCollision(collisionState_, pair.indexA, pair.indexB, distance, cp, overlap)) {
      collidingPairs.emplace_back(pair.indexA, pair.indexB);
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
  for (const auto& pair : validPairs_) {
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, distance, cp, overlap)) {
      continue;
    }
    error += sqr(overlap) * kCollisionWeight * this->weight_;
  }

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
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, distance, cp, overlap)) {
      continue;
    }

    error += sqr(overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> position_i =
        collisionState_.origin[pair.indexA] + collisionState_.direction[pair.indexA] * cp[0];
    const Vector3<T> position_j =
        collisionState_.origin[pair.indexB] + collisionState_.direction[pair.indexB] * cp[1];
    const Vector3<T> direction = position_i - position_j;
    const T wgt = -T(2) * kCollisionWeight * this->weight_ * (overlap / distance);

    accumulateGradientAlongChain(
        state,
        position_i,
        direction,
        wgt,
        collisionGeometry_[pair.indexA].parent,
        pair.commonAncestor,
        gradient);
    accumulateGradientAlongChain(
        state,
        position_j,
        direction,
        -wgt,
        collisionGeometry_[pair.indexB].parent,
        pair.commonAncestor,
        gradient);
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
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!checkCollision(collisionState_, pair.indexA, pair.indexB, distance, cp, overlap)) {
      continue;
    }

    error += sqr(overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> position_i =
        collisionState_.origin[pair.indexA] + collisionState_.direction[pair.indexA] * cp[0];
    const Vector3<T> position_j =
        collisionState_.origin[pair.indexB] + collisionState_.direction[pair.indexB] * cp[1];
    const Vector3<T> direction = position_i - position_j;
    const T fac = T(2) / distance * wgt;
    const int row = pos++;

    accumulateJacobianAlongChain(
        state,
        position_i,
        direction,
        -fac,
        collisionGeometry_[pair.indexA].parent,
        pair.commonAncestor,
        jacobian,
        row);
    accumulateJacobianAlongChain(
        state,
        position_j,
        direction,
        fac,
        collisionGeometry_[pair.indexB].parent,
        pair.commonAncestor,
        jacobian,
        row);

    residual(row) = overlap * wgt;
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t CollisionErrorFunctionT<T>::getJacobianSize() const {
  return validPairs_.size();
}

template class CollisionErrorFunctionT<float>;
template class CollisionErrorFunctionT<double>;

} // namespace momentum
