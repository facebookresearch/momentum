/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/collision_error_function_stateless.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
CollisionErrorFunctionStatelessT<T>::CollisionErrorFunctionStatelessT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& cg)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry(cg) {
  updateCollisionPairs();
}

template <typename T>
CollisionErrorFunctionStatelessT<T>::CollisionErrorFunctionStatelessT(const Character& character)
    : CollisionErrorFunctionStatelessT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries")) {
  // Do nothing
}

template <typename T>
void CollisionErrorFunctionStatelessT<T>::updateCollisionPairs() {
  // clear collisions
  collisionPairs.clear();

  // create zero state
  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionState.update(state, collisionGeometry);

  // go over all possible pairs
  for (size_t i = 0; i < collisionGeometry.size(); ++i) {
    for (size_t j = i + 1; j < collisionGeometry.size(); ++j) {
      // check if pair naturally intersects
      T distance;
      Vector2<T> cp;
      T overlap;
      if (overlaps(
              collisionState.origin[i],
              collisionState.direction[i],
              collisionState.radius[i],
              collisionState.delta[i],
              collisionState.origin[j],
              collisionState.direction[j],
              collisionState.radius[j],
              collisionState.delta[j],
              distance,
              cp,
              overlap)) {
        continue;
      }

      // no, it doesn't. check

      // walk down the this->skeleton_ hierarchy for the collision pair until we find a common
      // ancestor
      size_t p0 = collisionGeometry[i].parent;
      size_t p1 = collisionGeometry[j].parent;
      size_t count = 0;

      // Find the common ancestor. We can always just walk up the higher joint index because the
      // parent always has to have a lower index than the child joint.
      while (p0 != p1) {
        if (p0 > p1) {
          p0 = this->skeleton_.joints[p0].parent;
        } else {
          p1 = this->skeleton_.joints[p1].parent;
        }
        count++;
      }

      const Vector3i collisionPair(
          gsl::narrow_cast<int>(i), gsl::narrow_cast<int>(j), gsl::narrow_cast<int>(p0));

      // Store collision pairs excluding those with the same parent or adjacent joints
      if (count > 1) {
        collisionPairs.push_back(collisionPair);
      }
    }
  }

  collisionActive.setConstant(collisionPairs.size(), false);
}

template <typename T>
std::vector<Vector2i> CollisionErrorFunctionStatelessT<T>::getCollisionPairs() const {
  std::vector<Vector2i> result;

  for (size_t i = 0; i < collisionPairs.size(); i++) {
    if (collisionActive[i]) {
      result.emplace_back(collisionPairs[i].template head<2>());
    }
  }

  return result;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getError(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // update collision state
  collisionState.update(state, collisionGeometry);
  collisionActive.setConstant(false);

  // loop over all pairs of collision objects and check
  double error = 0.0;

  for (size_t i = 0; i < collisionPairs.size(); i++) {
    const auto& pr = collisionPairs[i];
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState.origin[pr[0]],
            collisionState.direction[pr[0]],
            collisionState.radius[pr[0]],
            collisionState.delta[pr[0]],
            collisionState.origin[pr[1]],
            collisionState.direction[pr[1]],
            collisionState.radius[pr[1]],
            collisionState.delta[pr[1]],
            distance,
            cp,
            overlap)) {
      continue;
    }

    collisionActive[i] = true;
    error += sqr(overlap) * kCollisionWeight * this->weight_;
  }

  // return error
  return error;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getGradient(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // update collision state
  collisionState.update(state, collisionGeometry);
  collisionActive.setConstant(false);

  double error = 0;
  for (size_t i = 0; i < collisionPairs.size(); ++i) {
    const auto& pr = collisionPairs[i];
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState.origin[pr[0]],
            collisionState.direction[pr[0]],
            collisionState.radius[pr[0]],
            collisionState.delta[pr[0]],
            collisionState.origin[pr[1]],
            collisionState.direction[pr[1]],
            collisionState.radius[pr[1]],
            collisionState.delta[pr[1]],
            distance,
            cp,
            overlap)) {
      continue;
    }

    collisionActive[i] = true;
    error += sqr(overlap) * kCollisionWeight * this->weight_;

    // calculate collision resolve direction. this is what we need to push joint parent i in.
    // the direction for joint parent j is the inverse
    const Vector3<T> position_i =
        (collisionState.origin[pr[0]] + collisionState.direction[pr[0]] * cp[0]);
    const Vector3<T> position_j =
        (collisionState.origin[pr[1]] + collisionState.direction[pr[1]] * cp[1]);
    const Vector3<T> direction = position_i - position_j;
    const T overlapFraction = overlap / distance;

    // calculate weight
    const T wgt = -T(2) * kCollisionWeight * this->weight_ * overlapFraction;

    // -----------------------------------
    //  process first joint
    // -----------------------------------
    size_t jointIndex = collisionGeometry[pr[0]].parent;
    while (jointIndex != kInvalidIndex && jointIndex != gsl::narrow_cast<size_t>(pr[2])) {
      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Vector3<T> posd = position_i - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getTranslationDerivative(d)) * wgt;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * wgt;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = direction.dot(jointState.getScaleDerivative(posd)) * wgt;
        // explicitly multiply with the parameter transform to generate parameter space
        // gradients
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    // -----------------------------------
    //  process second joint
    // -----------------------------------
    jointIndex = collisionGeometry[pr[1]].parent;
    while (jointIndex != kInvalidIndex && jointIndex != gsl::narrow_cast<size_t>(pr[2])) {
      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Vector3<T> posd = position_j - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getTranslationDerivative(d)) * -wgt;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * -wgt;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          const auto maxIndex =
              this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
               index < maxIndex;
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = direction.dot(jointState.getScaleDerivative(posd)) * -wgt;
        // explicitly multiply with the parameter transform to generate parameter space
        // gradients
        const auto maxIndex =
            this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < maxIndex;
             ++index) {
          gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  // return error
  return error;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getJacobian(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(collisionPairs.size()),
      "Jacobian rows mismatch: Actual {}, Expected {}",
      jacobian.rows(),
      gsl::narrow<Eigen::Index>(collisionPairs.size()));

  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(collisionPairs.size()),
      "Residual rows mismatch: Actual {}, Expected {}",
      residual.rows(),
      gsl::narrow<Eigen::Index>(collisionPairs.size()));

  if (state.jointState.empty()) {
    return 0.0;
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // update collision state
  {
    MT_PROFILE_EVENT("Collision: updateState");

    collisionState.update(state, collisionGeometry);
    collisionActive.setConstant(false);
  }

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);

  int pos = 0;
  double error = 0;

  // loop over all pairs of collision objects and check
  for (size_t i = 0; i < collisionPairs.size(); ++i) {
    const auto& pr = collisionPairs[i];
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState.origin[pr[0]],
            collisionState.direction[pr[0]],
            collisionState.radius[pr[0]],
            collisionState.delta[pr[0]],
            collisionState.origin[pr[1]],
            collisionState.direction[pr[1]],
            collisionState.radius[pr[1]],
            collisionState.delta[pr[1]],
            distance,
            cp,
            overlap)) {
      continue;
    }

    collisionActive[i] = true;
    error += sqr(overlap) * kCollisionWeight * this->weight_;

    // calculate collision resolve direction. this is what we need to push joint parent i in.
    // the direction for joint parent j is the inverse
    const Vector3<T> position_i =
        (collisionState.origin[pr[0]] + collisionState.direction[pr[0]] * cp[0]);
    const Vector3<T> position_j =
        (collisionState.origin[pr[1]] + collisionState.direction[pr[1]] * cp[1]);
    const Vector3<T> direction = position_i - position_j;
    const T inverseDistance = T(1) / distance;

    // calculate constant factor
    const T fac = T(2) * inverseDistance * wgt;

    // get position in jacobian
    const int row = pos++;

    // -----------------------------------
    //  process first joint
    // -----------------------------------
    size_t jointIndex = collisionGeometry[pr[0]].parent;
    while (jointIndex != kInvalidIndex && jointIndex != gsl::narrow_cast<size_t>(pr[2])) {
      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Vector3<T> posd = position_i - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getTranslationDerivative(d)) * -fac;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * -fac;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
               ++index) {
            jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = direction.dot(jointState.getScaleDerivative(posd)) * -fac;
        // explicitly multiply with the parameter transform to generate parameter space
        // gradients
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    // -----------------------------------
    //  process second joint
    // -----------------------------------
    jointIndex = collisionGeometry[pr[1]].parent;
    while (jointIndex != kInvalidIndex && jointIndex != gsl::narrow_cast<size_t>(pr[2])) {
      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Vector3<T> posd = position_j - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getTranslationDerivative(d)) * fac;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient
          const T val = direction.dot(jointState.getRotationDerivative(d, posd)) * fac;
          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
               ++index) {
            jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = direction.dot(jointState.getScaleDerivative(posd)) * fac;
        // explicitly multiply with the parameter transform to generate parameter space
        // gradients
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    residual(row) = overlap * wgt;
  }

  usedRows = pos;

  // return error
  return error;
}

template <typename T>
size_t CollisionErrorFunctionStatelessT<T>::getJacobianSize() const {
  return collisionPairs.size();
}

template class CollisionErrorFunctionStatelessT<float>;
template class CollisionErrorFunctionStatelessT<double>;

} // namespace momentum
