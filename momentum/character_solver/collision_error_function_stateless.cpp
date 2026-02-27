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
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
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
      if (!isValidCollisionPair(collisionState, collisionGeometry, this->skeleton_, i, j)) {
        continue;
      }

      // Find the common ancestor, handling kInvalidIndex parents safely
      const size_t p0 = collisionGeometry[i].parent;
      const size_t p1 = collisionGeometry[j].parent;
      const size_t lca = this->skeleton_.commonAncestor(p0, p1);

      // Store the LCA as int for the collision pair. When lca == kInvalidIndex (no common
      // ancestor, e.g. world-fixed capsule), we store -1. The gradient/Jacobian loops compare
      // jointIndex against gsl::narrow_cast<size_t>(pr[2]); since those loops already terminate
      // when jointIndex == kInvalidIndex (the root's parent), the pr[2] sentinel is never
      // actually compared for world-fixed capsules — it just needs to not accidentally match
      // a valid joint index.
      const int lcaInt = (lca == kInvalidIndex) ? -1 : gsl::narrow_cast<int>(lca);
      collisionPairs.push_back(
          Vector3i(gsl::narrow_cast<int>(i), gsl::narrow_cast<int>(j), lcaInt));
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
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
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
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
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

    // Effective radii at the closest points (for scale derivative correction below)
    const T radiusA_at_cp = collisionState.radius[pr[0]][0] + cp[0] * collisionState.delta[pr[0]];
    const T radiusB_at_cp = collisionState.radius[pr[1]][0] + cp[1] * collisionState.delta[pr[1]];

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

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          gradient_jointParams_to_modelParams(
              direction.dot(jointState.getTranslationDerivative(d)) * wgt,
              paramIndex + d,
              this->parameterTransform_,
              gradient);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          gradient_jointParams_to_modelParams(
              direction.dot(jointState.getRotationDerivative(d, posd)) * wgt,
              paramIndex + 3 + d,
              this->parameterTransform_,
              gradient);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // Scale derivative has an extra radius term: d(overlap)/d(s) = d(radius)/d(s) -
        // (dir/dist)·d(pos)/d(s)
        gradient_jointParams_to_modelParams(
            (direction.dot(jointState.getScaleDerivative(posd)) -
             distance * radiusA_at_cp * ln2<T>()) *
                wgt,
            paramIndex + 6,
            this->parameterTransform_,
            gradient);
      }

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

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          gradient_jointParams_to_modelParams(
              direction.dot(jointState.getTranslationDerivative(d)) * -wgt,
              paramIndex + d,
              this->parameterTransform_,
              gradient);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          gradient_jointParams_to_modelParams(
              direction.dot(jointState.getRotationDerivative(d, posd)) * -wgt,
              paramIndex + 3 + d,
              this->parameterTransform_,
              gradient);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        gradient_jointParams_to_modelParams(
            (direction.dot(jointState.getScaleDerivative(posd)) +
             distance * radiusB_at_cp * ln2<T>()) *
                -wgt,
            paramIndex + 6,
            this->parameterTransform_,
            gradient);
      }

      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    // -----------------------------------
    //  process scale derivatives above common ancestor
    // -----------------------------------
    // Translation and rotation derivatives cancel above the LCA (both capsule endpoints
    // move identically), but scale derivatives do not: the two endpoints are at different
    // distances from the scaled joint, and scaling also changes capsule radii.
    {
      const size_t commonAncestor = gsl::narrow_cast<size_t>(pr[2]);
      const T netScaleVal =
          (direction.squaredNorm() - distance * (radiusA_at_cp + radiusB_at_cp)) * ln2<T>() * wgt;
      size_t ancestorIndex = commonAncestor;
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

  // return error
  return error;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
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

    // Effective radii at the closest points (for scale derivative correction below)
    const T radiusA_at_cp = collisionState.radius[pr[0]][0] + cp[0] * collisionState.delta[pr[0]];
    const T radiusB_at_cp = collisionState.radius[pr[1]][0] + cp[1] * collisionState.delta[pr[1]];

    // calculate constant factor
    const T fac = inverseDistance * wgt;

    // get position in jacobian
    const int row = pos++;
    auto jacobianRow = jacobian.middleRows(row, 1);

    // -----------------------------------
    //  process first joint
    // -----------------------------------
    size_t jointIndex = collisionGeometry[pr[0]].parent;
    while (jointIndex != kInvalidIndex && jointIndex != gsl::narrow_cast<size_t>(pr[2])) {
      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Vector3<T> posd = position_i - jointState.translation();

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          jacobian_jointParams_to_modelParams<T>(
              direction.dot(jointState.getTranslationDerivative(d)) * -fac,
              paramIndex + d,
              this->parameterTransform_,
              jacobianRow);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          jacobian_jointParams_to_modelParams<T>(
              direction.dot(jointState.getRotationDerivative(d, posd)) * -fac,
              paramIndex + 3 + d,
              this->parameterTransform_,
              jacobianRow);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // Scale derivative has an extra radius term
        jacobian_jointParams_to_modelParams<T>(
            direction.dot(jointState.getScaleDerivative(posd)) * -fac +
                wgt * radiusA_at_cp * ln2<T>(),
            paramIndex + 6,
            this->parameterTransform_,
            jacobianRow);
      }

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

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          jacobian_jointParams_to_modelParams<T>(
              direction.dot(jointState.getTranslationDerivative(d)) * fac,
              paramIndex + d,
              this->parameterTransform_,
              jacobianRow);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          jacobian_jointParams_to_modelParams<T>(
              direction.dot(jointState.getRotationDerivative(d, posd)) * fac,
              paramIndex + 3 + d,
              this->parameterTransform_,
              jacobianRow);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        jacobian_jointParams_to_modelParams<T>(
            direction.dot(jointState.getScaleDerivative(posd)) * fac +
                wgt * radiusB_at_cp * ln2<T>(),
            paramIndex + 6,
            this->parameterTransform_,
            jacobianRow);
      }

      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    // -----------------------------------
    //  process scale derivatives above common ancestor
    // -----------------------------------
    // Translation and rotation derivatives cancel above the LCA; scale does not.
    {
      const size_t commonAncestor = gsl::narrow_cast<size_t>(pr[2]);
      const T netScaleVal = -fac * ln2<T>() * direction.squaredNorm() +
          wgt * (radiusA_at_cp + radiusB_at_cp) * ln2<T>();
      size_t ancestorIndex = commonAncestor;
      while (ancestorIndex != kInvalidIndex) {
        const size_t paramIndex = ancestorIndex * kParametersPerJoint;
        if (this->activeJointParams_[paramIndex + 6]) {
          jacobian_jointParams_to_modelParams<T>(
              netScaleVal, paramIndex + 6, this->parameterTransform_, jacobianRow);
        }
        ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
      }
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
