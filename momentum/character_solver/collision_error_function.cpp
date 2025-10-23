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
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry_(cg), bvh_() {
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
  // Clear any existing collision pairs
  excludingPairIds_.clear();
  jacobianSize_ = 0;

  // Create a zero state for the skeleton
  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);

  // Update the collision state based on the zero state and collision geometry
  collisionState_.update(state, collisionGeometry_);

  // Get the size of collisionGeometry_
  const auto n = collisionGeometry_.size();

  // Build BVH
  std::vector<axel::BoundingBox<T>> aabbs(n);
  for (size_t i = 0; i < n; ++i) {
    auto& aabb = aabbs[i];
    aabb.id = i;
    updateAabb(
        aabb, collisionState_.origin[i], collisionState_.direction[i], collisionState_.radius[i]);
  }
  bvh_.setBoundingBoxes(aabbs);

  // Exit early as there are no geometry objects to process
  if (n == 0) {
    return;
  }

  // Check if the pair intersects in the initial pose
  const auto numPotentialPairs = n * (n - 1) / 2;
  excludingPairIds_.reserve(numPotentialPairs);
  T distance;
  Vector2<T> cp;
  T overlap;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const std::pair<size_t, size_t> pairId = std::make_pair(i, j);

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
        excludingPairIds_.insert(pairId);
        continue;
      }

      // Walk down the this->skeleton_ hierarchy for the collision pair until we find a common
      // ancestor
      size_t p0 = collisionGeometry_[i].parent;
      size_t p1 = collisionGeometry_[j].parent;
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

      // Store collision pairs excluding those with the same parent or adjacent joints
      if (count > 1) {
        jacobianSize_++;
      } else {
        excludingPairIds_.insert(pairId);
      }
    }
  }
  // TODO: Consider using BVH for faster intersection checks
}

template <typename T>
void CollisionErrorFunctionT<T>::computeBroadPhase(const SkeletonStateT<T>& state) {
  // Update collision state
  {
    MT_PROFILE_EVENT("Collision: updateState");
    collisionState_.update(state, collisionGeometry_);
  }

  // Update the AABBs for each geometry object
  for (auto& aabb : bvh_.getPrimitives()) {
    const auto& index = aabb.id;
    updateAabb(
        aabb,
        collisionState_.origin[index],
        collisionState_.direction[index],
        collisionState_.radius[index]);
  }

  // Update BVH
  bvh_.setBoundingBoxes(bvh_.getPrimitives());
}

template <typename T>
std::vector<Vector2i> CollisionErrorFunctionT<T>::getCollisionPairs() const {
  std::vector<Vector2i> collidingPairs;

  bvh_.traverseOverlappingPairs([&](auto indexA, auto indexB) -> bool {
    const std::pair<size_t, size_t> pairId = std::make_pair(indexA, indexB);

    // Skip excluded pairs
    if (excludingPairIds_.find(pairId) != excludingPairIds_.end()) {
      return true; // continue traversing
    }

    // Perform narrow-phase collision check
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState_.origin[indexA],
            collisionState_.direction[indexA],
            collisionState_.radius[indexA],
            collisionState_.delta[indexA],
            collisionState_.origin[indexB],
            collisionState_.direction[indexB],
            collisionState_.radius[indexB],
            collisionState_.delta[indexB],
            distance,
            cp,
            overlap)) {
      return true; // continue traversing
    }

    collidingPairs.emplace_back(indexA, indexB);

    return true; // continue traversal
  });

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

  // Check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  // Traverse all the overlapping pairs and compute the error
  double error = 0;
  bvh_.traverseOverlappingPairs([&](auto indexA, auto indexB) -> bool {
    const std::pair<size_t, size_t> pairId = std::make_pair(indexA, indexB);

    // Skip excluded pairs
    if (excludingPairIds_.find(pairId) != excludingPairIds_.end()) {
      return true; // continue traversing
    }

    // Perform narrow-phase collision check
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState_.origin[indexA],
            collisionState_.direction[indexA],
            collisionState_.radius[indexA],
            collisionState_.delta[indexA],
            collisionState_.origin[indexB],
            collisionState_.direction[indexB],
            collisionState_.radius[indexB],
            collisionState_.delta[indexB],
            distance,
            cp,
            overlap)) {
      return true; // continue traversing
    }

    error += sqr(overlap) * kCollisionWeight * this->weight_;

    return true; // continue traversal
  });

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

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  // Traverse all the overlapping pairs and compute the error
  double error = 0;
  bvh_.traverseOverlappingPairs([&](auto indexA, auto indexB) -> bool {
    const std::pair<size_t, size_t> pairId = std::make_pair(indexA, indexB);

    // Skip excluded pairs
    if (excludingPairIds_.find(pairId) != excludingPairIds_.end()) {
      return true; // continue traversing
    }

    // Perform narrow-phase collision check
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState_.origin[indexA],
            collisionState_.direction[indexA],
            collisionState_.radius[indexA],
            collisionState_.delta[indexA],
            collisionState_.origin[indexB],
            collisionState_.direction[indexB],
            collisionState_.radius[indexB],
            collisionState_.delta[indexB],
            distance,
            cp,
            overlap)) {
      return true; // continue traversing
    }

    const auto& jointA = collisionGeometry_[indexA].parent;
    const auto& jointB = collisionGeometry_[indexB].parent;
    const size_t commonAncestor = this->skeleton_.commonAncestor(jointA, jointB);
    error += sqr(overlap) * kCollisionWeight * this->weight_;

    // calculate collision resolve direction. this is what we need to push joint parent i in.
    // the direction for joint parent j is the inverse
    const Vector3<T> position_i =
        (collisionState_.origin[indexA] + collisionState_.direction[indexA] * cp[0]);
    const Vector3<T> position_j =
        (collisionState_.origin[indexB] + collisionState_.direction[indexB] * cp[1]);
    const Vector3<T> direction = position_i - position_j;
    const T overlapFraction = overlap / distance;

    // calculate weight
    const T wgt = -T(2) * kCollisionWeight * this->weight_ * overlapFraction;

    // -----------------------------------
    //  process first joint
    // -----------------------------------
    size_t jointIndex = collisionGeometry_[indexA].parent;
    while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
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
    jointIndex = collisionGeometry_[indexB].parent;
    while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
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

    return true;
  });

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

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);

  int pos = 0;
  double error = 0;

  // loop over all pairs of collision objects and check
  bvh_.traverseOverlappingPairs([&](auto indexA, auto indexB) -> bool {
    const std::pair<size_t, size_t> pairId = std::make_pair(indexA, indexB);

    // Skip excluded pairs
    if (excludingPairIds_.find(pairId) != excludingPairIds_.end()) {
      return true; // continue traversing
    }

    // Perform narrow-phase collision check
    T distance;
    Vector2<T> cp;
    T overlap;
    if (!overlaps(
            collisionState_.origin[indexA],
            collisionState_.direction[indexA],
            collisionState_.radius[indexA],
            collisionState_.delta[indexA],
            collisionState_.origin[indexB],
            collisionState_.direction[indexB],
            collisionState_.radius[indexB],
            collisionState_.delta[indexB],
            distance,
            cp,
            overlap)) {
      return true; // continue traversing
    }

    const auto& jointA = collisionGeometry_[indexA].parent;
    const auto& jointB = collisionGeometry_[indexB].parent;
    const size_t commonAncestor = this->skeleton_.commonAncestor(jointA, jointB);
    error += sqr(overlap) * kCollisionWeight * this->weight_;

    // calculate collision resolve direction. this is what we need to push joint parent i in.
    // the direction for joint parent j is the inverse
    const Vector3<T> position_i =
        (collisionState_.origin[indexA] + collisionState_.direction[indexA] * cp[0]);
    const Vector3<T> position_j =
        (collisionState_.origin[indexB] + collisionState_.direction[indexB] * cp[1]);
    const Vector3<T> direction = position_i - position_j;
    const T inverseDistance = T(1) / distance;

    // calculate constant factor
    const T fac = T(2) * inverseDistance * wgt;

    // get position in jacobian
    const int row = pos++;

    // -----------------------------------
    //  process first joint
    // -----------------------------------
    size_t jointIndex = collisionGeometry_[indexA].parent;
    while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
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
    jointIndex = collisionGeometry_[indexB].parent;
    while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
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

    return true;
  });

  usedRows = pos;

  return error;
}

template <typename T>
size_t CollisionErrorFunctionT<T>::getJacobianSize() const {
  return jacobianSize_;
}

template class CollisionErrorFunctionT<float>;
template class CollisionErrorFunctionT<double>;

} // namespace momentum
