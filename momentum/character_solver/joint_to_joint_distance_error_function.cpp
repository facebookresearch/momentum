/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/joint_to_joint_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/math/constants.h"

namespace momentum {

template <typename T>
JointToJointDistanceErrorFunctionT<T>::JointToJointDistanceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : MultiSourceErrorFunctionT<T, 1>(skel, pt) {}

template <typename T>
JointToJointDistanceErrorFunctionT<T>::JointToJointDistanceErrorFunctionT(
    const Character& character)
    : JointToJointDistanceErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::addConstraint(
    size_t joint1,
    const Vector3<T>& offset1,
    size_t joint2,
    const Vector3<T>& offset2,
    T targetDistance,
    T weight) {
  MT_CHECK(joint1 < this->skeleton_.joints.size(), "Invalid joint1 index: {}", joint1);
  MT_CHECK(joint2 < this->skeleton_.joints.size(), "Invalid joint2 index: {}", joint2);
  MT_CHECK(joint1 != joint2, "joint1 and joint2 must be different");
  MT_CHECK(targetDistance >= 0, "targetDistance must be non-negative");

  constraints_.push_back(ConstraintType{joint1, offset1, joint2, offset2, targetDistance, weight});
  contributionsCacheValid_ = false;
}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
  contributionsCache_.clear();
  contributionsCacheValid_ = false;
}

template <typename T>
size_t JointToJointDistanceErrorFunctionT<T>::getNumConstraints() const {
  return constraints_.size();
}

template <typename T>
T JointToJointDistanceErrorFunctionT<T>::getConstraintWeight(size_t constrIndex) const {
  return constraints_[constrIndex].weight * kDistanceWeight;
}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::rebuildContributionsCache() const {
  contributionsCache_.clear();
  contributionsCache_.reserve(constraints_.size());

  for (const auto& constr : constraints_) {
    contributionsCache_.push_back({
        SourceT<T>::jointPoint(constr.joint1, constr.offset1),
        SourceT<T>::jointPoint(constr.joint2, constr.offset2),
    });
  }

  contributionsCacheValid_ = true;
}

template <typename T>
std::span<const SourceT<T>> JointToJointDistanceErrorFunctionT<T>::getContributions(
    size_t constrIndex) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");

  if (!contributionsCacheValid_) {
    rebuildContributionsCache();
  }

  return std::span<const SourceT<T>>(
      contributionsCache_[constrIndex].data(), contributionsCache_[constrIndex].size());
}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");
  MT_CHECK(worldVecs.size() == 2, "Expected 2 world vectors for distance constraint");

  const auto& constr = constraints_[constrIndex];

  // Compute distance: ||p1 - p2||
  const Eigen::Vector3<T> diff = worldVecs[0] - worldVecs[1];
  const T actualDistance = diff.norm();

  // Handle near-zero distance to avoid division by zero
  if (actualDistance < Eps<T>(1e-8f, 1e-14)) {
    f(0) = -constr.targetDistance;
    if (!dfdv.empty()) {
      MT_CHECK(dfdv.size() == 2, "dfdv size must be 2");
      dfdv[0].setZero();
      dfdv[1].setZero();
    }
    return;
  }

  // Residual: f = ||p1 - p2|| - targetDistance
  f(0) = actualDistance - constr.targetDistance;

  // Compute derivatives if requested
  if (!dfdv.empty()) {
    MT_CHECK(dfdv.size() == 2, "dfdv size must be 2");

    // df/dp1 = (p1 - p2) / ||p1 - p2|| = direction (1x3 row vector)
    // df/dp2 = -(p1 - p2) / ||p1 - p2|| = -direction
    const Eigen::Vector3<T> direction = diff / actualDistance;
    dfdv[0] = direction.transpose();
    dfdv[1] = -direction.transpose();
  }
}

template class JointToJointDistanceErrorFunctionT<float>;
template class JointToJointDistanceErrorFunctionT<double>;

} // namespace momentum
