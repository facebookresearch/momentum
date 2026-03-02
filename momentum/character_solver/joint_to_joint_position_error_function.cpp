/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/joint_to_joint_position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
JointToJointPositionErrorFunctionT<T>::JointToJointPositionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : MultiSourceErrorFunctionT<T, 3>(skel, pt) {}

template <typename T>
JointToJointPositionErrorFunctionT<T>::JointToJointPositionErrorFunctionT(
    const Character& character)
    : JointToJointPositionErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::addConstraint(
    size_t sourceJoint,
    const Vector3<T>& sourceOffset,
    size_t referenceJoint,
    const Vector3<T>& referenceOffset,
    const Vector3<T>& target,
    T weight,
    const std::string& name) {
  MT_CHECK(
      sourceJoint < this->skeleton_.joints.size(), "Invalid sourceJoint index: {}", sourceJoint);
  MT_CHECK(
      referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      referenceJoint);

  JointToJointPositionDataT<T> constraint;
  constraint.sourceJoint = sourceJoint;
  constraint.sourceOffset = sourceOffset;
  constraint.referenceJoint = referenceJoint;
  constraint.referenceOffset = referenceOffset;
  constraint.target = target;
  constraint.weight = weight;
  constraint.name = name;
  constraints_.push_back(constraint);
  contributionsCacheValid_ = false;
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::addConstraint(
    const JointToJointPositionDataT<T>& constraint) {
  MT_CHECK(
      constraint.sourceJoint < this->skeleton_.joints.size(),
      "Invalid sourceJoint index: {}",
      constraint.sourceJoint);
  MT_CHECK(
      constraint.referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      constraint.referenceJoint);
  constraints_.push_back(constraint);
  contributionsCacheValid_ = false;
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
  contributionsCache_.clear();
  contributionsCacheValid_ = false;
}

template <typename T>
size_t JointToJointPositionErrorFunctionT<T>::getNumConstraints() const {
  return constraints_.size();
}

template <typename T>
T JointToJointPositionErrorFunctionT<T>::getConstraintWeight(size_t constrIndex) const {
  return constraints_[constrIndex].weight;
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::rebuildContributionsCache() const {
  contributionsCache_.clear();
  contributionsCache_.reserve(constraints_.size());

  for (const auto& constr : constraints_) {
    // The contributions are:
    // [0] JointPoint: source world position
    // [1] JointPoint: reference world position (origin of reference frame)
    // [2] JointDirection: x-axis of reference frame (local x unit vector)
    // [3] JointDirection: y-axis of reference frame (local y unit vector)
    // [4] JointDirection: z-axis of reference frame (local z unit vector)
    contributionsCache_.push_back({
        SourceT<T>::jointPoint(constr.sourceJoint, constr.sourceOffset),
        SourceT<T>::jointPoint(constr.referenceJoint, constr.referenceOffset),
        SourceT<T>::jointDirection(constr.referenceJoint, Eigen::Vector3<T>::UnitX()),
        SourceT<T>::jointDirection(constr.referenceJoint, Eigen::Vector3<T>::UnitY()),
        SourceT<T>::jointDirection(constr.referenceJoint, Eigen::Vector3<T>::UnitZ()),
    });
  }

  contributionsCacheValid_ = true;
}

template <typename T>
std::span<const SourceT<T>> JointToJointPositionErrorFunctionT<T>::getContributions(
    size_t constrIndex) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");

  if (!contributionsCacheValid_) {
    rebuildContributionsCache();
  }

  return std::span<const SourceT<T>>(
      contributionsCache_[constrIndex].data(), contributionsCache_[constrIndex].size());
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");
  MT_CHECK(worldVecs.size() == 5, "Expected 5 world vectors for joint-to-joint position");

  const auto& constr = constraints_[constrIndex];

  // World vectors from contributions:
  // [0] srcWorldPos: world position of source joint point
  // [1] refWorldPos: world position of reference joint point (origin of reference frame)
  // [2] xAxis: world x-axis direction of reference frame
  // [3] yAxis: world y-axis direction of reference frame
  // [4] zAxis: world z-axis direction of reference frame
  const Eigen::Vector3<T>& srcWorldPos = worldVecs[0];
  const Eigen::Vector3<T>& refWorldPos = worldVecs[1];
  const Eigen::Vector3<T>& xAxis = worldVecs[2];
  const Eigen::Vector3<T>& yAxis = worldVecs[3];
  const Eigen::Vector3<T>& zAxis = worldVecs[4];

  // Difference vector in world frame
  const Eigen::Vector3<T> diff = srcWorldPos - refWorldPos;

  // Transform to reference frame: relativePos = R_ref^T * diff
  // where R_ref = [xAxis, yAxis, zAxis] (column vectors)
  // So R_ref^T * diff = [xAxis.dot(diff), yAxis.dot(diff), zAxis.dot(diff)]^T
  const Eigen::Vector3<T> relativePos(xAxis.dot(diff), yAxis.dot(diff), zAxis.dot(diff));

  // Residual: f = relativePos - target
  f = relativePos - constr.target;

  // Compute derivatives if requested
  if (!dfdv.empty()) {
    MT_CHECK(dfdv.size() == 5, "dfdv size must be 5");

    // df/d(srcWorldPos): derivative of relativePos w.r.t. srcWorldPos
    // relativePos_i = axis_i.dot(srcWorldPos - refWorldPos)
    // d(relativePos_i)/d(srcWorldPos) = axis_i^T
    // So df/d(srcWorldPos) = [xAxis; yAxis; zAxis]^T = R_ref^T
    Eigen::Matrix3<T> R_refT;
    R_refT.row(0) = xAxis.transpose();
    R_refT.row(1) = yAxis.transpose();
    R_refT.row(2) = zAxis.transpose();
    dfdv[0] = R_refT;

    // df/d(refWorldPos): derivative of relativePos w.r.t. refWorldPos
    // d(relativePos_i)/d(refWorldPos) = -axis_i^T
    // So df/d(refWorldPos) = -R_ref^T
    dfdv[1] = -R_refT;

    // df/d(xAxis): derivative of relativePos w.r.t. xAxis
    // Only affects first component: relativePos[0] = xAxis.dot(diff)
    // d(relativePos[0])/d(xAxis) = diff^T
    dfdv[2].setZero();
    dfdv[2].row(0) = diff.transpose();

    // df/d(yAxis): derivative of relativePos w.r.t. yAxis
    // Only affects second component: relativePos[1] = yAxis.dot(diff)
    dfdv[3].setZero();
    dfdv[3].row(1) = diff.transpose();

    // df/d(zAxis): derivative of relativePos w.r.t. zAxis
    // Only affects third component: relativePos[2] = zAxis.dot(diff)
    dfdv[4].setZero();
    dfdv[4].row(2) = diff.transpose();
  }
}

template class JointToJointPositionErrorFunctionT<float>;
template class JointToJointPositionErrorFunctionT<double>;

} // namespace momentum
