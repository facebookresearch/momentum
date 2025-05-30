/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skeleton.h>

#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/common/checks.h>

namespace momentum {

template <typename T>
SkeletonT<T>::SkeletonT(JointList joints_in) : joints(std::move(joints_in)) {
  // Ensure some invariants of the skeleton:
  for (size_t iJoint = 0; iJoint < joints.size(); ++iJoint) {
    MT_CHECK(joints[iJoint].parent == kInvalidIndex || joints[iJoint].parent < iJoint);
  }
}

template <typename T>
size_t SkeletonT<T>::getJointIdByName(const std::string_view name) const {
  for (size_t i = 0; i < joints.size(); i++) {
    if (joints[i].name == name) {
      return i;
    }
  }
  return kInvalidIndex;
}

template <typename T>
std::vector<std::string> SkeletonT<T>::getJointNames() const {
  std::vector<std::string> result;
  result.reserve(joints.size());
  for (const auto& j : joints) {
    result.emplace_back(j.name);
  }
  return result;
}

template <typename T>
std::vector<size_t> SkeletonT<T>::getChildrenJoints(const size_t jointId, const bool recursive)
    const {
  MT_THROW_IF_T(
      jointId >= joints.size(),
      std::out_of_range,
      "Out of bounds getChildrenJoints query. Requested index: {}. Number of joints: {}",
      jointId,
      joints.size());

  std::vector<size_t> childrenJoints;
  std::vector<int> jointDistance(joints.size(), -1);
  jointDistance[jointId] = 0;

  // traversal assuming parentJoint < childJoint
  for (size_t jointIter = jointId + 1; jointIter < joints.size(); ++jointIter) {
    const auto jointParent = joints[jointIter].parent;
    const int distParent = (jointParent == kInvalidIndex) ? -1 : jointDistance[jointParent];
    if ((recursive && distParent >= 0) || (!recursive && distParent == 0)) {
      jointDistance[jointIter] = distParent + 1;
      childrenJoints.push_back(jointIter);
    }
  }

  return childrenJoints;
}

template <typename T>
bool SkeletonT<T>::isAncestor(size_t jointId, size_t ancestorJointId) const {
  MT_CHECK(jointId < joints.size(), "{} vs {}", jointId, joints.size());
  MT_CHECK(ancestorJointId < joints.size(), "{} vs {}", ancestorJointId, joints.size());
  while (jointId != kInvalidIndex) {
    if (jointId == ancestorJointId) {
      return true;
    }
    jointId = joints[jointId].parent;
  }

  return false;
}

template <typename T>
size_t SkeletonT<T>::commonAncestor(size_t joint1, size_t joint2) const {
  MT_CHECK(joint1 == kInvalidIndex || joint1 < joints.size());
  MT_CHECK(joint2 == kInvalidIndex || joint2 < joints.size());

  while (joint1 != kInvalidIndex && joint2 != kInvalidIndex && joint1 != joint2) {
    if (joint1 < joint2) {
      joint2 = joints[joint2].parent;
    } else {
      joint1 = joints[joint1].parent;
    }
  }

  return joint1;
}

template struct SkeletonT<float>;
template struct SkeletonT<double>;

} // namespace momentum
