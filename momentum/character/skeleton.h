/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/common/exception.h>

#include <string>
#include <string_view>

namespace momentum {

/// The skeletal structure of a momentum Character.
template <typename T>
struct SkeletonT {
  /// The list of joints in this skeleton, ordered such that every joint's parent
  /// appears earlier in the list (parent index < child index, or kInvalidIndex for roots).
  JointList joints;

  SkeletonT() = default;

  /// @pre Every joint's parent index is either less than its own index or kInvalidIndex.
  /// @throws if the hierarchy invariant is violated.
  explicit SkeletonT(JointList joints);

  SkeletonT(const SkeletonT& other) = default;
  SkeletonT(SkeletonT&& other) noexcept = default;
  SkeletonT& operator=(const SkeletonT& other) = default;
  SkeletonT& operator=(SkeletonT&& other) noexcept = default;
  ~SkeletonT() = default;

  /// Returns the index of a joint with the given name, or kInvalidIndex if not found.
  [[nodiscard]] size_t getJointIdByName(std::string_view name) const;

  [[nodiscard]] std::vector<std::string> getJointNames() const;

  /// Returns indices of child joints of the specified joint.
  ///
  /// @param jointId Index of the joint to find children for
  /// @param recursive If true, returns all descendants; if false, only direct children
  /// @throws std::out_of_range if jointId is invalid
  [[nodiscard]] std::vector<size_t> getChildrenJoints(size_t jointId, bool recursive = true) const;

  /// Returns true if `ancestorJointId` is an ancestor of `jointId`.
  ///
  /// A joint is considered to be its own ancestor (`isAncestor(id, id)` returns true).
  [[nodiscard]] bool isAncestor(size_t jointId, size_t ancestorJointId) const;

  /// Returns true if the two joints are the same, or one is the direct parent of the other.
  ///
  /// Returns false if either joint is kInvalidIndex, since world-fixed entities have no
  /// joint ancestry and are never considered adjacent to skeleton joints.
  [[nodiscard]] bool isSameOrAdjacentJoints(size_t joint1, size_t joint2) const;

  /// Returns the lowest common ancestor of two joints in the hierarchy, or kInvalidIndex
  /// if either joint is kInvalidIndex.
  [[nodiscard]] size_t commonAncestor(size_t joint1, size_t joint2) const;

  /// Returns a copy of this skeleton with all numeric values converted to scalar type `U`.
  template <typename U>
  [[nodiscard]] SkeletonT<U> cast() const {
    if constexpr (std::is_same_v<T, U>) {
      return *this;
    } else {
      SkeletonT<U> newSkeleton;
      newSkeleton.joints = ::momentum::cast<U>(joints);
      return newSkeleton;
    }
  }
};

} // namespace momentum
