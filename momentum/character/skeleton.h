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
  /// The list of joints in this skeleton.
  JointList joints;

  /// Default constructor
  SkeletonT() = default;

  /// Constructor that validates joint hierarchy.
  /// Ensures parent indices are valid (parent < child index or kInvalidIndex).
  explicit SkeletonT(JointList joints);

  /// Copy constructor
  SkeletonT(const SkeletonT& other) = default;

  /// Move constructor
  SkeletonT(SkeletonT&& other) noexcept = default;

  /// Copy assignment operator
  SkeletonT& operator=(const SkeletonT& other) = default;

  /// Move assignment operator
  SkeletonT& operator=(SkeletonT&& other) noexcept = default;

  /// Returns the index of a joint with the given name, or kInvalidIndex if not found.
  [[nodiscard]] size_t getJointIdByName(std::string_view name) const;

  /// Returns a vector containing all joint names in the skeleton.
  [[nodiscard]] std::vector<std::string> getJointNames() const;

  /// Returns indices of child joints for the specified joint.
  ///
  /// @param jointId Index of the joint to find children for
  /// @param recursive If true, returns all descendants; if false, only direct children
  /// @throws std::out_of_range if jointId is invalid
  [[nodiscard]] std::vector<size_t> getChildrenJoints(size_t jointId, bool recursive = true) const;

  /// Determines if one joint is an ancestor of another in the hierarchy.
  ///
  /// Returns true if ancestorJointId is an ancestor of jointId.
  /// A joint is considered to be its own ancestor (isAncestor(id, id) returns true).
  [[nodiscard]] bool isAncestor(size_t jointId, size_t ancestorJointId) const;

  /// Finds the closest common ancestor of two joints in the hierarchy.
  ///
  /// Returns the index of the joint that is the lowest common ancestor
  /// in the hierarchy for the two specified joints.
  [[nodiscard]] size_t commonAncestor(size_t joint1, size_t joint2) const;

  /// Converts the skeleton to use a different scalar type.
  ///
  /// Returns a copy of this skeleton with all numeric values converted to type U.
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
