/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/constants.h>
#include <momentum/math/types.h>

namespace momentum {

/// A simplified version of a Maya joint with pre-rotation only (no post-rotation).
template <typename T>
struct JointT {
  using Scalar = T;

  std::string name{"uninitialized"};

  /// Index of the parent joint in the skeleton hierarchy, or `kInvalidIndex` for the root.
  size_t parent{kInvalidIndex};

  // TODO: does it make sense to save its own index too?

  /// Pre-rotation from the parent joint's axes to this joint's local axes.
  /// Applied before the joint's own rotation parameters.
  /// @note Default-initialized via `=` rather than braces due to a CUDA 11
  /// compiler bug with brace-initialization of Eigen's Quaternion.
  Quaternion<T> preRotation = Quaternion<T>::Identity();

  /// Constant translation offset from the parent joint's origin to this joint's
  /// origin, expressed in the parent's local frame.
  Vector3<T> translationOffset = Vector3<T>::Zero();

  /// Checks if the current joint is approximately equal to the provided joint.
  [[nodiscard]] bool isApprox(
      const JointT<T>& joint,
      const T& rotTol = Eps<T>(1e-4, 1e-10),
      const T& tranTol = Eigen::NumTraits<Scalar>::dummy_precision()) const {
    if (name != joint.name) {
      return false;
    }

    if (parent != joint.parent) {
      return false;
    }

    if (!preRotation.toRotationMatrix().isApprox(joint.preRotation.toRotationMatrix(), rotTol)) {
      return false;
    }

    if (!translationOffset.isApprox(joint.translationOffset, tranTol)) {
      return false;
    }

    return true;
  }

  /// Casts the current joint to another scalar type.
  template <typename U>
  [[nodiscard]] auto cast() const {
    if constexpr (std::is_same_v<T, U>) {
      return *this;
    } else {
      JointT<U> newJoint;
      newJoint.name = name;
      newJoint.parent = parent;
      newJoint.preRotation = preRotation.template cast<U>();
      newJoint.translationOffset = translationOffset.template cast<U>();
      return newJoint;
    }
  }
};

/// A list of joints (e.g., of a skeleton).
template <typename T>
using JointListT = std::vector<JointT<T>>;
using JointList = JointListT<float>;
using JointListd = JointListT<double>;

} // namespace momentum
