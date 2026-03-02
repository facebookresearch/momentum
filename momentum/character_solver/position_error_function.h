/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// 3D position constraint data
template <typename T>
struct PositionDataT : ConstraintData {
  /// Positional offset in the parent joint space
  Vector3<T> offset;
  /// Target position
  Vector3<T> target;

  explicit PositionDataT(
      const Vector3<T>& inOffset,
      const Vector3<T>& inTarget,
      size_t pIndex,
      float w,
      const std::string& n = "")
      : ConstraintData(pIndex, w, n), offset(inOffset), target(inTarget) {}
};

/// The PositionErrorFunction computes the 3D positional errors from a list of Constraints.
/// Each constraint specifies a locator on the skeleton (parent joint and offset), and its target 3D
/// position.
template <typename T>
class PositionErrorFunctionT : public JointErrorFunctionT<T, PositionDataT<T>> {
 public:
  /// Constructor
  ///
  /// @param[in] skel: character skeleton
  /// @param[in] pt: character parameter transformation
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit PositionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, PositionDataT<T>>(skel, pt, lossAlpha, lossC) {}

  /// A convenience api where character contains info of the skeleton and parameter transform.
  ///
  /// @param[in] character: character definition
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit PositionErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : PositionErrorFunctionT(character.skeleton, character.parameterTransform, lossAlpha, lossC) {
  }

  /// Default constant weight in MarkerErrorFunction. This can be used for backwards compatibility
  /// in setWeight().
  static constexpr T kLegacyWeight = 1e-4f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector3<T>& f,
      std::array<Vector3<T>, 1>& v,
      std::array<Matrix3<T>, 1>& dfdv) const final;
};

} // namespace momentum
