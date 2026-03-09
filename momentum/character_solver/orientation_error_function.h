/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// Constraint data on 3x3 rotation represented in quaternions
template <typename T>
struct OrientationDataT : ConstraintData {
  /// Rotation offset in the parent space
  Eigen::Quaternion<T> offset;
  /// Target rotation in global space
  Eigen::Quaternion<T> target;

  /// param[in] inOffset: input rotation offset in the parent space
  /// param[in] inTarget: input target rotation in global space
  /// param[in] pIndex: joint index of the parent joint
  /// param[in] w: weight
  /// param[in] n: name of this constraint
  explicit OrientationDataT(
      const Eigen::Quaternion<T>& inOffset,
      const Eigen::Quaternion<T>& inTarget,
      size_t pIndex,
      float w,
      const std::string& n = "")
      : ConstraintData(pIndex, w, n),
        offset(inOffset.normalized()),
        target(inTarget.normalized()) {}
};

/// The OrientationErrorFunction minimizes the F-norm of the element-wise difference of a 3x3
/// rotation matrix to a target rotation. Therefore, function f has 9 numbers (FuncDim), and
/// NumVec=3 for the 3 rotation axis.
template <typename T>
class OrientationErrorFunctionT : public JointErrorFunctionT<T, OrientationDataT<T>, 9, 3, 0> {
 public:
  explicit OrientationErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, OrientationDataT<T>, 9, 3, 0>(skel, pt, lossAlpha, lossC) {}

  explicit OrientationErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : OrientationErrorFunctionT(
            character.skeleton,
            character.parameterTransform,
            lossAlpha,
            lossC) {}

  /// Default constant weight in MarkerErrorFunction. This can be used for backwards compatibility
  /// in setWeight().
  static constexpr T kLegacyWeight = 1e-1f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector<T, 9>& f,
      std::array<Vector3<T>, 3>& v,
      std::array<Eigen::Matrix<T, 9, 3>, 3>& dfdv) const final;
};

} // namespace momentum
