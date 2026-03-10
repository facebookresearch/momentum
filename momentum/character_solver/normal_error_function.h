/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// Normal constraint data: data needed to compute the point-to-plane distance for
/// "data-to-template" in ICP.
template <typename T>
struct NormalDataT : ConstraintData {
  /// The template point defined in the local space
  Vector3<T> localPoint;
  /// The template normal defined in the local space
  Vector3<T> localNormal;
  /// The data point in world space
  Vector3<T> globalPoint;

  explicit NormalDataT(
      const Vector3<T>& inLocalPt,
      const Vector3<T>& inLocalNorm,
      const Vector3<T>& inTargetPt,
      size_t pIndex,
      float w,
      const std::string& n = {})
      : ConstraintData(pIndex, w, n),
        localPoint(inLocalPt),
        localNormal(inLocalNorm.normalized()),
        globalPoint(inTargetPt) {}
};

/// The NormalErrorFunction computes a "point-to-plane" (signed) distance from a target point to the
/// plane defined by a local point and a local normal vector. The name is not informative but chosen
/// for backwards compatibility purpose.
template <typename T>
class NormalErrorFunctionT : public JointErrorFunctionT<T, NormalDataT<T>, 1, 2, 1> {
 public:
  explicit NormalErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, NormalDataT<T>, 1, 2, 1>(skel, pt, lossAlpha, lossC) {}

  explicit NormalErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : NormalErrorFunctionT<T>(
            character.skeleton,
            character.parameterTransform,
            lossAlpha,
            lossC) {}

  /// Default constant weight in MarkerErrorFunction. This can be used for backwards compatibility
  /// in setWeight().
  static constexpr T kLegacyWeight = 1e-4f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector<T, 1>& f,
      std::array<Vector3<T>, 2>& v,
      std::array<Eigen::Matrix<T, 1, 3>, 2>& dfdv) const final;
};
} // namespace momentum
