/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// Aim constraint data: a local ray (point, dir) passes a global point target
template <typename T>
struct AimDataT : ConstraintData {
  /// The origin of the local ray
  Vector3<T> localPoint;
  /// The direction of the local ray
  Vector3<T> localDir;
  /// The global aim target
  Vector3<T> globalTarget;

  explicit AimDataT(
      const Vector3<T>& inLocalPt,
      const Vector3<T>& inLocalDir,
      const Vector3<T>& inTarget,
      size_t pIndex,
      float w,
      const std::string& n = {})
      : ConstraintData(pIndex, w, n),
        localPoint(inLocalPt),
        localDir(inLocalDir.normalized()),
        globalTarget(inTarget) {}
};

/// The AimDistErrorFunction computes the distance from the aim target point to an aim ray (an
/// original and a positive direction). The distance is minimized if the target lies on the ray (ie.
/// on the positive direction), or when the target point is close to the ray origin. Another way to
/// look at this loss is to minimize the element-wise differences between the aim vector's
/// projection to the ray direction and the aim vector itself (therefore a shorter aim vector also
/// has a smaller norm).
template <typename T>
class AimDistErrorFunctionT : public JointErrorFunctionT<T, AimDataT<T>, 3, 2, 1> {
 public:
  explicit AimDistErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, AimDataT<T>, 3, 2, 1>(skel, pt, lossAlpha, lossC) {}

  explicit AimDistErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : AimDistErrorFunctionT<T>(
            character.skeleton,
            character.parameterTransform,
            lossAlpha,
            lossC) {}

  /// Default constant weight in MarkerErrorFunction (same as fixed axis). This can be used for
  /// backwards compatibility in setWeight().
  static constexpr T kLegacyWeight = 1e-1f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector3<T>& f,
      std::array<Vector3<T>, 2>& v,
      std::array<Eigen::Matrix3<T>, 2>& dfdv) const final;
};

/// The AimDirErrorFunction

/// The AimDirErrorFunction computes the element-wise differences between the normalized aim vector
/// and the ray direction. Because the aim vector is normalized, the distance to the ray origin
/// doesn't affect the loss as in AimDistErrorFunction. When the aim vector is close to zero length,
/// we will clamp it to zero.
template <typename T>
class AimDirErrorFunctionT : public JointErrorFunctionT<T, AimDataT<T>, 3, 2, 1> {
 public:
  explicit AimDirErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, AimDataT<T>, 3, 2, 1>(skel, pt, lossAlpha, lossC) {}

  explicit AimDirErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : AimDirErrorFunctionT<T>(
            character.skeleton,
            character.parameterTransform,
            lossAlpha,
            lossC) {}

  /// Default constant weight in MarkerErrorFunction (same as fixed axis). This can be used for
  /// backwards compatibility in setWeight().
  static constexpr T kLegacyWeight = 1e-1f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector3<T>& f,
      std::array<Vector3<T>, 2>& v,
      std::array<Eigen::Matrix3<T>, 2>& dfdv) const final;
};
} // namespace momentum
