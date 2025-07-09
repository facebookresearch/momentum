/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/constraint_error_function.h>

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

/// Base class for all types of aim constraints.
template <typename T>
class AimBaseErrorFunctionT : public ConstraintErrorFunctionT<T, AimDataT<T>, 3, 2, 1> {
 protected:
  explicit AimBaseErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha,
      const T& lossC)
      : ConstraintErrorFunctionT<T, AimDataT<T>, 3, 2, 1>(skel, pt, lossAlpha, lossC) {}
};

/// The AimDistErrorFunction minimizes the distance between a ray (origin, direction) defined in
/// joint-local space and the world-space target point.   The residual is defined as the distance
/// between the tartet position and its projection onto the ray.  Note that the ray is only defined
/// for positive t values, meaning that if the target point is _behind_ the ray origin, its
/// projection will be at the ray origin where t=0.
template <typename T>
class AimDistErrorFunctionT : public AimBaseErrorFunctionT<T> {
 public:
  explicit AimDistErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : AimBaseErrorFunctionT<T>(skel, pt, lossAlpha, lossC) {}

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
      optional_ref<std::array<Vector3<T>, 2>> v = {},
      optional_ref<std::array<Eigen::Matrix3<T>, 2>> dfdv = {}) const final;
};

/// The AimDirErrorFunction minimizes the element-wise difference between a ray (origin, direction)
/// defined in joint-local space and the normalized vector connecting the ray origin to the
/// world-space target point.  If the vector has near-zero length, the residual is set to zero to
/// avoid divide-by-zero.
template <typename T>
class AimDirErrorFunctionT : public AimBaseErrorFunctionT<T> {
 public:
  explicit AimDirErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : AimBaseErrorFunctionT<T>(skel, pt, lossAlpha, lossC) {}

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
      optional_ref<std::array<Vector3<T>, 2>> v = {},
      optional_ref<std::array<Eigen::Matrix3<T>, 2>> dfdv = {}) const final;
};
} // namespace momentum
