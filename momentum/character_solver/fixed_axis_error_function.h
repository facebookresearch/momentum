/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// Axis constraint data
template <typename T>
struct FixedAxisDataT : ConstraintData {
  /// axis in the parent's coordinate frame
  Vector3<T> localAxis;
  /// target axis in the global frame
  Vector3<T> globalAxis;

  explicit FixedAxisDataT(
      const Vector3<T>& local,
      const Vector3<T>& global,
      size_t pIndex,
      float w,
      const std::string& n = "")
      : ConstraintData(pIndex, w, n),
        localAxis(local.normalized()),
        globalAxis(global.normalized()) {}
};

// The fixed angle error minimizes the difference between a local axis and a target direction in the
// global frame. These error functions differ in how they calculate the difference between two axis.

/// Diff: compute the element wise differences between the two axis
template <typename T>
class FixedAxisDiffErrorFunctionT : public JointErrorFunctionT<T, FixedAxisDataT<T>, 3, 1, 0> {
 public:
  explicit FixedAxisDiffErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, FixedAxisDataT<T>, 3, 1, 0>(skel, pt, lossAlpha, lossC) {}

  explicit FixedAxisDiffErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : FixedAxisDiffErrorFunctionT<T>(
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
      Vector3<T>& f,
      std::array<Vector3<T>, 1>& v,
      std::array<Matrix3<T>, 1>& dfdv) const final;
};

/// Cos: compute the dot product between the two axis - cosine of the angle between them
template <typename T>
class FixedAxisCosErrorFunctionT : public JointErrorFunctionT<T, FixedAxisDataT<T>, 1, 1, 0> {
 public:
  explicit FixedAxisCosErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, FixedAxisDataT<T>, 1, 1, 0>(skel, pt, lossAlpha, lossC) {}

  explicit FixedAxisCosErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : FixedAxisCosErrorFunctionT<T>(
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
      Vector<T, 1>& f,
      std::array<Vector3<T>, 1>& v,
      std::array<Eigen::Matrix<T, 1, 3>, 1>& dfdv) const final;
};

/// Angle: compute the angle

/// Angle: compute the angle between the two axis using acosine of their dot product.
template <typename T>
class FixedAxisAngleErrorFunctionT : public JointErrorFunctionT<T, FixedAxisDataT<T>, 1, 1, 0> {
 public:
  explicit FixedAxisAngleErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : JointErrorFunctionT<T, FixedAxisDataT<T>, 1, 1, 0>(skel, pt, lossAlpha, lossC) {}

  explicit FixedAxisAngleErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : FixedAxisAngleErrorFunctionT<T>(
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
      Vector<T, 1>& f,
      std::array<Vector3<T>, 1>& v,
      std::array<Eigen::Matrix<T, 1, 3>, 1>& dfdv) const final;
};

} // namespace momentum
