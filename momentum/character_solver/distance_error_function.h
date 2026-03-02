/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/locator.h>
#include <momentum/character_solver/joint_error_function.h>

namespace momentum {

/// Distance constraint data for the DistanceErrorFunction.
/// The constraint minimizes |distance(p_joint, origin) - target|^2
template <typename T>
struct DistanceConstraintDataT : ConstraintData {
  /// Origin point in world space
  Vector3<T> origin = Vector3<T>::Zero();
  /// Target distance in world space
  T target{};
  /// Relative offset to the parent joint
  Vector3<T> offset = Vector3<T>::Zero();

  /// Default constructor
  DistanceConstraintDataT() : ConstraintData(kInvalidIndex, 0.0f) {}

  /// Constructor with explicit parameters
  explicit DistanceConstraintDataT(
      const Vector3<T>& inOffset,
      const Vector3<T>& inOrigin,
      const T inTarget,
      size_t pIndex,
      float w,
      const std::string& n = "")
      : ConstraintData(pIndex, w, n), origin(inOrigin), target(inTarget), offset(inOffset) {}

  /// Factory method to create constraint from a locator
  static DistanceConstraintDataT<T> createFromLocator(const momentum::Locator& locator);
};

/// The DistanceErrorFunction computes the distance error between a point on the skeleton and a
/// target origin. The error is the difference between the actual distance and the target distance.
///
/// f = distance(T * offset, origin) - target
///
/// where T is the global transformation of the parent joint, offset is the local offset in joint
/// space, origin is the target point in world space, and target is the desired distance.
template <typename T>
class DistanceErrorFunctionT : public JointErrorFunctionT<T, DistanceConstraintDataT<T>, 1, 1, 1> {
 public:
  using Base = JointErrorFunctionT<T, DistanceConstraintDataT<T>, 1, 1, 1>;
  using typename Base::DfdvType;
  using typename Base::FuncType;
  using typename Base::VType;

  /// Constructor
  ///
  /// @param[in] skel: character skeleton
  /// @param[in] pt: parameter transformation
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit DistanceErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : Base(skel, pt, lossAlpha, lossC) {}

  /// A convenience constructor where character contains info of the skeleton and parameter
  /// transform.
  ///
  /// @param[in] character: character definition
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit DistanceErrorFunctionT(
      const Character& character,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : DistanceErrorFunctionT(character.skeleton, character.parameterTransform, lossAlpha, lossC) {
  }

  /// Default constant weight in DistanceErrorFunction. This can be used for backwards
  /// compatibility in setWeight().
  static constexpr T kLegacyWeight = 1.0f;

  /// Distance weight constant for backward compatibility with diff_ik code.
  static constexpr T kDistanceWeight = 1.0f;

  /// Returns true if there are no constraints.
  /// @return true if the constraint list is empty
  [[nodiscard]] bool empty() const {
    return this->constraints_.empty();
  }

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      FuncType& f,
      std::array<VType, 1>& v,
      std::array<DfdvType, 1>& dfdv) const final;
};

} // namespace momentum
