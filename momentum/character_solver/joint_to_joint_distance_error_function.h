/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/multi_source_error_function.h>

#include <Eigen/Core>

#include <vector>

namespace momentum {

/// Constraint that enforces a distance relationship between two points attached to different
/// joints.
///
/// Each point is specified by a joint index and an offset in the local coordinate system of that
/// joint.
template <typename T>
struct JointToJointDistanceConstraintT {
  /// Index of the first joint.
  size_t joint1 = kInvalidIndex;

  /// Offset from joint1 in the local coordinate system of joint1.
  Vector3<T> offset1 = Vector3<T>::Zero();

  /// Index of the second joint.
  size_t joint2 = kInvalidIndex;

  /// Offset from joint2 in the local coordinate system of joint2.
  Vector3<T> offset2 = Vector3<T>::Zero();

  /// Target distance between the two points (in world space).
  T targetDistance = T(0);

  /// Weight for this constraint.
  T weight = T(1);

  template <typename T2>
  JointToJointDistanceConstraintT<T2> cast() const {
    return {
        this->joint1,
        this->offset1.template cast<T2>(),
        this->joint2,
        this->offset2.template cast<T2>(),
        static_cast<T2>(this->targetDistance),
        static_cast<T2>(this->weight)};
  }
};

/// Error function that penalizes deviation from a target distance between two points attached to
/// different joints.
///
/// This is useful for enforcing distance constraints between different parts of a character, such
/// as maintaining a fixed distance between hands or ensuring two joints stay a certain distance
/// apart.
///
/// The residual is:
///   f = ||p1 - p2|| - targetDistance
///
/// where p1 and p2 are the world-space positions of the two joint points.
///
/// @tparam T Scalar type (float or double)
template <typename T>
class JointToJointDistanceErrorFunctionT : public MultiSourceErrorFunctionT<T, 1> {
 public:
  using typename MultiSourceErrorFunctionT<T, 1>::FuncType;
  using typename MultiSourceErrorFunctionT<T, 1>::DfdvType;
  using ConstraintType = JointToJointDistanceConstraintT<T>;

  explicit JointToJointDistanceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);

  explicit JointToJointDistanceErrorFunctionT(const Character& character);

  ~JointToJointDistanceErrorFunctionT() override = default;

  /// Add a constraint between two joints with local offsets.
  void addConstraint(
      size_t joint1,
      const Vector3<T>& offset1,
      size_t joint2,
      const Vector3<T>& offset2,
      T targetDistance,
      T weight = T(1));

  /// Clear all constraints.
  void clearConstraints();

  /// Get all constraints.
  [[nodiscard]] const std::vector<ConstraintType>& getConstraints() const {
    return constraints_;
  }

  /// Returns the number of constraints.
  [[nodiscard]] size_t getNumConstraints() const override;

  /// Returns the per-constraint weight, including the kDistanceWeight factor.
  /// The total weight applied is: constr.weight * kDistanceWeight * this->weight_
  [[nodiscard]] T getConstraintWeight(size_t constrIndex) const override;

  /// Returns the contributions for a constraint (2 JointPoint contributions).
  [[nodiscard]] std::span<const SourceT<T>> getContributions(size_t constrIndex) const override;

  /// Computes the residual and derivatives.
  ///
  /// The residual is f = ||p1 - p2|| - targetDistance.
  ///
  /// @param constrIndex Index of the constraint
  /// @param state Current skeleton state
  /// @param meshState Current mesh state (unused)
  /// @param worldVecs Pre-computed world positions of the two joint points
  /// @param f Output: residual (1x1)
  /// @param dfdv Output: derivative df/dv for each joint point
  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const override;

  /// Default weight for distance constraints.
  static constexpr T kDistanceWeight = 1e-2f;

 private:
  std::vector<ConstraintType> constraints_;

  // Cached contributions per constraint (2 JointPoints per constraint)
  mutable std::vector<std::array<SourceT<T>, 2>> contributionsCache_;
  mutable bool contributionsCacheValid_ = false;

  void rebuildContributionsCache() const;
};

} // namespace momentum
