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

#include <string>
#include <vector>

namespace momentum {

/// Constraint data for a center of mass target.
///
/// Represents a target position for the center of mass (CoM) computed from
/// a weighted sum of joint positions. Each constraint can specify a different
/// set of joints and masses to allow for partial body CoM computation.
///
/// @section Example
/// @code
/// CenterOfMassConstraintT<float> constraint;
/// constraint.jointIndices = {0, 1, 2, 3};  // Spine joints
/// constraint.masses = {1.0f, 2.0f, 1.5f, 1.0f};
/// constraint.target = Eigen::Vector3f(0, 1.0f, 0);  // Target CoM position
/// constraint.weight = 1.0f;
/// @endcode
template <typename T>
struct CenterOfMassConstraintT {
  /// Indices of joints contributing to this CoM constraint
  std::vector<size_t> jointIndices;

  /// Mass values for each joint (must match jointIndices size)
  std::vector<T> masses;

  /// Target position for the center of mass
  Eigen::Vector3<T> target = Eigen::Vector3<T>::Zero();

  /// Weight of this constraint
  T weight = T(1);

  /// Optional name for debugging
  std::string name;

  /// Returns the total mass (sum of all masses)
  [[nodiscard]] T totalMass() const;
};

/// Error function that constrains the center of mass to a target position.
///
/// The center of mass is computed as:
///   CoM = (Σ m_k * p_k) / (Σ m_k)
///
/// where m_k is the mass at joint k and p_k is the world position of joint k.
///
/// The residual is:
///   f = CoM - target
///
/// This uses the MultiSourceErrorFunctionT base class with N JointPoint contributions,
/// one per joint, demonstrating the architecture for new error functions.
///
/// @section Usage
/// @code
/// CenterOfMassErrorFunctionT<float> errorFn(skeleton, parameterTransform);
///
/// CenterOfMassConstraintT<float> constraint;
/// constraint.jointIndices = {0, 1, 2};
/// constraint.masses = {10.0f, 5.0f, 3.0f};
/// constraint.target = Eigen::Vector3f(0, 1.0f, 0);
/// constraint.weight = 1.0f;
///
/// errorFn.addConstraint(constraint);
/// @endcode
///
/// @tparam T Scalar type (float or double)
template <typename T>
class CenterOfMassErrorFunctionT : public MultiSourceErrorFunctionT<T, 3> {
 public:
  using typename MultiSourceErrorFunctionT<T, 3>::FuncType;
  using typename MultiSourceErrorFunctionT<T, 3>::DfdvType;
  using ConstraintType = CenterOfMassConstraintT<T>;

  /// Constructs the error function.
  ///
  /// @param skeleton The skeleton structure
  /// @param parameterTransform Maps joint parameters to model parameters
  CenterOfMassErrorFunctionT(
      const Skeleton& skeleton,
      const ParameterTransform& parameterTransform);

  ~CenterOfMassErrorFunctionT() override = default;

  /// Adds a center of mass constraint.
  ///
  /// @param constraint The constraint to add
  void addConstraint(const ConstraintType& constraint);

  /// Sets all constraints, replacing any existing ones.
  ///
  /// @param constraints The new list of constraints
  void setConstraints(std::span<const ConstraintType> constraints);

  /// Clears all constraints.
  void clearConstraints();

  /// Returns the current constraints.
  [[nodiscard]] const std::vector<ConstraintType>& getConstraints() const;

  /// Returns the number of constraints.
  [[nodiscard]] size_t getNumConstraints() const override;

  [[nodiscard]] T getConstraintWeight(size_t constrIndex) const override;

  /// Returns the contributions for a constraint.
  ///
  /// For CoM, each joint is a JointPoint contribution at the joint's origin.
  [[nodiscard]] std::span<const SourceT<T>> getContributions(size_t constrIndex) const override;

  /// Computes the residual and derivatives.
  ///
  /// The residual is f = CoM - target, where CoM is the mass-weighted average
  /// of joint positions.
  ///
  /// @param constrIndex Index of the constraint
  /// @param state Current skeleton state
  /// @param meshState Current mesh state (unused)
  /// @param worldVecs Pre-computed world positions of all joints in this constraint
  /// @param f Output: residual (3x1)
  /// @param dfdv Output: derivative df/dv for each joint (scaled by normalized mass)
  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const override;

 private:
  std::vector<ConstraintType> constraints_;

  // Cached contributions per constraint (built when constraints are added)
  mutable std::vector<std::vector<SourceT<T>>> contributionsCache_;
  mutable bool contributionsCacheValid_ = false;

  void rebuildContributionsCache() const;
};

} // namespace momentum
