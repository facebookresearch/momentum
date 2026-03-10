/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_derivative.h>
#include <momentum/character_solver/skeleton_error_function.h>

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
///   CoM = (sum m_k * p_k) / (sum m_k)
///
/// where m_k is the mass at joint k and p_k is the world position of joint k.
///
/// The residual is:
///   f = CoM - target
///
/// This uses SkeletonDerivativeT for efficient hierarchy walks, similar to
/// CameraProjectionErrorFunctionT.
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
class CenterOfMassErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  using ConstraintType = CenterOfMassConstraintT<T>;

  /// Constructs the error function.
  ///
  /// @param skeleton The skeleton structure
  /// @param parameterTransform Maps joint parameters to model parameters
  CenterOfMassErrorFunctionT(
      const Skeleton& skeleton,
      const ParameterTransform& parameterTransform);

  ~CenterOfMassErrorFunctionT() override = default;

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

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
  [[nodiscard]] size_t getNumConstraints() const;

 private:
  std::vector<ConstraintType> constraints_;
  SkeletonDerivativeT<T> skeletonDerivative_;
};

} // namespace momentum
