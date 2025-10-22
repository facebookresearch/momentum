/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skinned_locator.h>
#include <momentum/character_solver/error_function_utils.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/plane_error_function.h>
#include <momentum/character_solver/position_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/fwd.h>

namespace momentum {

/// Constraint structure for skinned locator error function.
///
/// Represents a constraint that pulls a skinned locator towards a target position
/// in world space with a specified weight.
template <typename T>
struct SkinnedLocatorConstraintT {
  int locatorIndex = -1;
  T weight = 1;
  Eigen::Vector3<T> targetPosition;

  template <typename T2>
  SkinnedLocatorConstraintT<T2> cast() const {
    return {
        this->locatorIndex,
        static_cast<T2>(this->weight),
        this->targetPosition.template cast<T2>()};
  }
};

/// Error function for skinned locator constraints.
///
/// This error function computes the error between skinned locator positions and their
/// target positions. Skinned locators are points that are deformed by the character's
/// skeleton and can be used to constrain the character to specific world positions.
/// The error is computed as the weighted squared distance between the current skinned
/// locator position and the target position.
template <typename T>
class SkinnedLocatorErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  /// Constructs a skinned locator error function for the given character.
  explicit SkinnedLocatorErrorFunctionT(const Character& character_in);
  ~SkinnedLocatorErrorFunctionT() override;

  SkinnedLocatorErrorFunctionT(const SkinnedLocatorErrorFunctionT<T>& other) = delete;
  SkinnedLocatorErrorFunctionT<T>& operator=(const SkinnedLocatorErrorFunctionT<T>& other) = delete;
  SkinnedLocatorErrorFunctionT(SkinnedLocatorErrorFunctionT<T>&& other) = delete;
  SkinnedLocatorErrorFunctionT<T>& operator=(SkinnedLocatorErrorFunctionT<T>&& other) = delete;

  /// Computes the total error for all skinned locator constraints.
  ///
  /// @param modelParameters The current model parameters
  /// @param state The current skeleton state
  /// @return The sum of weighted squared distances between locators and targets
  [[nodiscard]] double getError(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */) final;

  /// Computes the gradient of the error function with respect to model parameters.
  ///
  /// @param modelParameters The current model parameters
  /// @param state The current skeleton state
  /// @param gradient Output gradient vector to be filled
  /// @return The total error value
  double getGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  /// Computes the Jacobian matrix and residual vector for the error function.
  ///
  /// @param modelParameters The current model parameters
  /// @param state The current skeleton state
  /// @param jacobian Output Jacobian matrix to be filled
  /// @param residual Output residual vector to be filled
  /// @param usedRows Number of rows used in the Jacobian (output parameter)
  /// @return The total error value
  double getJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  /// Returns the number of rows needed for the Jacobian matrix.
  [[nodiscard]] size_t getJacobianSize() const final;

  /// Adds a new skinned locator constraint.
  ///
  /// @param locatorIndex Index of the skinned locator in the character
  /// @param weight Weight for this constraint
  /// @param targetPosition Target position in world space
  void addConstraint(int locatorIndex, T weight, const Eigen::Vector3<T>& targetPosition);

  /// Removes all constraints.
  void clearConstraints();

  /// Sets all constraints at once, replacing any existing constraints.
  ///
  /// @param constraints Vector of constraints to set
  void setConstraints(const std::vector<SkinnedLocatorConstraintT<T>>& constraints);

  /// Returns the current list of constraints.
  [[nodiscard]] const std::vector<SkinnedLocatorConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Returns the character associated with this error function.
  [[nodiscard]] const Character* getCharacter() const override {
    return &character_;
  }

 private:
  // Get the parameter index for a skinned locator, or -1 if not parameterized
  [[nodiscard]] int getSkinnedLocatorParameterIndex(int locatorIndex) const {
    if (locatorIndex <
        static_cast<int>(character_.parameterTransform.skinnedLocatorParameters.size())) {
      return character_.parameterTransform.skinnedLocatorParameters[locatorIndex];
    }
    return -1;
  }

  // Calculate the derivative of the world position with respect to the rest position
  void calculateDWorldPos(
      const SkeletonStateT<T>& state,
      const SkinnedLocatorConstraintT<T>& constr,
      const Eigen::Vector3<T>& d_restPos,
      Eigen::Vector3<T>& d_worldPos) const;
  double calculatePositionJacobian(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const SkinnedLocatorConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  double calculatePositionGradient(
      const ModelParametersT<T>& modelParameters,
      const SkeletonStateT<T>& state,
      const SkinnedLocatorConstraintT<T>& constr,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  // Calculate the skinned locator position in world space
  Eigen::Vector3<T> calculateSkinnedLocatorPosition(
      const SkeletonStateT<T>& state,
      int locatorIndex,
      const Eigen::Vector3<T>& locatorRestPos) const;
  Eigen::Vector3<T> getLocatorRestPosition(const ModelParametersT<T>& modelParams, int locatorIndex)
      const;

  const Character& character_;

  std::vector<SkinnedLocatorConstraintT<T>> constraints_;
};

} // namespace momentum
