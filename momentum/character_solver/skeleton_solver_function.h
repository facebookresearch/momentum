/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/solver/solver_function.h>
#include <span>

namespace momentum {

template <typename T>
class SkeletonSolverFunctionT : public SolverFunctionT<T> {
 public:
  SkeletonSolverFunctionT(
      const Character& character,
      const ParameterTransformT<T>& parameterTransform,
      std::span<const std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions = {});
  ~SkeletonSolverFunctionT() override;

  double getError(const Eigen::VectorX<T>& parameters) final;

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) final;

  double getJacobian(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& jacobian,
      Eigen::VectorX<T>& residual,
      size_t& actualRows) final;

  double getJtJR(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& jtj,
      Eigen::VectorX<T>& jtr) override;

  // overriding this to get a mix of JtJs and analytical Hessians from skeleton_ errorFunctions_
  double getSolverDerivatives(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& hess,
      Eigen::VectorX<T>& grad) override;

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& delta) final;
  void setEnabledParameters(const ParameterSet& ps) final;

  void addErrorFunction(std::shared_ptr<SkeletonErrorFunctionT<T>> solvable);
  void clearErrorFunctions();

  [[nodiscard]] const std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>& getErrorFunctions()
      const;

  /// Provides access to the full character (replaces getSkeleton and getParameterTransform)
  [[nodiscard]] const Character& getCharacter() const {
    return character_;
  }

  /// Legacy compatibility (delegates to character_.skeleton)
  [[nodiscard]] const Skeleton* getSkeleton() const {
    return &character_.skeleton;
  }

  /// Provides access to the parameter transform
  [[nodiscard]] const ParameterTransformT<T>* getParameterTransform() const {
    return &parameterTransform_;
  }

  /// Updates mesh state if any error functions require it
  void updateMeshState(const ModelParametersT<T>& parameters, const SkeletonStateT<T>& state);

  /// Checks if any error functions require mesh state
  [[nodiscard]] bool needsMeshState() const;

 private:
  const Character& character_;
  const ParameterTransformT<T>& parameterTransform_;
  std::unique_ptr<SkeletonStateT<T>> state_;
  std::unique_ptr<MeshStateT<T>> meshState_;
  bool needsMeshState_;
  VectorX<bool> activeJointParams_;

  Eigen::MatrixX<T> tJacobian_;
  Eigen::VectorX<T> tResidual_;

  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions_;
};

} // namespace momentum
