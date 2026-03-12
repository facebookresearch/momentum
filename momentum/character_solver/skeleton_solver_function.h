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
#include <momentum/character/skeleton_state.h>
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

  // Block-wise Jacobian interface (getJacobian and getJtJR now use these via parent class)
  void initializeJacobianComputation(const Eigen::VectorX<T>& parameters) override;
  [[nodiscard]] size_t getJacobianBlockCount() const override;
  [[nodiscard]] size_t getJacobianBlockSize(size_t blockIndex) const override;
  double computeJacobianBlock(
      const Eigen::VectorX<T>& parameters,
      size_t blockIndex,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
      Eigen::Ref<Eigen::VectorX<T>> residualBlock,
      size_t& actualRows) override;
  void finalizeJacobianComputation() override;

  // overriding this to get a mix of JtJs and analytical Hessians from skeleton_ errorFunctions_
  double getSolverDerivatives(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& hess,
      Eigen::VectorX<T>& grad) override;

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& delta) final;
  void setEnabledParameters(const ParameterSet& ps) final;
  void initialize() override;

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

  /// True when all skeleton joints need both transforms and derivatives.
  /// Enables a fast path that bypasses per-joint bitset checks.
  /// Defaults to true so that before initialize() is called, the baseline code path is used.
  bool allJointsActive_{true};

  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions_;

  /// Updates the skeleton state; inlined so the compiler can eliminate the branch
  /// in the common case where allJointsActive_ is true.
  void updateSkeletonState(const Eigen::VectorX<T>& parameters, bool computeDeriv) {
    if (allJointsActive_) {
      // Fast path: identical to baseline — no per-joint bitset checks.
      state_->set(parameterTransform_.apply(parameters), character_.skeleton, computeDeriv);
    } else {
      updateSkeletonStateSelective(parameters, computeDeriv);
    }
  }

  /// Slow path for selective joint computation; outlined to keep the fast path lean.
  void updateSkeletonStateSelective(const Eigen::VectorX<T>& parameters, bool computeDeriv);

  /// Joints that need transforms in any solver computation; computed once at initialization
  JointSet activeJointXform_;

  /// Joints that need derivatives in any solver computation; computed once at initialization
  JointSet activeJointDeriv_;

  JointSet solverActiveJoints_;
  ParameterSet enabledParameters_;
};

} // namespace momentum
