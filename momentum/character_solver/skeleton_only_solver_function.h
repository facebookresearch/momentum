/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/solver/solver_function.h>

#include <memory>
#include <vector>

namespace momentum {

/// Solver function on a skeleton and parameter transform alone, for callers without a mesh.
///
/// Sums the skeleton error functions added to it, as SkeletonSolverFunctionT does, but takes no
/// Character, so it links none of the mesh deformation (skinning) code. Error functions that need a
/// mesh are not supported. Optional per-parameter bounds end each update on [lower, upper].
///
/// It solves as SkeletonSolverFunctionT without line search: getError() returns the unrounded sum,
/// where SkeletonSolverFunctionT rounds it to float. It does not override getSolverDerivatives(),
/// which GaussNewtonSolverT does not use.
template <typename T>
class SkeletonOnlySolverFunctionT : public SolverFunctionT<T> {
 public:
  /// @param skeleton The skeleton the error functions are evaluated on; must outlive this object.
  /// @param parameterTransform The skeleton's parameter transform; must outlive this object.
  SkeletonOnlySolverFunctionT(
      const Skeleton& skeleton,
      const ParameterTransformT<T>& parameterTransform);
  ~SkeletonOnlySolverFunctionT() override;

  SkeletonOnlySolverFunctionT(const SkeletonOnlySolverFunctionT&) = delete;
  SkeletonOnlySolverFunctionT& operator=(const SkeletonOnlySolverFunctionT&) = delete;
  SkeletonOnlySolverFunctionT(SkeletonOnlySolverFunctionT&&) = delete;
  SkeletonOnlySolverFunctionT& operator=(SkeletonOnlySolverFunctionT&&) = delete;

  double getError(const Eigen::VectorX<T>& parameters) final;

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) final;

  void initializeJacobianComputation(const Eigen::VectorX<T>& parameters) final;
  [[nodiscard]] size_t getJacobianBlockCount() const final;
  [[nodiscard]] size_t getJacobianBlockSize(size_t blockIndex) const final;
  double computeJacobianBlock(
      const Eigen::VectorX<T>& parameters,
      size_t blockIndex,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
      Eigen::Ref<Eigen::VectorX<T>> residualBlock,
      size_t& actualRows) final;

  /// Subtracts @p delta, then ends each parameter on its bounds when bounds are set.
  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& delta) final;
  void setEnabledParameters(const ParameterSet& ps) final;

  /// Adds an error function; throws if it needs a mesh.
  void addErrorFunction(std::shared_ptr<SkeletonErrorFunctionT<T>> errorFunction);

  /// Ends each update on [lower, upper], one entry per model parameter (infinite for unbounded).
  /// Throws unless lower <= upper for every parameter (NaN included).
  void setParameterBounds(Eigen::VectorX<T> lower, Eigen::VectorX<T> upper);

 private:
  const Skeleton& skeleton_;
  const ParameterTransformT<T>& parameterTransform_;
  std::unique_ptr<SkeletonStateT<T>> state_;
  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions_;
  Eigen::VectorX<T> lowerBounds_;
  Eigen::VectorX<T> upperBounds_;
};

} // namespace momentum
