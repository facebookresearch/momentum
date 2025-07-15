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

namespace momentum {

template <typename T>
class SkeletonSolverFunctionT : public SolverFunctionT<T> {
 public:
  SkeletonSolverFunctionT(const Skeleton* skel, const ParameterTransformT<T>* parameterTransform);

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

  [[nodiscard]] const Skeleton* getSkeleton() const {
    return skeleton_;
  }
  [[nodiscard]] const ParameterTransformT<T>* getParameterTransform() const {
    return parameterTransform_;
  }

 private:
  const Skeleton* skeleton_;
  const ParameterTransformT<T>* parameterTransform_;
  std::unique_ptr<SkeletonStateT<T>> state_;
  VectorX<bool> activeJointParams_;

  Eigen::MatrixX<T> tJacobian_;
  Eigen::VectorX<T> tResidual_;

  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions_;
};

} // namespace momentum
