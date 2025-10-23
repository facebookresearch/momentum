/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>

#include <unordered_map>

namespace momentum {

template <typename T>
class UnionErrorFunctionT : public SkeletonErrorFunctionT<T>,
                            public virtual FullyDifferentiableSkeletonErrorFunctionT<T> {
 public:
  UnionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>& errorFunctions);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;
  [[nodiscard]] size_t getJacobianSize() const final;

  std::vector<std::string> inputs() const override;
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> result) const override;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) override;
  Eigen::Index getInputSize(const std::string& name) const override;
  const char* name() const override {
    return name_.c_str();
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& modelParams,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) override;

 private:
  std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> errorFunctions_;
  std::vector<std::shared_ptr<FullyDifferentiableSkeletonErrorFunctionT<T>>> diffErrorFunctions_;
  std::string name_;

  // For each input, which error functions support that input
  std::unordered_map<std::string, std::vector<size_t>> inputs_;
};

} // namespace momentum
