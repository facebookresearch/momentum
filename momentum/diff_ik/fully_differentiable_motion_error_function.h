/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/model_parameters_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

#include <Eigen/Core>

namespace momentum {

template <typename T>
class FullyDifferentiableMotionErrorFunctionT : public FullyDifferentiableSkeletonErrorFunctionT<T>,
                                                public ModelParametersErrorFunctionT<T> {
 public:
  FullyDifferentiableMotionErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  ~FullyDifferentiableMotionErrorFunctionT() override;

  [[nodiscard]] std::vector<std::string> inputs() const final;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const final;
  [[nodiscard]] const char* name() const final {
    return "ModelParametersErrorFunction";
  }

  static constexpr const char* kTargetParameters = "target_parameters";
  static constexpr const char* kTargetWeights = "target_weights";

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) final;

 protected:
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const final;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) final;
};

} // namespace momentum
