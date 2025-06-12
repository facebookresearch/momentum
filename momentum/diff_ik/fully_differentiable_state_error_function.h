/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

#include <Eigen/Core>

namespace momentum {

template <typename T>
class FullyDifferentiableStateErrorFunctionT : public FullyDifferentiableSkeletonErrorFunctionT<T>,
                                               public StateErrorFunctionT<T> {
 public:
  FullyDifferentiableStateErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  ~FullyDifferentiableStateErrorFunctionT() override;

  [[nodiscard]] std::vector<std::string> inputs() const final;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const final;
  [[nodiscard]] const char* name() const final {
    return "StateErrorFunction";
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) final;

 protected:
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const final;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) final;

 private:
  template <typename JetType>
  JetType calculateGradient_dot(
      const SkeletonStateT<T>& state,
      size_t iJoint,
      const Eigen::Vector3<JetType>& targetTranslation,
      const Eigen::Quaternion<JetType>& targetRotation,
      const JetType& targetPositionWeight,
      const JetType& targetRotationWeight,
      Eigen::Ref<const Eigen::VectorX<T>> vec);
};

} // namespace momentum
