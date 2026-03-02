/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <momentum/character/locator.h>
#include <momentum/character_solver/projection_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiableProjectionErrorFunction);

template <typename T>
class FullyDifferentiableProjectionErrorFunctionT
    : public momentum::ProjectionErrorFunctionT<T>,
      public virtual momentum::FullyDifferentiableSkeletonErrorFunctionT<T> {
 public:
  static constexpr const char* kParents = "parents";
  static constexpr const char* kOffsets = "offsets";
  static constexpr const char* kWeights = "weights";
  static constexpr const char* kTargets = "targets";
  static constexpr const char* kProjections = "projections";

  FullyDifferentiableProjectionErrorFunctionT(
      const momentum::Skeleton& skel,
      const momentum::ParameterTransform& pt,
      T nearClip = T(1));

  [[nodiscard]] std::vector<std::string> inputs() const override;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const override;
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> result) const override;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) override;
  [[nodiscard]] const char* name() const override {
    return "ProjectionErrorFunction";
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const momentum::ModelParametersT<T>& modelParams,
      const momentum::SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) override;

 protected:
  template <typename JetType>
  JetType constraintGradient_dot(
      const momentum::ModelParametersT<T>& modelParams,
      const momentum::SkeletonStateT<T>& skelState,
      int parentJointIndex_cons,
      const Eigen::Vector3<JetType>& offset_cons,
      const JetType& weight_cons,
      const Eigen::Matrix<JetType, 3, 4>& projection_cons,
      const Eigen::Vector2<JetType>& target_cons,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) const;

  using momentum::ProjectionErrorFunctionT<T>::_nearClip;
  using momentum::ProjectionErrorFunctionT<T>::kProjectionWeight;
};

} // namespace momentum
