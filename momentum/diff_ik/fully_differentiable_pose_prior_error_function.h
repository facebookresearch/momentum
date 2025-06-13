/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

#include <Eigen/Core>

namespace momentum {

template <typename T>
class FullyDifferentiablePosePriorErrorFunctionT
    : public FullyDifferentiableSkeletonErrorFunctionT<T>,
      public PosePriorErrorFunctionT<T> {
 public:
  FullyDifferentiablePosePriorErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      std::vector<std::string> names);
  ~FullyDifferentiablePosePriorErrorFunctionT() override;

  void setPosePrior(
      const VectorX<T>& pi,
      const MatrixX<T>& mmu,
      const std::vector<MatrixX<T>>& W,
      const VectorX<T>& sigma);

  static constexpr const char* kPi = "pi";
  static constexpr const char* kMu = "mu";
  static constexpr const char* kW = "W";
  static constexpr const char* kSigma = "sigma";
  static constexpr const char* kParameterIndices = "parameter_indices";

  [[nodiscard]] std::vector<std::string> inputs() const final;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const final;
  [[nodiscard]] const char* name() const final {
    return "MarkerErrorFunction";
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) final;

 protected:
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const final;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) final;

  // pi_ are the mixture weights
  Eigen::VectorX<T> pi_;

  // The columns of mu are the means for the different mixtures
  Eigen::MatrixX<T> mmu_;

  // Each matrix W is the basis for a mixture:
  std::vector<Eigen::MatrixX<T>> W_;

  // The covariance matrix is (sigma^2 * I + W^T * W)^{-1}
  // Note that this is sigma, not sigma squared, because the scaling should
  // be more reasonable (sigma is the same order of magnitude as the mu and
  // W arrays).
  Eigen::VectorX<T> sigma_;

  const std::vector<std::string> names_;
};

} // namespace momentum
