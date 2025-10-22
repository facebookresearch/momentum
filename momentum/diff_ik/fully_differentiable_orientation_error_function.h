/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

#include <vector>

namespace momentum {

template <typename T>
struct OrientationConstraintT {
  Quaternion<T> offset; ///< Constraint rotation offset in local joint space
  Quaternion<T> target; ///< Target orientation in global space
  size_t parent; ///< Parent joint index this constraint is under
  float weight; ///< Weight of the constraint
  std::string name; ///< Name of the constraint

  explicit OrientationConstraintT(
      const Quaternion<T>& offset,
      const Quaternion<T>& target,
      size_t parent,
      float w,
      const std::string& n = "")
      : offset(offset), target(target), parent(parent), weight(w), name(n) {
    // Empty
  }
};

template <typename T>
class FullyDifferentiableOrientationErrorFunctionT
    : public FullyDifferentiableSkeletonErrorFunctionT<T>,
      public SkeletonErrorFunctionT<T> {
 public:
  static constexpr T kOrientationWeight = 1e-1f;
  static constexpr const char* kParents = "parents";
  static constexpr const char* kOffsets = "offsets";
  static constexpr const char* kWeights = "weights";
  static constexpr const char* kTargets = "targets";

  FullyDifferentiableOrientationErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  ~FullyDifferentiableOrientationErrorFunctionT() override = default;

  [[nodiscard]] std::vector<std::string> inputs() const final;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const final;
  [[nodiscard]] const char* name() const final {
    return "OrientationErrorFunction";
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) final;

  double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<Eigen::MatrixX<T>> jacobian,
      Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void addConstraint(const OrientationConstraintT<T>& constr);

 private:
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const final;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) final;

  T calculateOrientationGradient(
      const SkeletonStateT<T>& state,
      const OrientationConstraintT<T>& constr,
      Eigen::VectorX<T>& jGrad) const;

  template <typename JetType>
  JetType calculateOrientationGradient_dot(
      const SkeletonStateT<T>& state,
      size_t constrParent,
      const JetType& constrWeight,
      const Eigen::Quaternion<JetType>& constrOrientationOffset,
      const Eigen::Quaternion<JetType>& constrOrientationTarget,
      Eigen::Ref<const Eigen::VectorX<T>> vec) const;

  [[nodiscard]] T calculateOrientationJacobian(
      const SkeletonStateT<T>& state,
      const OrientationConstraintT<T>& constr,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  std::vector<OrientationConstraintT<T>> constraints_;
  Eigen::VectorX<T> jointGrad_;
};

} // namespace momentum
