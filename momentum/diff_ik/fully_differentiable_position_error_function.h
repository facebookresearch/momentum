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
struct PositionConstraintT {
  Vector3<T> offset; ///< Positional offset in the parent joint space
  Vector3<T> target; ///< Target position
  size_t parent; ///< Parent joint index this constraint is under
  float weight; ///< Weight of the constraint
  std::string name; ///< Name of the constraint

  explicit PositionConstraintT(
      const Vector3<T>& offset,
      const Vector3<T>& target,
      size_t parent,
      float w,
      const std::string& n = "")
      : offset(offset), target(target), parent(parent), weight(w), name(n) {
    // Empty
  }
};

template <typename T>
struct PositionConstraintStateT {
  std::vector<Eigen::Vector3<T>>
      position; // the current position of all locators according to a given skeleton state

  PositionConstraintStateT() = default;

  PositionConstraintStateT(
      const SkeletonStateT<T>& skeletonState,
      const std::vector<PositionConstraintT<T>>& referenceConstraints) {
    update(skeletonState, referenceConstraints);
  }

  void update(
      const SkeletonStateT<T>& skeletonState,
      const std::vector<PositionConstraintT<T>>& referenceConstraints);
};

template <typename T>
class FullyDifferentiablePositionErrorFunctionT
    : public FullyDifferentiableSkeletonErrorFunctionT<T>,
      public SkeletonErrorFunctionT<T> {
 public:
  static constexpr T kPositionWeight = 1e-4f;
  static constexpr const char* kParents = "parents";
  static constexpr const char* kOffsets = "offsets";
  static constexpr const char* kWeights = "weights";
  static constexpr const char* kTargets = "targets";

  FullyDifferentiablePositionErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  ~FullyDifferentiablePositionErrorFunctionT() override = default;

  [[nodiscard]] std::vector<std::string> inputs() const final;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const final;
  [[nodiscard]] const char* name() const final {
    return "PositionErrorFunction";
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

  void addConstraint(const PositionConstraintT<T>& constr);

  [[nodiscard]] const std::vector<PositionConstraintT<T>>& getConstraints() const;

 private:
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> value) const final;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) final;

  T calculatePositionGradient(
      const SkeletonStateT<T>& state,
      const PositionConstraintT<T>& constr,
      const Eigen::Vector3<T>& pos,
      Eigen::VectorX<T>& jGrad) const;

  template <typename JetType>
  [[nodiscard]] JetType calculatePositionGradient_dot(
      const SkeletonStateT<T>& state,
      size_t iConstr,
      size_t constrParent,
      const JetType& constrWeight,
      const Eigen::Vector3<JetType>& constr_offset,
      const Eigen::Vector3<JetType>& constr_target,
      Eigen::Ref<const Eigen::VectorX<T>> vec) const;

  [[nodiscard]] T calculatePositionJacobian(
      const SkeletonStateT<T>& state,
      const PositionConstraintT<T>& constr,
      const Eigen::Vector3<T>& pos,
      Ref<Eigen::MatrixX<T>> jac,
      Ref<Eigen::VectorX<T>> res) const;

  std::vector<PositionConstraintT<T>> constraints_;
  PositionConstraintStateT<T> constraintsState_;
  Eigen::VectorX<T> jointGrad_;
};

} // namespace momentum
