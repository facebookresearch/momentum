/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "momentum/character/locator.h"
#include "momentum/character_solver/skeleton_error_function.h"

namespace momentum {

// The distance constraint is defined as ||(p_joint - origin)^2 - target||^2
template <typename T>
struct DistanceConstraintDataT {
  Eigen::Vector3<T> origin = Eigen::Vector3<T>::Zero(); // origin in world space
  T target{}; // distance target in world space
  size_t parent{}; // parent joint of the constraint
  Eigen::Vector3<T> offset; // relative offset to the parent
  T weight{}; // constraint weight
  // comment for now
  static DistanceConstraintDataT<T> createFromLocator(const momentum::Locator& locator);
};

template <typename T>
class DistanceErrorFunctionT : public momentum::SkeletonErrorFunctionT<T> {
 public:
  DistanceErrorFunctionT(const momentum::Skeleton& skel, const momentum::ParameterTransform& pt);

  [[nodiscard]] double getError(
      const momentum::ModelParametersT<T>& params,
      const momentum::SkeletonStateT<T>& state,
      const momentum::MeshStateT<T>& meshState) final;
  double getGradient(
      const momentum::ModelParametersT<T>& params,
      const momentum::SkeletonStateT<T>& state,
      const momentum::MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;
  double getJacobian(
      const momentum::ModelParametersT<T>& params,
      const momentum::SkeletonStateT<T>& state,
      const momentum::MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;
  [[nodiscard]] size_t getJacobianSize() const final;

  void addConstraint(const DistanceConstraintDataT<T>& constr) {
    constraints_.push_back(constr);
  }
  void clearConstraints() {
    constraints_.clear();
  }
  void setConstraints(std::vector<DistanceConstraintDataT<T>> constr) {
    constraints_ = std::move(constr);
  }
  [[nodiscard]] bool empty() const {
    return constraints_.empty();
  }
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  [[nodiscard]] const std::vector<DistanceConstraintDataT<T>>& getConstraints() const {
    return constraints_;
  }

 protected:
  // TODO: what should we use here?
  const T kDistanceWeight = 1.0f;

  std::vector<DistanceConstraintDataT<T>> constraints_;
};

} // namespace momentum
