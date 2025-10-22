/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <momentum/character/locator.h>
#include <momentum/character_solver/skeleton_error_function.h>

namespace momentum {

template <typename T>
struct ProjectionConstraintDataT {
  // Project a 3d point (after skinning) to (rxz, ryz, z) where
  // (rx,ry) is the residual.
  //
  // This encodes the target 2d point, the view and camera transforms, and the
  // camera distortion (if any).
  //
  // This projection model was copied from the hand IK FullProjectionConstraint.
  // Ideally we would refactor somehow to make them share more code but
  // the kinematic models are so different this would take some work.
  Eigen::Matrix<T, 3, 4> projection; // Projection matrix

  size_t parent{}; // parent joint of the constraint
  Eigen::Vector3<T> offset = Eigen::Vector3<T>::Zero(); // relative offset to the parent
  T weight = 1; // constraint weight

  Eigen::Vector2<T> target = Eigen::Vector2<T>::Zero();

  static ProjectionConstraintDataT<T> createFromLocator(const momentum::Locator& locator);
};

template <typename T>
class ProjectionErrorFunctionT : public momentum::SkeletonErrorFunctionT<T> {
 public:
  ProjectionErrorFunctionT(
      const momentum::Skeleton& skel,
      const momentum::ParameterTransform& pt,
      T nearClip = T(1));

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

  void addConstraint(const ProjectionConstraintDataT<T>& data);

  void clearConstraints() {
    constraints_.clear();
  }
  [[nodiscard]] bool empty() const {
    return constraints_.empty();
  }
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }
  void setConstraints(std::vector<ProjectionConstraintDataT<T>> constraints) {
    constraints_ = std::move(constraints);
  }

  [[nodiscard]] const std::vector<ProjectionConstraintDataT<T>>& getConstraints() const {
    return constraints_;
  }

 protected:
  std::vector<ProjectionConstraintDataT<T>> constraints_;

  // Projection error is roughly in radians, so a value near 1 is reasonable here:
  static constexpr T kProjectionWeight = 1.0f;

  // Ignore projection constraints involving joints closer than this distance.
  // Prevents divide-by-zero in the projection matrix and bad behavior close to
  // the camera.
  T _nearClip = 1.0f;
};

} // namespace momentum
