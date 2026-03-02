/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <momentum/character/locator.h>
#include <momentum/character_solver/joint_error_function.h>

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

/// ProjectionErrorFunctionT computes a 2D perspective projection error.
///
/// For each constraint, it transforms a 3D offset attached to a joint into world space,
/// projects it using a 3x4 projection matrix, and computes the 2D residual against a target.
///
/// @tparam T Scalar type (float or double)
template <typename T>
class ProjectionErrorFunctionT
    : public JointErrorFunctionT<T, ProjectionConstraintDataT<T>, 2, 1, 1> {
 public:
  using Base = JointErrorFunctionT<T, ProjectionConstraintDataT<T>, 2, 1, 1>;
  using typename Base::DfdvType;
  using typename Base::FuncType;
  using typename Base::VType;

  /// Constructor
  ///
  /// @param[in] skel Character skeleton
  /// @param[in] pt Parameter transformation
  /// @param[in] nearClip Distance threshold below which constraints are ignored (prevents
  ///                     divide-by-zero and bad behavior near the camera)
  ProjectionErrorFunctionT(
      const momentum::Skeleton& skel,
      const momentum::ParameterTransform& pt,
      T nearClip = T(1));

  /// @return Whether the constraint list is empty
  [[nodiscard]] bool empty() const {
    return this->constraints_.empty();
  }

  void setConstraints(std::vector<ProjectionConstraintDataT<T>> constraints) {
    this->constraints_ = std::move(constraints);
  }

 protected:
  /// Evaluates the projection error function for a single constraint.
  ///
  /// Computes f = (x/z - target_x, y/z - target_y) where (x, y, z) is the projected point.
  /// If z < nearClip, returns zero (constraint ignored).
  ///
  /// @param[in] constrIndex Index of the constraint to evaluate
  /// @param[in] state Joint state of the constraint's parent joint
  /// @param[out] f Output residual of dimension 2
  /// @param[out] v Output world-space vectors
  /// @param[out] dfdv The 2x3 Jacobian df/dv
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      FuncType& f,
      std::array<VType, 1>& v,
      std::array<DfdvType, 1>& dfdv) const override;

  // Projection error is roughly in radians, so a value near 1 is reasonable here:
  static constexpr T kProjectionWeight = 1.0f;

  // Ignore projection constraints involving joints closer than this distance.
  // Prevents divide-by-zero in the projection matrix and bad behavior close to
  // the camera.
  T nearClip_ = 1.0f;
};

} // namespace momentum
