/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>
#include <momentum/character_solver/joint_error_function.h>
#include <momentum/character_solver/multi_source_error_function.h>

namespace momentum {

/// Constraint data for camera projection.
///
/// Each constraint specifies a 3D point (attached to a skeleton joint via parent + offset)
/// and a target 2D pixel location.
template <typename T>
struct CameraProjectionConstraintDataT : public ConstraintData {
  Eigen::Vector3<T> offset = Eigen::Vector3<T>::Zero(); ///< Local offset from parent joint
  Eigen::Vector2<T> target = Eigen::Vector2<T>::Zero(); ///< Target 2D pixel position

  CameraProjectionConstraintDataT() : ConstraintData(kInvalidIndex, 0.0f) {}

  CameraProjectionConstraintDataT(
      size_t parentJoint,
      const Eigen::Vector3<T>& localOffset,
      const Eigen::Vector2<T>& targetPixel,
      float w,
      const std::string& n = "")
      : ConstraintData(parentJoint, w, n), offset(localOffset), target(targetPixel) {}
};

/// Camera projection error function using IntrinsicsModelT.
///
/// Projects 3D joint-attached points through a camera with configurable
/// intrinsics model (pinhole, OpenCV distortion, etc.) and penalizes
/// the 2D pixel error against target positions.
///
/// Supports both bone-parented cameras (camera attached to a skeleton joint)
/// and static cameras (fixed world-space camera).
///
/// For bone-parented cameras, each constraint decomposes into 5 contributions:
///   [0] JointPoint(constraint.parent) — the 3D point
///   [1] JointPoint(cameraParent) — camera translation (negative sign in dfdv)
///   [2-4] JointDirection(cameraParent) — camera X/Y/Z axes
///
/// For static cameras, each constraint has 1 contribution:
///   [0] JointPoint(constraint.parent) — the 3D point
template <typename T>
class CameraProjectionConstraintErrorFunctionT : public MultiSourceErrorFunctionT<T, 2> {
 public:
  using Base = MultiSourceErrorFunctionT<T, 2>;
  using typename Base::DfdvType;
  using typename Base::FuncType;

  /// Constructor for bone-parented camera.
  ///
  /// @param character Character with skeleton and parameter transform
  /// @param intrinsicsModel Camera intrinsics model for projection
  /// @param cameraParent Joint index the camera is attached to
  /// @param cameraOffset Transform from camera parent joint to eye space
  /// @param lossAlpha Alpha parameter for generalized loss
  /// @param lossC C parameter for generalized loss
  CameraProjectionConstraintErrorFunctionT(
      const Character& character,
      std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
      size_t cameraParent,
      const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  /// Constructor for static camera.
  ///
  /// @param character Character with skeleton and parameter transform
  /// @param intrinsicsModel Camera intrinsics model for projection
  /// @param eyeFromWorld Transform from world space to eye space
  /// @param lossAlpha Alpha parameter for generalized loss
  /// @param lossC C parameter for generalized loss
  CameraProjectionConstraintErrorFunctionT(
      const Character& character,
      std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
      const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1));

  ~CameraProjectionConstraintErrorFunctionT() override = default;

  /// Returns contributions for the given constraint.
  ///
  /// For bone-parented: 5 contributions (1 JointPoint for 3D point + 1 JointPoint for camera
  /// position + 3 JointDirections for camera axes). For static: 1 contribution (1 JointPoint).
  [[nodiscard]] std::span<const SourceT<T>> getContributions(size_t constrIndex) const override;

  void evalFunction(
      size_t constrIndex,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      std::span<const Eigen::Vector3<T>> worldVecs,
      FuncType& f,
      std::span<DfdvType> dfdv) const final;

  /// Override gradient for bone-parented cameras to avoid numerical cancellation
  /// from the 5-contribution decomposition. Uses a stable 2-contribution formulation
  /// where the camera joint's effect is captured by a single virtual point at p_world.
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) override;

  /// Override Jacobian for bone-parented cameras (same stability fix as getGradient).
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) override;

  /// Whether this is a bone-parented camera (vs static).
  [[nodiscard]] bool isBoneParented() const {
    return isBoneParented_;
  }

  /// Adds a constraint.
  void addConstraint(const CameraProjectionConstraintDataT<T>& constr) {
    constraints_.push_back(constr);
    invalidateContributionsCache();
  }

  /// Clears all constraints.
  void clearConstraints() {
    constraints_.clear();
    invalidateContributionsCache();
  }

  /// Returns the constraints.
  [[nodiscard]] const std::vector<CameraProjectionConstraintDataT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Returns the number of constraints.
  [[nodiscard]] size_t getNumConstraints() const override {
    return constraints_.size();
  }

  /// Returns the per-constraint weight.
  [[nodiscard]] T getConstraintWeight(size_t constrIndex) const override {
    return static_cast<T>(constraints_[constrIndex].weight);
  }

 private:
  std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel_;
  bool isBoneParented_;
  size_t cameraParent_;
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset_;
  std::vector<CameraProjectionConstraintDataT<T>> constraints_;

  // Cached contributions per constraint
  mutable std::vector<std::vector<SourceT<T>>> perConstraintContributions_;
  mutable size_t contributionsCacheSize_ = 0;

  void invalidateContributionsCache() {
    contributionsCacheSize_ = 0;
  }

  void ensureContributionsCache() const;
};

} // namespace momentum
