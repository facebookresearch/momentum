/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint_state.h>
#include <momentum/character/types.h>
#include <momentum/math/types.h>

namespace momentum {

/// Measures of similarity between two skeleton states
///
/// Contains both per-joint error metrics and aggregate statistics
struct StateSimilarity {
  /// Position error for each joint (in distance units)
  VectorXf positionError;

  /// Orientation error for each joint (in radians)
  VectorXf orientationError;

  /// Root mean square of position errors across all joints
  float positionRMSE{0.0f};

  /// Root mean square of orientation errors across all joints
  float orientationRMSE{0.0f};

  /// Maximum position error across all joints
  float positionMax{0.0f};

  /// Maximum orientation error across all joints
  float orientationMax{0.0f};
};

/// Represents the complete state of a skeleton
///
/// Stores joint parameters and computed joint states for an entire skeleton.
/// Provides methods for updating the state based on new parameters and
/// comparing different skeleton states.
template <typename T>
struct SkeletonStateT {
  /// Joint parameters for all joints in the skeleton
  JointParametersT<T> jointParameters;

  /// Computed joint states for all joints in the skeleton
  JointStateListT<T> jointState;

  /// Creates an empty skeleton state
  SkeletonStateT() noexcept = default;

  /// Creates a skeleton state from joint parameters and a reference skeleton
  ///
  /// @param parameters Joint parameters for all joints in the skeleton
  /// @param referenceSkeleton The skeleton structure defining joint hierarchy
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  SkeletonStateT(
      const JointParametersT<T>& parameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  /// Creates a skeleton state from rvalue joint parameters and a reference skeleton
  ///
  /// @param parameters Joint parameters for all joints in the skeleton (moved from)
  /// @param referenceSkeleton The skeleton structure defining joint hierarchy
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  SkeletonStateT(
      JointParametersT<T>&& parameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  /// Creates a skeleton state from another skeleton state with a different scalar type
  ///
  /// @tparam T2 Source scalar type
  /// @param rhs Source skeleton state to copy from
  template <typename T2>
  explicit SkeletonStateT(const SkeletonStateT<T2>& rhs);

  /// Updates the skeleton state with new joint parameters
  ///
  /// @param jointParameters New joint parameters for all joints
  /// @param referenceSkeleton The skeleton structure defining joint hierarchy
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  void set(
      const JointParametersT<T>& jointParameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  /// Updates the skeleton state with new rvalue joint parameters
  ///
  /// @param jointParameters New joint parameters for all joints (moved from)
  /// @param referenceSkeleton The skeleton structure defining joint hierarchy
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  void set(
      JointParametersT<T>&& jointParameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  /// Updates the skeleton state from another skeleton state with a different scalar type
  ///
  /// @tparam T2 Source scalar type
  /// @param rhs Source skeleton state to copy from
  template <typename T2>
  void set(const SkeletonStateT<T2>& rhs);

  /// Compares two skeleton states and returns similarity metrics
  ///
  /// @param state1 First skeleton state to compare
  /// @param state2 Second skeleton state to compare
  /// @return Similarity metrics between the two states
  [[nodiscard]] static StateSimilarity compare(
      const SkeletonStateT<T>& state1,
      const SkeletonStateT<T>& state2);

  /// Extracts global transforms for all joints in the skeleton
  ///
  /// @return List of global transforms for all joints
  [[nodiscard]] TransformListT<T> toTransforms() const;

  /// Converts the skeleton state to a different scalar type
  ///
  /// @tparam T2 Target scalar type
  /// @return Skeleton state with the target scalar type
  template <typename T2>
  [[nodiscard]] SkeletonStateT<T2> cast() const {
    SkeletonStateT<T2> result;
    result.set(*this);
    return result;
  }

 private:
  /// Updates the joint states based on the current joint parameters
  ///
  /// @param referenceSkeleton The skeleton structure defining joint hierarchy
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  void set(const Skeleton& referenceSkeleton, bool computeDeriv);

  /// Copies joint states from another skeleton state with a different scalar type
  ///
  /// @tparam T2 Source scalar type
  /// @param rhs Source skeleton state to copy from
  template <typename T2>
  void copy(const SkeletonStateT<T2>& rhs);
};

/// Computes the relative transform between two joints in a skeleton
///
/// This transform maps points from joint A's local space to joint B's local space.
/// It is computed by finding the common ancestor of both joints and combining
/// the transforms along the path, which is more numerically stable than
/// computing (T_B)^{-1} * T_A directly.
///
/// @param jointA Source joint index
/// @param jointB Target joint index
/// @param referenceSkeleton The skeleton structure defining joint hierarchy
/// @param skelState Current state of the skeleton
/// @return Transform from joint A's local space to joint B's local space
template <typename T>
TransformT<T> transformAtoB(
    size_t jointA,
    size_t jointB,
    const Skeleton& referenceSkeleton,
    const SkeletonStateT<T>& skelState);

/// @name Inverse FK: Skeleton State → Joint Parameters
///
/// Four variants exist for converting skeleton state back to joint parameters:
///
// clang-format off
/// | Scenario                          | Function                                              |
/// |-----------------------------------|-------------------------------------------------------|
/// | General 3-axis joints             | `skeletonStateToJointParameters(state, skeleton)`     |
/// | Joints with <3 rotation DOF       | `skeletonStateToJointParametersRespectingActiveParameters(state, skeleton, activeJointParams)` |
/// | Starting from transforms          | `skeletonStateToJointParameters(transforms, skeleton)` |
/// | Transforms + constrained DOF      | `skeletonStateToJointParametersRespectingActiveParameters(transforms, skeleton, activeJointParams)` |
// clang-format on
///
/// The "RespectingActiveParameters" variants use `rotationMatrixToOneAxisEuler` or
/// `rotationMatrixToTwoAxisEuler` for joints with restricted DOF, avoiding Euler
/// angle ambiguity that the standard 3-axis extraction suffers from.
/// @{

/// Invert the skeleton state (global transforms in world space) back to joint parameters (Euler
/// angles in local space).  Note that this conversion is not unique due to the non-uniqueness of
/// Euler angle conversion.
///
/// @warning Euler angle extraction is numerically unstable near gimbal lock
/// (±90° pitch / Y-axis rotation). Near these configurations:
/// - Float precision: round-trip FK → inverse FK can shift angles by ~1e-3 radians
/// - Double precision: ~1e-10 radians
/// - The extracted angles may "flip" to an equivalent but numerically different representation
///
/// For joints with fewer than 3 rotation DOF, prefer
/// `skeletonStateToJointParametersRespectingActiveParameters()` which uses
/// constrained Euler extraction and avoids the ambiguity entirely.
template <typename T>
[[nodiscard]] JointParametersT<T> skeletonStateToJointParameters(
    const SkeletonStateT<T>& state,
    const Skeleton& skeleton);

template <typename T>
[[nodiscard]] JointParametersT<T> skeletonStateToJointParameters(
    const TransformListT<T>& state,
    const Skeleton& skeleton);

/// Converts skeleton state back to joint parameters, respecting active joint parameter constraints.
///
/// This function addresses the limitation of the original `skeletonStateToJointParameters` which
/// always uses 3-axis Euler rotations. Many joints only support 1-axis or 2-axis rotations, so
/// this function uses the provided `activeJointParams` to determine how many rotation axes are
/// active for each joint and applies the appropriate conversion method:
/// - 0 active rotation axes: Sets all rotation parameters to zero
/// - 1 active rotation axis: Uses `rotationMatrixToOneAxisEuler`
/// - 2 active rotation axes: Uses `rotationMatrixToTwoAxisEuler`
/// - 3 active rotation axes: Uses the standard 3-axis Euler conversion
///
/// @tparam T The scalar type (float or double)
/// @param[in] state The skeleton state to convert
/// @param[in] skeleton The skeleton defining joint hierarchy and properties
/// @param[in] activeJointParams Boolean array indicating which joint parameters are active.
///            Size should be skeleton.joints.size() * kParametersPerJoint
/// @return Joint parameters with rotations respecting the active parameter constraints
template <typename T>
[[nodiscard]] JointParametersT<T> skeletonStateToJointParametersRespectingActiveParameters(
    const SkeletonStateT<T>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);

/// Converts transform list back to joint parameters, respecting active joint parameter constraints.
///
/// This function addresses the limitation of the original `skeletonStateToJointParameters` which
/// always uses 3-axis Euler rotations. Many joints only support 1-axis or 2-axis rotations, so
/// this function uses the provided `activeJointParams` to determine how many rotation axes are
/// active for each joint and applies the appropriate conversion method:
/// - 0 active rotation axes: Sets all rotation parameters to zero
/// - 1 active rotation axis: Uses `rotationMatrixToOneAxisEuler`
/// - 2 active rotation axes: Uses `rotationMatrixToTwoAxisEuler`
/// - 3 active rotation axes: Uses the standard 3-axis Euler conversion
///
/// @tparam T The scalar type (float or double)
/// @param[in] state The transform list to convert
/// @param[in] skeleton The skeleton defining joint hierarchy and properties
/// @param[in] activeJointParams Boolean array indicating which joint parameters are active.
///            Size should be skeleton.joints.size() * kParametersPerJoint
/// @return Joint parameters with rotations respecting the active parameter constraints
template <typename T>
[[nodiscard]] JointParametersT<T> skeletonStateToJointParametersRespectingActiveParameters(
    const TransformListT<T>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);

/// @}

} // namespace momentum
