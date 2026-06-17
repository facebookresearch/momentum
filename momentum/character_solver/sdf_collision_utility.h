/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/error_function_utils.h>
#include <momentum/common/checks.h>
#include <momentum/math/mesh.h>
#include <momentum/math/transform.h>

#include <axel/BoundingBox.h>

#include <array>
#include <span>

namespace momentum::detail_sdf_collision {

/// Shared, stateless helpers for SDF-collision error functions.
///
/// These free functions hold the geometry queries (collider transforms, bounding boxes) and the
/// joint chain-rule accumulation that are common to both the single-frame
/// `SDFCollisionErrorFunctionT` and the cross-frame (sequence) collision error functions. Keeping
/// them as free functions lets both error functions share the same derivation without depending on
/// each other's class.
///
/// The `SdfColliderType` is duck-typed; see `SDFCollisionErrorFunctionT` for the required
/// interface.

/// Returns the common ancestor joint of a vertex's skinning chain and an SDF collider's parent
/// joint, or `kInvalidIndex` when the collider has no parent joint.
///
/// This is the joint at which the vertex and collider hierarchies meet; derivatives above it cancel
/// because both the vertex and the collider move together, so it bounds the chains that need to be
/// walked.
inline size_t findCommonAncestorForVertex(
    const SkinWeights& skinWeights,
    const Skeleton& skeleton,
    size_t vertexIndex,
    size_t sdfParent) {
  if (sdfParent == kInvalidIndex) {
    return kInvalidIndex;
  }
  size_t minSkinnedJoint = kInvalidIndex;
  for (uint32_t k = 0; k < kMaxSkinJoints; ++k) {
    const auto w = skinWeights.weight(vertexIndex, k);
    if (w > 0) {
      const auto joint = skinWeights.index(vertexIndex, k);
      if (minSkinnedJoint == kInvalidIndex || joint < minSkinnedJoint) {
        minSkinnedJoint = joint;
      }
    }
  }
  if (minSkinnedJoint == kInvalidIndex) {
    return kInvalidIndex;
  }
  return skeleton.commonAncestor(minSkinnedJoint, sdfParent);
}

/// Returns the world-to-collider transform for a collider given a skeleton state.
///
/// World-fixed colliders (`parentJoint() == kInvalidIndex`) use only their local-to-parent
/// transform; joint-attached colliders are first composed with their parent joint transform.
template <typename T, typename SdfColliderType>
TransformT<T> computeWorldToColliderTransform(
    const SdfColliderType& collider,
    const SkeletonStateT<T>& state) {
  TransformT<T> parentTransform;
  const auto parentJoint = collider.parentJoint();
  if (parentJoint != kInvalidIndex) {
    MT_CHECK(parentJoint < state.jointState.size(), "Invalid parent joint index: {}", parentJoint);
    parentTransform = state.jointState[parentJoint].transform;
  }
  auto colliderWorldTransform =
      parentTransform * collider.localToParentTransform().template cast<T>();
  return colliderWorldTransform.inverse();
}

/// Returns the world-space axis-aligned bounding box of a collider given a skeleton state.
///
/// Transforms the collider's eight local-bound corners into world space. Returns an empty box for
/// invalid colliders.
template <typename T, typename SdfColliderType>
axel::BoundingBox<T> computeColliderWorldBoundingBox(
    const SdfColliderType& collider,
    const SkeletonStateT<T>& state) {
  if (!collider.isValid()) {
    return axel::BoundingBox<T>();
  }

  TransformT<T> colliderWorldTransform;
  const auto parentJoint = collider.parentJoint();
  if (parentJoint == kInvalidIndex) {
    colliderWorldTransform = collider.localToParentTransform().template cast<T>();
  } else {
    MT_CHECK(parentJoint < state.jointState.size(), "Invalid parent joint index: {}", parentJoint);
    colliderWorldTransform = state.jointState[parentJoint].transform *
        collider.localToParentTransform().template cast<T>();
  }

  const auto& localBounds = collider.bounds();
  const auto& minCorner = localBounds.min();
  const auto& maxCorner = localBounds.max();

  std::array<Eigen::Vector3<T>, 8> corners = {
      {{minCorner.x(), minCorner.y(), minCorner.z()},
       {maxCorner.x(), minCorner.y(), minCorner.z()},
       {minCorner.x(), maxCorner.y(), minCorner.z()},
       {maxCorner.x(), maxCorner.y(), minCorner.z()},
       {minCorner.x(), minCorner.y(), maxCorner.z()},
       {maxCorner.x(), minCorner.y(), maxCorner.z()},
       {minCorner.x(), maxCorner.y(), maxCorner.z()},
       {maxCorner.x(), maxCorner.y(), maxCorner.z()}}};

  const auto firstWorldCorner = colliderWorldTransform * corners[0];
  axel::BoundingBox<T> worldBounds(firstWorldCorner, T(0));
  for (size_t i = 1; i < corners.size(); ++i) {
    worldBounds.extend(colliderWorldTransform * corners[i]);
  }
  return worldBounds;
}

/// Returns the world-space axis-aligned bounding box enclosing the given posed-mesh vertices.
///
/// Returns an empty box when `vertexIndices` is empty.
template <typename T>
axel::BoundingBox<T> computeBoneBoundingBox(
    std::span<const size_t> vertexIndices,
    const MeshT<T>& posedMesh) {
  if (vertexIndices.empty()) {
    return axel::BoundingBox<T>();
  }
  axel::BoundingBox<T> bbox(posedMesh.vertices[vertexIndices[0]], T(0));
  for (size_t i = 1; i < vertexIndices.size(); ++i) {
    bbox.extend(posedMesh.vertices[vertexIndices[i]]);
  }
  return bbox;
}

/// Accumulates the gradient contribution of a single joint onto the model-parameter gradient.
///
/// Projects the constraint `direction` onto the joint's translation/rotation/scale derivatives and
/// maps the result from joint-parameter to model-parameter space. `posd` is the constrained point
/// relative to the joint origin.
template <typename T>
void accumulateJointGradient(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    const VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  for (size_t d = 0; d < 3; d++) {
    if (activeJointParams[paramIndex + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          parameterTransform,
          gradient);
    }
    if (activeJointParams[paramIndex + 3 + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          parameterTransform,
          gradient);
    }
  }
  if (activeJointParams[paramIndex + 6]) {
    gradient_jointParams_to_modelParams(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        parameterTransform,
        gradient);
  }
}

/// Accumulates the Jacobian contribution of a single joint onto one residual row.
///
/// Like `accumulateJointGradient` but writes a single-row Jacobian block (`jacobianRow`) in
/// model-parameter space rather than a gradient vector.
template <typename T>
void accumulateJointJacobian(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    const VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) {
  for (size_t d = 0; d < 3; d++) {
    if (activeJointParams[paramIndex + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          parameterTransform,
          jacobianRow);
    }
    if (activeJointParams[paramIndex + 3 + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          parameterTransform,
          jacobianRow);
    }
  }
  if (activeJointParams[paramIndex + 6]) {
    jacobian_jointParams_to_modelParams<T>(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        parameterTransform,
        jacobianRow);
  }
}

/// Accumulates the gradient contribution of a collider's joint hierarchy onto the gradient.
///
/// Walks from the collider's parent joint up to (but not including) `commonAncestor`, accumulating
/// each joint's contribution for the collider-side of the constraint at `sdfSurfacePoint`.
template <typename T>
void accumulateColliderHierarchyGradient(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();
    accumulateJointGradient(
        jointState, paramIndex, posd, direction, activeJointParams, parameterTransform, gradient);
    jointIndex = skeleton.joints[jointIndex].parent;
  }
}

/// Accumulates the Jacobian contribution of a collider's joint hierarchy onto one residual row.
///
/// Like `accumulateColliderHierarchyGradient` but writes a single-row Jacobian block.
template <typename T>
void accumulateColliderHierarchyJacobian(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();
    accumulateJointJacobian(
        jointState,
        paramIndex,
        posd,
        direction,
        activeJointParams,
        parameterTransform,
        jacobianRow);
    jointIndex = skeleton.joints[jointIndex].parent;
  }
}

} // namespace momentum::detail_sdf_collision
