/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/skinned_locator.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

#include <span>
#include <vector>

namespace momentum {

/// @file linear_skinning.h
/// Functions for linear blend skinning (Skeletal Subspace Deformation)
///
/// Linear blend skinning (LBS), also known as Skeletal Subspace Deformation (SSD),
/// is a technique used in character animation to deform a mesh based on an underlying
/// skeleton. Each vertex in the mesh is influenced by one or more joints, with weights
/// determining how much influence each joint has on the vertex's final position.
///
/// This file provides functions to:
/// - Apply forward skinning (transform vertices from bind pose to animated pose)
/// - Apply inverse skinning (transform vertices from animated pose back to bind pose)
/// - Compute inverse skinning transformations for specific vertices
///
/// Key concepts:
/// - Bind pose: The reference pose of the character, where skin weights are defined
/// - Skin weights: Per-vertex weights defining how much influence each joint has
/// - Joint transformations: Current transformations of the skeleton's joints
/// - Forward skinning: Applying joint transformations to deform the mesh
/// - Inverse skinning: Reversing the deformation to return to bind pose

/// Computes the skinning transforms from joint states and inverse bind poses
///
/// This is a helper function that computes the combined transformation matrices
/// used for skinning. Each transform is computed as: jointState.transform * inverseBindPose.
///
/// @param jointState Span of joint states containing current transformations
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @return Vector of 4x4 transformation matrices for skinning
template <typename T>
std::vector<Eigen::Matrix4<T>> computeSkinningTransforms(
    typename DeduceSpanType<const JointStateT<T>>::type jointState,
    const TransformationListT<T>& inverseBindPose);

/// Applies forward SSD (linear blend skinning) to points using precomputed skinning transforms
///
/// This is the core skinning function that applies precomputed transformation matrices
/// to transform points from bind pose to animated pose.
///
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform
/// @param skinningTransforms Precomputed skinning transformation matrices
/// @return Vector of transformed points
template <typename T>
std::vector<Vector3<T>> applySSD(
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const Eigen::Matrix4<T>>::type skinningTransforms);

/// Applies forward SSD to points using joint states, returning new points
///
/// This overload transforms individual points and returns a new vector of transformed points.
/// Use this when you need to transform arbitrary points without modifying a mesh.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform
/// @param jointState Span of joint states containing current transformations
/// @return Vector of transformed points
template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const JointStateT<T>>::type jointState);

/// Applies forward SSD to points using skeleton state, returning new points
///
/// This overload transforms individual points using the full skeleton state.
/// Convenience wrapper that extracts joint states from the skeleton state.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform
/// @param state Current skeleton state containing joint transformations
/// @return Vector of transformed points
template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const SkeletonStateT<T>& state);

/// Applies forward SSD to points using precomputed world transforms, returning new points
///
/// This convenience overload accepts world-space transforms directly instead of joint states.
/// Useful when you already have computed world transforms from another source.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform
/// @param worldTransforms World-space transforms for each joint
/// @return Vector of transformed points
template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const TransformationListT<T>& worldTransforms);

/// Applies forward SSD to points using a span of world transforms, returning new points
///
/// This overload accepts world-space transforms as a span for greater flexibility.
/// Useful when you have transforms in a contiguous array that isn't a std::vector.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform
/// @param worldTransforms Span of world-space transforms for each joint
/// @return Vector of transformed points
template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const TransformT<T>>::type worldTransforms);

/// Applies forward SSD to a mesh using precomputed skinning transforms
///
/// This is the core mesh skinning function that applies precomputed transformation matrices
/// to transform both vertices and normals from bind pose to animated pose.
///
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param skinningTransforms Precomputed skinning transformation matrices
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const Eigen::Matrix4<T>>::type skinningTransforms,
    MeshT<T>& outputMesh);

/// Applies forward SSD to a mesh using joint states, modifying output mesh
///
/// This overload accepts a span of joint states directly.
/// Use this when you have joint transformations but not a complete skeleton state.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param jointState Span of joint states containing transformations
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const JointStateT<T>>::type jointState,
    MeshT<T>& outputMesh);

/// Applies forward SSD to a mesh using skeleton state, modifying output mesh
///
/// This overload transforms both vertices and normals of the mesh, writing results to outputMesh.
/// Use this when working with complete meshes and full skeleton state.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param state Current skeleton state containing joint transformations
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const SkeletonStateT<T>& state,
    MeshT<T>& outputMesh);

/// Applies forward SSD to a mesh using precomputed world transforms
///
/// This convenience overload accepts world-space transforms directly instead of joint states.
/// Useful when you already have computed world transforms from another source.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param worldTransforms World-space transforms for each joint
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const TransformationListT<T>& worldTransforms,
    MeshT<T>& outputMesh);

/// Applies forward SSD to a mesh using a span of world transforms
///
/// This overload accepts world-space transforms as a span for greater flexibility.
/// Useful when you have transforms in a contiguous array that isn't a std::vector.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param worldTransforms Span of world-space transforms for each joint
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const TransformT<T>>::type worldTransforms,
    MeshT<T>& outputMesh);

/// Computes the inverse SSD transformation for a specific vertex
///
/// Returns a transformation matrix that converts from skinned space back to bind pose space.
/// This is useful for operations that need to work in the original bind pose space.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param state Current skeleton state containing joint transformations
/// @param index Index of the vertex to compute inverse transformation for
/// @return Inverse transformation matrix for the specified vertex
Affine3f getInverseSSDTransformation(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    const SkeletonState& state,
    size_t index);

/// Applies inverse SSD to points, returning new points in bind pose space
///
/// This overload transforms points from skinned space back to bind pose space,
/// returning a new vector of transformed points.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform (in skinned space)
/// @param state Current skeleton state containing joint transformations
/// @return Vector of transformed points in bind pose space
std::vector<Vector3f> applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const SkeletonState& state);

/// Applies inverse SSD to points, storing results in a mesh
///
/// This overload transforms points from skinned space back to bind pose space,
/// writing the results directly to the provided mesh's vertices.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param points Input points to transform (in skinned space)
/// @param state Current skeleton state containing joint transformations
/// @param mesh Output mesh to store the transformed vertices
void applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const SkeletonState& state,
    Mesh& mesh);

/// Computes the world position of a skinned locator using linear blend skinning
///
/// This function applies linear blend skinning to transform a locator's rest position
/// to its world position based on the current skeleton state. The position is computed
/// by blending the transforms from each parent joint weighted by the corresponding
/// skin weights.
///
/// @param locator The skinned locator containing parent joints and weights
/// @param restPosition The locator's position in rest pose (bind pose)
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param state Current skeleton state containing joint transformations
/// @return The locator's world position
template <typename T>
Vector3<T> getSkinnedLocatorPosition(
    const SkinnedLocator& locator,
    const Vector3<T>& restPosition,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<T>& state);

/// Computes the world position of a skinned locator using linear blend skinning
///
/// Convenience overload that uses the locator's stored rest position.
///
/// @param locator The skinned locator containing parent joints, weights, and rest position
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param state Current skeleton state containing joint transformations
/// @return The locator's world position
template <typename T>
Vector3<T> getSkinnedLocatorPosition(
    const SkinnedLocator& locator,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<T>& state);

} // namespace momentum
