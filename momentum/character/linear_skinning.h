/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

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

/// Applies forward SSD (linear blend skinning) to points, returning new points
///
/// This overload transforms individual points and returns a new vector of transformed points.
/// Use this when you need to transform arbitrary points without modifying a mesh.
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

/// Applies forward SSD to a mesh using raw joint state list, modifying output mesh
///
/// This overload accepts a JointStateList directly instead of a full SkeletonState.
/// Use this when you have joint transformations but not a complete skeleton state.
///
/// @param inverseBindPose Inverse bind pose transformations for each joint
/// @param skin Skin weights defining influence of each joint on vertices
/// @param mesh Input mesh to transform
/// @param state List of joint states containing transformations
/// @param outputMesh Output mesh to store the transformed result
template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const JointStateListT<T>& state,
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

} // namespace momentum
