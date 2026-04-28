/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>

namespace momentum {

/// Scales the character (mesh and skeleton) by the desired amount.
///
/// Note that this should primarily be used when transforming the character into different units. If
/// you simply want to apply an identity-specific scale to the character, you should use the
/// 'scale_global' parameter in the ParameterTransform class.
[[nodiscard]] Character scaleCharacter(const Character& character, float scale);

/// Transforms the character (mesh and skeleton) by the desired transformation matrix.
///
/// @pre `xform` must be a rigid transform (no scale or shear).
///
/// Note that this should primarily be used for transforming models between different coordinate
/// spaces (e.g. y-up vs. z-up). If you want to move a character around the scene, you should
/// preferably use the model parameters.
[[nodiscard]] Character transformCharacter(const Character& character, const Affine3f& xform);

/// Replaces the subtree of `tgtCharacter`'s skeleton rooted at `tgtRootJoint` with the subtree of
/// `srcCharacter`'s skeleton rooted at `srcRootJoint`.
///
/// This function is typically used to swap one character's hand skeleton with another, for example.
///
/// @return A new Character that is identical to tgtCharacter except that the skeleton under
/// tgtRootJoint has been replaced by the part of srcCharacter's skeleton rooted at srcRootJoint.
[[nodiscard]] Character replaceSkeletonHierarchy(
    const Character& srcCharacter,
    const Character& tgtCharacter,
    const std::string& srcRootJoint,
    const std::string& tgtRootJoint);

/// Removes the specified joints and any joints parented beneath them from the character.
///
/// Currently, it is necessary to remove child joints to prevent dangling joints. Mesh points
/// skinned to the removed joints are re-skinned to their parent joint in the hierarchy.
[[nodiscard]] Character removeJoints(
    const Character& character,
    momentum::span<const size_t> jointsToRemove);

/// Maps the input ModelParameter motion to a target character by matching model parameter names.
/// Mismatched names will be discarded (source) or set to zero (target).
///
/// @return A matrix of model parameters for the target character (one column per frame).
MatrixXf mapMotionToCharacter(
    const MotionParameters& inputMotion,
    const Character& targetCharacter);

/// Maps the input JointParameter vector to a target character by matching joint names. Mismatched
/// names will be discarded (source) or set to zero (target). For every matched joint, all 7
/// parameters will be copied over.
JointParameters mapIdentityToCharacter(
    const IdentityParameters& inputIdentity,
    const Character& targetCharacter);

/// Reduces the character's mesh to only include the specified vertices and associated faces.
///
/// @pre `activeVertices.size() == character.mesh->vertices.size()`
template <typename T>
[[nodiscard]] CharacterT<T> reduceMeshByVertices(
    const CharacterT<T>& character,
    const std::vector<bool>& activeVertices);

/// Reduces the character's mesh to only include the specified faces and associated vertices.
///
/// @pre `activeFaces.size() == character.mesh->faces.size()`
template <typename T>
[[nodiscard]] CharacterT<T> reduceMeshByFaces(
    const CharacterT<T>& character,
    const std::vector<bool>& activeFaces);

/// Reduces a standalone mesh to only include the specified vertices and associated faces.
///
/// This is the mesh-only version of reduceMeshByVertices that works without a Character.
/// It handles vertices, normals, colors, confidence, faces, lines, texcoords, texcoord_faces,
/// and texcoord_lines with proper index remapping and compaction.
///
/// @pre `activeVertices.size() == mesh.vertices.size()`
template <typename T>
[[nodiscard]] MeshT<T> reduceMeshByVertices(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeVertices);

/// Reduces a standalone mesh to only include the specified faces and associated vertices.
///
/// This is the mesh-only version of reduceMeshByFaces that works without a Character.
/// It handles vertices, normals, colors, confidence, faces, lines, texcoords, texcoord_faces,
/// and texcoord_lines with proper index remapping and compaction.
///
/// @pre `activeFaces.size() == mesh.faces.size()`
template <typename T>
[[nodiscard]] MeshT<T> reduceMeshByFaces(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeFaces);

/// Converts a vertex selection to a face selection. A face is marked active only if every one of
/// its vertices is active.
template <typename T>
[[nodiscard]] std::vector<bool> verticesToFaces(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeVertices);

/// Converts a face selection to a vertex selection. A vertex is marked active if it is referenced
/// by any active face.
template <typename T>
[[nodiscard]] std::vector<bool> facesToVertices(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeFaces);

/// Converts a vertex selection to a polygon selection. A polygon is marked active only if every
/// one of its vertices is active.
template <typename T>
[[nodiscard]] std::vector<bool> verticesToPolys(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeVertices);

/// Converts a polygon selection to a vertex selection. A vertex is marked active if it is
/// referenced by any active polygon.
template <typename T>
[[nodiscard]] std::vector<bool> polysToVertices(
    const MeshT<T>& mesh,
    const std::vector<bool>& activePolys);

/// Reduces a standalone mesh to only include the specified polygons and associated vertices.
///
/// Converts polygon selection to vertex selection and face selection, then reduces the mesh.
/// Both triangulated faces and polygon data are filtered and remapped.
///
/// @pre `activePolys.size() == mesh.polygons.size()`
template <typename T>
[[nodiscard]] MeshT<T> reduceMeshByPolys(
    const MeshT<T>& mesh,
    const std::vector<bool>& activePolys);

/// Reduces the character's mesh to only include the specified polygons and associated vertices.
///
/// Converts polygon selection to vertex selection and face selection, then reduces the mesh.
/// Both triangulated faces and polygon data are filtered and remapped.
///
/// @pre `activePolys.size() == character.mesh->polygons.size()`
template <typename T>
[[nodiscard]] CharacterT<T> reduceMeshByPolys(
    const CharacterT<T>& character,
    const std::vector<bool>& activePolys);

/// Result of addRigidTransformNode containing the new character and indices of the added joint
/// and model parameters.
// TODO: Rename `boneIndex` to `jointIndex` to match the rest of the codebase's terminology
// ("joint" is used throughout; "bone" appears only here).
struct RigidTransformNodeResult {
  Character character;
  size_t boneIndex;
  size_t parameterStartIndex;
};

/// Adds a new root-level joint with 6 rigid DOF parameters (tx/ty/tz/rx/ry/rz) to a character.
///
/// The new joint is parented to kInvalidIndex (root-level) and gets 6 new model parameters
/// that map directly to the joint's translation and rotation parameters.
///
/// @param[in] name Name for the new joint (also used as prefix for parameter names).
[[nodiscard]] RigidTransformNodeResult addRigidTransformNode(
    const Character& character,
    const std::string& name,
    const Eigen::Vector3f& translationOffset = Eigen::Vector3f::Zero(),
    const Eigen::Quaternionf& preRotation = Eigen::Quaternionf::Identity());

} // namespace momentum
