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
///
/// @param[in] character Character to be scaled
/// @param[in] scale Scale factor to apply
/// @return A new Character object that has been scaled
[[nodiscard]] Character scaleCharacter(const Character& character, float scale);

/// Transforms the character (mesh and skeleton) by the desired transformation matrix.  The
/// transformation matrix should not have any scale or shear.
///
/// Note that this should primarily be used for transforming models between different coordinate
/// spaces (e.g. y-up vs. z-up). If you want to move a character around the scene, you should
/// preferably use the model parameters.
///
/// @param[in] character Character to be transformed
/// @param[in] xform Transformation to apply
/// @return A new Character object that has been transformed
[[nodiscard]] Character transformCharacter(const Character& character, const Affine3f& xform);

/// Replaces the part of target_character's skeleton rooted at target_root with the part of
/// source_character's skeleton rooted at source_root.
///
/// This function is typically used to swap one character's hand skeleton with another, for example.
///
/// @param[in] srcCharacter The source character whose skeleton will be copied.
/// @param[in] tgtCharacter The target character whose skeleton will be replaced.
/// @param[in] srcRootJoint Root of the source skeleton hierarchy to be copied.
/// @param[in] tgtRootJoint Root of the target skeleton hierarchy to be replaced.
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
///
/// @param[in] character The character from which joints will be removed.
/// @param[in] jointsToRemove A vector of joint indices to be removed.
/// @return A new Character object that is identical to the original except for the removal of
/// specified joints.
[[nodiscard]] Character removeJoints(
    const Character& character,
    momentum::span<const size_t> jointsToRemove);

/// Maps the input ModelParameter motion to a target character by matching model parameter names.
/// Mismatched names will be discarded (source) or set to zero (target).
///
/// @param[in] inputMotion Input ModelParameter motion with names.
/// @param[in] targetCharacter Target character that defines its own ModelParameters.
/// @return A matrix of model parameters for the target character.
MatrixXf mapMotionToCharacter(
    const MotionParameters& inputMotion,
    const Character& targetCharacter);

/// Maps the input JointParameter vector to a target character by matching joint names. Mismatched
/// names will be discarded (source) or set to zero (target). For every matched joint, all 7
/// parameters will be copied over.
///
/// @param[in] inputIdentity Input JointParameter vector with joint names.
/// @param[in] targetCharacter Target character that defines its own Joints.
/// @return A vector of joint parameters for the target character.
JointParameters mapIdentityToCharacter(
    const IdentityParameters& inputIdentity,
    const Character& targetCharacter);

/// Reduces the mesh to only include the specified vertices and associated faces
///
/// @param[in] character Character to be reduced
/// @param[in] activeVertices Boolean vector indicating which vertices to keep
/// @return A new character with mesh reduced to the specified vertices
template <typename T>
[[nodiscard]] CharacterT<T> reduceMeshByVertices(
    const CharacterT<T>& character,
    const std::vector<bool>& activeVertices);

/// Reduces the mesh to only include the specified faces and associated vertices
///
/// @param[in] character Character to be reduced
/// @param[in] activeFaces Boolean vector indicating which faces to keep
/// @return A new character with mesh reduced to the specified faces
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
/// @param[in] mesh The mesh to reduce
/// @param[in] activeVertices Boolean vector indicating which vertices to keep
/// @return A new mesh containing only the specified vertices and their associated faces
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
/// @param[in] mesh The mesh to reduce
/// @param[in] activeFaces Boolean vector indicating which faces to keep
/// @return A new mesh containing only the specified faces and their referenced vertices
template <typename T>
[[nodiscard]] MeshT<T> reduceMeshByFaces(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeFaces);

/// Converts vertex selection to face selection
///
/// @param[in] character Character containing the mesh
/// @param[in] activeVertices Boolean vector indicating which vertices are active
/// @return Boolean vector indicating which faces only contain active vertices
template <typename T>
[[nodiscard]] std::vector<bool> verticesToFaces(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeVertices);

/// Converts face selection to vertex selection
///
/// @param[in] mesh The mesh containing the faces
/// @param[in] activeFaces Boolean vector indicating which faces are active
/// @return Boolean vector indicating which vertices are used by active faces
template <typename T>
[[nodiscard]] std::vector<bool> facesToVertices(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeFaces);

/// Converts vertex selection to polygon selection
///
/// A polygon is active if all its vertices are active.
///
/// @param[in] mesh The mesh containing the polygon data
/// @param[in] activeVertices Boolean vector indicating which vertices are active
/// @return Boolean vector indicating which polygons only contain active vertices
template <typename T>
[[nodiscard]] std::vector<bool> verticesToPolys(
    const MeshT<T>& mesh,
    const std::vector<bool>& activeVertices);

/// Converts polygon selection to vertex selection
///
/// @param[in] mesh The mesh containing the polygon data
/// @param[in] activePolys Boolean vector indicating which polygons are active
/// @return Boolean vector indicating which vertices are used by active polygons
template <typename T>
[[nodiscard]] std::vector<bool> polysToVertices(
    const MeshT<T>& mesh,
    const std::vector<bool>& activePolys);

/// Reduces a standalone mesh to only include the specified polygons and associated vertices.
///
/// Converts polygon selection to vertex selection and face selection, then reduces the mesh.
/// Both triangulated faces and polygon data are filtered and remapped.
///
/// @param[in] mesh The mesh to reduce
/// @param[in] activePolys Boolean vector indicating which polygons to keep
/// @return A new mesh containing only the specified polygons and their referenced vertices
template <typename T>
[[nodiscard]] MeshT<T> reduceMeshByPolys(
    const MeshT<T>& mesh,
    const std::vector<bool>& activePolys);

/// Reduces the mesh to only include the specified polygons and associated vertices
///
/// Converts polygon selection to vertex selection and face selection, then reduces the mesh.
/// Both triangulated faces and polygon data are filtered and remapped.
///
/// @param[in] character Character to be reduced
/// @param[in] activePolys Boolean vector indicating which polygons to keep
/// @return A new character with mesh reduced to the specified polygons
template <typename T>
[[nodiscard]] CharacterT<T> reduceMeshByPolys(
    const CharacterT<T>& character,
    const std::vector<bool>& activePolys);

/// Result of addRigidTransformNode containing the new character and indices of the added bone and
/// parameters.
struct RigidTransformNodeResult {
  Character character;
  size_t boneIndex;
  size_t parameterStartIndex;
};

/// Adds a new root-level bone with 6 rigid DOF parameters (tx/ty/tz/rx/ry/rz) to a character.
///
/// The new joint is parented to kInvalidIndex (root-level) and gets 6 new model parameters
/// that map directly to the joint's translation and rotation parameters.
///
/// @param[in] character Character to add the rigid transform node to
/// @param[in] name Name for the new joint (also used as prefix for parameter names)
/// @param[in] translationOffset Translation offset for the new joint
/// @param[in] preRotation Pre-rotation for the new joint
/// @return A RigidTransformNodeResult containing the new character, the index of the new bone,
///         and the start index of the new model parameters
[[nodiscard]] RigidTransformNodeResult addRigidTransformNode(
    const Character& character,
    const std::string& name,
    const Eigen::Vector3f& translationOffset = Eigen::Vector3f::Zero(),
    const Eigen::Quaternionf& preRotation = Eigen::Quaternionf::Identity());

} // namespace momentum
