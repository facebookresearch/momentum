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
[[nodiscard]] Character transformCharacter(const Character& character, const Transform& xform);

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
    gsl::span<const size_t> jointsToRemove);

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
VectorXf mapIdentityToCharacter(
    const IdentityParameters& inputIdentity,
    const Character& targetCharacter);

} // namespace momentum
