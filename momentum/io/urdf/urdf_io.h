/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <span>

namespace momentum {

/// @file urdf_io.h
/// @brief URDF character loader for Momentum.
///
/// ## URDF-to-Momentum Frame Convention
///
/// URDF and Momentum organize skeletons differently, so this loader has to choose what a
/// Momentum "joint" node means.
///
/// **URDF convention:**
/// - A URDF `joint` connects a parent link to a child link.
/// - The joint contributes the parent-to-child bind-pose transform (`<origin>`) and the
///   motion axis (`<axis>`).
/// - Data attached to the child link — visuals, collisions, inertials, and descendant joint
///   origins — is authored in the child link frame.
///
/// **Momentum convention:**
/// - Each skeleton node stores a local frame (`preRotation` + translation offset) plus
///   parameters on fixed local principal axes: `rx`/`ry`/`rz`, `tx`/`ty`/`tz`.
///
/// **Loader choice: child-link-centric import**
/// - The imported Momentum node corresponds to the URDF child link and carries that link's
///   incoming URDF joint.
/// - `preRotation` is set to the URDF joint origin rotation only, so the imported local frame
///   stays equal to the URDF child link frame.
/// - This keeps child-link-authored data directly compatible with the imported skeleton.
///
/// **Consequence for DOF mapping:**
/// - DOFs are mapped onto the matching Momentum principal axis inside the preserved link frame.
///   For example, a revolute joint with axis `(0,0,1)` maps to `rz`.
/// - The exposed model parameter preserves the URDF joint scalar coordinate. If the URDF axis
///   points along `-X`, `-Y`, or `-Z`, the parameter transform therefore carries a `-1`
///   coefficient into the corresponding local Momentum principal axis.
/// - In other words, under this link-preserving convention the exposed model parameter is not
///   always numerically equal to the local positive-axis `rx`/`ry`/`rz` entry used by Momentum
///   FK. For negative-axis joints, a positive model parameter produces a negative local
///   principal-axis rotation.
/// - Because the model parameter stays in URDF joint coordinates, revolute and prismatic limits
///   and mimic relations remain in the same coordinate as the imported model parameter.
/// - We do not bake an extra axis-alignment rotation into `preRotation`. Doing so would make
///   the imported local frame differ from the URDF child link frame, and every child-link-
///   authored quantity would need frame conversion.
///
/// **Limitation:** Only principal axes (±X, ±Y, ±Z) are supported. Non-principal axes
/// (e.g., `(0.707, 0, 0.707)`) would require an additional axis-alignment frame or a more
/// general joint representation, so the loader rejects them.
///
/// **Units:** URDF uses meters; Momentum uses centimeters. All positions are converted
/// automatically.

/// Loads a character from a URDF file.
///
/// Parses the skeleton (joints, hierarchy, limits), parameter transform, and visual meshes.
/// Mesh files referenced by `<visual>` elements (STL, OBJ) are resolved relative to the
/// URDF file's directory.
///
/// @param filepath Path to the URDF file.
/// @return The loaded character with skeleton, parameter transform, limits, and optionally
///   mesh with skin weights.
template <typename T = float>
CharacterT<T> loadUrdfCharacter(const filesystem::path& filepath);

/// Loads a character from a URDF file using the provided byte data.
///
/// Mesh loading is not supported when loading from bytes because there is no
/// file path to resolve relative mesh references. Use the overload with
/// meshBasePath to enable mesh loading.
template <typename T = float>
[[nodiscard]] CharacterT<T> loadUrdfCharacter(std::span<const std::byte> bytes);

/// Loads a character from a URDF file using the provided byte data, with a base
/// path for resolving mesh file references.
///
/// @param bytes The raw URDF file contents.
/// @param meshBasePath The directory to use for resolving relative mesh file paths.
template <typename T = float>
[[nodiscard]] CharacterT<T> loadUrdfCharacter(
    std::span<const std::byte> bytes,
    const filesystem::path& meshBasePath);

} // namespace momentum
