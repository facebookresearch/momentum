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
/// URDF and Momentum have different conventions for representing joint degrees of freedom.
/// This loader maps between them as follows:
///
/// **URDF convention:**
/// - Each joint specifies an arbitrary `axis` direction (e.g., `<axis xyz="0 0 1"/>`)
///   that defines the direction of rotation (revolute) or translation (prismatic).
/// - The axis is just a direction vector — it does NOT define the local coordinate system.
/// - The link frame is defined solely by the joint's `<origin>` transform (xyz + rpy).
/// - Visual meshes are positioned relative to the link frame.
///
/// **Momentum convention:**
/// - Joint DOFs are applied along fixed local axes: `rx`/`ry`/`rz` for rotation,
///   `tx`/`ty`/`tz` for translation.
/// - `preRotation` defines the parent-to-child frame rotation at bind pose.
///
/// **How this loader bridges the two:**
/// - DOFs are mapped to the **natural principal axis** matching the URDF axis direction.
///   For example, a revolute joint with axis `(0,0,1)` maps to `rz`, not `rx`.
/// - `preRotation` is set to the URDF joint origin rotation **only** — no axis alignment
///   rotation is baked in. This means the Momentum joint frame coincides with the URDF
///   link frame.
/// - Visual meshes can be transformed using the joint world transform directly, with no
///   frame correction needed.
///
/// **Why this matters for animation:**
/// - URDF has no official animation format. Motion data typically comes from BVH or other
///   formats that express rotations in the joint's local frame.
/// - Because we preserve the URDF link frame (no axis remapping), animation data can be
///   applied directly to the corresponding `rx`/`ry`/`rz` parameter without any frame
///   conversion.
/// - If axis alignment were baked into `preRotation`, external animation data would need
///   to account for the internal frame rotation, which no external tool would know to do.
///
/// **Limitation:** Only principal axes (±X, ±Y, ±Z) are supported. Non-principal axes
/// (e.g., `(0.707, 0, 0.707)`) will throw an error.
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
