/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string_view>

namespace momentum {

// ============================================================================
// GLTF File Format
// ============================================================================

/// File format in which the character is saved
enum class GltfFileFormat {
  Extension = 0, // The file extension is used for deduction (e.g. ".gltf" --> ASCII)
  GltfBinary = 1, // Binary format (generally .glb)
  GltfAscii = 2, // ASCII format (generally .gltf)
};

/// Options for GLTF file export
struct GltfOptions {
  /// Include GLTF extensions in the output.
  bool extensions = true;
  /// Include collision geometry in the output.
  bool collisions = true;
  /// Include locators in the output.
  bool locators = true;
  /// Include mesh geometry in the output.
  bool mesh = true;
  /// Include blend shapes in the output.
  bool blendShapes = true;
};

// ============================================================================
// FBX Coordinate System Options
// =====================================================================

/// Specifies which canonical axis represents up in the system (typically Y or Z).
/// Maps to fbxsdk::FbxAxisSystem::EUpVector
enum class FBXUpVector { XAxis = 1, YAxis = 2, ZAxis = 3 };

/// Vector with origin at the screen pointing toward the camera.
/// This is a subset of enum EUpVector because axis cannot be repeated.
/// We use the system of "parity" to define this vector because its value (X,Y or
/// Z axis) really depends on the up-vector.
/// Maps to fbxsdk::FbxAxisSystem::EFrontVector
enum class FBXFrontVector { ParityEven = 1, ParityOdd = 2 };

/// Specifies the third vector of the system.
/// Maps to fbxsdk::FbxAxisSystem::ECoordSystem
enum class FBXCoordSystem { RightHanded, LeftHanded };

/// A struct containing the up, front vectors and coordinate system for FBX export.
struct FBXCoordSystemInfo {
  /// Default to the same orientations as FbxAxisSystem::eMayaYUp
  FBXUpVector upVector = FBXUpVector::YAxis;
  FBXFrontVector frontVector = FBXFrontVector::ParityOdd;
  FBXCoordSystem coordSystem = FBXCoordSystem::RightHanded;
};

// ============================================================================
// Unified File Save Options
// ============================================================================

/// Unified options for saving files in both FBX and GLTF formats.
///
/// This struct consolidates save options that were previously scattered across
/// multiple function parameters. Format-specific options (e.g., FBX coordinate
/// system, GLTF extensions) are included but only used by their respective formats.
struct FileSaveOptions {
  // ---- Common Options (used by both FBX and GLTF) ----

  /// Include mesh geometry in the output (default: true)
  bool mesh = true;

  /// Include locators in the output (default: true)
  bool locators = true;

  /// Include collision geometry in the output (default: true)
  bool collisions = true;

  /// Include blend shapes in the output (default: true)
  bool blendShapes = true;

  /// Permissive mode: allow saving mesh-only characters without skin weights (default: false)
  bool permissive = false;

  // ---- FBX-Specific Options ----

  /// FBX coordinate system configuration (default: Maya Y-up)
  FBXCoordSystemInfo coordSystemInfo = {};

  /// Optional namespace prefix for FBX node names (e.g., "ns" becomes "ns:")
  /// Only used for FBX output (default: empty = no namespace)
  std::string_view fbxNamespace = "";

  // ---- GLTF-Specific Options ----

  /// Enable GLTF extensions (default: true)
  /// Only used for GLTF output
  bool extensions = true;

  /// GLTF file format selection (default: Extension)
  /// Only used for GLTF output
  GltfFileFormat gltfFileFormat = GltfFileFormat::Extension;
};

} // namespace momentum
