/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/marker.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/file_save_options.h>
#include <momentum/math/types.h>

#include <span>

namespace momentum {

/// Builder class for creating FBX files with multiple characters and animations.
///
/// FbxBuilder allows you to incrementally construct an FBX scene by adding
/// skinned characters, rigid bodies, motions, and marker data. This is useful
/// for creating scenes with multiple characters/objects in a single FBX file.
///
/// Example usage:
/// @code
///   FbxBuilder builder;
///   builder.addCharacter(bodyCharacter);
///   builder.addMotion(bodyCharacter, fps, motion);
///   builder.addRigidBody(propCharacter, "controller", 0);
///   builder.addMotionWithJointParams(propCharacter, fps, jointParams);
///   builder.save("output.fbx");
/// @endcode
class FbxBuilder final {
 public:
  FbxBuilder();

  ~FbxBuilder();

  FbxBuilder(FbxBuilder const&) = delete;

  void operator=(FbxBuilder const&) = delete;

  FbxBuilder(FbxBuilder&&) noexcept;

  FbxBuilder& operator=(FbxBuilder&&) noexcept;

  /// Add a skinned character (skeleton + mesh with skin weights) to the scene.
  ///
  /// Creates the skeleton hierarchy, mesh with skin deformer, locators,
  /// collision geometry, and metadata.
  ///
  /// @param character The character to add.
  /// @param options File save options controlling mesh, locators, collisions.
  void addCharacter(const Character& character, const FileSaveOptions& options = FileSaveOptions());

  /// Add a rigid body (skeleton + mesh parented to a joint, no skinning).
  ///
  /// For props/objects where the entire mesh moves rigidly with a joint.
  /// The mesh is parented under the specified joint node instead of the scene
  /// root, so it inherits the skeleton's animation without needing skin weights.
  ///
  /// @param character The character to add as a rigid body.
  /// @param name Optional name override. Uses character.name if empty.
  /// @param parentJoint Index of the joint to parent the mesh under.
  /// @param options File save options (mesh option is respected).
  void addRigidBody(
      const Character& character,
      const std::string& name = "",
      size_t parentJoint = 0,
      const FileSaveOptions& options = FileSaveOptions());

  /// Add model-parameter animation for a previously added character.
  ///
  /// Converts model parameters to joint parameters via CharacterState, then
  /// creates animation curves. Also extracts and animates blend shape weights
  /// when the character has blend shapes.
  ///
  /// @param character The character to animate (must match a previously added character).
  /// @param fps Animation frame rate in frames per second.
  /// @param motion Model parameters matrix with shape (nFrames x nModelParams).
  /// @param offsets Identity/offset parameters for the bind pose.
  void addMotion(
      const Character& character,
      float fps,
      const MatrixXf& motion = {},
      const VectorXf& offsets = {});

  /// Add joint-parameter animation for a previously added character.
  ///
  /// Uses joint parameters directly, bypassing the activeJointParams check.
  /// This is useful for rigid bodies or characters with simple parameterization.
  ///
  /// @param character The character to animate (must match a previously added character).
  /// @param fps Animation frame rate in frames per second.
  /// @param jointParams Joint parameters matrix with shape (nFrames x nJointParams).
  void addMotionWithJointParams(const Character& character, float fps, const MatrixXf& jointParams);

  /// Add an animated mesh (mesh node with per-frame transform, no skeleton).
  ///
  /// Creates a standalone mesh node and animates its transform directly from
  /// the root joint parameters. This is for props/objects that should appear
  /// as a simple animated mesh without any skeleton hierarchy.
  ///
  /// @param character The character providing the mesh geometry.
  /// @param name Name for the mesh node.
  /// @param fps Animation frame rate in frames per second.
  /// @param jointParams Joint parameters matrix with shape (nFrames x nJointParams).
  void addAnimatedMesh(
      const Character& character,
      const std::string& name,
      float fps,
      const MatrixXf& jointParams);

  /// Add a marker sequence to the scene.
  ///
  /// @param fps Frame rate for the marker animation.
  /// @param markerSequence Marker data per frame: markerSequence[frame][marker].
  void addMarkerSequence(float fps, std::span<const std::vector<Marker>> markerSequence);

  /// Save the FBX scene to a file.
  ///
  /// After saving, the builder is in a moved-from state and should not be used.
  ///
  /// @param filename Output file path.
  void save(const filesystem::path& filename);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace momentum
