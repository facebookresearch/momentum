/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/skeleton.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/file_save_options.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/math/mesh.h>
#include <momentum/math/types.h>

#include <fx/gltf.h>

namespace fx::gltf {
struct Document;
} // namespace fx::gltf

namespace momentum {

using MotionParameters = std::tuple<std::vector<std::string>, MatrixXf>;
using IdentityParameters = std::tuple<std::vector<std::string>, JointParameters>;

/// Helper class to build a glb scene. It supports adding multiple characters
/// and motions for each character.
/// By default, momentum extension is added to the glb nodes. This is required if you
/// want to correctly load the exported character back.
///
/// @note A scene may hold multiple characters, but the momentum loaders can only read a
/// single-character scene back (loading a multi-character scene throws or drops all but the
/// first). Names returned by addCharacter are handles for use within this builder session, not
/// persistent identifiers recoverable on reload.
///
/// @note Character identity is the *exported* name, not character.name. addCharacter dedupes names,
/// so the exported name it RETURNS may differ from character.name. Pass that returned name as
/// `characterName` to addMotion/addSkeletonStates to target a specific character or instance; an
/// empty characterName resolves to the first character added under character.name.
class GltfBuilder final {
 public:
  GltfBuilder();

  ~GltfBuilder();

  GltfBuilder(GltfBuilder const&) = delete;

  void operator=(GltfBuilder const&) = delete;

  GltfBuilder(GltfBuilder&&) noexcept;

  GltfBuilder& operator=(GltfBuilder&&) noexcept;

  /// Specify how marker mesh is represented in the glb file
  enum class MarkerMesh : uint8_t { None, UnitCube };

  /// Add a character to the scene. Each character will have a root node with the character's
  /// name as the parent of the skeleton root and the character mesh.
  /// positionOffset and rotationOffset can be provided as an initial offset to the character.
  /// Duplicate names are made unique by appending a numeric suffix.
  ///
  /// @return The name used for the exported character. This is a within-session handle: pass it as
  /// characterName to addMotion/addSkeletonStates to target this specific character. It is not a
  /// persistent identifier recoverable on reload.
  std::string addCharacter(
      const Character& character,
      const Vector3f& positionOffset = Vector3f::Zero(),
      const Quaternionf& rotationOffset = Quaternionf::Identity(),
      const FileSaveOptions& options = FileSaveOptions());

  /// Add a static mesh, such as an environment or a target scan
  void addMesh(const Mesh& mesh, const std::string& name, bool addColor = false);

  void setFps(float fps);

  /// Add a motion to the provided character. If addCharacter is not called before adding
  /// the motion, the character will be automatically added with the default settings.
  ///
  /// @param[in] character The character to add motion for.
  /// @param[in] fps Frame rate of the motion in frames per second.
  /// @param[in] motion Motion parameters (parameter names and poses matrix).
  /// @param[in] offsets Identity/offset parameters (joint names and offsets vector).
  /// @param[in] addExtensions Whether to add momentum extensions to the document.
  /// @param[in] customName Custom name for the animation (default is "default").
  /// @param[in] timestamps Per-frame timestamps. Size should match motion columns.
  /// @param[in] characterName Exported name returned by addCharacter, selecting which exported
  /// character to animate. Pass it to target a specific instance when the same Character object was
  /// added more than once, or a duplicate whose name was auto-suffixed. When empty, resolves to the
  /// character exported under character.name (the first added under that name); if none exists, the
  /// character is auto-added with default settings.
  void addMotion(
      const Character& character,
      float fps = 120.0f,
      const MotionParameters& motion = {},
      const IdentityParameters& offsets = {},
      bool addExtensions = true,
      const std::string& customName = "default",
      std::span<const int64_t> timestamps = {},
      const std::string& characterName = "");

  /// Add a skeleton states to the provided character. If addCharacter is not called before adding
  /// the skeleton states, the character will be automatically added with the default settings.
  ///
  /// @param[in] characterName Exported name returned by addCharacter, selecting which exported
  /// character to animate (see addMotion). When empty, resolves to the character exported under
  /// character.name; if none exists, the character is auto-added with default settings.
  void addSkeletonStates(
      const Character& character,
      float fps,
      std::span<const SkeletonState> skeletonStates,
      const std::string& customName = "default",
      const std::string& characterName = "");

  /// Add marker data to the file
  ///
  /// @param[in] fps The frame rate of the motion capture data in frames per second.
  /// @param[in] markerSequence A 2D vector specifying the Marker data (name/position/occlusion) for
  /// all markers across all captured frames. Size: [numFrames][numMarkers]
  /// @param[in] markerMesh Optional parameter specifying the MarkerMesh type (default is
  /// MarkerMesh::None).
  /// @param[in] animName Optional parameter specifying the animation name (default is "default").
  void addMarkerSequence(
      float fps,
      std::span<const std::vector<momentum::Marker>> markerSequence,
      MarkerMesh markerMesh = MarkerMesh::UnitCube,
      const std::string& animName = "default");

  // Save the file with the provided filename. If the fileFormat is 'GltfFileFormat::Auto',
  // will deduct the file format by filename.
  // When embedResources is true, it will set all the existing buffers to embed the data.
  void save(
      const filesystem::path& filename,
      GltfFileFormat fileFormat = GltfFileFormat::Auto,
      bool embedResources = false);

  static void save(
      fx::gltf::Document& document,
      const filesystem::path& filename,
      GltfFileFormat fileFormat = GltfFileFormat::Auto,
      bool embedResources = false);

  /// Set all existing buffers to embed resources.
  void forceEmbedResources();

  static void forceEmbedResources(fx::gltf::Document& document);

  /// Allow copy the document, but do not allow modify the file
  /// outside of the builder to keep metadata consistent
  const fx::gltf::Document& getDocument();

  size_t getNumCharacters();

  size_t getCharacterRootIndex(const std::string& name);

  size_t getNumJoints(const std::string& name);

  size_t getNumMotions();

  float getFps();

  std::vector<size_t> getCharacterMotions(const std::string& characterName);

 private:
  /// Resolve which exported character an addMotion/addSkeletonStates call targets (see the
  /// "Character name model" note in gltf_builder.cpp) and validate its skeleton matches @p
  /// character; returns the resolved exported name. For an empty @p characterName whose source name
  /// was never added, the character is auto-added with default settings.
  std::string resolveCharacterForMotion(
      const Character& character,
      const std::string& characterName);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace momentum
