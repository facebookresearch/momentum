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
/// @note addCharacter() exports each character under a unique name: character.name, with a numeric
/// suffix when that name is already in use. Pass the returned name as `characterName` to
/// addMotion() and addSkeletonStates() to target that character; an empty `characterName` targets
/// the character added from character.name that kept that name, or else the first one added from
/// character.name.
/// loadGltfCharacters() reads every character back under its exported name.
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
  ///
  /// @return The unique name the character is exported under (see the class notes).
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
  /// The momentum extension keeps one motion per character, so adding another motion to the same
  /// character replaces the stored model parameters and timestamps, and the identity if @p offsets
  /// is not empty. A motion without frames stores nothing, not even the identity.
  ///
  /// @param[in] character The character to add motion for.
  /// @param[in] fps Frame rate of the motion in frames per second.
  /// @param[in] motion Motion parameters (parameter names and poses matrix).
  /// @param[in] offsets Identity/offset parameters (joint names and offsets vector).
  /// @param[in] addExtensions Whether to add momentum extensions to the document.
  /// @param[in] customName Name of the glTF animation that receives the motion (default is
  /// "default"). Motions added with the same name share one animation, which glTF viewers play as
  /// one clip, so give the characters of one recording the same name.
  /// @param[in] timestamps Per-frame timestamps: empty, or one per motion column; otherwise the
  /// function throws before it changes the document.
  /// @param[in] characterName Name returned by addCharacter() of the character to animate. When
  /// empty, the character added from character.name that kept that name, or else the first one
  /// added from character.name, which is added with default settings if there is none.
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
  /// @param[in] customName Name of the glTF animation that receives the skeleton states; see
  /// addMotion().
  /// @param[in] characterName Name of the character to animate; see addMotion().
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
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace momentum
