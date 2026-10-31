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
#include <momentum/math/mesh.h>
#include <momentum/math/types.h>

#include <map>
#include <span>
#include <string>
#include <variant>

namespace momentum {

/// Value of a user-defined FBX property.
using FbxUserPropertyValue = std::variant<bool, int, float, std::string>;

/// User-defined FBX properties, keyed by property name.
///
/// These are written to a node as user-defined properties, which is how DCC
/// tools surface them as an object's "custom attributes". Note this is unrelated
/// to the FBX SDK's `FbxNodeAttribute`, which is the geometry attached to a node.
using FbxUserProperties = std::map<std::string, FbxUserPropertyValue>;

/// Builder class for creating FBX files with multiple characters and animations.
///
/// FbxBuilder allows you to incrementally construct an FBX scene by adding
/// skinned characters, rigid bodies, motions, and marker data. This is useful
/// for creating scenes with multiple characters/objects in a single FBX file.
///
/// @note addCharacter(), addRigidBody() and addAnimatedMesh() export each object under a unique
/// name: the requested name (character.name, or the `name` argument of addRigidBody() and
/// addAnimatedMesh()), with a numeric suffix when that name is already in use. The exported name is
/// the object's name in the file. A renamed character's joint and attribute nodes get the FBX
/// namespace "<exportedName>:" (loaders remove it when stripping namespaces), and its skinned mesh
/// is "<exportedName>_mesh" instead of "body_mesh". A rigid body with an explicit `name` also gets
/// the namespace. Pass the returned name as `characterName` to addMotion() and
/// addMotionWithJointParams() to target that character; an empty `characterName` targets the
/// character added from character.name that kept that name, or else the first one added from
/// character.name. An animated mesh is not a motion target. loadFbxCharacters() reads every
/// character and animated mesh back under its exported name.
///
/// A non-empty FileSaveOptions::fbxNamespace given to addCharacter() or addRigidBody() prefixes the
/// name of every node in the saved file with "<namespace>:", as saveFbx() does. A file has one
/// namespace, so a different non-empty namespace in a later call throws.
///
/// A file also has one frame rate. Each call that writes animation (addMotion(),
/// addMotionWithJointParams(), addAnimatedMesh() and addMarkerSequence()) sets it, so the last such
/// call wins, and the loaders read every motion at that frame rate: a motion written at another
/// frame rate comes back resampled, with another frame count. Give every motion of a file the same
/// frame rate.
///
/// Example usage:
/// @code
///   FbxBuilder builder;
///   const std::string bodyName = builder.addCharacter(bodyCharacter);
///   builder.addMotion(bodyCharacter, fps, motion, {}, bodyName);
///   const std::string propName = builder.addRigidBody(propCharacter, "controller", 0);
///   builder.addMotionWithJointParams(propCharacter, fps, jointParams, propName);
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
  /// @param options File save options: mesh, locators, collisions, coordinate system, and the FBX
  /// namespace (see the class notes).
  /// @return The unique name the character is exported under (see the class notes).
  std::string addCharacter(
      const Character& character,
      const FileSaveOptions& options = FileSaveOptions());

  /// Add a rigid body (skeleton + mesh parented to a joint, no skinning).
  ///
  /// For props/objects where the entire mesh moves rigidly with a joint.
  /// The mesh is parented under the specified joint node instead of the scene
  /// root, so it inherits the skeleton's animation without needing skin weights.
  ///
  /// @param character The character to add as a rigid body.
  /// @param name Optional requested base name. Uses character.name if empty.
  /// @param parentJoint Index of the joint to parent the mesh under.
  /// @param options File save options: the mesh option and the FBX namespace (see the class notes).
  /// @return The unique name the rigid body is exported under (see the class notes).
  std::string addRigidBody(
      const Character& character,
      const std::string& name = "",
      size_t parentJoint = 0,
      const FileSaveOptions& options = FileSaveOptions());

  /// Add model-parameter animation for a previously added character.
  ///
  /// Converts model parameters to joint parameters via CharacterState, then
  /// creates animation curves. Also extracts and animates blend shape weights
  /// when the character has blend shapes. A joint parameter that no model parameter drives gets a
  /// curve when its value is not zero, such as an offset from @p offsets. A later call for the same
  /// character replaces its motion; a motion without columns writes nothing and keeps the earlier
  /// one.
  ///
  /// @param character The character to animate (must match a previously added character).
  /// @param fps Animation frame rate in frames per second.
  /// @param motion Model parameters with shape (nModelParams x nFrames), one column per frame. The
  /// number of rows must be character.parameterTransform.numAllModelParameters(); otherwise the
  /// function throws.
  /// @param offsets Identity/offset joint parameters, added to the joint parameters of every frame.
  /// When its size is not numJointParameters(), the offsets of the character's earlier motion are
  /// used, as GltfBuilder does, or the bind pose (zero) if there is none.
  /// @param characterName Name returned by addCharacter() or addRigidBody() of the character to
  /// animate. When empty, the character added from character.name that kept that name, or else the
  /// first one added from character.name; throws if there is none.
  void addMotion(
      const Character& character,
      double fps,
      const MatrixXf& motion = {},
      const VectorXf& offsets = {},
      const std::string& characterName = "");

  /// Add joint-parameter animation for a previously added character.
  ///
  /// Uses joint parameters directly, bypassing the activeJointParams check.
  /// This is useful for rigid bodies or characters with simple parameterization. A later call for
  /// the same character replaces its motion.
  ///
  /// @param character The character to animate (must match a previously added character).
  /// @param fps Animation frame rate in frames per second.
  /// @param jointParams Joint parameters with shape (nJointParams x nFrames), one column per frame.
  /// The number of rows must be character.parameterTransform.numJointParameters(); otherwise the
  /// function throws.
  /// @param characterName Name of the character to animate; see addMotion().
  void addMotionWithJointParams(
      const Character& character,
      double fps,
      const MatrixXf& jointParams,
      const std::string& characterName = "");

  /// Add an animated mesh (mesh node with per-frame transform, no skeleton).
  ///
  /// Creates a standalone mesh node and animates its transform directly from
  /// the root joint parameters. This is for props/objects that should appear
  /// as a simple animated mesh without any skeleton hierarchy.
  ///
  /// @param character The character providing the mesh geometry.
  /// @param name Requested name for the mesh node.
  /// @param fps Animation frame rate in frames per second.
  /// @param jointParams Joint parameters with shape (nJointParams x nFrames), one column per frame.
  /// Only the first joint's rows are used: tx, ty, tz, rx, ry, rz and, if present, the scale (unit
  /// scale without it). Fewer than 6 rows throws.
  /// @param userProperties User-defined properties to write onto the mesh node,
  ///   surfaced by DCC tools as the object's custom attributes.
  /// @return The unique name the mesh node is exported under (see the class notes).
  std::string addAnimatedMesh(
      const Character& character,
      const std::string& name,
      double fps,
      const MatrixXf& jointParams,
      const FbxUserProperties& userProperties = {});

  /// Add an animated mesh from a Mesh object (no Character needed).
  ///
  /// Creates a standalone mesh node and animates its transform from joint
  /// parameters. The joint params follow momentum convention: 7 values per
  /// joint (tx, ty, tz, rx, ry, rz, sc). Only the first joint's parameters
  /// are used (root transform).
  ///
  /// @param mesh The mesh geometry.
  /// @param name Requested name for the mesh node.
  /// @param fps Animation frame rate in frames per second.
  /// @param jointParams Joint parameters with shape (nJointParams x nFrames), one column per frame.
  /// Only the first joint's rows are used: tx, ty, tz, rx, ry, rz and, if present, the scale (unit
  /// scale without it). Fewer than 6 rows throws.
  /// @param translationOffset Rest-pose translation offset added to the
  ///   translation channels (defaults to zero).
  /// @param userProperties User-defined properties to write onto the mesh node,
  ///   surfaced by DCC tools as the object's custom attributes.
  /// @return The unique name the mesh node is exported under (see the class notes).
  std::string addAnimatedMesh(
      const Mesh& mesh,
      const std::string& name,
      double fps,
      const MatrixXf& jointParams,
      const Vector3f& translationOffset = Vector3f::Zero(),
      const FbxUserProperties& userProperties = {});

  /// Add a marker sequence to the scene.
  ///
  /// FBX stores no confidence: every visible observation is written, and the loaders read each one
  /// with confidence 1. (The glTF writer leaves out observations with a confidence below 0.001 and
  /// stores the confidence of the others.)
  ///
  /// @param fps Frame rate for the marker animation.
  /// @param markerSequence Marker data per frame: markerSequence[frame][marker].
  void addMarkerSequence(double fps, std::span<const std::vector<Marker>> markerSequence);

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
