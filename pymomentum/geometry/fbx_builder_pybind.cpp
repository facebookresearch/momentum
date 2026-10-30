/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/fbx_builder_pybind.h"

#include <momentum/character/character.h>
#include <momentum/character/marker.h>
#include <momentum/io/fbx/fbx_builder.h>
#include <momentum/math/mesh.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

// A Python motion (numFrames x numParameters) in the momentum layout (numParameters x numFrames);
// empty when the motion is not given or has no values.
mm::MatrixXf toMomentumLayout(const std::optional<Eigen::MatrixXf>& motion) {
  if (!motion.has_value() || motion->size() == 0) {
    return {};
  }
  return motion->transpose();
}

} // namespace

void registerFbxBuilderBindings(pybind11::module& m) {
  py::class_<mm::FbxBuilder>(
      m,
      "FbxBuilder",
      R"(A builder class for creating FBX files with multiple characters and animations.

FbxBuilder allows you to incrementally construct an FBX scene by adding
skinned characters, rigid bodies, motions, and marker data. This is useful
for creating scenes with multiple characters/objects in a single FBX file.

Character names: ``add_character`` / ``add_rigid_body`` de-duplicate names (appending
``_1``, ``_2``, ...) and return the exported name, which may differ from ``character.name``.
Pass that returned name as ``character_name`` to the ``add_motion*`` methods to target a
specific character or instance (e.g. the same character added more than once).

Note: FBX saving requires the Autodesk FBX SDK and is only available on
Linux x86_64, Windows x86_64, and macOS.)")
      .def(py::init<>(), R"(Create a new FbxBuilder.)")
      .def(
          "add_character",
          &mm::FbxBuilder::addCharacter,
          R"(Add a skinned character (skeleton + mesh with skin weights) to the scene.

Creates the skeleton hierarchy, mesh with skin deformer, locators,
collision geometry, and metadata.

:param character: The character to add.
:param options: File save options: mesh, locators, collisions, coordinate system and FBX
    namespace. A non-empty ``fbx_namespace`` prefixes the name of every node in the saved file; a
    file has one namespace, so a different non-empty namespace in a later call raises an error.
:return: The exported character name. A numeric suffix is added when the requested name is already in use.)",
          py::arg("character"),
          py::arg("options") = mm::FileSaveOptions())
      .def(
          "add_rigid_body",
          &mm::FbxBuilder::addRigidBody,
          R"(Add a rigid body (skeleton + mesh parented to a joint, no skinning).

For props/objects where the entire mesh moves rigidly with a joint.
The mesh is parented under the specified joint node instead of the scene
root, so it inherits the skeleton's animation without needing skin weights.

:param character: The character to add as a rigid body.
:param name: Optional requested base name. Uses character.name if empty. A numeric
    suffix is appended if the resulting name is already in use.
:param parent_joint: Index of the joint to parent the mesh under. Defaults to 0
    (the root joint).
:param options: File save options: the mesh option and the FBX namespace (see
    :meth:`add_character`).
:return: The exported rigid-body name. A numeric suffix is added when the requested name is already in use.)",
          py::arg("character"),
          py::arg("name") = "",
          py::arg("parent_joint") = 0,
          py::arg("options") = mm::FileSaveOptions())
      .def(
          "add_motion",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             double fps,
             const std::optional<Eigen::MatrixXf>& motion,
             const std::optional<Eigen::VectorXf>& offsets,
             const std::string& characterName) {
            builder.addMotion(
                character,
                fps,
                toMomentumLayout(motion),
                offsets.value_or(mm::VectorXf()),
                characterName);
          },
          R"(Add model-parameter animation for a previously added character.

Converts model parameters to joint parameters, then creates animation
curves. Also extracts and animates blend shape weights when applicable.

:param character: The character to animate (must match a previously added character).
:param fps: Animation frame rate in frames per second.
:param motion: Model parameters matrix with shape (nFrames, nModelParams). None for no animation.
:param offsets: Identity/offset parameters for the bind pose. None to use default bind pose.
:param character_name: Which exported character to animate (see "Character names"). When empty, resolves to the character added from the source name that kept that name, or else the first one added from it.)",
          py::arg("character"),
          py::arg("fps") = 120.0f,
          py::arg("motion") = std::nullopt,
          py::arg("offsets") = std::nullopt,
          py::arg("character_name") = "")
      .def(
          "add_motion_with_joint_params",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             double fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const std::string& characterName) {
            builder.addMotionWithJointParams(
                character, fps, toMomentumLayout(jointParams), characterName);
          },
          R"(Add joint-parameter animation for a previously added character.

Uses joint parameters directly, bypassing the activeJointParams check.
This is useful for rigid bodies or characters with simple parameterization.

:param character: The character to animate (must match a previously added character).
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams). None for no animation.
:param character_name: Which exported character to animate (see "Character names"). When empty, resolves to the character added from the source name that kept that name, or else the first one added from it.)",
          py::arg("character"),
          py::arg("fps") = 120.0f,
          py::arg("joint_params") = std::nullopt,
          py::arg("character_name") = "")
      .def(
          "add_animated_mesh",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             const std::string& name,
             double fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const mm::FbxUserProperties& userProperties) {
            return builder.addAnimatedMesh(
                character, name, fps, toMomentumLayout(jointParams), userProperties);
          },
          R"(Add an animated mesh (no skeleton) to the scene.

Creates a standalone mesh node and animates its transform directly from
the root joint parameters. Use this for props/objects that should appear
as a simple animated mesh without any skeleton hierarchy.

:param character: The character providing the mesh geometry.
:param name: Requested name for the mesh node.
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams). Only the first
    joint's columns are used: tx, ty, tz, rx, ry, rz and, if present, the scale (unit scale
    without it). Fewer than 6 columns raises an error.
:param user_properties: Mapping of name to bool/int/float/str, written onto the
    mesh node as user-defined FBX properties. DCC tools surface these as the
    object's custom attributes.
:return: The unique name the mesh node is exported under: ``name``, with a numeric suffix
    if a character, rigid body or animated mesh already uses it.)",
          py::arg("character"),
          py::arg("name"),
          py::arg("fps") = 120.0f,
          py::arg("joint_params") = std::nullopt,
          py::arg("user_properties") = mm::FbxUserProperties{})
      .def(
          "add_animated_mesh",
          [](mm::FbxBuilder& builder,
             const mm::Mesh& mesh,
             const std::string& name,
             double fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const std::optional<Eigen::Vector3f>& translationOffset,
             const mm::FbxUserProperties& userProperties) {
            return builder.addAnimatedMesh(
                mesh,
                name,
                fps,
                toMomentumLayout(jointParams),
                translationOffset.value_or(Eigen::Vector3f::Zero()),
                userProperties);
          },
          R"(Add an animated mesh from a Mesh object (no Character needed).

Creates a standalone mesh node and animates its transform from the root joint
parameters.

:param mesh: The mesh geometry.
:param name: Requested name for the mesh node.
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams). Only the first
    joint's columns are used: tx, ty, tz, rx, ry, rz and, if present, the scale (unit scale
    without it). Fewer than 6 columns raises an error.
:param translation_offset: Rest-pose translation offset added to the translation
    channels (defaults to zero).
:param user_properties: Mapping of name to bool/int/float/str, written onto the
    mesh node as user-defined FBX properties. DCC tools surface these as the
    object's custom attributes.
:return: The unique name the mesh node is exported under: ``name``, with a numeric suffix
    if a character, rigid body or animated mesh already uses it.)",
          py::arg("mesh"),
          py::arg("name"),
          py::arg("fps") = 120.0f,
          py::arg("joint_params") = std::nullopt,
          py::arg("translation_offset") = std::nullopt,
          py::arg("user_properties") = mm::FbxUserProperties{})
      .def(
          "add_marker_sequence",
          [](mm::FbxBuilder& builder,
             double fps,
             const std::vector<std::vector<mm::Marker>>& markerSequence) {
            builder.addMarkerSequence(fps, std::span(markerSequence));
          },
          R"(Add a marker sequence to the scene.

:param fps: Frame rate for the marker animation.
:param marker_sequence: Marker data per frame: marker_sequence[frame][marker].)",
          py::arg("fps"),
          py::arg("marker_sequence"))
      .def(
          "save",
          [](mm::FbxBuilder& builder, const std::string& filename) { builder.save(filename); },
          R"(Save the FBX scene to a file.

After saving, the builder is consumed and should not be used further.

:param filename: Output file path.)",
          py::arg("filename"));
}

} // namespace pymomentum
