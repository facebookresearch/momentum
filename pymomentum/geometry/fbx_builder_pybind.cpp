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

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void registerFbxBuilderBindings(pybind11::module& m) {
  py::class_<mm::FbxBuilder>(
      m,
      "FbxBuilder",
      R"(A builder class for creating FBX files with multiple characters and animations.

FbxBuilder allows you to incrementally construct an FBX scene by adding
skinned characters, rigid bodies, motions, and marker data. This is useful
for creating scenes with multiple characters/objects in a single FBX file.

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
:param options: File save options controlling mesh, locators, collisions.
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
:param options: File save options (mesh option is respected).
:return: The exported rigid-body name. A numeric suffix is added when the requested name is already in use.)",
          py::arg("character"),
          py::arg("name") = "",
          py::arg("parent_joint") = 0,
          py::arg("options") = mm::FileSaveOptions())
      .def(
          "add_motion",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             float fps,
             const std::optional<Eigen::MatrixXf>& motion,
             const std::optional<Eigen::VectorXf>& offsets,
             const std::optional<std::string>& characterName) {
            // pymomentum convention: (nFrames x nParams), C++ convention: (nParams x nFrames)
            mm::MatrixXf transposedMotion;
            if (motion.has_value() && motion->size() > 0) {
              transposedMotion = motion->transpose();
            }
            mm::VectorXf actualOffsets;
            if (offsets.has_value()) {
              actualOffsets = offsets.value();
            }
            builder.addMotion(
                character, fps, transposedMotion, actualOffsets, characterName.value_or(""));
          },
          R"(Add model-parameter animation for a previously added character.

Converts model parameters to joint parameters, then creates animation
curves. Also extracts and animates blend shape weights when applicable.

:param character: The character to animate (must match a previously added character).
:param fps: Animation frame rate in frames per second.
:param motion: Model parameters matrix with shape (nFrames, nModelParams). None for no animation.
:param offsets: Identity/offset parameters for the bind pose. None to use default bind pose.
:param character_name: Exported name returned by :meth:`add_character` or :meth:`add_rigid_body`, selecting which exported character to animate. Pass it to target a specific instance when the same character was added more than once, or a duplicate whose name was auto-suffixed. When omitted, resolves to the character exported under the source name.)",
          py::arg("character"),
          py::arg("fps") = 120.0f,
          py::arg("motion") = std::nullopt,
          py::arg("offsets") = std::nullopt,
          py::arg("character_name") = std::nullopt)
      .def(
          "add_motion_with_joint_params",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             float fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const std::optional<std::string>& characterName) {
            // pymomentum convention: (nFrames x nParams), C++ convention: (nParams x nFrames)
            mm::MatrixXf transposedJointParams;
            if (jointParams.has_value() && jointParams->size() > 0) {
              transposedJointParams = jointParams->transpose();
            }
            builder.addMotionWithJointParams(
                character, fps, transposedJointParams, characterName.value_or(""));
          },
          R"(Add joint-parameter animation for a previously added character.

Uses joint parameters directly, bypassing the activeJointParams check.
This is useful for rigid bodies or characters with simple parameterization.

:param character: The character to animate (must match a previously added character).
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams). None for no animation.
:param character_name: Exported name returned by :meth:`add_character` or :meth:`add_rigid_body`, selecting which exported character to animate. Pass it to target a specific instance when the same character was added more than once, or a duplicate whose name was auto-suffixed. When omitted, resolves to the character exported under the source name.)",
          py::arg("character"),
          py::arg("fps") = 120.0f,
          py::arg("joint_params") = std::nullopt,
          py::arg("character_name") = std::nullopt)
      .def(
          "add_animated_mesh",
          [](mm::FbxBuilder& builder,
             const mm::Character& character,
             const std::string& name,
             float fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const mm::FbxUserProperties& userProperties) {
            mm::MatrixXf transposedJointParams;
            if (jointParams.has_value() && jointParams->size() > 0) {
              transposedJointParams = jointParams->transpose();
            }
            builder.addAnimatedMesh(character, name, fps, transposedJointParams, userProperties);
          },
          R"(Add an animated mesh (no skeleton) to the scene.

Creates a standalone mesh node and animates its transform directly from
the root joint parameters. Use this for props/objects that should appear
as a simple animated mesh without any skeleton hierarchy.

:param character: The character providing the mesh geometry.
:param name: Name for the mesh node.
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams).
:param user_properties: Mapping of name to bool/int/float/str, written onto the
    mesh node as user-defined FBX properties. DCC tools surface these as the
    object's custom attributes.)",
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
             float fps,
             const std::optional<Eigen::MatrixXf>& jointParams,
             const std::optional<Eigen::Vector3f>& translationOffset,
             const mm::FbxUserProperties& userProperties) {
            mm::MatrixXf transposedJointParams;
            if (jointParams.has_value() && jointParams->size() > 0) {
              transposedJointParams = jointParams->transpose();
            }
            builder.addAnimatedMesh(
                mesh,
                name,
                fps,
                transposedJointParams,
                translationOffset.value_or(Eigen::Vector3f::Zero()),
                userProperties);
          },
          R"(Add an animated mesh from a Mesh object (no Character needed).

Creates a standalone mesh node and animates its transform from joint
parameters. Only the root transform (first 7 joint params) is used.

:param mesh: The mesh geometry.
:param name: Name for the mesh node.
:param fps: Animation frame rate in frames per second.
:param joint_params: Joint parameters matrix with shape (nFrames, nJointParams).
:param translation_offset: Rest-pose translation offset added to the translation
    channels (defaults to zero).
:param user_properties: Mapping of name to bool/int/float/str, written onto the
    mesh node as user-defined FBX properties. DCC tools surface these as the
    object's custom attributes.)",
          py::arg("mesh"),
          py::arg("name"),
          py::arg("fps") = 120.0f,
          py::arg("joint_params") = std::nullopt,
          py::arg("translation_offset") = std::nullopt,
          py::arg("user_properties") = mm::FbxUserProperties{})
      .def(
          "add_marker_sequence",
          [](mm::FbxBuilder& builder,
             float fps,
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
