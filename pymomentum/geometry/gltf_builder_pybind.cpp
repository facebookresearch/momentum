/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/gltf_builder_pybind.h"
#include "pymomentum/geometry/momentum_io.h"

#include <momentum/character/character.h>
#include <momentum/character/fwd.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/io/gltf/gltf_builder.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>
#include <sstream>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void registerGltfBuilderBindings(pybind11::module& m) {
  // =====================================================
  // momentum::GltfBuilder::MarkerMesh enum
  // =====================================================
  py::enum_<mm::GltfBuilder::MarkerMesh>(m, "MarkerMesh")
      .value("NoMesh", mm::GltfBuilder::MarkerMesh::None) // None is a reserved
                                                          // work in Python.
      .value("UnitCube", mm::GltfBuilder::MarkerMesh::UnitCube);

  // =====================================================
  // momentum::GltfBuilder
  // - constructor with fps
  // - getFps() / setFps()
  // - addCharacter()
  // - addMesh()
  // - addMotion()
  // - addSkeletonStates()
  // - addMarkerSequence()
  // - save()
  // - to_bytes()
  // =====================================================

  py::class_<mm::GltfBuilder>(
      m,
      "GltfBuilder",
      R"(A builder class for creating GLTF files with multiple characters and animations.

The GltfBuilder allows you to incrementally construct a GLTF scene by adding characters,
meshes, motions, and marker data. This is useful for creating complex scenes with multiple
characters or combining different types of data into a single GLTF file.)")
      .def(
          py::init([](float fps) {
            auto builder = std::make_unique<mm::GltfBuilder>();
            builder->setFps(fps);
            return builder;
          }),
          R"(Create a new GltfBuilder with the specified frame rate.

:param fps: Frame rate in frames per second for animations.)",
          py::arg("fps") = 120.0f)
      .def_property(
          "fps",
          &mm::GltfBuilder::getFps,
          &mm::GltfBuilder::setFps,
          R"(The frame rate in frames per second used for animations.

This property controls the timing of all animations added to the GLTF file.
Setting this value will affect subsequently added motions and animations.

:type: float)")
      .def(
          "add_character",
          [](mm::GltfBuilder& builder,
             const mm::Character& character,
             const std::optional<Eigen::Vector3f>& positionOffset,
             const std::optional<Eigen::Vector4f>& rotationOffset,
             const std::optional<mm::GltfOptions>& options) {
            // Use defaults if not provided
            Eigen::Vector3f actualPositionOffset = positionOffset.value_or(Eigen::Vector3f::Zero());
            Eigen::Vector4f actualRotationOffset =
                rotationOffset.value_or(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
            mm::GltfOptions actualOptions = options.value_or(mm::GltfOptions{});

            // Convert Vector4f (x,y,z,w) to Quaternionf (w,x,y,z)
            mm::Quaternionf quaternionOffset(
                actualRotationOffset[3], // w
                actualRotationOffset[0], // x
                actualRotationOffset[1], // y
                actualRotationOffset[2]); // z

            builder.addCharacter(character, actualPositionOffset, quaternionOffset, actualOptions);
          },
          R"(Add a character to the GLTF scene.

Each character will have a root node with the character's name as the parent
of the skeleton root and the character mesh. Position and rotation offsets
can be provided as an initial transform for the character.

:param character: The character to add to the scene.
:param position_offset: Translation offset for the character's root node. Defaults to zero vector if None.
:param rotation_offset: Rotation offset as a quaternion in (x,y,z,w) format. Defaults to identity quaternion if None.
:param add_extensions: Whether to add momentum extensions to GLTF nodes.
:param add_collisions: Whether to add collision geometry to the scene.
:param add_locators: Whether to add locator data to the scene.
:param add_mesh: Whether to add the character's mesh to the scene.)",
          py::arg("character"),
          py::arg("position_offset") = std::nullopt,
          py::arg("rotation_offset") = std::nullopt,
          py::arg("options") = std::nullopt)
      .def(
          "add_mesh",
          &mm::GltfBuilder::addMesh,
          R"(Add a static mesh to the GLTF scene.

This can be used to add environment meshes, target scans, or other static
geometry that doesn't require animation. The mesh will be added as a separate
node in the scene with the specified name.

:param mesh: The mesh to add to the scene.
:param name: Name for the mesh node in the GLTF scene.
:param add_color: Whether to include vertex colors if present in the mesh.)",
          py::arg("mesh"),
          py::arg("name"),
          py::arg("add_color") = false)
      .def(
          "add_motion",
          [](mm::GltfBuilder& builder,
             const mm::Character& character,
             float fps,
             const std::optional<mm::MotionParameters>& motion,
             const std::optional<mm::IdentityParameters>& offsets,
             bool addExtensions,
             const std::string& customName) {
            // Apply same validation and transposition as
            // saveGLTFCharacterToFile
            mm::MotionParameters transposedMotion;
            if (motion.has_value()) {
              const auto& [parameters, poses] = motion.value();
              MT_THROW_IF(
                  poses.cols() != parameters.size(),
                  "Expected motion parameters to be n_frames x {}, but got {} x {}",
                  parameters.size(),
                  poses.rows(),
                  poses.cols());
            }

            builder.addMotion(
                character,
                fps,
                pymomentum::transpose(motion.value_or(mm::MotionParameters{})),
                offsets.value_or(mm::IdentityParameters{}),
                addExtensions,
                customName);
          },
          R"(Add a motion sequence to the specified character.

If addCharacter has not been called before adding the motion, the character
will be automatically added with default settings. The motion data contains
model parameters that animate the character over time.

:param character: The character to add motion for.
:param fps: Frame rate in frames per second for the motion data.
:param motion: Optional motion parameters as a tuple of (parameter_names, motion_data).
               Motion data should be a matrix with shape [n_frames x n_parameters].
:param offsets: Optional identity parameters as a tuple of (joint_names, offset_data).
                Offset data should be a vector with shape [n_joints * 7].
:param add_extensions: Whether to add momentum extensions to GLTF nodes.
:param custom_name: Custom name for the animation in the GLTF file.)",
          py::arg("character"),
          py::arg("fps") = 120.0f,
          py::arg("motion") = std::optional<mm::MotionParameters>{},
          py::arg("offsets") = std::optional<mm::IdentityParameters>{},
          py::arg("add_extensions") = true,
          py::arg("custom_name") = "default")
      .def(
          "add_skeleton_states",
          [](mm::GltfBuilder& builder,
             const mm::Character& character,
             float fps,
             const py::array_t<float>& skeletonStates,
             const std::string& customName) {
            // Use the shared utility function for conversion
            std::vector<mm::SkeletonState> skelStates =
                pymomentum::arrayToSkeletonStates(skeletonStates, character);

            // Call the addSkeletonStates method
            builder.addSkeletonStates(character, fps, std::span(skelStates), customName);
          },
          R"(Add skeleton states animation to the specified character.

If addCharacter has not been called before adding the skeleton states, the character
will be automatically added with default settings. The skeleton states contain
per-joint transforms that define the character's pose over time.

:param character: The character to add skeleton states for.
:param fps: Frame rate in frames per second for the skeleton state data.
:param skeleton_states: Skeleton states as a 3D array with shape [nFrames, nJoints, 8].
                       Each joint state contains [tx, ty, tz, rx, ry, rz, rw, s] where
                       translation is (tx,ty,tz), rotation is quaternion (rx,ry,rz,rw)
                       in (x,y,z,w) format, and s is scale.
:param custom_name: Custom name for the animation in the GLTF file.)",
          py::arg("character"),
          py::arg("fps"),
          py::arg("skeleton_states"),
          py::arg("custom_name") = "default")
      .def(
          "add_marker_sequence",
          [](mm::GltfBuilder& builder,
             float fps,
             const std::vector<std::vector<mm::Marker>>& markerSequence,
             mm::GltfBuilder::MarkerMesh markerMesh,
             const std::string& animName) {
            builder.addMarkerSequence(fps, std::span(markerSequence), markerMesh, animName);
          },
          R"(Add marker sequence animation data to the GLTF scene.

This method adds motion capture marker data to the GLTF file. The marker data
represents 3D positions of markers over time, which can be used for motion capture
analysis or visualization. Optional marker mesh visualization can be added as unit cubes.

:param fps: Frame rate in frames per second for the marker sequence data.
:param marker_sequence: A 2D list/array with shape [numFrames][numMarkers] containing
                       Marker objects for each frame. Each Marker contains name,
                       position, and occlusion status.
:param marker_mesh: Type of mesh to represent markers visually using :class:`MarkerMesh` enum.
                   Default is MarkerMesh.None for no visual representation.
                   MarkerMesh.UnitCube displays markers as unit cubes.
:param anim_name: Custom name for the marker animation in the GLTF file.)",
          py::arg("fps"),
          py::arg("marker_sequence"),
          py::arg("marker_mesh") = mm::GltfBuilder::MarkerMesh::None,
          py::arg("anim_name") = "default")
      .def(
          "save",
          [](mm::GltfBuilder& builder,
             const std::string& filename,
             const std::optional<mm::GltfFileFormat>& fileFormat) {
            mm::GltfFileFormat actualFileFormat = fileFormat.value_or(mm::GltfFileFormat::Auto);
            builder.save(filename, actualFileFormat);
          },
          R"(Save the GLTF scene to a file.

This method writes the constructed GLTF scene to the specified file. The file format
can be explicitly specified or automatically deduced from the file extension.

:param filename: Path where to save the GLTF file.
:param file_format: Optional file format specification using GltfFileFormat enum.
                   If not provided, format will be deduced from filename extension.)",
          py::arg("filename"),
          py::arg("file_format") = std::optional<mm::GltfFileFormat>{})
      .def(
          "to_bytes",
          [](mm::GltfBuilder& builder,
             const std::optional<mm::GltfFileFormat>& fileFormat) -> py::bytes {
            // Get a copy of the document
            fx::gltf::Document doc = builder.getDocument();

            // Use ostringstream to serialize the document to bytes
            std::ostringstream output(std::ios::binary | std::ios::out);
            fx::gltf::Save(
                doc,
                output,
                {},
                fileFormat.value_or(mm::GltfFileFormat::Binary) != mm::GltfFileFormat::Ascii);

            // Convert to Python bytes
            const std::string& str = output.str();
            return py::bytes(str);
          },
          R"(Convert the GLTF scene to bytes in memory.

This method serializes the constructed GLTF scene to a byte array without
writing to disk. This is useful for programmatic processing, network transmission,
or when you need the GLTF data as bytes for other purposes.

:return: The GLTF scene as bytes. For Binary format, this will be GLB binary data.
         For Ascii format, this will be JSON text encoded as UTF-8 bytes.)",
          py::arg("file_format") = mm::GltfFileFormat::Binary);
}

} // namespace pymomentum
