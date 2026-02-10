/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/renderer/mesh_processing.h>
#include <pymomentum/renderer/momentum_render.h>
#include <pymomentum/renderer/software_rasterizer.h>
#include <pymomentum/tensor_momentum/tensor_parameter_transform.h>
#include <pymomentum/tensor_momentum/tensor_skeleton_state.h>
#include <pymomentum/torch_bridge.h>

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/text_rasterizer.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <optional>

namespace py = pybind11;

using namespace pymomentum;

PYBIND11_MODULE(renderer, m) {
  // TODO more explanation
  m.attr("__name__") = "pymomentum.renderer";
  m.doc() = "Functions for rendering momentum models.";

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  pybind11::module_::import("torch"); // @dep=//caffe2:torch
  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbsource//arvr/libraries/pymomentum:geometry
  py::module_ camera_module = pybind11::module_::import(
      "pymomentum.camera"); // @dep=fbsource//arvr/libraries/pymomentum:camera

  py::class_<momentum::rasterizer::PhongMaterial>(
      m,
      "PhongMaterial",
      "A Phong shading material model with diffuse, specular, and emissive components. "
      "Supports both solid colors and texture maps for realistic surface rendering. "
      "The Phong model provides smooth shading with controllable highlights and surface properties.")
      .def_property(
          "diffuse_color",
          [](const momentum::rasterizer::PhongMaterial& mat) { return mat.diffuseColor; },
          [](momentum::rasterizer::PhongMaterial& mat, const Eigen::Vector3f& color) {
            mat.diffuseColor = color;
          })
      .def_property(
          "specular_color",
          [](const momentum::rasterizer::PhongMaterial& mat) { return mat.specularColor; },
          [](momentum::rasterizer::PhongMaterial& mat, const Eigen::Vector3f& color) {
            mat.specularColor = color;
          })
      .def_property(
          "emissive_color",
          [](const momentum::rasterizer::PhongMaterial& mat) { return mat.emissiveColor; },
          [](momentum::rasterizer::PhongMaterial& mat, const Eigen::Vector3f& color) {
            mat.emissiveColor = color;
          })
      .def_property(
          "specular_exponent",
          [](const momentum::rasterizer::PhongMaterial& mat) { return mat.specularExponent; },
          [](momentum::rasterizer::PhongMaterial& mat, const float exponent) {
            mat.specularExponent = exponent;
          })
      .def(py::init<>())
      .def(
          py::init(
              [](const std::optional<Eigen::Vector3f>& diffuseColor,
                 const std::optional<Eigen::Vector3f>& specularColor,
                 const std::optional<float>& specularExponent,
                 const std::optional<Eigen::Vector3f>& emissiveColor,
                 const std::optional<py::array_t<const float>>& diffuseTexture,
                 const std::optional<py::array_t<const float>>& emissiveTexture)
                  -> momentum::rasterizer::PhongMaterial {
                return pymomentum::createPhongMaterial(
                    diffuseColor,
                    specularColor,
                    specularExponent,
                    emissiveColor,
                    diffuseTexture,
                    emissiveTexture);
              }),
          R"(Create a Phong material with customizable properties.

          :param diffuse_color: RGB diffuse color values (0-1 range)
          :param specular_color: RGB specular color values (0-1 range)
          :param specular_exponent: Specular highlight sharpness
          :param emissive_color: RGB emissive color values (0-1 range)
          :param diffuse_texture: Optional diffuse texture as a numpy array
          :param emissive_texture: Optional emissive texture as a numpy array
          )",
          py::arg("diffuse_color") = std::optional<Eigen::Vector3f>{},
          py::arg("specular_color") = std::optional<Eigen::Vector3f>{},
          py::arg("specular_exponent") = std::optional<float>{},
          py::arg("emissive_color") = std::optional<Eigen::Vector3f>{},
          py::arg("diffuse_texture") = std::optional<py::array_t<const float>>{},
          py::arg("emissive_texture") = std::optional<py::array_t<const float>>{})
      .def("__repr__", [](const momentum::rasterizer::PhongMaterial& self) {
        return fmt::format(
            "PhongMaterial(diffuse=({:.2f}, {:.2f}, {:.2f}), specular=({:.2f}, {:.2f}, {:.2f}), exponent={:.2f}, emissive=({:.2f}, {:.2f}, {:.2f}))",
            self.diffuseColor.x(),
            self.diffuseColor.y(),
            self.diffuseColor.z(),
            self.specularColor.x(),
            self.specularColor.y(),
            self.specularColor.z(),
            self.specularExponent,
            self.emissiveColor.x(),
            self.emissiveColor.y(),
            self.emissiveColor.z());
      });

  py::enum_<momentum::rasterizer::LightType>(m, "LightType", "Type of light to use in rendering.")
      .value("Ambient", momentum::rasterizer::LightType::Ambient)
      .value("Directional", momentum::rasterizer::LightType::Directional)
      .value("Point", momentum::rasterizer::LightType::Point);

  py::enum_<momentum::rasterizer::HorizontalAlignment>(
      m, "HorizontalAlignment", "Horizontal text alignment options.")
      .value("Left", momentum::rasterizer::HorizontalAlignment::Left)
      .value("Center", momentum::rasterizer::HorizontalAlignment::Center)
      .value("Right", momentum::rasterizer::HorizontalAlignment::Right);

  py::enum_<momentum::rasterizer::VerticalAlignment>(
      m, "VerticalAlignment", "Vertical text alignment options.")
      .value("Top", momentum::rasterizer::VerticalAlignment::Top)
      .value("Center", momentum::rasterizer::VerticalAlignment::Center)
      .value("Bottom", momentum::rasterizer::VerticalAlignment::Bottom);
  py::class_<momentum::rasterizer::Light>(
      m,
      "Light",
      "A light source for 3D rendering supporting point, directional, and ambient lighting. "
      "Point lights emit from a specific position, directional lights simulate distant sources "
      "like the sun, and ambient lights provide uniform illumination from all directions.")
      .def(py::init<>())
      .def_property(
          "position",
          [](const momentum::rasterizer::Light& light) { return light.position; },
          [](momentum::rasterizer::Light& light, const Eigen::Vector3f& position) {
            light.position = position;
          })
      .def_property(
          "color",
          [](const momentum::rasterizer::Light& light) { return light.color; },
          [](momentum::rasterizer::Light& light, const Eigen::Vector3f& color) {
            light.color = color;
          })
      .def_property(
          "type",
          [](const momentum::rasterizer::Light& light) { return light.type; },
          [](momentum::rasterizer::Light& light, const momentum::rasterizer::LightType& type) {
            light.type = type;
          })
      .def(
          "__repr__",
          [](const momentum::rasterizer::Light& self) {
            std::string typeStr;
            switch (self.type) {
              case momentum::rasterizer::LightType::Ambient:
                typeStr = "Ambient";
                break;
              case momentum::rasterizer::LightType::Directional:
                typeStr = "Directional";
                break;
              case momentum::rasterizer::LightType::Point:
                typeStr = "Point";
                break;
            }
            return fmt::format(
                "Light(type={}, position=({:.2f}, {:.2f}, {:.2f}), color=({:.2f}, {:.2f}, {:.2f}))",
                typeStr,
                self.position.x(),
                self.position.y(),
                self.position.z(),
                self.color.x(),
                self.color.y(),
                self.color.z());
          })
      .def_static(
          "create_point_light",
          [](const Eigen::Vector3f& position, std::optional<Eigen::Vector3f>& color) {
            return momentum::rasterizer::createPointLight(
                position, color.value_or(Eigen::Vector3f::Ones()));
          },
          py::arg("position"),
          py::arg("color") = std::optional<Eigen::Vector3f>{})
      .def_static(
          "create_directional_light",
          [](const Eigen::Vector3f& direction, std::optional<Eigen::Vector3f>& color) {
            return momentum::rasterizer::createDirectionalLight(
                direction, color.value_or(Eigen::Vector3f::Ones()));
          },
          py::arg("direction"),
          py::arg("color") = std::optional<Eigen::Vector3f>{})
      .def_static(
          "create_ambient_light",
          [](const std::optional<Eigen::Vector3f>& color) {
            return momentum::rasterizer::createAmbientLight(
                color.value_or(Eigen::Vector3f::Ones()));
          },
          py::arg("color") = std::optional<Eigen::Vector3f>{})
      .def(
          "transform",
          [](const momentum::rasterizer::Light& light, const Eigen::Matrix4f& xf) {
            return momentum::rasterizer::transformLight(light, Eigen::Affine3f(xf));
          },
          "Transform the light using the passed-in transform.",
          py::arg("xf"));

  m.def(
      "build_cameras_for_body",
      &buildCamerasForBody,
      R"(Build a batched vector of cameras that roughly face the body (default: face the front of the body).  If you pass in multiple frames of animation, the camera will ensure all frames are visible.

:param character: Character to use.
:param jointParameters: torch.Tensor of size (nBatch x [nFrames] x nJointParameters) or size (nJointParameters); can be computed from the modelParameters using :math:`ParameterTransform.apply`.
:param imageHeight: Height of the target image.
:param imageWidth: Width of the target image.
:param focalLength_mm: 35mm-equivalent focal length; e.g. focalLength=50 corresponds to a "normal" lens.
:param horizontal: whether the cameras are placed horizontally, assuming the Y axis is the world up direction.
:param camera_angle: what direction the camera looks at the body. default: 0, looking at front of body. pi/2: at left side of body.
:return: List of cameras, one for each element of the batch.)",
      py::arg("character"),
      py::arg("joint_parameters"),
      py::arg("image_height"),
      py::arg("image_width"),
      py::arg("focal_length_mm") = 50.0f,
      py::arg("horizontal") = false,
      py::arg("camera_angle") = 0.0f);

  m.def(
      "create_camera_for_body",
      &createCameraForBody,
      R"(Create a camera that roughly faces the body (default: face the front of the body).  If you pass in multiple frames of animation, the camera will ensure all frames are visible.

:param character: Character to use.
:param skeleton_states: numpy.ndarray of skeleton states with shape (nJoints, 8), (nBatch, nJoints, 8), or (nBatch, nFrames, nJoints, 8). Each joint has 8 values: (tx, ty, tz, rx, ry, rz, rw, s) representing translation, quaternion rotation (x, y, z, w), and uniform scale.
:param image_height: Height of the target image.
:param image_width: Width of the target image.
:param focal_length_mm: 35mm-equivalent focal length; e.g. focal_length_mm=50 corresponds to a "normal" lens.  Specified in millimeters because this is what photographers use.
:param horizontal: whether the cameras are placed horizontally, assuming the Y axis is the world up direction.
:param camera_angle: what direction the camera looks at the body. default: 0, looking at front of body. pi/2: at left side of body.
:return: a :class:`Camera`.)",
      py::arg("character"),
      py::arg("skeleton_states"),
      py::arg("image_height"),
      py::arg("image_width"),
      py::arg("focal_length_mm") = 50.0f,
      py::arg("horizontal") = false,
      py::arg("camera_angle") = 0.0f);

  m.def(
      "build_cameras_for_hand",
      &buildCamerasForHand,
      R"(Build a vector of cameras that roughly face inward from the front of the hand.

:param wristTransformation: Wrist transformation.
:param imageHeight: Height of the target image.
:param imageWidth: Width of the target image.
:return: List of cameras, one for each element of the batch.)",
      py::arg("wrist_transformation"),
      py::arg("image_height"),
      py::arg("image_width"));

  m.def(
      "build_cameras_for_hand_surface",
      &buildCamerasForHandSurface,
      R"(Build a vector of cameras that face over the plane.

:param imageHeight: Height of the target image.
:param imageWidth: Width of the target image.
:return: List of cameras, one for each element of the batch.)",
      py::arg("wrist_transformation"),
      py::arg("image_height"),
      py::arg("image_width"));

  m.def(
      "triangulate",
      &triangulate,
      R"(triangulate the polygon mesh.

:param faceOffests: numpy.ndarray defining the starting and end points of each polygon
:param facesIndices: numpy.ndarray of the face indices to vertex
)",
      py::arg("face_indices"),
      py::arg("face_offsets"));

  m.def(
      "subdivide_mesh",
      &subdivideMesh,
      R"(Subdivide the triangle mesh.

:param vertices: n x 3 numpy.ndarray of vertex positions.
:param normals: n x 3 numpy.ndarray of vertex normals.
:param triangles: n x 3 numpy.ndarray of triangles.
:param texture_coordinates: n x numpy.ndarray or texture coordinates.
:param texture_triangles: n x numpy.ndarray or texture triangles (see :func:`rasterize_mesh` for more details).).
:param levels: Maximum levels to subdivide (default = 1)
:param max_edge_length: Stop subdividing when the longest edge is shorter than this length.
:return: A tuple [vertices, normals, triangles, texture_coordinates, texture_triangles].
)",
      py::arg("vertices"),
      py::arg("normals"),
      py::arg("triangles"),
      py::arg("texture_coordinates") = std::optional<RowMatrixf>{},
      py::arg("texture_triangles") = std::optional<RowMatrixi>{},
      py::arg("levels") = 1,
      py::arg("max_edge_length") = 0);

  m.def(
      "rasterize_mesh",
      &rasterizeMesh,
      R"(Rasterize the triangle mesh using the passed-in camera onto a given RGB+depth buffer.  Uses an optimized cross-platform SIMD implementation.

Notes:

* You can rasterize multiple meshes to the same depth buffer by calling this function multiple times.
* To simplify the SIMD implementation, the width of the depth buffer must be a multiple of 8.  If you want to render a resolution that is not a multiple of 8, allocate an appropriately padded depth buffer (e.g. using :meth:`create_rasterizer_buffers`) and then extract the smaller image at the end.

:param vertex_positions: (nVert x 3) Tensor of vertex positions.
:param vertex_normals: (nVert x 3) Tensor of vertex normals.
:param triangles: (nVert x 3) Tensor of triangles.
:param camera: Camera to render from.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param material: Material to render with (assumes solid color for now).
:param texture_coordinates: Texture coordinates used with the mesh, indexed by texture_triangles (if present) or triangles otherwise.
:param texture_triangles: Triangles in texture coordinate space.  Must have the same number of triangles as the triangles input and is assumed to match the regular triangles input if not present.  This allows discontinuities in the texture map without needing to break up the mesh.
:param per_vertex_diffuse_color: A per-vertex diffuse color to use instead of the material's diffuse color.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param surface_normals_buffer: Buffer to render eye-space surface normals to; can be reused for multiple renders.  Should have dimensions [height x width x 3].
:param vertex_index_buffer: Optional buffer to rasterize the vertex indices to.  Useful for e.g. computing parts.
:param triangle_index_buffer: Optional buffer to rasterize the triangle indices to.  Useful for e.g. computing parts.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
)",
      py::arg("vertex_positions"),
      py::arg("vertex_normals"),
      py::arg("triangles"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("vertex_index_buffer") = std::optional<at::Tensor>{},
      py::arg("triangle_index_buffer") = std::optional<at::Tensor>{},
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("texture_coordinates") = std::optional<at::Tensor>{},
      py::arg("texture_triangles") = std::optional<at::Tensor>{},
      py::arg("per_vertex_diffuse_color") = std::optional<at::Tensor>{},
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_wireframe",
      &rasterizeWireframe,
      R"(Rasterize the triangle mesh as a wireframe.

See detailed notes under :meth:`rasterize_mesh`, above.

:param vertex_positions: (nVert x 3) Tensor of vertex positions.
:param triangles: (nVert x 3) Tensor of triangles.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param thickness: Thickness of the wireframe lines.
:param color: Wireframe color.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
)",
      py::arg("vertex_positions"),
      py::arg("triangles"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("thickness") = 1.0f,
      py::arg("color") = std::optional<Eigen::Vector3f>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_spheres",
      &rasterizeSpheres,
      R"(Rasterize spheres using the passed-in camera onto a given RGB+depth buffer.  Uses an optimized cross-platform SIMD implementation.

See detailed notes under :meth:`rasterize_mesh`, above.

:param center: (nSpheres x 3) Tensor of sphere centers.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param radius: (nSpheres) Optional Tensor of per-sphere radius values (defaults to 1).
:param color: (nSpheres x 3) optional Tensor of per-sphere colors (defaults to using the passed-in material).
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param subdivision_level: How many subdivision levels; more levels means more triangles, smoother spheres, but slower rendering.
)",
      py::arg("center"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("radius") = std::optional<at::Tensor>{},
      py::arg("color") = std::optional<at::Tensor>{},
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::kw_only(),
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("subdivision_level") = 2);

  m.def(
      "rasterize_camera_frustum",
      &rasterizeCameraFrustum,
      R"(Rasterize the camera frustum.

:param camera_frustum: Camera frustum to render.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param line_thickness: Thickness of the lines.
:param distance: Distance to project the frustum out into space (defaults to 10cm).
:param num_samples: Number of samples to use for computing the boundaries of the frustum (defaults to 20).
:param color: Color to use for the frustum (defaults to white).
:param model_matrix: Additional matrix to apply to the frustum.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something
    else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.
  )",
      py::arg("camera_frustum"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::arg("line_thickness") = 1.0f,
      py::arg("distance") = 10.0f,
      py::arg("num_samples") = 20,
      py::arg("color") = std::optional<Eigen::Vector3f>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_cylinders",
      &rasterizeCylinders,
      R"(Rasterize cylinders using the passed-in camera onto a given RGB+depth buffer.  Uses an optimized cross-platform SIMD implementation.

A cylinder is defined as extending from start_position to end_position with the radius provided by radius.

See detailed notes under :meth:`rasterize_mesh`, above.

:param start_position: (nCylinders x 3) torch.Tensor of starting positions.
:param end_position: (nCylinders x 3) torch.Tensor of ending positions.
:param camera: Camera to render from.
:param radius: (nSpheres) Optional tensor of per-cylinder radius values (defaults to 1).
:param color: (nVert x 3) Optional Tensor of per-cylinder colors (defaults to using the material parameter).
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param length_subdivisions: How many subdivisions along length; longer cylinders may need more to avoid looking chunky.
:param radius_subdivisions: How many subdivisions around cylinder radius; good values are between 16 and 64.
)",
      py::arg("start_position"),
      py::arg("end_position"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("radius") = std::optional<at::Tensor>{},
      py::arg("color") = std::optional<at::Tensor>{},
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::kw_only(),
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("length_subdivisions") = 16,
      py::arg("radius_subdivisions") = 16);

  m.def(
      "rasterize_capsules",
      &rasterizeCapsules,
      R"(Rasterize capsules using the passed-in camera onto a given RGB+depth buffer.

A capsule is defined as extending along the x axis in the local space defined by the transform.  It
has two radius values, one for the start and end of the capsule, and the ends of the capsules are spheres.

:param transformation: (nCapsules x 4 x 4) torch.Tensor of transformations from capsule-local space (oriented along the x axis) to world space.
:param radius: (nCapsules x 2) torch.Tensor of per-capsule start and end radius values.
:param length: (nCapsules) torch.Tensor of per-capsule length values.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param surface_normals_buffer: Buffer to render eye-space surface normals to; can be reused for multiple renders.  Should have dimensions [height x width x 3].
:param material: Material to render with (assumes solid color for now).
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param cylinder_length_subdivisions: How many subdivisions along cylinder length; longer cylinders may need more to avoid looking chunky.
:param cylinder_radius_subdivisions: How many subdivisions around cylinder radius; good values are between 16 and 64.
  )",
      py::arg("transformation"),
      py::arg("radius"),
      py::arg("length"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::kw_only(),
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("cylinder_length_subdivisions") = 16,
      py::arg("cylinder_radius_subdivisions") = 16);

  m.def(
      "rasterize_transforms",
      &rasterizeTransforms,
      R"(Rasterize a set of transforms as little frames using arrows.

  :param transforms: (n x 4 x 4) torch.Tensor of transforms to render.
  :param camera: Camera to render from.
  :param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
  :param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
  :param surface_normals_buffer: Buffer to render eye-space surface normals to; can be reused for multiple renders.
  :param scale: Scale of the arrows.
  :param material: Material to render with.  If not specified, then red/green/blue are used for the x/y/z axes.
  :param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
  :param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
  :param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
  :param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something
      else and avoid depth fighting.  Defaults to 0.
  :param image_offset: Offset by (x, y) pixels in image space.
  :param length_subdivisions: How many subdivisions along length; longer cylinders may need more to avoid looking chunky.
  :param radius_subdivisions: How many subdivisions around cylinder radius; good values are between 16 and 64. )",
      py::arg("transforms"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::kw_only(),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("scale") = 1.0f,
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("length_subdivisions") = 16,
      py::arg("radius_subdivisions") = 16);

  py::enum_<SkeletonStyle>(
      m,
      "SkeletonStyle",
      R"(Rendering style options for skeleton visualization. Different styles are optimized for different use cases, "
      "from technical debugging to publication-quality figures.)")
      .value(
          "Pipes",
          SkeletonStyle::Pipes,
          "Render joints as spheres with fixed-radius pipes connecting them. Useful when rendering a skeleton under a mesh.")
      .value(
          "Octahedrons",
          SkeletonStyle::Octahedrons,
          "Render joints as octahedrons. This gives much more sense of the joint orientations and is useful when rendering just the skeleton.")
      .value(
          "Lines",
          SkeletonStyle::Lines,
          "Render joints as lines. This would look nice in e.g. a paper figure. Note that all sizes are in pixels when this is used.");

  m.def(
      "rasterize_skeleton",
      &rasterizeSkeleton,
      R"(Rasterize the skeleton onto a given RGB+depth buffer by placing spheres at joints and connecting joints with cylinders.

See detailed notes under :meth:`rasterize_mesh`, above.

:param character: :class:`pymomentum.geometry.Character` whose skeleton to rasterize.
:param skeleton_state: State of the skeleton.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param sphere_material: Material to use for spheres at joints.
:param cylinder_material: Material to use for cylinders at joints.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param active_joints: Bool tensor specifying which joints to render.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param sphere_radius: Radius for spheres at joints.
:param cylinder_radius: Radius for cylinders between joints.
:param sphere_subdivision_level: How many subdivision levels; more levels means more triangles, smoother spheres, but slower rendering.  Good values are between 1 and 3.
:param cylinder_length_subdivisions: How many subdivisions along cylinder length; longer cylinders may need more to avoid looking chunky.
:param cylinder_radius_subdivisions: How many subdivisions around cylinder radius; good values are between 16 and 64.
)",
      py::arg("character"),
      py::arg("skeleton_state"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("sphere_material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("cylinder_material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("active_joints") = std::optional<at::Tensor>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("sphere_radius") = 2.0f,
      py::arg("cylinder_radius") = 1.0f,
      py::arg("sphere_subdivision_level") = 2,
      py::arg("cylinder_length_subdivisions") = 16,
      py::arg("cylinder_radius_subdivisions") = 16,
      py::arg("style") = SkeletonStyle::Pipes);

  m.def(
      "rasterize_character",
      &rasterizeCharacter,
      R"(Rasterize the posed character using the passed-in camera onto a given RGB+depth buffer.  Uses an optimized cross-platform SIMD implementation.

See detailed notes under :meth:`rasterize_mesh`, above.

:param character: :class:`pymomentum.geometry.Character` whose skeleton to rasterize.
:param skeleton_state: State of the skeleton.
:param camera: Camera to render from.
:param material: Material to render with (assumes solid color for now).
:param per_vertex_diffuse_color: A per-vertex diffuse color to use instead of the material's diffuse color.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Additional matrix to apply to the model.  Unlike the camera transforms, it is allowed to have scaling and/or shearing.
:param back_face_culling: Enable back-face culling (speeds up the render).
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param vertex_index_buffer: Optional buffer to rasterize the vertex indices to.  Useful for e.g. computing parts.
:param triangle_index_buffer: Optional buffer to rasterize the triangle indices to.  Useful for e.g. computing parts.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param wireframe_color: If provided, color to use for the wireframe (defaults to no wireframe).
)",
      py::arg("character"),
      py::arg("skeleton_state"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("vertex_index_buffer") = std::optional<at::Tensor>{},
      py::arg("triangle_index_buffer") = std::optional<at::Tensor>{},
      py::arg("material") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("per_vertex_diffuse_color") = std::optional<at::Tensor>{},
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("wireframe_color") = std::optional<Eigen::Vector3f>{});

  m.def(
      "rasterize_checkerboard",
      &rasterizeCheckerboard,
      R"(Rasterize a checkerboard floor in the x-z plane (with y up).

See detailed notes under :meth:`rasterize_mesh`, above.

:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param material1: Material to use for even checks.
:param material2: Material to use for odd checks.
:param lights: Lights to use in renderering, in world-space.  If none are given, a default light setup is used.
:param model_matrix: Matrix to use to transform the plane from the origin.
:param back_face_culling: Cull back-facing triangles.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
:param width: Width of the plane in x/z.
:param num_checks: Number of checks in each axis.
:param subdivisions: How much to divide up each check.
)",
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("surface_normals_buffer") = std::optional<at::Tensor>{},
      py::arg("material1") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("material2") = std::optional<momentum::rasterizer::PhongMaterial>{},
      py::arg("lights") = std::optional<std::vector<momentum::rasterizer::Light>>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{},
      py::arg("width") = 50.0f,
      py::arg("num_checks") = 10,
      py::arg("subdivisions") = 1);

  m.def(
      "rasterize_lines",
      &rasterizeLines,
      R"(Rasterize lines to the provided RGB and z buffers.

The advantage of using rasterization to draw lines instead of just drawing them with e.g. opencv is that it will use the correct camera model and respect the z buffer.

See detailed notes under :meth:`rasterize_mesh`, above.

:param positions: (nLines x 2 x 3) torch.Tensor of start/end positions.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param width: Width of the lines, currently is the same across all lines.
:param color: Line color, currently is shared across all lines.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
)",
      py::arg("positions"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("thickness") = 1.0f,
      py::arg("color") = std::optional<Eigen::Vector3f>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_circles",
      &rasterizeCircles,
      R"(Rasterize circles to the provided RGB and z buffers.

The advantage of using rasterization to draw circles instead of just drawing them with e.g. opencv is that it will use the correct camera model and respect the z buffer.

See detailed notes under :meth:`rasterize_mesh`, above.

:param positions: (nCircles x 3) torch.Tensor of circle centers.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: RGB-buffer to render geometry onto; can be reused for multiple renders.
:param line_thickness: Thickness of the circle outline.
:param radius: Radius of the circle.
:param line_color: Color of the outline, is transparent if not provided.
:param fill_color: Line color, is transparent if not provided.
:param near_clip: Clip any triangles closer than this depth.  Defaults to 0.1.
:param depth_offset: Offset the depth values.  Nonzero values can be used to render something slightly in front of something else and avoid depth fighting.  Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.  Can be used to render e.g. two characters next to each other for comparison without needing to create a special camera.
)",
      py::arg("positions"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("line_thickness") = 1.0f,
      py::arg("radius") = 3.0f,
      py::arg("line_color") = std::optional<Eigen::Vector3f>{},
      py::arg("fill_color") = std::optional<Eigen::Vector3f>{},
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  // 2D rasterization functions that operate directly in image space
  // without camera projection or z-buffer
  m.def(
      "rasterize_lines_2d",
      &rasterizeLines2D,
      R"(Rasterize lines directly in 2D image space without camera projection or z-buffer.

:param positions: (nLines x 4) torch.Tensor of line start/end positions in image space [start_x, start_y, end_x, end_y].
:param rgb_buffer: RGB-buffer to render geometry onto.
:param thickness: Thickness of the lines.
:param color: Line color.
:param z_buffer: Optional Z-buffer to write zeros to for alpha matting.
:param image_offset: Offset by (x, y) pixels in image space.
)",
      py::arg("positions"),
      py::arg("rgb_buffer"),
      py::arg("thickness") = 1.0f,
      py::arg("color") = std::optional<Eigen::Vector3f>{},
      py::arg("z_buffer") = std::optional<at::Tensor>{},
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_circles_2d",
      &rasterizeCircles2D,
      R"(Rasterize circles directly in 2D image space without camera projection or z-buffer.

:param positions: (nCircles x 2) torch.Tensor of circle centers in image space [x, y].
:param rgb_buffer: RGB-buffer to render geometry onto.
:param line_thickness: Thickness of the circle outline.
:param radius: Radius of the circle.
:param line_color: Color of the outline, is transparent if not provided.
:param fill_color: Fill color, is transparent if not provided.
:param z_buffer: Optional Z-buffer to write zeros to for alpha matting.
:param image_offset: Offset by (x, y) pixels in image space.
)",
      py::arg("positions"),
      py::arg("rgb_buffer"),
      py::arg("line_thickness") = 1.0f,
      py::arg("radius") = 3.0f,
      py::arg("line_color") = std::optional<Eigen::Vector3f>{},
      py::arg("fill_color") = std::optional<Eigen::Vector3f>{},
      py::arg("z_buffer") = std::optional<at::Tensor>{},
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "create_z_buffer",
      &createZBuffer,
      R"(Creates a padded buffer suitable for rasterization.

:param camera: Camera to render from.

:return: A z_buffer torch.Tensor (height, padded_width) suitable for use in the :meth:`rasterize` function.
)",
      py::arg("camera"),
      py::arg("far_clip") = FLT_MAX);

  m.def(
      "create_rgb_buffer",
      &createRGBBuffer,
      R"(Creates a padded RGB buffer suitable for rasterization.

:param camera: Camera to render from.
:param background_color: Background color, defaults to all-black (0, 0, 0).

:return: A rgb_buffer torch.Tensor (height, padded_width, 3) suitable for use in the :meth:`rasterize` function. After rasterization, use `rgb_buffer[:, 0 : camera.image_width, :]` to get the rendered image.
)",
      py::arg("camera"),
      py::arg("background_color") = std::optional<Eigen::Vector3f>{});

  m.def(
      "create_index_buffer",
      &createIndexBuffer,
      R"(Creates a padded RGB buffer suitable for storing triangle or vertex indices during rasterization.

:param camera: Camera to render from.

:return: An integer tensor (height, padded_width) suitable for passing in as an index buffer to the :meth:`rasterize` function.
)",
      py::arg("camera"));

  m.def(
      "alpha_matte",
      &alphaMatte,
      R"(Use alpha matting to overlay a rasterized image onto a background image.

This function includes a few features which simplify using it with the rasterizer:
1. Depth buffer is automatically converted to an alpha matte.
2. Supersampled images are handled correctly: if your rgb_buffer is an integer multiple of the target image, it will automatically be smoothed and converted to fractional alpha.
3. You can apply an additional global alpha on top of the per-pixel alpha.

:param depth_buffer: A z-buffer as generated by :meth:`create_z_buffer`.
:param rgb_buffer: An rgb_buffer as created by :meth:`create_rgb_buffer`.
:param target_image: A target RGB image to overlay the rendered image onto.  Will be written in place.
:param alpha: A global alpha between 0 and 1 to multiply the source image by.  Defaults to 1.
)",
      py::arg("depth_buffer"),
      py::arg("rgb_buffer"),
      py::arg("target_image"),
      py::arg("alpha") = 1.0f);

  m.def(
      "create_shadow_projection_matrix",
      [](const momentum::rasterizer::Light& light,
         const std::optional<Eigen::Vector3f>& planeNormal_in,
         const std::optional<Eigen::Vector3f>& planeOrigin_in) {
        const Eigen::Vector3f planeNormal =
            planeNormal_in.value_or(Eigen::Vector3f::UnitY()).normalized();
        const Eigen::Vector3f planeOrigin = planeOrigin_in.value_or(Eigen::Vector3f::Zero());
        const Eigen::Vector4f planeVec(
            planeNormal.x(), planeNormal.y(), planeNormal.z(), -planeNormal.dot(planeOrigin));

        Eigen::Vector4f lightVec = Eigen::Vector4f::Zero();
        switch (light.type) {
          case momentum::rasterizer::LightType::Directional:
            lightVec =
                Eigen::Vector4f(light.position.x(), light.position.y(), light.position.z(), 0.0f);
            break;
          case momentum::rasterizer::LightType::Point:
            lightVec =
                Eigen::Vector4f(light.position.x(), light.position.y(), light.position.z(), 1.0f);
            break;
          case momentum::rasterizer::LightType::Ambient:
            throw std::runtime_error("Shadows not supported for ambient lights");
        }

        Eigen::Matrix4f shadowMat = lightVec.dot(planeVec) * Eigen::Matrix4f::Identity() -
            (lightVec * planeVec.transpose());
        return shadowMat;
      },
      R"(Create a modelview matrix that when passed to rasterize_mesh will project all the vertices to the passed-in plane.

This is useful for rendering shadows using the classic projection shadows technique from OpenGL.

:param light: The light to use to cast shadows.
:param plane_normal: The normal vector of the plane (defaults to y-up, (0, 1, 0)).
:param plane_origin: A point on the plane, defaults to the origin (0, 0, 0).

:return: a 4x4 matrix that can be passed to the rasterizer function.)",
      py::arg("light"),
      py::arg("plane_normal") = std::optional<Eigen::Vector3f>{},
      py::arg("plane_origin") = std::optional<Eigen::Vector3f>{});

  m.def(
      "rasterize_text",
      &rasterizeText,
      R"(Rasterize text at 3D world positions.

Projects 3D positions to image space using the camera and renders text strings at those locations using an embedded bitmap font.

:param positions: (nTexts x 3) torch.Tensor of 3D positions in world coordinates.
:param texts: List of strings to render at each position.
:param camera: Camera to render from.
:param z_buffer: Z-buffer to render geometry onto; can be reused for multiple renders.
:param rgb_buffer: Optional RGB-buffer to render geometry onto.
:param color: RGB color for the text. Defaults to white (1, 1, 1).
:param text_scale: Integer scaling factor for text size (1 = 1 pixel per font pixel). Defaults to 1.
:param horizontal_alignment: Horizontal text alignment (Left, Center, or Right). Defaults to Left.
:param vertical_alignment: Vertical text alignment (Top, Center, or Bottom). Defaults to Top.
:param model_matrix: Additional matrix to apply to the model. Defaults to identity matrix.
:param near_clip: Clip any text closer than this depth. Defaults to 0.1.
:param depth_offset: Offset the depth values. Defaults to 0.
:param image_offset: Offset by (x, y) pixels in image space.
)",
      py::arg("positions"),
      py::arg("texts"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::optional<at::Tensor>{},
      py::kw_only(),
      py::arg("color") = Eigen::Vector3f(1.0f, 1.0f, 1.0f),
      py::arg("text_scale") = 1,
      py::arg("horizontal_alignment") = momentum::rasterizer::HorizontalAlignment::Left,
      py::arg("vertical_alignment") = momentum::rasterizer::VerticalAlignment::Top,
      py::arg("model_matrix") = std::optional<Eigen::Matrix4f>{},
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f,
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});

  m.def(
      "rasterize_text_2d",
      &rasterizeText2D,
      R"(Rasterize text directly in 2D image space without camera projection or depth testing.

:param positions: (nTexts x 2) torch.Tensor of 2D positions in image coordinates.
:param texts: List of strings to render at each position.
:param rgb_buffer: RGB-buffer to render geometry onto.
:param color: RGB color for the text. Defaults to white (1, 1, 1).
:param text_scale: Integer scaling factor for text size (1 = 1 pixel per font pixel). Defaults to 1.
:param horizontal_alignment: Horizontal text alignment (Left, Center, or Right). Defaults to Left.
:param vertical_alignment: Vertical text alignment (Top, Center, or Bottom). Defaults to Top.
:param z_buffer: Optional Z-buffer to write zeros to for alpha matting.
:param image_offset: Offset by (x, y) pixels in image space.
)",
      py::arg("positions"),
      py::arg("texts"),
      py::arg("rgb_buffer"),
      py::arg("color") = Eigen::Vector3f(1.0f, 1.0f, 1.0f),
      py::arg("text_scale") = 1,
      py::arg("horizontal_alignment") = momentum::rasterizer::HorizontalAlignment::Left,
      py::arg("vertical_alignment") = momentum::rasterizer::VerticalAlignment::Top,
      py::arg("z_buffer") = std::optional<at::Tensor>{},
      py::arg("image_offset") = std::optional<Eigen::Vector2f>{});
}
