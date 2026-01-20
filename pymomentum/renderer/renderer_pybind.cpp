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

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/rasterizer/camera.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/text_rasterizer.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <optional>

namespace py = pybind11;
namespace mm = momentum;

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

  // Bind IntrinsicsModel and its derived classes
  py::class_<
      momentum::rasterizer::IntrinsicsModel,
      std::shared_ptr<momentum::rasterizer::IntrinsicsModel>>(
      m, "IntrinsicsModel", "Base class for camera intrinsics models")
      .def_property_readonly(
          "image_width",
          &momentum::rasterizer::IntrinsicsModel::imageWidth,
          "Width of the image in pixels")
      .def_property_readonly(
          "image_height",
          &momentum::rasterizer::IntrinsicsModel::imageHeight,
          "Height of the image in pixels")
      .def_property_readonly(
          "fx", &momentum::rasterizer::IntrinsicsModel::fx, "Focal length in x direction (pixels)")
      .def_property_readonly(
          "fy", &momentum::rasterizer::IntrinsicsModel::fy, "Focal length in y direction (pixels)")
      .def(
          "__repr__",
          [](const momentum::rasterizer::IntrinsicsModel& self) {
            return fmt::format(
                "IntrinsicsModel(image_size=({}, {}), focal_length=({:.2f}, {:.2f}))",
                self.imageWidth(),
                self.imageHeight(),
                self.fx(),
                self.fy());
          })
      .def(
          "project",
          [](const momentum::rasterizer::IntrinsicsModel& intrinsics,
             const py::array_t<float>& points) -> py::array_t<float> {
            if (points.ndim() != 2 || points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto pts_acc = points.unchecked<2>();

            for (size_t i = 0; i < points.shape(0); i += momentum::rasterizer::kSimdPacketSize) {
              momentum::rasterizer::FloatP px, py, pz;
              auto nPtsCur = std::min(momentum::rasterizer::kSimdPacketSize, points.shape(0) - i);
              for (int k = 0; k < nPtsCur; ++k) {
                px[k] = pts_acc(i + k, 0);
                py[k] = pts_acc(i + k, 1);
                pz[k] = pts_acc(i + k, 2);
              }

              const auto [res, valid] =
                  intrinsics.project(momentum::rasterizer::Vector3fP(px, py, pz));

              for (int k = 0; k < nPtsCur; ++k) {
                res_acc(i + k, 0) = res.x()[k];
                res_acc(i + k, 1) = res.y()[k];
                res_acc(i + k, 2) = res.z()[k];
              }
            }
            return result;
          },
          R"(Project 3D points in camera space to 2D image coordinates.

:param points: (N x 3) array of 3D points in camera coordinate space to project.
:return: (N x 3) array of projected points where columns are [x, y, depth] in image coordinates.)",
          py::arg("points"))
      .def(
          "upsample",
          &momentum::rasterizer::IntrinsicsModel::upsample,
          R"(Create a new intrinsics model upsampled by the given factor.

:param factor: Upsampling factor (e.g., 2.0 doubles the resolution).
:return: A new IntrinsicsModel instance with upsampled parameters.)",
          py::arg("factor"))
      .def(
          "downsample",
          &momentum::rasterizer::IntrinsicsModel::downsample,
          R"(Create a new intrinsics model downsampled by the given factor.

:param factor: Downsampling factor (e.g., 2.0 halves the resolution).
:return: A new IntrinsicsModel instance with downsampled parameters.)",
          py::arg("factor"))
      .def(
          "crop",
          &momentum::rasterizer::IntrinsicsModel::crop,
          R"(Create a new intrinsics model cropped to a sub-region of the image.

:param top: Top offset in pixels.
:param left: Left offset in pixels.
:param width: New width in pixels after cropping.
:param height: New height in pixels after cropping.
:return: A new IntrinsicsModel instance with cropped parameters.)",
          py::arg("top"),
          py::arg("left"),
          py::arg("width"),
          py::arg("height"))
      .def(
          "resize",
          &momentum::rasterizer::IntrinsicsModel::resize,
          R"(Create a new intrinsics model resized to new image dimensions.

:param image_width: New image width in pixels.
:param image_height: New image height in pixels.
:return: A new IntrinsicsModel instance with resized parameters.)",
          py::arg("image_width"),
          py::arg("image_height"));

  py::class_<momentum::rasterizer::OpenCVDistortionParameters>(
      m, "OpenCVDistortionParameters", "OpenCV distortion parameters")
      .def(py::init<>(), "Initialize with default parameters (no distortion)")
      .def_readwrite(
          "k1",
          &momentum::rasterizer::OpenCVDistortionParameters::k1,
          "Radial distortion coefficient k1")
      .def_readwrite(
          "k2",
          &momentum::rasterizer::OpenCVDistortionParameters::k2,
          "Radial distortion coefficient k2")
      .def_readwrite(
          "k3",
          &momentum::rasterizer::OpenCVDistortionParameters::k3,
          "Radial distortion coefficient k3")
      .def_readwrite(
          "k4",
          &momentum::rasterizer::OpenCVDistortionParameters::k4,
          "Radial distortion coefficient k4")
      .def_readwrite(
          "k5",
          &momentum::rasterizer::OpenCVDistortionParameters::k5,
          "Radial distortion coefficient k5")
      .def_readwrite(
          "k6",
          &momentum::rasterizer::OpenCVDistortionParameters::k6,
          "Radial distortion coefficient k6")
      .def_readwrite(
          "p1",
          &momentum::rasterizer::OpenCVDistortionParameters::p1,
          "Tangential distortion coefficient p1")
      .def_readwrite(
          "p2",
          &momentum::rasterizer::OpenCVDistortionParameters::p2,
          "Tangential distortion coefficient p2")
      .def_readwrite(
          "p3",
          &momentum::rasterizer::OpenCVDistortionParameters::p3,
          "Tangential distortion coefficient p3")
      .def_readwrite(
          "p4",
          &momentum::rasterizer::OpenCVDistortionParameters::p4,
          "Tangential distortion coefficient p4")
      .def("__repr__", [](const momentum::rasterizer::OpenCVDistortionParameters& self) {
        return fmt::format(
            "OpenCVDistortionParameters(k1={:.4f}, k2={:.4f}, k3={:.4f}, k4={:.4f}, k5={:.4f}, k6={:.4f}, p1={:.4f}, p2={:.4f}, p3={:.4f}, p4={:.4f})",
            self.k1,
            self.k2,
            self.k3,
            self.k4,
            self.k5,
            self.k6,
            self.p1,
            self.p2,
            self.p3,
            self.p4);
      });

  // Factory function for PinholeIntrinsicsModel to avoid template ambiguity
  auto pinholeFactory =
      [](int32_t imageWidth,
         int32_t imageHeight,
         std::optional<float> fx,
         std::optional<float> fy,
         std::optional<float> cx,
         std::optional<float> cy) -> std::shared_ptr<momentum::rasterizer::PinholeIntrinsicsModel> {
    // Default focal length calculation: "normal" lens is a 50mm lens on
    // a 35mm camera body
    const float focal_length_cm = 5.0f;
    const float film_width_cm = 3.6f;
    const float default_focal_length_pixels = (focal_length_cm / film_width_cm) * (float)imageWidth;

    if (fx.has_value() != fy.has_value()) {
      throw std::runtime_error("fx and fy must be both specified or both omitted");
    }

    if (cx.has_value() != cy.has_value()) {
      throw std::runtime_error("cx and cy must be both specified or both omitted");
    }

    return std::make_shared<momentum::rasterizer::PinholeIntrinsicsModel>(
        imageWidth,
        imageHeight,
        fx.value_or(default_focal_length_pixels),
        fy.value_or(default_focal_length_pixels),
        cx.value_or(imageWidth / 2.0f),
        cy.value_or(imageHeight / 2.0f));
  };

  py::class_<
      momentum::rasterizer::PinholeIntrinsicsModel,
      momentum::rasterizer::IntrinsicsModel,
      std::shared_ptr<momentum::rasterizer::PinholeIntrinsicsModel>>(
      m, "PinholeIntrinsicsModel", "Pinhole camera intrinsics model without distortion")
      .def(
          py::init(pinholeFactory),
          R"(Create a pinhole camera model with specified focal lengths and image dimensions.

:param image_width: Width of the image in pixels.
:param image_height: Height of the image in pixels.
:param fx: Focal length in x direction (pixels). Defaults to computed value based on 50mm equivalent lens.
:param fy: Focal length in y direction (pixels). Defaults to computed value based on 50mm equivalent lens.
:param cx: Principal point x-coordinate (pixels). Defaults to image center if not provided.
:param cy: Principal point y-coordinate (pixels). Defaults to image center if not provided.
:return: A new PinholeIntrinsicsModel instance.)",
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("fx") = std::nullopt,
          py::arg("fy") = std::nullopt,
          py::arg("cx") = std::nullopt,
          py::arg("cy") = std::nullopt)
      .def_property_readonly(
          "cx",
          &momentum::rasterizer::PinholeIntrinsicsModel::cx,
          "Principal point x-coordinate (pixels)")
      .def_property_readonly(
          "cy",
          &momentum::rasterizer::PinholeIntrinsicsModel::cy,
          "Principal point y-coordinate (pixels)")
      .def("__repr__", [](const momentum::rasterizer::PinholeIntrinsicsModel& self) {
        return fmt::format(
            "PinholeIntrinsicsModel(image_size=({}, {}), fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f})",
            self.imageWidth(),
            self.imageHeight(),
            self.fx(),
            self.fy(),
            self.cx(),
            self.cy());
      });

  // Factory function for OpenCVIntrinsicsModel to avoid template ambiguity
  auto opencvFactory =
      [](int32_t imageWidth,
         int32_t imageHeight,
         std::optional<float> fx,
         std::optional<float> fy,
         std::optional<float> cx,
         std::optional<float> cy,
         std::optional<momentum::rasterizer::OpenCVDistortionParameters> distortionParams)
      -> std::shared_ptr<momentum::rasterizer::OpenCVIntrinsicsModel> {
    // Default focal length calculation: "normal" lens is a 50mm
    // lens on a 35mm camera body
    const float focal_length_cm = 5.0f;
    const float film_width_cm = 3.6f;
    const float default_focal_length_pixels = (focal_length_cm / film_width_cm) * (float)imageWidth;
    return std::make_shared<momentum::rasterizer::OpenCVIntrinsicsModel>(
        imageWidth,
        imageHeight,
        fx.value_or(default_focal_length_pixels),
        fy.value_or(default_focal_length_pixels),
        cx.value_or(imageWidth / 2.0f),
        cy.value_or(imageHeight / 2.0f),
        distortionParams.value_or(momentum::rasterizer::OpenCVDistortionParameters{}));
  };

  py::class_<
      momentum::rasterizer::OpenCVIntrinsicsModel,
      momentum::rasterizer::IntrinsicsModel,
      std::shared_ptr<momentum::rasterizer::OpenCVIntrinsicsModel>>(
      m, "OpenCVIntrinsicsModel", "OpenCV camera intrinsics model with distortion")
      .def(
          py::init(opencvFactory),
          R"(Create an OpenCV camera model with specified parameters and optional distortion.

:param image_width: Width of the image in pixels.
:param image_height: Height of the image in pixels.
:param fx: Focal length in x direction (pixels).
:param fy: Focal length in y direction (pixels).
:param cx: Principal point x-coordinate (pixels).
:param cy: Principal point y-coordinate (pixels).
:param distortion_params: Optional OpenCV distortion parameters. Defaults to no distortion if not provided.
:return: A new OpenCVIntrinsicsModel instance.)",
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("fx"),
          py::arg("fy"),
          py::arg("cx"),
          py::arg("cy"),
          py::arg("distortion_params") = std::nullopt)
      .def_property_readonly(
          "cx",
          &momentum::rasterizer::OpenCVIntrinsicsModel::cx,
          "Principal point x-coordinate (pixels)")
      .def_property_readonly(
          "cy",
          &momentum::rasterizer::OpenCVIntrinsicsModel::cy,
          "Principal point y-coordinate (pixels)")
      .def("__repr__", [](const momentum::rasterizer::OpenCVIntrinsicsModel& self) {
        return fmt::format(
            "OpenCVIntrinsicsModel(image_size=({}, {}), fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f})",
            self.imageWidth(),
            self.imageHeight(),
            self.fx(),
            self.fy(),
            self.cx(),
            self.cy());
      });

  // Factory function for Camera to avoid template ambiguity
  auto cameraFactory =
      [](std::shared_ptr<const momentum::rasterizer::IntrinsicsModel> intrinsics,
         const std::optional<Eigen::Matrix4f>& eye_from_world) -> momentum::rasterizer::Camera {
    return momentum::rasterizer::Camera(
        intrinsics, Eigen::Affine3f(eye_from_world.value_or(Eigen::Matrix4f::Identity())));
  };

  // Bind Camera class
  py::class_<momentum::rasterizer::Camera>(m, "Camera", "Camera for rendering")
      .def(
          py::init(cameraFactory),
          R"(Create a camera with specified intrinsics and pose.

:param intrinsics_model: Camera intrinsics model defining focal length, principal point, and image dimensions.
:param eye_from_world: Optional 4x4 transformation matrix from world space to camera/eye space. Defaults to identity matrix if not provided.
:return: A new Camera instance with the specified intrinsics and pose.)",
          py::arg("intrinsics_model"),
          py::arg("eye_from_world") = std::nullopt)
      .def(
          "__repr__",
          [](const momentum::rasterizer::Camera& self) {
            Eigen::Vector3f position = self.eyeFromWorld().inverse().translation();
            return fmt::format(
                "Camera(image_size=({}, {}), focal_length=({:.2f}, {:.2f}), position=({:.2f}, {:.2f}, {:.2f}))",
                self.imageWidth(),
                self.imageHeight(),
                self.fx(),
                self.fy(),
                position.x(),
                position.y(),
                position.z());
          })
      .def_property_readonly(
          "image_width", &momentum::rasterizer::Camera::imageWidth, "Width of the image in pixels")
      .def_property_readonly(
          "image_height",
          &momentum::rasterizer::Camera::imageHeight,
          "Height of the image in pixels")
      .def_property_readonly(
          "fx", &momentum::rasterizer::Camera::fx, "Focal length in x direction (pixels)")
      .def_property_readonly(
          "fy", &momentum::rasterizer::Camera::fy, "Focal length in y direction (pixels)")
      .def_property_readonly(
          "intrinsics_model",
          &momentum::rasterizer::Camera::intrinsicsModel,
          "The camera's intrinsics model")
      .def_property(
          "T_eye_from_world",
          [](const momentum::rasterizer::Camera& self) -> Eigen::Matrix4f {
            return self.eyeFromWorld().matrix();
          },
          [](momentum::rasterizer::Camera& self, const Eigen::Matrix4f& value) {
            self.setEyeFromWorld(Eigen::Affine3f(value));
          },
          "Transform from world space to camera/eye space")
      .def_property(
          "T_world_from_eye",
          [](const momentum::rasterizer::Camera& self) -> Eigen::Matrix4f {
            return self.worldFromEye().matrix();
          },
          [](momentum::rasterizer::Camera& self, const Eigen::Matrix4f& value) {
            self.setEyeFromWorld(Eigen::Affine3f(value));
          },
          "Transform from world space to camera/eye space")
      .def_property_readonly(
          "center_of_projection",
          [](const momentum::rasterizer::Camera& self) -> Eigen::Vector3f {
            return (self.eyeFromWorld().inverse().translation()).eval();
          },
          "Position of the camera center in world space")
      .def(
          "look_at",
          [](const momentum::rasterizer::Camera& self,
             const Eigen::Vector3f& position,
             const std::optional<Eigen::Vector3f>& target,
             const std::optional<Eigen::Vector3f>& up) {
            return self.lookAt(
                position,
                target.value_or(Eigen::Vector3f::Zero()),
                up.value_or(Eigen::Vector3f::UnitY()));
          },
          R"(Position the camera to look at a specific target point.

:param position: 3D position where the camera should be placed.
:param target: 3D point the camera should look at. Defaults to origin (0,0,0) if not provided.
:param up: Up vector for camera orientation. Defaults to (0,1,0) if not provided.
:return: A new Camera instance positioned to look at the target.)",
          py::arg("position"),
          py::arg("target") = std::nullopt,
          py::arg("up") = std::nullopt)
      .def(
          "frame",
          [](const momentum::rasterizer::Camera& self,
             const py::array_t<float>& points,
             float min_z,
             float edge_padding) -> momentum::rasterizer::Camera {
            if (points.ndim() != 2 || points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            // Convert py::array_t<float> to std::vector<Eigen::Vector3f>
            std::vector<Eigen::Vector3f> eigenPoints;
            eigenPoints.reserve(points.shape(0));

            auto pts_acc = points.unchecked<2>();
            for (size_t i = 0; i < points.shape(0); ++i) {
              eigenPoints.emplace_back(pts_acc(i, 0), pts_acc(i, 1), pts_acc(i, 2));
            }

            return self.framePoints(eigenPoints, min_z, edge_padding);
          },
          R"(Adjust the camera position to ensure all specified points are in view.

:param points: (N x 3) array of 3D points that should be visible in the camera view.
:param min_z: Minimum distance from camera to maintain. Defaults to 0.1.
:param edge_padding: Padding factor to add around the points as a fraction of the image size. Defaults to 0.05.
:return: A new Camera instance positioned to frame all the specified points.)",
          py::arg("points"),
          py::arg("min_z") = 0.1f,
          py::arg("edge_padding") = 0.05f)
      .def_property_readonly(
          "world_space_principle_axis",
          [](const momentum::rasterizer::Camera& self) -> Eigen::Vector3f {
            // The principle axis is the direction the camera is looking
            // In camera space, this is the positive Z axis (0, 0, 1)
            Eigen::Vector3f cameraSpacePrincipalAxis = Eigen::Vector3f::UnitZ();
            return (self.worldFromEye().linear() * cameraSpacePrincipalAxis).eval();
          },
          "Camera world-space principal axis (direction the camera is looking)")
      .def(
          "upsample",
          [](const momentum::rasterizer::Camera& self, float factor) {
            return momentum::rasterizer::Camera(
                self.intrinsicsModel()->upsample(factor), self.eyeFromWorld());
          },
          R"(Create a new camera with upsampled resolution by the given factor.

:param factor: Upsampling factor (e.g., 2.0 doubles the resolution).
:return: A new Camera instance with upsampled intrinsics and same pose.)",
          py::arg("factor"))
      .def(
          "crop",
          [](const momentum::rasterizer::Camera& self,
             int32_t top,
             int32_t left,
             int32_t width,
             int32_t height) { return self.crop(top, left, width, height); },
          R"(Create a new camera with cropped image region.

:param top: Top offset in pixels.
:param left: Left offset in pixels.
:param width: New width in pixels after cropping.
:param height: New height in pixels after cropping.
:return: A new Camera instance with cropped intrinsics and same pose.)",
          py::arg("top"),
          py::arg("left"),
          py::arg("width"),
          py::arg("height"))
      .def(
          "project",
          [](const momentum::rasterizer::Camera& self,
             const py::array_t<float>& world_points) -> py::array_t<float> {
            if (world_points.ndim() != 2 || world_points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{world_points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto pts_acc = world_points.unchecked<2>();

            for (size_t i = 0; i < world_points.shape(0); ++i) {
              // Extract world point
              Eigen::Vector3f world_pt(pts_acc(i, 0), pts_acc(i, 1), pts_acc(i, 2));

              // Use the camera's project method directly
              auto [projected_pt, is_valid] = self.project(world_pt);

              res_acc(i, 0) = projected_pt.x();
              res_acc(i, 1) = projected_pt.y();
              res_acc(i, 2) = projected_pt.z();
            }
            return result;
          },
          R"(Project 3D points from world space to 2D image coordinates.

:param world_points: (N x 3) array of 3D points in world coordinate space to project.
:return: (N x 3) array of projected points where columns are [x, y, depth] in image coordinates.)",
          py::arg("world_points"))
      .def(
          "unproject",
          [](const momentum::rasterizer::Camera& self,
             const py::array_t<float>& image_points) -> py::array_t<float> {
            if (image_points.ndim() != 2 || image_points.shape(1) != 3) {
              throw std::runtime_error(
                  "Expected image_points to be a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{image_points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto img_acc = image_points.unchecked<2>();

            for (size_t i = 0; i < image_points.shape(0); ++i) {
              // Create 3D image point (u, v, z)
              const Eigen::Vector3f image_point(img_acc(i, 0), img_acc(i, 1), img_acc(i, 2));

              // Use the camera's unproject method to get world coordinates
              // directly
              auto [world_pt, is_valid] = self.unproject(image_point);

              res_acc(i, 0) = world_pt.x();
              res_acc(i, 1) = world_pt.y();
              res_acc(i, 2) = world_pt.z();
            }
            return result;
          },
          R"(Unproject 3D image coordinates to 3D points in world space.

:param image_points: (N x 3) array of 3D points in image coordinates [x, y, depth].
:return: (N x 3) array of 3D points in world coordinate space.)",
          py::arg("image_points"));

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
