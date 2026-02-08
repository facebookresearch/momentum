/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/rasterizer.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <optional>

namespace py = pybind11;

PYBIND11_MODULE(camera, m) {
  m.attr("__name__") = "pymomentum.camera";
  m.doc() = "Camera classes for projection and rendering in pymomentum.";

  // Bind IntrinsicsModel and its derived classes
  py::class_<momentum::IntrinsicsModel, std::shared_ptr<momentum::IntrinsicsModel>>(
      m, "IntrinsicsModel", "Base class for camera intrinsics models")
      .def_property_readonly(
          "image_width", &momentum::IntrinsicsModel::imageWidth, "Width of the image in pixels")
      .def_property_readonly(
          "image_height", &momentum::IntrinsicsModel::imageHeight, "Height of the image in pixels")
      .def_property_readonly(
          "fx", &momentum::IntrinsicsModel::fx, "Focal length in x direction (pixels)")
      .def_property_readonly(
          "fy", &momentum::IntrinsicsModel::fy, "Focal length in y direction (pixels)")
      .def(
          "__repr__",
          [](const momentum::IntrinsicsModel& self) {
            return fmt::format(
                "IntrinsicsModel(image_size=({}, {}), focal_length=({:.2f}, {:.2f}))",
                self.imageWidth(),
                self.imageHeight(),
                self.fx(),
                self.fy());
          })
      .def(
          "project",
          [](const momentum::IntrinsicsModel& intrinsics,
             const py::array_t<float>& points) -> py::array_t<float> {
            if (points.ndim() != 2 || points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto pts_acc = points.unchecked<2>();

            for (py::ssize_t i = 0; i < points.shape(0);
                 i += momentum::rasterizer::kSimdPacketSize) {
              momentum::rasterizer::FloatP px, py, pz;
              auto nPtsCur = std::min(
                  static_cast<py::ssize_t>(momentum::rasterizer::kSimdPacketSize),
                  points.shape(0) - i);
              for (py::ssize_t k = 0; k < nPtsCur; ++k) {
                px[k] = pts_acc(i + k, 0);
                py[k] = pts_acc(i + k, 1);
                pz[k] = pts_acc(i + k, 2);
              }

              const auto [res, valid] =
                  intrinsics.project(momentum::rasterizer::Vector3fP(px, py, pz));

              for (py::ssize_t k = 0; k < nPtsCur; ++k) {
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
          &momentum::IntrinsicsModel::upsample,
          R"(Create a new intrinsics model upsampled by the given factor.

:param factor: Upsampling factor (e.g., 2.0 doubles the resolution).
:return: A new IntrinsicsModel instance with upsampled parameters.)",
          py::arg("factor"))
      .def(
          "downsample",
          &momentum::IntrinsicsModel::downsample,
          R"(Create a new intrinsics model downsampled by the given factor.

:param factor: Downsampling factor (e.g., 2.0 halves the resolution).
:return: A new IntrinsicsModel instance with downsampled parameters.)",
          py::arg("factor"))
      .def(
          "crop",
          &momentum::IntrinsicsModel::crop,
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
          &momentum::IntrinsicsModel::resize,
          R"(Create a new intrinsics model resized to new image dimensions.

:param image_width: New image width in pixels.
:param image_height: New image height in pixels.
:return: A new IntrinsicsModel instance with resized parameters.)",
          py::arg("image_width"),
          py::arg("image_height"));

  py::class_<momentum::OpenCVDistortionParameters>(
      m, "OpenCVDistortionParameters", "OpenCV distortion parameters")
      .def(py::init<>(), "Initialize with default parameters (no distortion)")
      .def_readwrite(
          "k1", &momentum::OpenCVDistortionParameters::k1, "Radial distortion coefficient k1")
      .def_readwrite(
          "k2", &momentum::OpenCVDistortionParameters::k2, "Radial distortion coefficient k2")
      .def_readwrite(
          "k3", &momentum::OpenCVDistortionParameters::k3, "Radial distortion coefficient k3")
      .def_readwrite(
          "k4", &momentum::OpenCVDistortionParameters::k4, "Radial distortion coefficient k4")
      .def_readwrite(
          "k5", &momentum::OpenCVDistortionParameters::k5, "Radial distortion coefficient k5")
      .def_readwrite(
          "k6", &momentum::OpenCVDistortionParameters::k6, "Radial distortion coefficient k6")
      .def_readwrite(
          "p1", &momentum::OpenCVDistortionParameters::p1, "Tangential distortion coefficient p1")
      .def_readwrite(
          "p2", &momentum::OpenCVDistortionParameters::p2, "Tangential distortion coefficient p2")
      .def_readwrite(
          "p3", &momentum::OpenCVDistortionParameters::p3, "Tangential distortion coefficient p3")
      .def_readwrite(
          "p4", &momentum::OpenCVDistortionParameters::p4, "Tangential distortion coefficient p4")
      .def("__repr__", [](const momentum::OpenCVDistortionParameters& self) {
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
         std::optional<float> cy) -> std::shared_ptr<momentum::PinholeIntrinsicsModel> {
    // Default focal length calculation: "normal" lens is a 50mm lens on
    // a 35mm camera body
    const float focal_length_cm = 5.0f;
    const float film_width_cm = 3.6f;
    const float default_focal_length_pixels =
        (focal_length_cm / film_width_cm) * static_cast<float>(imageWidth);

    if (fx.has_value() != fy.has_value()) {
      throw std::runtime_error("fx and fy must be both specified or both omitted");
    }

    if (cx.has_value() != cy.has_value()) {
      throw std::runtime_error("cx and cy must be both specified or both omitted");
    }

    return std::make_shared<momentum::PinholeIntrinsicsModel>(
        imageWidth,
        imageHeight,
        fx.value_or(default_focal_length_pixels),
        fy.value_or(default_focal_length_pixels),
        cx.value_or(imageWidth / 2.0f),
        cy.value_or(imageHeight / 2.0f));
  };

  py::class_<
      momentum::PinholeIntrinsicsModel,
      momentum::IntrinsicsModel,
      std::shared_ptr<momentum::PinholeIntrinsicsModel>>(
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
          "cx", &momentum::PinholeIntrinsicsModel::cx, "Principal point x-coordinate (pixels)")
      .def_property_readonly(
          "cy", &momentum::PinholeIntrinsicsModel::cy, "Principal point y-coordinate (pixels)")
      .def("__repr__", [](const momentum::PinholeIntrinsicsModel& self) {
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
  auto opencvFactory = [](int32_t imageWidth,
                          int32_t imageHeight,
                          std::optional<float> fx,
                          std::optional<float> fy,
                          std::optional<float> cx,
                          std::optional<float> cy,
                          std::optional<momentum::OpenCVDistortionParameters> distortionParams)
      -> std::shared_ptr<momentum::OpenCVIntrinsicsModel> {
    // Default focal length calculation: "normal" lens is a 50mm
    // lens on a 35mm camera body
    const float focal_length_cm = 5.0f;
    const float film_width_cm = 3.6f;
    const float default_focal_length_pixels =
        (focal_length_cm / film_width_cm) * static_cast<float>(imageWidth);
    return std::make_shared<momentum::OpenCVIntrinsicsModel>(
        imageWidth,
        imageHeight,
        fx.value_or(default_focal_length_pixels),
        fy.value_or(default_focal_length_pixels),
        cx.value_or(imageWidth / 2.0f),
        cy.value_or(imageHeight / 2.0f),
        distortionParams.value_or(momentum::OpenCVDistortionParameters{}));
  };

  py::class_<
      momentum::OpenCVIntrinsicsModel,
      momentum::IntrinsicsModel,
      std::shared_ptr<momentum::OpenCVIntrinsicsModel>>(
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
          "cx", &momentum::OpenCVIntrinsicsModel::cx, "Principal point x-coordinate (pixels)")
      .def_property_readonly(
          "cy", &momentum::OpenCVIntrinsicsModel::cy, "Principal point y-coordinate (pixels)")
      .def("__repr__", [](const momentum::OpenCVIntrinsicsModel& self) {
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
      [](std::shared_ptr<const momentum::IntrinsicsModel> intrinsics,
         const std::optional<Eigen::Matrix4f>& eye_from_world) -> momentum::Camera {
    return momentum::Camera(
        intrinsics, Eigen::Affine3f(eye_from_world.value_or(Eigen::Matrix4f::Identity())));
  };

  // Bind Camera class
  py::class_<momentum::Camera>(m, "Camera", "Camera for rendering")
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
          [](const momentum::Camera& self) {
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
          "image_width", &momentum::Camera::imageWidth, "Width of the image in pixels")
      .def_property_readonly(
          "image_height", &momentum::Camera::imageHeight, "Height of the image in pixels")
      .def_property_readonly("fx", &momentum::Camera::fx, "Focal length in x direction (pixels)")
      .def_property_readonly("fy", &momentum::Camera::fy, "Focal length in y direction (pixels)")
      .def_property_readonly(
          "intrinsics_model", &momentum::Camera::intrinsicsModel, "The camera's intrinsics model")
      .def_property(
          "T_eye_from_world",
          [](const momentum::Camera& self) -> Eigen::Matrix4f {
            return self.eyeFromWorld().matrix();
          },
          [](momentum::Camera& self, const Eigen::Matrix4f& value) {
            self.setEyeFromWorld(Eigen::Affine3f(value));
          },
          "Transform from world space to camera/eye space")
      .def_property(
          "T_world_from_eye",
          [](const momentum::Camera& self) -> Eigen::Matrix4f {
            return self.worldFromEye().matrix();
          },
          [](momentum::Camera& self, const Eigen::Matrix4f& value) {
            self.setEyeFromWorld(Eigen::Affine3f(value));
          },
          "Transform from world space to camera/eye space")
      .def_property_readonly(
          "center_of_projection",
          [](const momentum::Camera& self) -> Eigen::Vector3f {
            return (self.eyeFromWorld().inverse().translation()).eval();
          },
          "Position of the camera center in world space")
      .def(
          "look_at",
          [](const momentum::Camera& self,
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
          [](const momentum::Camera& self,
             const py::array_t<float>& points,
             float min_z,
             float edge_padding) -> momentum::Camera {
            if (points.ndim() != 2 || points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            // Convert py::array_t<float> to std::vector<Eigen::Vector3f>
            std::vector<Eigen::Vector3f> eigenPoints;
            eigenPoints.reserve(points.shape(0));

            auto pts_acc = points.unchecked<2>();
            for (py::ssize_t i = 0; i < points.shape(0); ++i) {
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
          [](const momentum::Camera& self) -> Eigen::Vector3f {
            // The principle axis is the direction the camera is looking
            // In camera space, this is the positive Z axis (0, 0, 1)
            Eigen::Vector3f cameraSpacePrincipalAxis = Eigen::Vector3f::UnitZ();
            return (self.worldFromEye().linear() * cameraSpacePrincipalAxis).eval();
          },
          "Camera world-space principal axis (direction the camera is looking)")
      .def(
          "upsample",
          [](const momentum::Camera& self, float factor) {
            return momentum::Camera(self.intrinsicsModel()->upsample(factor), self.eyeFromWorld());
          },
          R"(Create a new camera with upsampled resolution by the given factor.

:param factor: Upsampling factor (e.g., 2.0 doubles the resolution).
:return: A new Camera instance with upsampled intrinsics and same pose.)",
          py::arg("factor"))
      .def(
          "crop",
          [](const momentum::Camera& self,
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
          [](const momentum::Camera& self,
             const py::array_t<float>& world_points) -> py::array_t<float> {
            if (world_points.ndim() != 2 || world_points.shape(1) != 3) {
              throw std::runtime_error("Expected a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{world_points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto pts_acc = world_points.unchecked<2>();

            for (py::ssize_t i = 0; i < world_points.shape(0); ++i) {
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
          [](const momentum::Camera& self,
             const py::array_t<float>& image_points) -> py::array_t<float> {
            if (image_points.ndim() != 2 || image_points.shape(1) != 3) {
              throw std::runtime_error(
                  "Expected image_points to be a 2D array of shape (nPoints, 3)");
            }

            py::array_t<float> result(std::vector<py::ssize_t>{image_points.shape(0), 3});
            auto res_acc = result.mutable_unchecked<2>();
            auto img_acc = image_points.unchecked<2>();

            for (py::ssize_t i = 0; i < image_points.shape(0); ++i) {
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
}
