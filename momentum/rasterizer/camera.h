/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/matrix.h>

#include <momentum/rasterizer/fwd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

namespace momentum::rasterizer {

/// Base class for camera intrinsics models.
/// This abstract class defines the interface for different camera models
/// such as pinhole and OpenCV distortion models.
template <typename T>
class IntrinsicsModelT {
 public:
  /// Constructor for intrinsics model.
  /// @param imageWidth Width of the image in pixels
  /// @param imageHeight Height of the image in pixels
  IntrinsicsModelT(int32_t imageWidth, int32_t imageHeight)
      : imageWidth_(imageWidth), imageHeight_(imageHeight) {}
  virtual ~IntrinsicsModelT() = default;

  /// Get the focal length in the x direction.
  /// @return Focal length fx in pixels
  [[nodiscard]] virtual T fx() const = 0;

  /// Get the focal length in the y direction.
  /// @return Focal length fy in pixels
  [[nodiscard]] virtual T fy() const = 0;

  /// Project 3D points to 2D image coordinates.
  /// @param point 3D points in camera coordinate space
  /// @return Pair of projected 2D points and validity mask
  [[nodiscard]] virtual std::pair<Vector3xP<T>, typename PacketType_t<T>::MaskType> project(
      const Vector3xP<T>& point) const = 0;

  /// Project a single 3D point to 2D image coordinates (Eigen version).
  /// @param point 3D point in camera coordinate space
  /// @return Pair of projected 2D point and validity flag
  [[nodiscard]] virtual std::pair<Eigen::Vector3<T>, bool> project(
      const Eigen::Vector3<T>& point) const = 0;

  /// Compute the Jacobian of the projection function with respect to 3D camera coordinates.
  /// @param point 3D point in camera coordinate space
  /// @return Tuple of (projected point, Jacobian matrix, valid flag)
  /// The Jacobian is a 3x3 matrix where:
  /// - Row 0: [du/dx, du/dy, du/dz]
  /// - Row 1: [dv/dx, dv/dy, dv/dz]
  /// - Row 2: [0, 0, 1] (for homogeneous coordinates)
  [[nodiscard]] virtual std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool> projectJacobian(
      const Eigen::Vector3<T>& point) const = 0;

  /// Unproject 3D image points to 3D camera points using Newton's method.
  /// @param imagePoint 3D point in image coordinates (u, v, z) where z is the desired depth
  /// @param maxIterations Maximum number of Newton iterations (default: 10)
  /// @param tolerance Convergence tolerance for the residual (default: 1e-6)
  /// @return Pair of 3D point in camera coordinates and validity flag
  [[nodiscard]] virtual std::pair<Eigen::Vector3<T>, bool> unproject(
      const Eigen::Vector3<T>& imagePoint,
      int maxIterations = 10,
      T tolerance = T(1e-6)) const = 0;

  /// Resample the intrinsics by a given factor.
  /// @param factor Resampling factor (>1 for upsampling, <1 for downsampling)
  /// @return New intrinsics model with resampled parameters
  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> resample(T factor) const;

  /// Downsample the intrinsics by a given factor.
  /// @param factor Downsampling factor (e.g., 2.0 halves the resolution)
  /// @return New intrinsics model with downsampled parameters
  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> downsample(T factor) const;

  /// Upsample the intrinsics by a given factor.
  /// @param factor Upsampling factor (e.g., 2.0 doubles the resolution)
  /// @return New intrinsics model with upsampled parameters
  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> upsample(T factor) const;

  /// Resize the intrinsics to new image dimensions.
  /// @param imageWidth New image width in pixels
  /// @param imageHeight New image height in pixels
  /// @return New intrinsics model with resized parameters
  [[nodiscard]] virtual std::shared_ptr<const IntrinsicsModelT<T>> resize(
      int32_t imageWidth,
      int32_t imageHeight) const = 0;

  /// Crop the intrinsics to a sub-region of the image.
  /// @param top Top offset in pixels
  /// @param left Left offset in pixels
  /// @param newWidth New width in pixels after cropping
  /// @param newHeight New height in pixels after cropping
  /// @return New intrinsics model with cropped parameters
  [[nodiscard]] virtual std::shared_ptr<const IntrinsicsModelT<T>>
  crop(int32_t top, int32_t left, int32_t newWidth, int32_t newHeight) const = 0;

  /// Get the image width.
  /// @return Image width in pixels
  [[nodiscard]] int32_t imageWidth() const {
    return imageWidth_;
  }

  /// Get the image height.
  /// @return Image height in pixels
  [[nodiscard]] int32_t imageHeight() const {
    return imageHeight_;
  }

 private:
  int32_t imageWidth_ = 640;
  int32_t imageHeight_ = 480;
};

/// Camera class that combines intrinsics and extrinsics for 3D rendering.
/// This class encapsulates both the camera's intrinsic parameters (focal length,
/// principal point, image dimensions) and extrinsic parameters (position and orientation).
template <typename T>
class CameraT {
 public:
  /// Default constructor creates a camera with identity transform.
  CameraT();

  /// Constructor with intrinsics model and optional transform.
  /// @param intrinsicsModel Shared pointer to the camera's intrinsics model
  /// @param eyeFromWorld Transform from world space to camera/eye space (defaults to identity)
  explicit CameraT(
      std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
      const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld =
          Eigen::Transform<T, 3, Eigen::Affine>::Identity());

  /// Get the image width from the intrinsics model.
  /// @return Image width in pixels
  [[nodiscard]] auto imageWidth() const {
    return intrinsicsModel_->imageWidth();
  }

  /// Get the image height from the intrinsics model.
  /// @return Image height in pixels
  [[nodiscard]] auto imageHeight() const {
    return intrinsicsModel_->imageHeight();
  }

  /// Get the focal length in x direction from the intrinsics model.
  /// @return Focal length fx in pixels
  [[nodiscard]] auto fx() const {
    return intrinsicsModel_->fx();
  }

  /// Get the focal length in y direction from the intrinsics model.
  /// @return Focal length fy in pixels
  [[nodiscard]] auto fy() const {
    return intrinsicsModel_->fy();
  }

  /// Get the intrinsics model.
  /// @return Shared pointer to the intrinsics model
  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel() const {
    return intrinsicsModel_;
  }

  /// Get the eye-from-world transformation matrix.
  /// @return Reference to the transformation from world space to camera space
  [[nodiscard]] const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld() const {
    return eyeFromWorld_;
  }

  /// Get the world-from-eye transformation matrix.
  /// @return Transformation from camera space to world space
  [[nodiscard]] Eigen::Transform<T, 3, Eigen::Affine> worldFromEye() const {
    return eyeFromWorld_.inverse();
  }

  /// Set the eye-from-world transformation matrix.
  /// @param eyeFromWorld New transformation from world space to camera space
  void setEyeFromWorld(const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld) {
    eyeFromWorld_ = eyeFromWorld;
  }

  /// Position the camera to look at a specific target point.
  ///
  /// @param position The position of the camera in world space
  /// @param target The target point to look at in world space
  /// @param up The up vector in world space (default is Y-up)
  /// @return A new camera with the updated transform
  [[nodiscard]] CameraT<T> lookAt(
      const Eigen::Vector3<T>& position,
      const Eigen::Vector3<T>& target = Eigen::Vector3<T>::Zero(),
      const Eigen::Vector3<T>& up = Eigen::Vector3<T>::UnitY()) const;

  /// Adjust the camera position to ensure all specified points are in view.
  /// This preserves the camera's orientation but adjusts its position.
  ///
  /// @param points The 3D points in world space that should be in view
  /// @param minZ The minimum Z distance from the camera
  /// @param edgePadding Padding from the edge of the frame (0.0-1.0)
  /// @return A new camera with the updated transform
  [[nodiscard]] CameraT<T> framePoints(
      const std::vector<Eigen::Vector3<T>>& points,
      T minZ = T(0.1),
      T edgePadding = T(0.05)) const;

  /// Project 3D world points to 2D image coordinates.
  /// @param worldPoints 3D points in world coordinate space (wide vector)
  /// @return Pair of projected 2D points and validity mask
  [[nodiscard]] std::pair<Vector3xP<T>, typename PacketType_t<T>::MaskType> project(
      const Vector3xP<T>& worldPoints) const;

  /// Project a single 3D world point to 2D image coordinates (Eigen version).
  /// @param worldPoint 3D point in world coordinate space
  /// @return Pair of projected 2D point and validity flag
  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> project(
      const Eigen::Vector3<T>& worldPoint) const;

  /// Compute the Jacobian of the projection function with respect to 3D world coordinates.
  /// @param worldPoint 3D point in world coordinate space
  /// @return Tuple of (projected point, Jacobian matrix, valid flag)
  /// The Jacobian is a 2x3 matrix where:
  /// - Row 0: [du/dX_world, du/dY_world, du/dZ_world]
  /// - Row 1: [dv/dX_world, dv/dY_world, dv/dZ_world]
  [[nodiscard]] std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 2, 3>, bool> projectJacobian(
      const Eigen::Vector3<T>& worldPoint) const;

  /// Unproject 3D image points to 3D world points using Newton's method.
  /// @param imagePoints 3D points in image coordinates (u, v, z) where z is the desired depth (wide
  /// vector)
  /// @param maxIterations Maximum number of Newton iterations (default: 10)
  /// @param tolerance Convergence tolerance for the residual (default: 1e-6)
  /// @return Pair of 3D points in world coordinates and validity mask
  [[nodiscard]] std::pair<Vector3xP<T>, typename PacketType_t<T>::MaskType>
  unproject(const Vector3xP<T>& imagePoints, int maxIterations = 10, T tolerance = T(1e-6)) const;

  /// Unproject 3D image point to 3D world point using Newton's method (Eigen version).
  /// @param imagePoint 3D point in image coordinates (u, v, z) where z is the desired depth
  /// @param maxIterations Maximum number of Newton iterations (default: 10)
  /// @param tolerance Convergence tolerance for the residual (default: 1e-6)
  /// @return Pair of 3D point in world coordinates and validity flag
  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> unproject(
      const Eigen::Vector3<T>& imagePoint,
      int maxIterations = 10,
      T tolerance = T(1e-6)) const;

  /// Create a cropped camera with a sub-region of the original image.
  /// @param top Top offset in pixels
  /// @param left Left offset in pixels
  /// @param newWidth New width in pixels after cropping
  /// @param newHeight New height in pixels after cropping
  /// @return New camera with cropped intrinsics and same pose
  [[nodiscard]] CameraT<T> crop(int32_t top, int32_t left, int32_t newWidth, int32_t newHeight)
      const {
    return CameraT<T>(intrinsicsModel_->crop(top, left, newWidth, newHeight), eyeFromWorld_);
  }

  /// Create a resized camera with new image dimensions.
  /// @param imageWidth New image width in pixels
  /// @param imageHeight New image height in pixels
  /// @return New camera with resized intrinsics and same pose
  [[nodiscard]] CameraT<T> resize(int32_t imageWidth, int32_t imageHeight) const {
    return CameraT<T>(intrinsicsModel_->resize(imageWidth, imageHeight), eyeFromWorld_);
  }

  [[nodiscard]] const IntrinsicsModelT<T>& getIntrinsicsModel() const {
    return *intrinsicsModel_;
  }

 private:
  /// Transform world points to camera/eye space using SIMD operations.
  /// @param worldPoints 3D points in world coordinate space (wide vector)
  /// @return 3D points in camera/eye coordinate space (wide vector)
  [[nodiscard]] Vector3xP<T> transformWorldToEye(const Vector3xP<T>& worldPoints) const;

  /// Transform a single world point to camera/eye space.
  /// @param worldPoint 3D point in world coordinate space
  /// @return 3D point in camera/eye coordinate space
  [[nodiscard]] Eigen::Vector3<T> transformWorldToEye(const Eigen::Vector3<T>& worldPoint) const;

  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld_ =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();

  std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel_;
};

/// OpenCV distortion parameters for camera lens distortion correction.
/// These parameters follow the OpenCV camera calibration model and include
/// both radial and tangential distortion coefficients.
template <typename T>
struct OpenCVDistortionParametersT {
  T k1 = T(0); ///< First radial distortion coefficient
  T k2 = T(0); ///< Second radial distortion coefficient
  T k3 = T(0); ///< Third radial distortion coefficient
  T k4 = T(0); ///< Fourth radial distortion coefficient (for fisheye model)
  T k5 = T(0); ///< Fifth radial distortion coefficient (for fisheye model)
  T k6 = T(0); ///< Sixth radial distortion coefficient (for fisheye model)

  T p1 = T(0); ///< First tangential distortion coefficient
  T p2 = T(0); ///< Second tangential distortion coefficient
  T p3 = T(0); ///< Third tangential distortion coefficient (for thin prism)
  T p4 = T(0); ///< Fourth tangential distortion coefficient (for thin prism)
};

/// PinholeIntrinsicsModel implements a basic pinhole camera model.
/// This model uses the following projection formula:
/// x = fx * X/Z + cx
/// y = fy * Y/Z + cy
///
/// Where:
/// - (X, Y, Z) is the 3D point in camera coordinates
/// - (x, y) is the projected 2D point in image coordinates
/// - (fx, fy) are the focal lengths
/// - (cx, cy) is the principal point
///
/// Unlike the OpenCVIntrinsicsModel, this model does not include any distortion.
template <typename T>
class PinholeIntrinsicsModelT : public IntrinsicsModelT<T> {
 public:
  PinholeIntrinsicsModelT(int32_t imageWidth, int32_t imageHeight, T fx, T fy, T cx, T cy);
  PinholeIntrinsicsModelT(int32_t imageWidth, int32_t imageHeight, T fx, T fy);

  [[nodiscard]] T fx() const final {
    return fx_;
  }

  [[nodiscard]] T fy() const final {
    return fy_;
  }

  [[nodiscard]] T cx() const {
    return cx_;
  }

  [[nodiscard]] T cy() const {
    return cy_;
  }

  [[nodiscard]] std::pair<Vector3xP<T>, typename PacketType_t<T>::MaskType> project(
      const Vector3xP<T>& point) const final;

  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> project(
      const Eigen::Vector3<T>& point) const final;

  [[nodiscard]] std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool> projectJacobian(
      const Eigen::Vector3<T>& point) const final;

  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> unproject(
      const Eigen::Vector3<T>& imagePoint,
      int maxIterations = 10,
      T tolerance = T(1e-6)) const final;

  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> resize(
      int32_t imageWidth,
      int32_t imageHeight) const final;

  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>>
  crop(int32_t top, int32_t left, int32_t newWidth, int32_t newHeight) const final;

 private:
  T fx_;
  T fy_;
  T cx_;
  T cy_;
};

/// OpenCVIntrinsicsModel implements the standard OpenCV camera model.
/// This model uses the following projection formula:
/// x = fx * X/Z + cx
/// y = fy * Y/Z + cy
///
/// Where:
/// - (X, Y, Z) is the 3D point in camera coordinates
/// - (x, y) is the projected 2D point in image coordinates
/// - (fx, fy) are the focal lengths
/// - (cx, cy) is the principal point
template <typename T>
class OpenCVIntrinsicsModelT : public IntrinsicsModelT<T> {
 public:
  OpenCVIntrinsicsModelT(
      int32_t imageWidth,
      int32_t imageHeight,
      T fx,
      T fy,
      T cx,
      T cy,
      const OpenCVDistortionParametersT<T>& params = OpenCVDistortionParametersT<T>{});

  [[nodiscard]] T fx() const final {
    return fx_;
  }
  [[nodiscard]] T fy() const final {
    return fy_;
  }
  [[nodiscard]] T cx() const {
    return cx_;
  }
  [[nodiscard]] T cy() const {
    return cy_;
  }

  /// Get the distortion parameters.
  /// @return Reference to the OpenCV distortion parameters
  [[nodiscard]] const OpenCVDistortionParametersT<T>& distortionParameters() const {
    return distortionParams_;
  }

  [[nodiscard]] std::pair<Vector3xP<T>, typename PacketType_t<T>::MaskType> project(
      const Vector3xP<T>& point) const final;

  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> project(
      const Eigen::Vector3<T>& point) const final;

  [[nodiscard]] std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool> projectJacobian(
      const Eigen::Vector3<T>& point) const final;

  [[nodiscard]] std::pair<Eigen::Vector3<T>, bool> unproject(
      const Eigen::Vector3<T>& imagePoint,
      int maxIterations = 10,
      T tolerance = T(1e-6)) const final;

  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>> resize(
      int32_t imageWidth,
      int32_t imageHeight) const final;

  [[nodiscard]] std::shared_ptr<const IntrinsicsModelT<T>>
  crop(int32_t top, int32_t left, int32_t newWidth, int32_t newHeight) const final;

 private:
  T fx_;
  T fy_;
  T cx_;
  T cy_;
  const OpenCVDistortionParametersT<T> distortionParams_;
};

// Type aliases for common instantiations
using IntrinsicsModel = IntrinsicsModelT<float>;
using IntrinsicsModeld = IntrinsicsModelT<double>;
using Camera = CameraT<float>;
using Camerad = CameraT<double>;
using PinholeIntrinsicsModel = PinholeIntrinsicsModelT<float>;
using PinholeIntrinsicsModeld = PinholeIntrinsicsModelT<double>;
using OpenCVIntrinsicsModel = OpenCVIntrinsicsModelT<float>;
using OpenCVIntrinsicsModeld = OpenCVIntrinsicsModelT<double>;
using OpenCVDistortionParameters = OpenCVDistortionParametersT<float>;
using OpenCVDistortionParametersd = OpenCVDistortionParametersT<double>;

} // namespace momentum::rasterizer
