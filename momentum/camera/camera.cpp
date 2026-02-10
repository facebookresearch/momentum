/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/camera/camera.h"

#include "momentum/camera/fwd.h"

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/matrix.h>

#include <cfloat>
#include <cmath>

namespace momentum {

// Default constructor is VGA res.
template <typename T>
CameraT<T>::CameraT()
    : intrinsicsModel_(
          std::make_shared<PinholeIntrinsicsModelT<T>>(
              640,
              480,
              (5.0 / 3.6) * 640,
              (5.0 / 3.6) * 640)) {}

// Constructor implementation for CameraT.
template <typename T>
CameraT<T>::CameraT(
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld)
    : eyeFromWorld_(eyeFromWorld), intrinsicsModel_(intrinsicsModel) {}

// Constructor for PinholeIntrinsicsModelT with explicit principal point.
template <typename T>
PinholeIntrinsicsModelT<T>::PinholeIntrinsicsModelT(
    int32_t imageWidth,
    int32_t imageHeight,
    T fx,
    T fy,
    T cx,
    T cy)
    : IntrinsicsModelT<T>(imageWidth, imageHeight), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

// Constructor for PinholeIntrinsicsModelT with principal point at image center.
template <typename T>
PinholeIntrinsicsModelT<T>::PinholeIntrinsicsModelT(
    int32_t imageWidth,
    int32_t imageHeight,
    T fx,
    T fy)
    : IntrinsicsModelT<T>(imageWidth, imageHeight),
      fx_(fx),
      fy_(fy),
      cx_(T(imageWidth) / T(2)),
      cy_(T(imageHeight) / T(2)) {}

// Constructor for OpenCVIntrinsicsModelT with optional distortion parameters.
template <typename T>
OpenCVIntrinsicsModelT<T>::OpenCVIntrinsicsModelT(
    int32_t imageWidth,
    int32_t imageHeight,
    T fx,
    T fy,
    T cx,
    T cy,
    const OpenCVDistortionParametersT<T>& params)
    : IntrinsicsModelT<T>(imageWidth, imageHeight),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      distortionParams_(params) {}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> PinholeIntrinsicsModelT<T>::project(
    const Vector3P<T>& point) const {
  // Normalize the point by dividing by z
  Packet<T> x = point.x() / point.z();
  Packet<T> y = point.y() / point.z();

  // Apply camera matrix to get pixel coordinates
  Packet<T> u = fx_ * x + this->cx();
  Packet<T> v = fy_ * y + this->cy();

  return {Vector3P<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> PinholeIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  // Normalize the point by dividing by z
  T x = point.x() / point.z();
  T y = point.y() / point.z();

  // Apply camera matrix to get pixel coordinates
  T u = fx_ * x + this->cx();
  T v = fy_ * y + this->cy();

  return {Eigen::Vector3<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool>
PinholeIntrinsicsModelT<T>::projectJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  // Check if point is in front of camera
  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T z_inv = T(1) / z;
  const T z_inv_sq = z_inv * z_inv;

  // Project the point
  const T u = fx_ * x * z_inv + cx_;
  const T v = fy_ * y * z_inv + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Compute Jacobian matrix (3x3)
  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  // Row 0: du/dx, du/dy, du/dz
  jacobian(0, 0) = fx_ * z_inv; // du/dx = fx / z
  jacobian(0, 1) = T(0); // du/dy = 0
  jacobian(0, 2) = -fx_ * x * z_inv_sq; // du/dz = -fx * x / z^2

  // Row 1: dv/dx, dv/dy, dv/dz
  jacobian(1, 0) = T(0); // dv/dx = 0
  jacobian(1, 1) = fy_ * z_inv; // dv/dy = fy / z
  jacobian(1, 2) = -fy_ * y * z_inv_sq; // dv/dz = -fy * y / z^2

  // Row 2: homogeneous coordinates (for completeness)
  jacobian(2, 0) = T(0); // dz/dx = 0
  jacobian(2, 1) = T(0); // dz/dy = 0
  jacobian(2, 2) = T(1); // dz/dz = 1

  return {projectedPoint, jacobian, true};
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> PinholeIntrinsicsModelT<T>::resize(
    int32_t imageWidth,
    int32_t imageHeight) const {
  T scaleX = T(imageWidth) / T(this->imageWidth());
  T scaleY = T(imageHeight) / T(this->imageHeight());

  // Apply the correct formula for camera center as noted in the comment:
  // cx = (old_cx + 0.5) * new_sizex / old_sizex - 0.5;
  // cy = (old_cy + 0.5) * new_sizey / old_sizey - 0.5;
  T old_cx = this->cx();
  T old_cy = this->cy();
  T new_cx = (old_cx + T(0.5)) * scaleX - T(0.5);
  T new_cy = (old_cy + T(0.5)) * scaleY - T(0.5);

  return std::make_shared<PinholeIntrinsicsModelT<T>>(
      imageWidth, imageHeight, fx_ * scaleX, fy_ * scaleY, new_cx, new_cy);
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> PinholeIntrinsicsModelT<T>::crop(
    int32_t top,
    int32_t left,
    int32_t newWidth,
    int32_t newHeight) const {
  // Ensure the crop doesn't exceed the original image dimensions
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Adjust the principal point by subtracting the crop offset
  T cameraCenter_cropped_cx = cx_ - T(left);
  T cameraCenter_cropped_cy = cy_ - T(top);

  return std::make_shared<PinholeIntrinsicsModelT<T>>(
      width, height, fx_, fy_, cameraCenter_cropped_cx, cameraCenter_cropped_cy);
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> PinholeIntrinsicsModelT<T>::unproject(
    const Eigen::Vector3<T>& imagePoint,
    int /*maxIterations*/,
    T /*tolerance*/) const {
  // For pinhole cameras, unprojection is straightforward - no distortion to invert
  const T u = imagePoint(0);
  const T v = imagePoint(1);
  const T depth = imagePoint(2);

  // Convert to normalized camera coordinates
  const T x = (u - cx_) / fx_;
  const T y = (v - cy_) / fy_;

  // Return 3D point in camera coordinates with the given depth
  return {Eigen::Vector3<T>(x * depth, y * depth, depth), depth > T(0)};
}

template <typename T>
Eigen::Index PinholeIntrinsicsModelT<T>::numIntrinsicParameters() const {
  return 4; // fx, fy, cx, cy
}

template <typename T>
Eigen::VectorX<T> PinholeIntrinsicsModelT<T>::getIntrinsicParameters() const {
  Eigen::VectorX<T> params(4);
  params << fx_, fy_, cx_, cy_;
  return params;
}

template <typename T>
void PinholeIntrinsicsModelT<T>::setIntrinsicParameters(
    const Eigen::Ref<const Eigen::VectorX<T>>& params) {
  fx_ = params(0);
  fy_ = params(1);
  cx_ = params(2);
  cy_ = params(3);
}

template <typename T>
std::shared_ptr<IntrinsicsModelT<T>> PinholeIntrinsicsModelT<T>::clone() const {
  return std::make_shared<PinholeIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_);
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
PinholeIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  // Check if point is in front of camera
  if (z <= T(0)) {
    Eigen::Matrix<T, 3, Eigen::Dynamic> zeroJacobian(3, 4);
    zeroJacobian.setZero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;
  const T xn = x * z_inv; // x/z (normalized coordinate)
  const T yn = y * z_inv; // y/z (normalized coordinate)

  // Project the point: u = fx * x/z + cx, v = fy * y/z + cy
  const T u = fx_ * xn + cx_;
  const T v = fy_ * yn + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Compute Jacobian matrix (3x4) with respect to [fx, fy, cx, cy]
  // u = fx * (x/z) + cx
  // v = fy * (y/z) + cy
  // depth = z (unchanged)
  //
  // du/dfx = x/z,  du/dfy = 0,    du/dcx = 1,  du/dcy = 0
  // dv/dfx = 0,    dv/dfy = y/z,  dv/dcx = 0,  dv/dcy = 1
  // dz/d*  = 0     (depth is unchanged by intrinsics)
  Eigen::Matrix<T, 3, Eigen::Dynamic> jacobian(3, 4);
  jacobian.setZero();

  // Row 0: du/d[fx, fy, cx, cy]
  jacobian(0, 0) = xn; // du/dfx = x/z
  jacobian(0, 1) = T(0); // du/dfy = 0
  jacobian(0, 2) = T(1); // du/dcx = 1
  jacobian(0, 3) = T(0); // du/dcy = 0

  // Row 1: dv/d[fx, fy, cx, cy]
  jacobian(1, 0) = T(0); // dv/dfx = 0
  jacobian(1, 1) = yn; // dv/dfy = y/z
  jacobian(1, 2) = T(0); // dv/dcx = 0
  jacobian(1, 3) = T(1); // dv/dcy = 1

  // Row 2: dz/d[fx, fy, cx, cy] = 0 (depth unchanged)
  // Already set to zero above

  return {projectedPoint, jacobian, true};
}

// Base class implementations for IntrinsicsModelT
template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> IntrinsicsModelT<T>::resample(T factor) const {
  return resize(
      static_cast<int32_t>(this->imageWidth() * factor),
      static_cast<int32_t>(this->imageHeight() * factor));
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> IntrinsicsModelT<T>::downsample(T factor) const {
  return resize(
      static_cast<int32_t>(this->imageWidth() / factor),
      static_cast<int32_t>(this->imageHeight() / factor));
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> IntrinsicsModelT<T>::upsample(T factor) const {
  return resize(
      static_cast<int32_t>(this->imageWidth() * factor),
      static_cast<int32_t>(this->imageHeight() * factor));
}

// OpenCVIntrinsicsModelT implementation
template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> OpenCVIntrinsicsModelT<T>::project(
    const Vector3P<T>& point) const {
  // Normalize the point by dividing by z
  Packet<T> invZ = T(1) / point.z();
  Packet<T> xp = point.x() * invZ;
  Packet<T> yp = point.y() * invZ;

  Packet<T> rsqr = drjit::square(xp) + drjit::square(yp);

  const auto& dp = distortionParams_;

  Packet<T> radialDistortion = T(1) +
      (rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3))) /
          (T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6)));

  Packet<T> xpp =
      xp * radialDistortion + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * drjit::square(xp));
  Packet<T> ypp =
      yp * radialDistortion + dp.p1 * (rsqr + T(2) * drjit::square(yp)) + T(2) * dp.p2 * xp * yp;

  // Apply camera matrix to get pixel coordinates
  Packet<T> u = fx_ * xpp + cx_;
  Packet<T> v = fy_ * ypp + cy_;

  return {Vector3P<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  // Normalize the point by dividing by z
  T invZ = T(1) / point.z();
  T xp = point.x() * invZ;
  T yp = point.y() * invZ;

  T rsqr = xp * xp + yp * yp;

  const auto& dp = distortionParams_;

  T radialDistortion = T(1) +
      (rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3))) /
          (T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6)));

  T xpp = xp * radialDistortion + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * xp * xp);
  T ypp = yp * radialDistortion + dp.p1 * (rsqr + T(2) * yp * yp) + T(2) * dp.p2 * xp * yp;

  // Apply camera matrix to get pixel coordinates
  T u = fx_ * xpp + cx_;
  T v = fy_ * ypp + cy_;

  return {Eigen::Vector3<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool>
OpenCVIntrinsicsModelT<T>::projectJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  // Check if point is in front of camera
  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T z_inv = T(1) / z;
  const T z_inv_sq = z_inv * z_inv;

  // Normalized coordinates
  const T xp = x * z_inv;
  const T yp = y * z_inv;
  const T rsqr = xp * xp + yp * yp;

  const auto& dp = distortionParams_;

  // Radial distortion components
  const T radial_num = T(1) + rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3));
  const T radial_den = T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6));
  const T radial_factor = radial_num / radial_den;

  // Tangential distortion
  const T xpp = xp * radial_factor + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * xp * xp);
  const T ypp = yp * radial_factor + dp.p1 * (rsqr + T(2) * yp * yp) + T(2) * dp.p2 * xp * yp;

  // Project the point
  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Derivatives of radial distortion factor with respect to rsqr
  const T dradial_num_drsqr = dp.k1 + rsqr * (T(2) * dp.k2 + rsqr * T(3) * dp.k3);
  const T dradial_den_drsqr = dp.k4 + rsqr * (T(2) * dp.k5 + rsqr * T(3) * dp.k6);
  const T dradial_factor_drsqr =
      (dradial_num_drsqr * radial_den - radial_num * dradial_den_drsqr) / (radial_den * radial_den);

  // Derivatives of distorted coordinates with respect to normalized coordinates
  const T dxpp_dxp =
      radial_factor + xp * dradial_factor_drsqr * T(2) * xp + T(2) * dp.p1 * yp + dp.p2 * T(6) * xp;
  const T dxpp_dyp = xp * dradial_factor_drsqr * T(2) * yp + T(2) * dp.p1 * xp + dp.p2 * T(2) * yp;
  const T dypp_dxp = yp * dradial_factor_drsqr * T(2) * xp + dp.p1 * T(2) * xp + T(2) * dp.p2 * yp;
  const T dypp_dyp =
      radial_factor + yp * dradial_factor_drsqr * T(2) * yp + dp.p1 * T(6) * yp + T(2) * dp.p2 * xp;

  // Compute Jacobian matrix (3x3)
  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  // Row 0: du/dx, du/dy, du/dz
  jacobian(0, 0) = fx_ * dxpp_dxp * z_inv; // du/dx
  jacobian(0, 1) = fx_ * dxpp_dyp * z_inv; // du/dy
  jacobian(0, 2) = -fx_ * (dxpp_dxp * x * z_inv_sq + dxpp_dyp * y * z_inv_sq); // du/dz

  // Row 1: dv/dx, dv/dy, dv/dz
  jacobian(1, 0) = fy_ * dypp_dxp * z_inv; // dv/dx
  jacobian(1, 1) = fy_ * dypp_dyp * z_inv; // dv/dy
  jacobian(1, 2) = -fy_ * (dypp_dxp * x * z_inv_sq + dypp_dyp * y * z_inv_sq); // dv/dz

  // Row 2: homogeneous coordinates (for completeness)
  jacobian(2, 0) = T(0); // dz/dx = 0
  jacobian(2, 1) = T(0); // dz/dy = 0
  jacobian(2, 2) = T(1); // dz/dz = 1

  return {projectedPoint, jacobian, true};
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> OpenCVIntrinsicsModelT<T>::resize(
    int32_t imageWidth,
    int32_t imageHeight) const {
  T scaleX = T(imageWidth) / T(this->imageWidth());
  T scaleY = T(imageHeight) / T(this->imageHeight());

  // Apply the correct formula for camera center as noted in the comment:
  // cx = (old_cx + 0.5) * new_sizex / old_sizex - 0.5;
  // cy = (old_cy + 0.5) * new_sizey / old_sizey - 0.5;
  T new_cx = (cx_ + T(0.5)) * scaleX - T(0.5);
  T new_cy = (cy_ + T(0.5)) * scaleY - T(0.5);

  return std::make_shared<OpenCVIntrinsicsModelT<T>>(
      imageWidth, imageHeight, fx_ * scaleX, fy_ * scaleY, new_cx, new_cy, distortionParams_);
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> OpenCVIntrinsicsModelT<T>::crop(
    int32_t top,
    int32_t left,
    int32_t newWidth,
    int32_t newHeight) const {
  // Ensure the crop doesn't exceed the original image dimensions
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Adjust the principal point by subtracting the crop offset
  T cameraCenter_cropped_cx = cx_ - T(left);
  T cameraCenter_cropped_cy = cy_ - T(top);

  return std::make_shared<OpenCVIntrinsicsModelT<T>>(
      width, height, fx_, fy_, cameraCenter_cropped_cx, cameraCenter_cropped_cy, distortionParams_);
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVIntrinsicsModelT<T>::unproject(
    const Eigen::Vector3<T>& imagePoint,
    int maxIterations,
    T tolerance) const {
  const T u = imagePoint(0);
  const T v = imagePoint(1);
  const T depth = imagePoint(2);

  // Check if depth is valid (positive)
  if (depth <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  // Initial guess: convert to normalized camera coordinates assuming no distortion
  const Eigen::Vector3<T> p_init((u - cx_) / fx_, (v - cy_) / fy_, depth);
  Eigen::Vector3<T> p_cur = p_init;

  // Newton's method with backtracking line search to solve the nonlinear projection equation
  for (int iter = 0; iter < maxIterations; ++iter) {
    // Use projectJacobian to get both the projected point and Jacobian
    const auto [projectedPoint, jacobian, isValid] = projectJacobian(p_cur);

    if (!isValid) {
      // Point is behind camera or other invalid condition
      return {p_init, false};
    }

    // Compute residual
    const Eigen::Vector<T, 2> residual =
        projectedPoint.template head<2>() - imagePoint.template head<2>();
    const T residual_norm = residual.norm();

    // Check convergence
    if (residual_norm < tolerance) {
      return {p_cur, true};
    }

    // Extract the 2x3 Jacobian matrix from the 3x3 matrix (ignore the third row)
    Eigen::Matrix<T, 2, 2> J = jacobian.template topLeftCorner<2, 2>();

    // Solve using QR decomposition for better numerical stability
    // We want to solve: J * delta = -residual (least squares)
    Eigen::Vector<T, 2> rhs = -residual;

    // Get the least squares solution using Eigen's QR decomposition (full Newton step)
    const Eigen::Vector<T, 2> delta = J.householderQr().solve(rhs);

    // Check if the solution is valid (no NaN or infinite values)
    if (!delta.allFinite()) {
      // QR solve failed, return failure
      return {p_init, false};
    }

    // Backtracking line search to ensure we make progress
    const T current_cost = residual_norm * residual_norm; // ||f(x)||^2
    const T alpha_init = T(1.0); // Start with full Newton step
    const T rho = T(0.5); // Backtracking factor
    const T c1 = T(1e-4); // Armijo condition parameter
    const int max_line_search_iters = 10;

    // Compute directional derivative: f'(x)^T * p = -||f(x)||^2 (since p = -J^T*f(x) for
    // Gauss-Newton)
    const T directional_derivative = -current_cost;

    Eigen::Vector3<T> p_new = p_cur;
    T alpha = alpha_init;
    T new_cost = std::numeric_limits<T>::max();
    bool line_search_success = false;

    for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter) {
      // Try step: x_new = x + alpha * delta
      p_new.template head<2>() = p_cur.template head<2>() + alpha * delta;

      // Evaluate cost at new point
      const auto [new_projectedPoint, new_isValid] = project(p_new);

      if (!new_isValid) {
        // Point went behind camera, reduce step size
        alpha *= rho;
        continue;
      }

      const Eigen::Vector<T, 2> residualNew =
          new_projectedPoint.template head<2>() - imagePoint.template head<2>();
      new_cost = residualNew.squaredNorm();

      // Armijo condition: f(x + alpha*p) <= f(x) + c1*alpha*f'(x)^T*p
      if (new_cost <= current_cost + c1 * alpha * directional_derivative) {
        line_search_success = true;
        break;
      }

      // Reduce step size
      alpha *= rho;
    }

    if (!line_search_success) {
      return {p_init, false};
    }

    // Update the 3D point
    p_cur = p_new;
  }

  // If we reach here, Newton's method didn't converge
  return {p_init, false};
}

template <typename T>
Eigen::Index OpenCVIntrinsicsModelT<T>::numIntrinsicParameters() const {
  return 14; // fx, fy, cx, cy, k1-k6, p1-p4
}

template <typename T>
Eigen::VectorX<T> OpenCVIntrinsicsModelT<T>::getIntrinsicParameters() const {
  Eigen::VectorX<T> params(14);
  params << fx_, fy_, cx_, cy_, distortionParams_.k1, distortionParams_.k2, distortionParams_.k3,
      distortionParams_.k4, distortionParams_.k5, distortionParams_.k6, distortionParams_.p1,
      distortionParams_.p2, distortionParams_.p3, distortionParams_.p4;
  return params;
}

template <typename T>
void OpenCVIntrinsicsModelT<T>::setIntrinsicParameters(
    const Eigen::Ref<const Eigen::VectorX<T>>& params) {
  fx_ = params(0);
  fy_ = params(1);
  cx_ = params(2);
  cy_ = params(3);
  distortionParams_.k1 = params(4);
  distortionParams_.k2 = params(5);
  distortionParams_.k3 = params(6);
  distortionParams_.k4 = params(7);
  distortionParams_.k5 = params(8);
  distortionParams_.k6 = params(9);
  distortionParams_.p1 = params(10);
  distortionParams_.p2 = params(11);
  distortionParams_.p3 = params(12);
  distortionParams_.p4 = params(13);
}

template <typename T>
std::shared_ptr<IntrinsicsModelT<T>> OpenCVIntrinsicsModelT<T>::clone() const {
  return std::make_shared<OpenCVIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_, distortionParams_);
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
OpenCVIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  // Check if point is in front of camera
  if (z <= T(0)) {
    Eigen::Matrix<T, 3, Eigen::Dynamic> zeroJacobian(3, 14);
    zeroJacobian.setZero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;

  // Normalized coordinates
  const T xp = x * z_inv;
  const T yp = y * z_inv;
  const T rsqr = xp * xp + yp * yp;

  const auto& dp = distortionParams_;

  // Radial distortion components
  // Note: the formula is radialDistortion = 1 + A/B where:
  //   A = rsqr * (k1 + rsqr * (k2 + rsqr * k3))
  //   B = 1 + rsqr * (k4 + rsqr * (k5 + rsqr * k6))
  const T A = rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3));
  const T B = T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6));
  const T radial_factor = T(1) + A / B;

  // Tangential distortion
  const T xpp = xp * radial_factor + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * xp * xp);
  const T ypp = yp * radial_factor + dp.p1 * (rsqr + T(2) * yp * yp) + T(2) * dp.p2 * xp * yp;

  // Project the point
  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Compute Jacobian matrix (3x14) with respect to
  // [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, p3, p4]
  Eigen::Matrix<T, 3, Eigen::Dynamic> jacobian(3, 14);
  jacobian.setZero();

  // Derivatives with respect to fx, fy, cx, cy (basic intrinsics)
  // u = fx * xpp + cx
  // v = fy * ypp + cy
  jacobian(0, 0) = xpp; // du/dfx
  jacobian(0, 2) = T(1); // du/dcx
  jacobian(1, 1) = ypp; // dv/dfy
  jacobian(1, 3) = T(1); // dv/dcy

  // Derivatives with respect to distortion parameters
  // radial_factor = 1 + A/B
  // d(radial_factor)/dk_i = d(A/B)/dk_i

  const T B_sq = B * B;

  // For numerator coefficients k1, k2, k3:
  // dA/dk1 = rsqr, dA/dk2 = rsqr^2, dA/dk3 = rsqr^3
  // d(A/B)/dk_i = (dA/dk_i) / B
  const T rsqr2 = rsqr * rsqr;
  const T rsqr3 = rsqr2 * rsqr;
  const T drf_dk1 = rsqr / B;
  const T drf_dk2 = rsqr2 / B;
  const T drf_dk3 = rsqr3 / B;

  // For denominator coefficients k4, k5, k6:
  // dB/dk4 = rsqr, dB/dk5 = rsqr^2, dB/dk6 = rsqr^3
  // d(A/B)/dk_i = -A * (dB/dk_i) / B^2
  const T drf_dk4 = -A * rsqr / B_sq;
  const T drf_dk5 = -A * rsqr2 / B_sq;
  const T drf_dk6 = -A * rsqr3 / B_sq;

  // d(xpp)/dk_i = xp * d(radial_factor)/dk_i
  // d(ypp)/dk_i = yp * d(radial_factor)/dk_i
  jacobian(0, 4) = fx_ * xp * drf_dk1; // du/dk1
  jacobian(1, 4) = fy_ * yp * drf_dk1; // dv/dk1
  jacobian(0, 5) = fx_ * xp * drf_dk2; // du/dk2
  jacobian(1, 5) = fy_ * yp * drf_dk2; // dv/dk2
  jacobian(0, 6) = fx_ * xp * drf_dk3; // du/dk3
  jacobian(1, 6) = fy_ * yp * drf_dk3; // dv/dk3
  jacobian(0, 7) = fx_ * xp * drf_dk4; // du/dk4
  jacobian(1, 7) = fy_ * yp * drf_dk4; // dv/dk4
  jacobian(0, 8) = fx_ * xp * drf_dk5; // du/dk5
  jacobian(1, 8) = fy_ * yp * drf_dk5; // dv/dk5
  jacobian(0, 9) = fx_ * xp * drf_dk6; // du/dk6
  jacobian(1, 9) = fy_ * yp * drf_dk6; // dv/dk6

  // Derivatives with respect to tangential distortion p1, p2
  // xpp = xp * radial_factor + 2*p1*xp*yp + p2*(rsqr + 2*xp^2)
  // ypp = yp * radial_factor + p1*(rsqr + 2*yp^2) + 2*p2*xp*yp

  // d(xpp)/dp1 = 2*xp*yp
  // d(ypp)/dp1 = rsqr + 2*yp^2
  jacobian(0, 10) = fx_ * T(2) * xp * yp; // du/dp1
  jacobian(1, 10) = fy_ * (rsqr + T(2) * yp * yp); // dv/dp1

  // d(xpp)/dp2 = rsqr + 2*xp^2
  // d(ypp)/dp2 = 2*xp*yp
  jacobian(0, 11) = fx_ * (rsqr + T(2) * xp * xp); // du/dp2
  jacobian(1, 11) = fy_ * T(2) * xp * yp; // dv/dp2

  // Derivatives with respect to p3, p4 (thin prism coefficients)
  // Note: The projection formula doesn't include p3 and p4 in the current implementation
  // so their derivatives are zero
  jacobian(0, 12) = T(0); // du/dp3
  jacobian(1, 12) = T(0); // dv/dp3
  jacobian(0, 13) = T(0); // du/dp4
  jacobian(1, 13) = T(0); // dv/dp4

  // Row 2: dz/d[all params] = 0 (depth unchanged)
  // Already set to zero

  return {projectedPoint, jacobian, true};
}

// OpenCVFisheyeIntrinsicsModelT implementation

template <typename T>
OpenCVFisheyeIntrinsicsModelT<T>::OpenCVFisheyeIntrinsicsModelT(
    int32_t imageWidth,
    int32_t imageHeight,
    T fx,
    T fy,
    T cx,
    T cy,
    const OpenCVFisheyeDistortionParametersT<T>& params)
    : IntrinsicsModelT<T>(imageWidth, imageHeight),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      distortionParams_(params) {}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> OpenCVFisheyeIntrinsicsModelT<T>::project(
    const Vector3P<T>& point) const {
  // Process each element in the packet individually since drjit::atan is not available
  Vector3P<T> result;

  for (size_t i = 0; i < Packet<T>::Size; ++i) {
    const T x = point.x()[i];
    const T y = point.y()[i];
    const T z = point.z()[i];

    if (z <= T(0)) {
      result.x()[i] = T(0);
      result.y()[i] = T(0);
      result.z()[i] = z;
      continue;
    }

    // Normalize the point by dividing by z
    const T invZ = T(1) / z;
    const T a = x * invZ;
    const T b = y * invZ;

    // Compute radius and angle
    const T rsqr = a * a + b * b;
    const T r = std::sqrt(rsqr);
    const T theta = std::atan(r);

    const auto& dp = distortionParams_;

    // Apply distortion
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;
    const T thetaD =
        theta * (T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8);

    // Compute scale factor
    const T scale = (r > T(1e-8)) ? thetaD / r : T(1);

    // Apply scale
    const T xpp = scale * a;
    const T ypp = scale * b;

    // Apply camera matrix to get pixel coordinates
    result.x()[i] = fx_ * xpp + cx_;
    result.y()[i] = fy_ * ypp + cy_;
    result.z()[i] = z;
  }

  return {result, point.z() > T(0)};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVFisheyeIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  const T z = point.z();

  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  // Normalize the point by dividing by z
  T invZ = T(1) / z;
  T a = point.x() * invZ;
  T b = point.y() * invZ;

  // Compute radius and angle
  T rsqr = a * a + b * b;
  T r = std::sqrt(rsqr);
  T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // Apply distortion: theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
  T theta2 = theta * theta;
  T theta4 = theta2 * theta2;
  T theta6 = theta4 * theta2;
  T theta8 = theta4 * theta4;
  T thetaD = theta * (T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8);

  // Compute scale factor: theta_d / r, handle r->0 singularity
  T scale = (r > T(1e-8)) ? thetaD / r : T(1);

  // Apply scale
  T xpp = scale * a;
  T ypp = scale * b;

  // Apply camera matrix to get pixel coordinates
  T u = fx_ * xpp + cx_;
  T v = fy_ * ypp + cy_;

  return {Eigen::Vector3<T>(u, v, z), true};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, 3>, bool>
OpenCVFisheyeIntrinsicsModelT<T>::projectJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T z_inv = T(1) / z;
  const T z_inv_sq = z_inv * z_inv;

  // Normalized coordinates
  const T a = x * z_inv;
  const T b = y * z_inv;
  const T rsqr = a * a + b * b;
  const T r = std::sqrt(rsqr);
  const T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // Distortion polynomial coefficients
  const T theta2 = theta * theta;
  const T theta4 = theta2 * theta2;
  const T theta6 = theta4 * theta2;
  const T theta8 = theta4 * theta4;
  const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
  const T thetaD = theta * poly;

  // Handle r->0 singularity
  const bool nearCenter = r < T(1e-8);
  const T scale = nearCenter ? T(1) : thetaD / r;

  // Distorted normalized coordinates
  const T xpp = scale * a;
  const T ypp = scale * b;

  // Project the point
  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Compute Jacobian matrix (3x3)
  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  if (nearCenter) {
    // Near optical axis: Jacobian simplifies to pinhole-like behavior
    jacobian(0, 0) = fx_ * z_inv;
    jacobian(0, 2) = -fx_ * x * z_inv_sq;
    jacobian(1, 1) = fy_ * z_inv;
    jacobian(1, 2) = -fy_ * y * z_inv_sq;
    jacobian(2, 2) = T(1);
    return {projectedPoint, jacobian, true};
  }

  // Derivative of poly with respect to theta
  const T dpoly_dtheta = T(2) * dp.k1 * theta + T(4) * dp.k2 * theta * theta2 +
      T(6) * dp.k3 * theta * theta4 + T(8) * dp.k4 * theta * theta6;
  const T dthetaD_dtheta = poly + theta * dpoly_dtheta;

  // Derivative of theta with respect to r: d(atan(r))/dr = 1/(1+r^2)
  const T dtheta_dr = T(1) / (T(1) + rsqr);

  // Derivative of scale with respect to r
  // scale = thetaD / r
  // d(scale)/dr = (d(thetaD)/dr * r - thetaD) / r^2
  //             = (dthetaD_dtheta * dtheta_dr * r - thetaD) / r^2
  const T r_inv = T(1) / r;
  const T dscale_dr = (dthetaD_dtheta * dtheta_dr * r - thetaD) * r_inv * r_inv;

  // Derivatives of r with respect to a, b
  // r = sqrt(a^2 + b^2)
  const T dr_da = a * r_inv;
  const T dr_db = b * r_inv;

  // Derivatives of xpp, ypp with respect to a, b
  // xpp = scale * a
  // dxpp/da = d(scale)/da * a + scale = dscale_dr * dr_da * a + scale
  // dxpp/db = d(scale)/db * a = dscale_dr * dr_db * a
  const T dxpp_da = dscale_dr * dr_da * a + scale;
  const T dxpp_db = dscale_dr * dr_db * a;
  const T dypp_da = dscale_dr * dr_da * b;
  const T dypp_db = dscale_dr * dr_db * b + scale;

  // Derivatives of a, b with respect to x, y, z
  // a = x/z, b = y/z
  // da/dx = 1/z, da/dy = 0, da/dz = -x/z^2
  // db/dx = 0, db/dy = 1/z, db/dz = -y/z^2

  // Chain rule: du/dx = fx * dxpp/dx = fx * (dxpp/da * da/dx + dxpp/db * db/dx)
  jacobian(0, 0) = fx_ * dxpp_da * z_inv;
  jacobian(0, 1) = fx_ * dxpp_db * z_inv;
  jacobian(0, 2) = fx_ * (dxpp_da * (-x * z_inv_sq) + dxpp_db * (-y * z_inv_sq));

  jacobian(1, 0) = fy_ * dypp_da * z_inv;
  jacobian(1, 1) = fy_ * dypp_db * z_inv;
  jacobian(1, 2) = fy_ * (dypp_da * (-x * z_inv_sq) + dypp_db * (-y * z_inv_sq));

  jacobian(2, 2) = T(1);

  return {projectedPoint, jacobian, true};
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> OpenCVFisheyeIntrinsicsModelT<T>::resize(
    int32_t imageWidth,
    int32_t imageHeight) const {
  T scaleX = T(imageWidth) / T(this->imageWidth());
  T scaleY = T(imageHeight) / T(this->imageHeight());

  // Apply the correct formula for camera center
  T new_cx = (cx_ + T(0.5)) * scaleX - T(0.5);
  T new_cy = (cy_ + T(0.5)) * scaleY - T(0.5);

  return std::make_shared<OpenCVFisheyeIntrinsicsModelT<T>>(
      imageWidth, imageHeight, fx_ * scaleX, fy_ * scaleY, new_cx, new_cy, distortionParams_);
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> OpenCVFisheyeIntrinsicsModelT<T>::crop(
    int32_t top,
    int32_t left,
    int32_t newWidth,
    int32_t newHeight) const {
  // Ensure the crop doesn't exceed the original image dimensions
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Adjust the principal point by subtracting the crop offset
  T cameraCenter_cropped_cx = cx_ - T(left);
  T cameraCenter_cropped_cy = cy_ - T(top);

  return std::make_shared<OpenCVFisheyeIntrinsicsModelT<T>>(
      width, height, fx_, fy_, cameraCenter_cropped_cx, cameraCenter_cropped_cy, distortionParams_);
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVFisheyeIntrinsicsModelT<T>::unproject(
    const Eigen::Vector3<T>& imagePoint,
    int maxIterations,
    T tolerance) const {
  const T u = imagePoint(0);
  const T v = imagePoint(1);
  const T depth = imagePoint(2);

  if (depth <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  // Convert to normalized distorted coordinates
  const T xpp = (u - cx_) / fx_;
  const T ypp = (v - cy_) / fy_;

  // Compute distorted radius (thetaD)
  const T rDistorted = std::sqrt(xpp * xpp + ypp * ypp);

  // Handle center case
  if (rDistorted < T(1e-8)) {
    return {Eigen::Vector3<T>(T(0), T(0), depth), true};
  }

  // Newton's method to solve: thetaD = theta * poly(theta)
  // f(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) - thetaD = 0
  const auto& dp = distortionParams_;
  T theta = rDistorted; // Initial guess

  for (int iter = 0; iter < maxIterations; ++iter) {
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;

    const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
    const T f = theta * poly - rDistorted;

    if (std::abs(f) < tolerance) {
      break;
    }

    // f'(theta) = poly + theta * dpoly/dtheta
    const T dpoly_dtheta = T(2) * dp.k1 * theta + T(4) * dp.k2 * theta * theta2 +
        T(6) * dp.k3 * theta * theta4 + T(8) * dp.k4 * theta * theta6;
    const T df = poly + theta * dpoly_dtheta;

    if (std::abs(df) < T(1e-12)) {
      break;
    }

    theta = theta - f / df;
  }

  // Compute undistorted radius: r = tan(theta)
  const T r = std::tan(theta);

  // Scale factor from distorted to undistorted
  // distorted = scale * undistorted, where scale = thetaD / r
  // So undistorted = distorted * r / thetaD
  const T unscale = (rDistorted > T(1e-8)) ? r / rDistorted : T(1);

  // Undistorted normalized coordinates
  const T a = xpp * unscale;
  const T b = ypp * unscale;

  // Convert to 3D point
  return {Eigen::Vector3<T>(a * depth, b * depth, depth), true};
}

template <typename T>
Eigen::Index OpenCVFisheyeIntrinsicsModelT<T>::numIntrinsicParameters() const {
  return 8; // fx, fy, cx, cy, k1, k2, k3, k4
}

template <typename T>
Eigen::VectorX<T> OpenCVFisheyeIntrinsicsModelT<T>::getIntrinsicParameters() const {
  Eigen::VectorX<T> params(8);
  params << fx_, fy_, cx_, cy_, distortionParams_.k1, distortionParams_.k2, distortionParams_.k3,
      distortionParams_.k4;
  return params;
}

template <typename T>
void OpenCVFisheyeIntrinsicsModelT<T>::setIntrinsicParameters(
    const Eigen::Ref<const Eigen::VectorX<T>>& params) {
  fx_ = params(0);
  fy_ = params(1);
  cx_ = params(2);
  cy_ = params(3);
  distortionParams_.k1 = params(4);
  distortionParams_.k2 = params(5);
  distortionParams_.k3 = params(6);
  distortionParams_.k4 = params(7);
}

template <typename T>
std::shared_ptr<IntrinsicsModelT<T>> OpenCVFisheyeIntrinsicsModelT<T>::clone() const {
  return std::make_shared<OpenCVFisheyeIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_, distortionParams_);
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
OpenCVFisheyeIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  if (z <= T(0)) {
    Eigen::Matrix<T, 3, Eigen::Dynamic> zeroJacobian(3, 8);
    zeroJacobian.setZero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;

  // Normalized coordinates
  const T a = x * z_inv;
  const T b = y * z_inv;
  const T rsqr = a * a + b * b;
  const T r = std::sqrt(rsqr);
  const T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // Distortion polynomial
  const T theta2 = theta * theta;
  const T theta4 = theta2 * theta2;
  const T theta6 = theta4 * theta2;
  const T theta8 = theta4 * theta4;
  const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
  const T thetaD = theta * poly;

  // Handle r->0 singularity
  const bool nearCenter = r < T(1e-8);
  const T scale = nearCenter ? T(1) : thetaD / r;

  // Distorted normalized coordinates
  const T xpp = scale * a;
  const T ypp = scale * b;

  // Project the point
  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Compute Jacobian matrix (3x8) with respect to [fx, fy, cx, cy, k1, k2, k3, k4]
  Eigen::Matrix<T, 3, Eigen::Dynamic> jacobian(3, 8);
  jacobian.setZero();

  // Derivatives with respect to fx, fy, cx, cy
  jacobian(0, 0) = xpp; // du/dfx
  jacobian(0, 2) = T(1); // du/dcx
  jacobian(1, 1) = ypp; // dv/dfy
  jacobian(1, 3) = T(1); // dv/dcy

  if (!nearCenter) {
    // Derivatives with respect to distortion parameters k1, k2, k3, k4
    // thetaD = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    // d(thetaD)/dk1 = theta * theta^2 = theta^3
    // d(thetaD)/dk2 = theta * theta^4 = theta^5
    // d(thetaD)/dk3 = theta * theta^6 = theta^7
    // d(thetaD)/dk4 = theta * theta^8 = theta^9

    // scale = thetaD / r
    // d(scale)/dk_i = d(thetaD)/dk_i / r

    const T r_inv = T(1) / r;
    const T dscale_dk1 = theta * theta2 * r_inv;
    const T dscale_dk2 = theta * theta4 * r_inv;
    const T dscale_dk3 = theta * theta6 * r_inv;
    const T dscale_dk4 = theta * theta8 * r_inv;

    // xpp = scale * a, ypp = scale * b
    // d(xpp)/dk_i = d(scale)/dk_i * a
    // d(ypp)/dk_i = d(scale)/dk_i * b

    jacobian(0, 4) = fx_ * dscale_dk1 * a; // du/dk1
    jacobian(1, 4) = fy_ * dscale_dk1 * b; // dv/dk1
    jacobian(0, 5) = fx_ * dscale_dk2 * a; // du/dk2
    jacobian(1, 5) = fy_ * dscale_dk2 * b; // dv/dk2
    jacobian(0, 6) = fx_ * dscale_dk3 * a; // du/dk3
    jacobian(1, 6) = fy_ * dscale_dk3 * b; // dv/dk3
    jacobian(0, 7) = fx_ * dscale_dk4 * a; // du/dk4
    jacobian(1, 7) = fy_ * dscale_dk4 * b; // dv/dk4
  }

  return {projectedPoint, jacobian, true};
}

template <typename T>
CameraT<T> CameraT<T>::lookAt(
    const Eigen::Vector3<T>& position,
    const Eigen::Vector3<T>& target,
    const Eigen::Vector3<T>& up) const {
  const Eigen::Vector3<T> diff = target - position;
  if (diff.norm() == T(0)) {
    // If target is the same as position, return the original camera
    return *this;
  }

  Eigen::Transform<T, 3, Eigen::Affine> eyeToWorldMat =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  eyeToWorldMat.translation() = position;

  Eigen::Vector3<T> zVec = diff.normalized();
  // Need to flip y upside down because y points down in image
  // coordinates (pixel 0,0 is in the top left)
  Eigen::Vector3<T> xVec = diff.cross(-up.normalized());
  if (xVec.norm() == T(0)) {
    // Up vector is parallel to the target position vector, ignore it and just
    // make sure we point the camera in the right direction
    Eigen::Quaternion<T> transform =
        Eigen::Quaternion<T>::FromTwoVectors(Eigen::Vector3<T>::UnitZ(), zVec);
    eyeToWorldMat.linear() = transform.toRotationMatrix();
  } else {
    Eigen::Vector3<T> yVec = xVec.cross(zVec).normalized();
    xVec = yVec.cross(zVec).normalized();
    eyeToWorldMat.linear().col(0) = xVec;
    eyeToWorldMat.linear().col(1) = yVec;
    eyeToWorldMat.linear().col(2) = zVec;
  }

  if (eyeToWorldMat.linear().determinant() < T(0.9)) {
    // Error in creating rotation matrix, return the original camera
    return *this;
  }

  CameraT<T> result = *this;
  result.setEyeFromWorld(eyeToWorldMat.inverse());
  return result;
}

template <typename T>
CameraT<T>
CameraT<T>::framePoints(const std::vector<Eigen::Vector3<T>>& points, T minZ, T edgePadding) const {
  if (points.empty()) {
    return *this;
  }

  const auto fx = this->fx();
  const auto fy = this->fy();

  const auto w = this->imageWidth();
  const auto h = this->imageHeight();

  const auto cx = w / T(2);
  const auto cy = h / T(2);

  // Calculate bounding box of points in eye space
  Eigen::AlignedBox<T, 3> bbox_eye;
  for (const auto& p_world : points) {
    // Transform world point to eye space
    bbox_eye.extend(eyeFromWorld_ * p_world);
  }

  // Create a new camera centered on the points
  CameraT<T> camera_recentered = *this;
  Eigen::Transform<T, 3, Eigen::Affine> newTransform =
      Eigen::Translation<T, 3>(
          -bbox_eye.center().x(), -bbox_eye.center().y(), -bbox_eye.min().z()) *
      eyeFromWorld_;
  camera_recentered.setEyeFromWorld(newTransform);

  // Calculate the maximum distance needed to ensure all points are in view
  const T max_x_pixel_diff = (T(1) - T(2) * edgePadding) * std::max(cx, T(w - 1) - cx);
  const T max_y_pixel_diff = (T(1) - T(2) * edgePadding) * std::max(cy, T(h - 1) - cy);

  T max_dz = std::numeric_limits<T>::lowest();
  for (const auto& p_world : points) {
    const Eigen::Vector3<T> p_eye = newTransform * p_world;

    // Make sure we're in front of the camera
    if (p_eye.z() < minZ) {
      max_dz = std::max(max_dz, minZ - p_eye.z());
    }
    max_dz = std::max(max_dz, (fx * std::abs(p_eye.x())) / max_x_pixel_diff - p_eye.z());
    max_dz = std::max(max_dz, (fy * std::abs(p_eye.y())) / max_y_pixel_diff - p_eye.z());
  }

  if (max_dz == std::numeric_limits<T>::lowest()) {
    return camera_recentered;
  }

  // Create the final camera with adjusted position
  CameraT<T> camera_final = camera_recentered;
  camera_final.setEyeFromWorld(
      Eigen::Translation<T, 3>(T(0), T(0), max_dz) * camera_recentered.eyeFromWorld());

  return camera_final;
}

template <typename T>
Vector3P<T> CameraT<T>::transformWorldToEye(const Vector3P<T>& worldPoints) const {
  // Transform all world points to camera space using SIMD operations
  const auto& R = eyeFromWorld_.linear();
  const auto& t = eyeFromWorld_.translation();

  // Apply rotation matrix: R * worldPoints
  Vector3P<T> eyePoints;
  eyePoints.x() = R(0, 0) * worldPoints.x() + R(0, 1) * worldPoints.y() + R(0, 2) * worldPoints.z();
  eyePoints.y() = R(1, 0) * worldPoints.x() + R(1, 1) * worldPoints.y() + R(1, 2) * worldPoints.z();
  eyePoints.z() = R(2, 0) * worldPoints.x() + R(2, 1) * worldPoints.y() + R(2, 2) * worldPoints.z();

  // Apply translation: eyePoints = R * worldPoints + t
  eyePoints.x() += t(0);
  eyePoints.y() += t(1);
  eyePoints.z() += t(2);

  return eyePoints;
}

template <typename T>
Eigen::Vector3<T> CameraT<T>::transformWorldToEye(const Eigen::Vector3<T>& worldPoint) const {
  // Transform world point to camera space using Eigen operations
  return eyeFromWorld_ * worldPoint;
}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> CameraT<T>::project(
    const Vector3P<T>& worldPoints) const {
  // Transform all world points to camera space using SIMD operations
  const Vector3P<T> eyePoints = transformWorldToEye(worldPoints);

  // Use the intrinsics model's SIMD project method directly
  return intrinsicsModel_->project(eyePoints);
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> CameraT<T>::project(const Eigen::Vector3<T>& worldPoint) const {
  // Transform world point to camera space using helper function
  const Eigen::Vector3<T> eyePoint = transformWorldToEye(worldPoint);

  // Use the intrinsics model to project to image coordinates
  return intrinsicsModel_->project(eyePoint);
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 2, 3>, bool> CameraT<T>::projectJacobian(
    const Eigen::Vector3<T>& worldPoint) const {
  // Transform world point to camera space using helper function
  const Eigen::Vector3<T> eyePoint = transformWorldToEye(worldPoint);

  // Get the Jacobian of projection with respect to camera coordinates
  const auto [projectedPoint, jacobian_eye, isValid] = intrinsicsModel_->projectJacobian(eyePoint);

  if (!isValid) {
    return {projectedPoint, Eigen::Matrix<T, 2, 3>::Zero(), false};
  }

  // Extract the 2x3 Jacobian matrix from the 3x3 matrix (ignore the third row for homogeneous
  // coordinates)
  const Eigen::Matrix<T, 2, 3> J_proj_eye = jacobian_eye.template topRows<2>();

  // Get the rotation part of the world-to-eye transform
  const Eigen::Matrix<T, 3, 3> R_eye_world = eyeFromWorld_.linear();

  // Apply chain rule: J_proj_world = J_proj_eye * R_eye_world
  const Eigen::Matrix<T, 2, 3> J_proj_world = J_proj_eye * R_eye_world;

  return {projectedPoint, J_proj_world, true};
}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType>
CameraT<T>::unproject(const Vector3P<T>& imagePoints, int maxIterations, T tolerance) const {
  using PacketT = Packet<T>;

  // Process each point in the packet individually using the intrinsics model's unproject method
  Vector3P<T> worldPoints;
  auto validMask = drjit::full<typename PacketT::MaskType>(true);

  for (size_t i = 0; i < PacketT::Size; ++i) {
    // Extract 3D image point for this element (u, v, z)
    Eigen::Vector3<T> imagePoint(imagePoints.x()[i], imagePoints.y()[i], imagePoints.z()[i]);

    // Unproject to camera space using the intrinsics model
    auto [eyePoint, isValid] = intrinsicsModel_->unproject(imagePoint, maxIterations, tolerance);

    // Check if unprojection was successful
    if (!isValid) {
      validMask[i] = false;
      // Set to zero for invalid points
      worldPoints.x()[i] = T(0);
      worldPoints.y()[i] = T(0);
      worldPoints.z()[i] = T(0);
      continue;
    }

    // Transform from camera space to world space
    const Eigen::Vector3<T> worldPoint = worldFromEye() * eyePoint;

    // Store the result
    worldPoints.x()[i] = worldPoint(0);
    worldPoints.y()[i] = worldPoint(1);
    worldPoints.z()[i] = worldPoint(2);
  }

  return {worldPoints, validMask};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool>
CameraT<T>::unproject(const Eigen::Vector3<T>& imagePoint, int maxIterations, T tolerance) const {
  // Unproject to camera space using the intrinsics model
  auto [eyePoint, isValid] = intrinsicsModel_->unproject(imagePoint, maxIterations, tolerance);

  if (!isValid) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  // Transform from camera space to world space
  const Eigen::Vector3<T> worldPoint = worldFromEye() * eyePoint;

  return {worldPoint, true};
}

// Explicit template instantiations
template class CameraT<float>;
template class CameraT<double>;
template class IntrinsicsModelT<float>;
template class IntrinsicsModelT<double>;
template class PinholeIntrinsicsModelT<float>;
template class PinholeIntrinsicsModelT<double>;
template class OpenCVIntrinsicsModelT<float>;
template class OpenCVIntrinsicsModelT<double>;
template class OpenCVFisheyeIntrinsicsModelT<float>;
template class OpenCVFisheyeIntrinsicsModelT<double>;

} // namespace momentum
