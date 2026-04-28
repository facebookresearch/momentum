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

#include <array>
#include <cfloat>
#include <cmath>

namespace momentum {

// Default is VGA resolution with a focal length corresponding to a 3.6 mm sensor and 5 mm lens.
template <typename T>
CameraT<T>::CameraT()
    : intrinsicsModel_(
          std::make_shared<PinholeIntrinsicsModelT<T>>(
              640,
              480,
              (5.0 / 3.6) * 640,
              (5.0 / 3.6) * 640)) {}

template <typename T>
CameraT<T>::CameraT(
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld)
    : eyeFromWorld_(eyeFromWorld), intrinsicsModel_(intrinsicsModel) {}

template <typename T>
PinholeIntrinsicsModelT<T>::PinholeIntrinsicsModelT(
    int32_t imageWidth,
    int32_t imageHeight,
    T fx,
    T fy,
    T cx,
    T cy)
    : IntrinsicsModelT<T>(imageWidth, imageHeight), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

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
  // TODO: project() does not guard against point.z() == 0; division by zero will produce NaN/Inf
  // before the validity mask filters them out.
  Packet<T> x = point.x() / point.z();
  Packet<T> y = point.y() / point.z();

  Packet<T> u = fx_ * x + this->cx();
  Packet<T> v = fy_ * y + this->cy();

  return {Vector3P<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> PinholeIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  // TODO: project() does not guard against point.z() == 0; division by zero will produce NaN/Inf
  // before the validity flag filters them out.
  T x = point.x() / point.z();
  T y = point.y() / point.z();

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

  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T z_inv = T(1) / z;
  const T z_inv_sq = z_inv * z_inv;

  const T u = fx_ * x * z_inv + cx_;
  const T v = fy_ * y * z_inv + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Pinhole projection Jacobian:
  //   u = fx * (x/z) + cx,  v = fy * (y/z) + cy,  z passes through unchanged.
  // TODO: jacobian(0,1), jacobian(1,0), jacobian(2,0), jacobian(2,1) are redundant assignments
  // since the matrix is zero-initialized; consider removing them in a separate code-change diff.
  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  jacobian(0, 0) = fx_ * z_inv;
  jacobian(0, 1) = T(0);
  jacobian(0, 2) = -fx_ * x * z_inv_sq;

  jacobian(1, 0) = T(0);
  jacobian(1, 1) = fy_ * z_inv;
  jacobian(1, 2) = -fy_ * y * z_inv_sq;

  jacobian(2, 0) = T(0);
  jacobian(2, 1) = T(0);
  jacobian(2, 2) = T(1);

  return {projectedPoint, jacobian, true};
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> PinholeIntrinsicsModelT<T>::resize(
    int32_t imageWidth,
    int32_t imageHeight) const {
  T scaleX = T(imageWidth) / T(this->imageWidth());
  T scaleY = T(imageHeight) / T(this->imageHeight());

  // Use the half-pixel-offset convention so that pixel centers map exactly between resolutions:
  //   new_cx = (old_cx + 0.5) * scaleX - 0.5
  //   new_cy = (old_cy + 0.5) * scaleY - 0.5
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
  // Clamp the crop region to stay inside the original image bounds.
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Shift the principal point so it stays at the same physical location after cropping.
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
  // Pinhole has no distortion to invert, so unprojection is closed-form.
  const T u = imagePoint(0);
  const T v = imagePoint(1);
  const T depth = imagePoint(2);

  const T x = (u - cx_) / fx_;
  const T y = (v - cy_) / fy_;

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
  auto result = std::make_shared<PinholeIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_);
  result->setName(this->name());
  return result;
}

template <typename T>
std::vector<std::string> PinholeIntrinsicsModelT<T>::getParameterNames() const {
  return {"fx", "fy", "cx", "cy"};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
PinholeIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  if (z <= T(0)) {
    Eigen::Matrix<T, 3, 4> zeroJacobian = Eigen::Matrix<T, 3, 4>::Zero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;
  const T xn = x * z_inv;
  const T yn = y * z_inv;

  const T u = fx_ * xn + cx_;
  const T v = fy_ * yn + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Jacobian with respect to [fx, fy, cx, cy]:
  //   du/dfx = x/z,  du/dfy = 0,    du/dcx = 1,  du/dcy = 0
  //   dv/dfx = 0,    dv/dfy = y/z,  dv/dcx = 0,  dv/dcy = 1
  //   dz/d*  = 0     (depth is independent of intrinsics)
  // TODO: the explicit T(0) assignments below are redundant given the zero-init; keep here for
  // readability symmetry with non-zero rows but consider removing in a separate code-change diff.
  Eigen::Matrix<T, 3, 4> jacobian = Eigen::Matrix<T, 3, 4>::Zero();

  jacobian(0, 0) = xn;
  jacobian(0, 1) = T(0);
  jacobian(0, 2) = T(1);
  jacobian(0, 3) = T(0);

  jacobian(1, 0) = T(0);
  jacobian(1, 1) = yn;
  jacobian(1, 2) = T(0);
  jacobian(1, 3) = T(1);

  return {projectedPoint, jacobian, true};
}

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

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> OpenCVIntrinsicsModelT<T>::project(
    const Vector3P<T>& point) const {
  // OpenCV rational distortion model: rational radial polynomial in r^2 plus tangential terms.
  // TODO: project() does not guard against point.z() == 0; division by zero will produce NaN/Inf
  // before the validity mask filters them out.
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

  Packet<T> u = fx_ * xpp + cx_;
  Packet<T> v = fy_ * ypp + cy_;

  return {Vector3P<T>(u, v, point.z()), point.z() > T(0)};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  // TODO: project() does not guard against point.z() == 0; division by zero will produce NaN/Inf
  // before the validity flag filters them out.
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

  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T z_inv = T(1) / z;
  const T z_inv_sq = z_inv * z_inv;

  const T xp = x * z_inv;
  const T yp = y * z_inv;
  const T rsqr = xp * xp + yp * yp;

  const auto& dp = distortionParams_;

  // Radial factor = num/den, where num and den are even-degree polynomials in r.
  const T radial_num = T(1) + rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3));
  const T radial_den = T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6));
  const T radial_factor = radial_num / radial_den;

  // Tangential terms (Brown-Conrady).
  const T xpp = xp * radial_factor + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * xp * xp);
  const T ypp = yp * radial_factor + dp.p1 * (rsqr + T(2) * yp * yp) + T(2) * dp.p2 * xp * yp;

  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // d(radial_factor)/d(rsqr) via the quotient rule.
  const T dradial_num_drsqr = dp.k1 + rsqr * (T(2) * dp.k2 + rsqr * T(3) * dp.k3);
  const T dradial_den_drsqr = dp.k4 + rsqr * (T(2) * dp.k5 + rsqr * T(3) * dp.k6);
  const T dradial_factor_drsqr =
      (dradial_num_drsqr * radial_den - radial_num * dradial_den_drsqr) / (radial_den * radial_den);

  // Partials of (xpp, ypp) with respect to (xp, yp); chain rule will combine with
  // d(xp,yp)/d(x,y,z).
  const T dxpp_dxp =
      radial_factor + xp * dradial_factor_drsqr * T(2) * xp + T(2) * dp.p1 * yp + dp.p2 * T(6) * xp;
  const T dxpp_dyp = xp * dradial_factor_drsqr * T(2) * yp + T(2) * dp.p1 * xp + dp.p2 * T(2) * yp;
  const T dypp_dxp = yp * dradial_factor_drsqr * T(2) * xp + dp.p1 * T(2) * xp + T(2) * dp.p2 * yp;
  const T dypp_dyp =
      radial_factor + yp * dradial_factor_drsqr * T(2) * yp + dp.p1 * T(6) * yp + T(2) * dp.p2 * xp;

  // TODO: jacobian(2,0) and jacobian(2,1) are redundant assignments since the matrix is
  // zero-initialized; consider removing them in a separate code-change diff.
  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  jacobian(0, 0) = fx_ * dxpp_dxp * z_inv;
  jacobian(0, 1) = fx_ * dxpp_dyp * z_inv;
  jacobian(0, 2) = -fx_ * (dxpp_dxp * x * z_inv_sq + dxpp_dyp * y * z_inv_sq);

  jacobian(1, 0) = fy_ * dypp_dxp * z_inv;
  jacobian(1, 1) = fy_ * dypp_dyp * z_inv;
  jacobian(1, 2) = -fy_ * (dypp_dxp * x * z_inv_sq + dypp_dyp * y * z_inv_sq);

  jacobian(2, 0) = T(0);
  jacobian(2, 1) = T(0);
  jacobian(2, 2) = T(1);

  return {projectedPoint, jacobian, true};
}

template <typename T>
std::shared_ptr<const IntrinsicsModelT<T>> OpenCVIntrinsicsModelT<T>::resize(
    int32_t imageWidth,
    int32_t imageHeight) const {
  T scaleX = T(imageWidth) / T(this->imageWidth());
  T scaleY = T(imageHeight) / T(this->imageHeight());

  // Use the half-pixel-offset convention (same as Pinhole resize).
  // Distortion parameters are unitless in normalized coordinates and stay unchanged.
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
  // Clamp the crop region to stay inside the original image bounds.
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Shift the principal point so it stays at the same physical location after cropping.
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

  if (depth <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  // Initial guess: ignore distortion. Distortion is small near the optical center, so this is
  // typically within Newton's basin of convergence.
  const Eigen::Vector3<T> p_init((u - cx_) / fx_, (v - cy_) / fy_, depth);
  Eigen::Vector3<T> p_cur = p_init;

  // Gauss-Newton with backtracking (Armijo) line search to invert the nonlinear projection.
  // On any failure (invalid point, singular Jacobian, line-search failure, no convergence) we
  // return the undistorted initial guess together with isValid=false.
  for (int iter = 0; iter < maxIterations; ++iter) {
    const auto [projectedPoint, jacobian, isValid] = projectJacobian(p_cur);

    if (!isValid) {
      return {p_init, false};
    }

    const Eigen::Vector<T, 2> residual =
        projectedPoint.template head<2>() - imagePoint.template head<2>();
    const T residual_norm = residual.norm();

    if (residual_norm < tolerance) {
      return {p_cur, true};
    }

    // Drop the depth pass-through row to get the 2x2 image-space Jacobian.
    Eigen::Matrix<T, 2, 2> J = jacobian.template topLeftCorner<2, 2>();

    // Solve J * delta = -residual via Householder QR for numerical stability.
    Eigen::Vector<T, 2> rhs = -residual;
    const Eigen::Vector<T, 2> delta = J.householderQr().solve(rhs);

    if (!delta.allFinite()) {
      return {p_init, false};
    }

    // Backtracking parameters.
    const T current_cost = residual_norm * residual_norm; // ||f(x)||^2
    const T alpha_init = T(1.0);
    const T rho = T(0.5);
    const T c1 = T(1e-4); // Armijo condition parameter
    const int max_line_search_iters = 10;

    // For Gauss-Newton with step delta = -J^+ f, f'(x)^T * delta = -||f(x)||^2.
    const T directional_derivative = -current_cost;

    Eigen::Vector3<T> p_new = p_cur;
    T alpha = alpha_init;
    T new_cost = std::numeric_limits<T>::max();
    bool line_search_success = false;

    for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter) {
      p_new.template head<2>() = p_cur.template head<2>() + alpha * delta;

      const auto [new_projectedPoint, new_isValid] = project(p_new);

      if (!new_isValid) {
        // Stepped behind the camera; shrink and retry.
        alpha *= rho;
        continue;
      }

      const Eigen::Vector<T, 2> residualNew =
          new_projectedPoint.template head<2>() - imagePoint.template head<2>();
      new_cost = residualNew.squaredNorm();

      // Armijo condition: f(x + alpha*p) <= f(x) + c1*alpha*f'(x)^T*p.
      if (new_cost <= current_cost + c1 * alpha * directional_derivative) {
        line_search_success = true;
        break;
      }

      alpha *= rho;
    }

    if (!line_search_success) {
      return {p_init, false};
    }

    p_cur = p_new;
  }

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
  auto result = std::make_shared<OpenCVIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_, distortionParams_);
  result->setName(this->name());
  return result;
}

template <typename T>
std::vector<std::string> OpenCVIntrinsicsModelT<T>::getParameterNames() const {
  return {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "p3", "p4"};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
OpenCVIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  if (z <= T(0)) {
    Eigen::Matrix<T, 3, 14> zeroJacobian = Eigen::Matrix<T, 3, 14>::Zero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;

  const T xp = x * z_inv;
  const T yp = y * z_inv;
  const T rsqr = xp * xp + yp * yp;

  const auto& dp = distortionParams_;

  // radial_factor = 1 + A/B, where
  //   A = rsqr * (k1 + rsqr * (k2 + rsqr * k3))
  //   B = 1 + rsqr * (k4 + rsqr * (k5 + rsqr * k6))
  const T A = rsqr * (dp.k1 + rsqr * (dp.k2 + rsqr * dp.k3));
  const T B = T(1) + rsqr * (dp.k4 + rsqr * (dp.k5 + rsqr * dp.k6));
  const T radial_factor = T(1) + A / B;

  const T xpp = xp * radial_factor + T(2) * dp.p1 * xp * yp + dp.p2 * (rsqr + T(2) * xp * xp);
  const T ypp = yp * radial_factor + dp.p1 * (rsqr + T(2) * yp * yp) + T(2) * dp.p2 * xp * yp;

  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Jacobian columns are ordered as [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, p3, p4],
  // matching getIntrinsicParameters() and getParameterNames().
  Eigen::Matrix<T, 3, 14> jacobian = Eigen::Matrix<T, 3, 14>::Zero();

  // d/d{fx, fy, cx, cy}: u = fx * xpp + cx,  v = fy * ypp + cy.
  jacobian(0, 0) = xpp;
  jacobian(0, 2) = T(1);
  jacobian(1, 1) = ypp;
  jacobian(1, 3) = T(1);

  // For radial coefficients k_i, only radial_factor depends on them, so
  //   d(xpp)/dk_i = xp * d(radial_factor)/dk_i, similarly for ypp.
  //
  // Numerator coefficients (k1, k2, k3): d(A/B)/dk_i = (dA/dk_i) / B with
  //   dA/dk1 = rsqr, dA/dk2 = rsqr^2, dA/dk3 = rsqr^3.
  // Denominator coefficients (k4, k5, k6): d(A/B)/dk_i = -A * (dB/dk_i) / B^2 with
  //   dB/dk4 = rsqr, dB/dk5 = rsqr^2, dB/dk6 = rsqr^3.
  const T B_sq = B * B;
  const T rsqr2 = rsqr * rsqr;
  const T rsqr3 = rsqr2 * rsqr;
  const T drf_dk1 = rsqr / B;
  const T drf_dk2 = rsqr2 / B;
  const T drf_dk3 = rsqr3 / B;
  const T drf_dk4 = -A * rsqr / B_sq;
  const T drf_dk5 = -A * rsqr2 / B_sq;
  const T drf_dk6 = -A * rsqr3 / B_sq;

  jacobian(0, 4) = fx_ * xp * drf_dk1;
  jacobian(1, 4) = fy_ * yp * drf_dk1;
  jacobian(0, 5) = fx_ * xp * drf_dk2;
  jacobian(1, 5) = fy_ * yp * drf_dk2;
  jacobian(0, 6) = fx_ * xp * drf_dk3;
  jacobian(1, 6) = fy_ * yp * drf_dk3;
  jacobian(0, 7) = fx_ * xp * drf_dk4;
  jacobian(1, 7) = fy_ * yp * drf_dk4;
  jacobian(0, 8) = fx_ * xp * drf_dk5;
  jacobian(1, 8) = fy_ * yp * drf_dk5;
  jacobian(0, 9) = fx_ * xp * drf_dk6;
  jacobian(1, 9) = fy_ * yp * drf_dk6;

  // Tangential distortion (Brown-Conrady):
  //   d(xpp)/dp1 = 2*xp*yp,           d(ypp)/dp1 = rsqr + 2*yp^2
  //   d(xpp)/dp2 = rsqr + 2*xp^2,     d(ypp)/dp2 = 2*xp*yp
  jacobian(0, 10) = fx_ * T(2) * xp * yp;
  jacobian(1, 10) = fy_ * (rsqr + T(2) * yp * yp);
  jacobian(0, 11) = fx_ * (rsqr + T(2) * xp * xp);
  jacobian(1, 11) = fy_ * T(2) * xp * yp;

  // TODO: p3 and p4 (thin-prism coefficients) are exposed in OpenCVDistortionParametersT but the
  // projection formula does not use them, so their derivatives are zero. Either implement
  // thin-prism distortion or remove p3/p4 from the parameter set. The explicit T(0) writes below
  // are also redundant given the zero-init.
  jacobian(0, 12) = T(0);
  jacobian(1, 12) = T(0);
  jacobian(0, 13) = T(0);
  jacobian(1, 13) = T(0);

  return {projectedPoint, jacobian, true};
}

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
      distortionParams_(params),
      maxRSquared_(std::numeric_limits<T>::max()) {
  // Initialize maxRSquared_ from the image bounds; callers may override via setMaxValidAngle().
  computeMaxValidAngleFromImageBounds();
}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> OpenCVFisheyeIntrinsicsModelT<T>::project(
    const Vector3P<T>& point) const {
  // Per-lane scalar loop because drjit::atan is not available; cannot vectorize the angular step.
  Vector3P<T> result;

  typename Packet<T>::MaskType validMask;
  for (size_t i = 0; i < Packet<T>::Size; ++i) {
    validMask[i] = true;
  }

  for (size_t i = 0; i < Packet<T>::Size; ++i) {
    const T x = point.x()[i];
    const T y = point.y()[i];
    const T z = point.z()[i];

    if (z <= T(0)) {
      result.x()[i] = T(0);
      result.y()[i] = T(0);
      result.z()[i] = z;
      validMask[i] = false;
      continue;
    }

    const T invZ = T(1) / z;
    const T a = x * invZ;
    const T b = y * invZ;

    const T rsqr = a * a + b * b;

    // Reject points outside the calibrated angular FOV (configured via setMaxValidAngle()).
    if (rsqr > maxRSquared_) {
      result.x()[i] = T(0);
      result.y()[i] = T(0);
      result.z()[i] = z;
      validMask[i] = false;
      continue;
    }

    const T r = std::sqrt(rsqr);
    const T theta = std::atan(r);

    const auto& dp = distortionParams_;

    // Equidistant fisheye distortion polynomial:
    //   thetaD = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;
    const T thetaD =
        theta * (T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8);

    // Avoid division by zero at the optical axis; thetaD/r -> 1 as r -> 0.
    const T scale = (r > T(1e-8)) ? thetaD / r : T(1);

    const T xpp = scale * a;
    const T ypp = scale * b;

    result.x()[i] = fx_ * xpp + cx_;
    result.y()[i] = fy_ * ypp + cy_;
    result.z()[i] = z;
  }

  return {result, validMask};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> OpenCVFisheyeIntrinsicsModelT<T>::project(
    const Eigen::Vector3<T>& point) const {
  const T z = point.z();

  if (z <= T(0)) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  T invZ = T(1) / z;
  T a = point.x() * invZ;
  T b = point.y() * invZ;

  T rsqr = a * a + b * b;

  // Reject points outside the calibrated angular FOV.
  if (rsqr > maxRSquared_) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  T r = std::sqrt(rsqr);
  T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
  T theta2 = theta * theta;
  T theta4 = theta2 * theta2;
  T theta6 = theta4 * theta2;
  T theta8 = theta4 * theta4;
  T thetaD = theta * (T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8);

  // Avoid division by zero at the optical axis; thetaD/r -> 1 as r -> 0.
  T scale = (r > T(1e-8)) ? thetaD / r : T(1);

  T xpp = scale * a;
  T ypp = scale * b;

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

  const T a = x * z_inv;
  const T b = y * z_inv;
  const T rsqr = a * a + b * b;

  if (rsqr > maxRSquared_) {
    return {Eigen::Vector3<T>::Zero(), Eigen::Matrix<T, 3, 3>::Zero(), false};
  }

  const T r = std::sqrt(rsqr);
  const T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // poly = 1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8;  thetaD = theta * poly.
  const T theta2 = theta * theta;
  const T theta4 = theta2 * theta2;
  const T theta6 = theta4 * theta2;
  const T theta8 = theta4 * theta4;
  const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
  const T thetaD = theta * poly;

  const bool nearCenter = r < T(1e-8);
  const T scale = nearCenter ? T(1) : thetaD / r;

  const T xpp = scale * a;
  const T ypp = scale * b;

  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  Eigen::Matrix<T, 3, 3> jacobian = Eigen::Matrix<T, 3, 3>::Zero();

  if (nearCenter) {
    // At the optical axis the fisheye reduces to the pinhole limit (scale -> 1, no cross-term).
    jacobian(0, 0) = fx_ * z_inv;
    jacobian(0, 2) = -fx_ * x * z_inv_sq;
    jacobian(1, 1) = fy_ * z_inv;
    jacobian(1, 2) = -fy_ * y * z_inv_sq;
    jacobian(2, 2) = T(1);
    return {projectedPoint, jacobian, true};
  }

  // Build d(scale)/dr via the chain rule:
  //   d(thetaD)/d(theta) = poly + theta * d(poly)/d(theta)
  //   d(theta)/dr        = 1 / (1 + r^2)        (since theta = atan(r))
  //   d(scale)/dr        = (d(thetaD)/dr * r - thetaD) / r^2     (quotient rule on thetaD/r)
  const T dpoly_dtheta = T(2) * dp.k1 * theta + T(4) * dp.k2 * theta * theta2 +
      T(6) * dp.k3 * theta * theta4 + T(8) * dp.k4 * theta * theta6;
  const T dthetaD_dtheta = poly + theta * dpoly_dtheta;
  const T dtheta_dr = T(1) / (T(1) + rsqr);
  const T r_inv = T(1) / r;
  const T dscale_dr = (dthetaD_dtheta * dtheta_dr * r - thetaD) * r_inv * r_inv;

  // r = sqrt(a^2 + b^2) => dr/da = a/r, dr/db = b/r.
  const T dr_da = a * r_inv;
  const T dr_db = b * r_inv;

  // (xpp, ypp) = scale * (a, b); product rule with d(scale)/d{a,b} = d(scale)/dr * d(r)/d{a,b}.
  const T dxpp_da = dscale_dr * dr_da * a + scale;
  const T dxpp_db = dscale_dr * dr_db * a;
  const T dypp_da = dscale_dr * dr_da * b;
  const T dypp_db = dscale_dr * dr_db * b + scale;

  // Compose with d(a, b)/d(x, y, z) where a = x/z, b = y/z:
  //   da/dx = 1/z, da/dy = 0, da/dz = -x/z^2; analogously for b.
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

  // Use the half-pixel-offset convention; distortion parameters are unitless and stay unchanged.
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
  // Clamp the crop region to stay inside the original image bounds.
  int32_t width = newWidth;
  if (left + width > this->imageWidth()) {
    width = this->imageWidth() - left;
  }

  int32_t height = newHeight;
  if (top + height > this->imageHeight()) {
    height = this->imageHeight() - top;
  }

  // Shift the principal point so it stays at the same physical location after cropping.
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

  // Strip intrinsics to recover normalized distorted coordinates (xpp, ypp); their length is
  // thetaD.
  const T xpp = (u - cx_) / fx_;
  const T ypp = (v - cy_) / fy_;

  const T rDistorted = std::sqrt(xpp * xpp + ypp * ypp);

  // The optical-axis ray maps to the optical center; bypass Newton to avoid 0/0.
  if (rDistorted < T(1e-8)) {
    return {Eigen::Vector3<T>(T(0), T(0), depth), true};
  }

  // Solve for theta in f(theta) = theta * poly(theta) - thetaD = 0 by Newton iteration.
  // Initial guess theta == rDistorted is exact when distortion is zero.
  // TODO: this loop silently returns the latest theta even when Newton fails to converge;
  // consider returning isValid=false on non-convergence to match OpenCV's project() behavior.
  const auto& dp = distortionParams_;
  T theta = rDistorted;

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

    // f'(theta) = poly + theta * d(poly)/d(theta).
    const T dpoly_dtheta = T(2) * dp.k1 * theta + T(4) * dp.k2 * theta * theta2 +
        T(6) * dp.k3 * theta * theta4 + T(8) * dp.k4 * theta * theta6;
    const T df = poly + theta * dpoly_dtheta;

    if (std::abs(df) < T(1e-12)) {
      break;
    }

    theta = theta - f / df;
  }

  // Equidistant model: project(theta) = scale * (a, b) with scale = thetaD/r and r = tan(theta).
  // Inverting that scale yields unscale = r/thetaD, applied to the distorted coords.
  const T r = std::tan(theta);
  const T unscale = (rDistorted > T(1e-8)) ? r / rDistorted : T(1);

  const T a = xpp * unscale;
  const T b = ypp * unscale;

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
  // Recompute max valid angle since distortion parameters changed
  computeMaxValidAngleFromImageBounds();
}

template <typename T>
std::shared_ptr<IntrinsicsModelT<T>> OpenCVFisheyeIntrinsicsModelT<T>::clone() const {
  auto result = std::make_shared<OpenCVFisheyeIntrinsicsModelT<T>>(
      this->imageWidth(), this->imageHeight(), fx_, fy_, cx_, cy_, distortionParams_);
  result->setName(this->name());
  return result;
}

template <typename T>
std::vector<std::string> OpenCVFisheyeIntrinsicsModelT<T>::getParameterNames() const {
  return {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"};
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 3, Eigen::Dynamic>, bool>
OpenCVFisheyeIntrinsicsModelT<T>::projectIntrinsicsJacobian(const Eigen::Vector3<T>& point) const {
  const T x = point(0);
  const T y = point(1);
  const T z = point(2);

  if (z <= T(0)) {
    Eigen::Matrix<T, 3, 8> zeroJacobian = Eigen::Matrix<T, 3, 8>::Zero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T z_inv = T(1) / z;

  const T a = x * z_inv;
  const T b = y * z_inv;
  const T rsqr = a * a + b * b;

  if (rsqr > maxRSquared_) {
    Eigen::Matrix<T, 3, 8> zeroJacobian = Eigen::Matrix<T, 3, 8>::Zero();
    return {Eigen::Vector3<T>::Zero(), zeroJacobian, false};
  }

  const T r = std::sqrt(rsqr);
  const T theta = std::atan(r);

  const auto& dp = distortionParams_;

  // poly = 1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8;  thetaD = theta * poly.
  const T theta2 = theta * theta;
  const T theta4 = theta2 * theta2;
  const T theta6 = theta4 * theta2;
  const T theta8 = theta4 * theta4;
  const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
  const T thetaD = theta * poly;

  const bool nearCenter = r < T(1e-8);
  const T scale = nearCenter ? T(1) : thetaD / r;

  const T xpp = scale * a;
  const T ypp = scale * b;

  const T u = fx_ * xpp + cx_;
  const T v = fy_ * ypp + cy_;
  Eigen::Vector3<T> projectedPoint(u, v, z);

  // Jacobian columns ordered as [fx, fy, cx, cy, k1, k2, k3, k4], matching getParameterNames().
  Eigen::Matrix<T, 3, 8> jacobian = Eigen::Matrix<T, 3, 8>::Zero();

  jacobian(0, 0) = xpp;
  jacobian(0, 2) = T(1);
  jacobian(1, 1) = ypp;
  jacobian(1, 3) = T(1);

  if (!nearCenter) {
    // Distortion-parameter derivatives. From thetaD = theta * poly we get d(thetaD)/dk_i
    // = theta^(2i+1), and scale = thetaD / r so d(scale)/dk_i = d(thetaD)/dk_i / r. Then
    // d(xpp)/dk_i = a * d(scale)/dk_i, d(ypp)/dk_i = b * d(scale)/dk_i.
    const T r_inv = T(1) / r;
    const T dscale_dk1 = theta * theta2 * r_inv;
    const T dscale_dk2 = theta * theta4 * r_inv;
    const T dscale_dk3 = theta * theta6 * r_inv;
    const T dscale_dk4 = theta * theta8 * r_inv;

    jacobian(0, 4) = fx_ * dscale_dk1 * a;
    jacobian(1, 4) = fy_ * dscale_dk1 * b;
    jacobian(0, 5) = fx_ * dscale_dk2 * a;
    jacobian(1, 5) = fy_ * dscale_dk2 * b;
    jacobian(0, 6) = fx_ * dscale_dk3 * a;
    jacobian(1, 6) = fy_ * dscale_dk3 * b;
    jacobian(0, 7) = fx_ * dscale_dk4 * a;
    jacobian(1, 7) = fy_ * dscale_dk4 * b;
  }
  // TODO: when nearCenter, distortion-parameter columns are left at zero. That matches the
  // limit (a, b -> 0 makes xpp, ypp insensitive to k_i to first order), but it differs subtly
  // from the projectJacobian() near-center branch which just falls back to pinhole. Worth a
  // unit test or comment update if the limit matters for solver behavior.

  return {projectedPoint, jacobian, true};
}

template <typename T>
T OpenCVFisheyeIntrinsicsModelT<T>::maxValidAngle() const {
  return std::atan(std::sqrt(maxRSquared_));
}

template <typename T>
void OpenCVFisheyeIntrinsicsModelT<T>::setMaxValidAngle(T angle) {
  const T tanAngle = std::tan(angle);
  maxRSquared_ = tanAngle * tanAngle;
}

template <typename T>
T OpenCVFisheyeIntrinsicsModelT<T>::maxRSquared() const {
  return maxRSquared_;
}

template <typename T>
void OpenCVFisheyeIntrinsicsModelT<T>::setMaxRSquared(T rsqr) {
  maxRSquared_ = rsqr;
}

template <typename T>
void OpenCVFisheyeIntrinsicsModelT<T>::computeMaxValidAngleFromImageBounds() {
  // The maximum theta the camera should accept is the angle whose distorted projection lands at
  // the farthest image corner. Find that pixel distance from the principal point first.
  const std::array<std::array<T, 2>, 4> corners = {
      {{T(0), T(0)},
       {T(this->imageWidth()), T(0)},
       {T(0), T(this->imageHeight())},
       {T(this->imageWidth()), T(this->imageHeight())}}};

  T maxDistSqr = T(0);
  for (const auto& corner : corners) {
    const T dx = corner[0] - cx_;
    const T dy = corner[1] - cy_;
    const T distSqr = dx * dx + dy * dy;
    maxDistSqr = std::max(maxDistSqr, distSqr);
  }
  const T maxDist = std::sqrt(maxDistSqr);

  // Convert pixel distance to normalized distorted radius. fx_ and fy_ are mixed via their
  // average; this is an approximation that ignores anisotropy and slight off-axis stretch.
  const T avgF = (fx_ + fy_) / T(2);
  if (avgF <= T(0)) {
    maxRSquared_ = std::numeric_limits<T>::max();
    return;
  }
  const T targetThetaD = maxDist / avgF;

  // Newton iteration to invert thetaD = theta * (1 + k1*theta^2 + ... + k4*theta^8) for theta.
  const auto& dp = distortionParams_;
  T theta = targetThetaD; // Exact when distortion coefficients are zero.

  constexpr int maxIter = 20;
  constexpr T tol = T(1e-10);

  for (int iter = 0; iter < maxIter; ++iter) {
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;

    const T poly = T(1) + dp.k1 * theta2 + dp.k2 * theta4 + dp.k3 * theta6 + dp.k4 * theta8;
    const T thetaD = theta * poly;

    // d(thetaD)/d(theta) = poly + theta * d(poly)/d(theta), where
    // d(poly)/d(theta) = 2*k1*theta + 4*k2*theta^3 + 6*k3*theta^5 + 8*k4*theta^7.
    const T dPolyDTheta = T(2) * dp.k1 * theta + T(4) * dp.k2 * theta2 * theta +
        T(6) * dp.k3 * theta4 * theta + T(8) * dp.k4 * theta6 * theta;
    const T dThetaDDTheta = poly + theta * dPolyDTheta;

    if (std::abs(dThetaDDTheta) < T(1e-12)) {
      break;
    }

    const T error = thetaD - targetThetaD;
    const T delta = error / dThetaDDTheta;
    theta -= delta;

    if (std::abs(delta) < tol) {
      break;
    }
  }

  // Clamp to [0, ~pi); tan() blows up beyond this and the equidistant model is undefined.
  theta = std::max(T(0), std::min(theta, T(3.1)));

  // Cache the squared tangent so the validity check in project() can stay multiplication-only.
  const T tanTheta = std::tan(theta);
  maxRSquared_ = tanTheta * tanTheta;
}

template <typename T>
CameraT<T> CameraT<T>::lookAt(
    const Eigen::Vector3<T>& position,
    const Eigen::Vector3<T>& target,
    const Eigen::Vector3<T>& up) const {
  const Eigen::Vector3<T> diff = target - position;
  if (diff.norm() == T(0)) {
    // Target coincides with position; the look direction is undefined, so leave the camera as-is.
    return *this;
  }

  Eigen::Transform<T, 3, Eigen::Affine> eyeToWorldMat =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  eyeToWorldMat.translation() = position;

  Eigen::Vector3<T> zVec = diff.normalized();
  // Image y points down (pixel (0,0) is in the top-left), so flip the world up vector when
  // building the camera basis.
  Eigen::Vector3<T> xVec = diff.cross(-up.normalized());
  if (xVec.norm() == T(0)) {
    // Up is parallel to the look direction, so any rotation about z is valid; pick the one that
    // aligns +Z with the look direction.
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

  // Sanity check: a proper rotation has determinant 1; bail out if numerics produced something
  // degenerate.
  if (eyeToWorldMat.linear().determinant() < T(0.9)) {
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

  // Use the geometric image center, ignoring any principal-point offset, for framing math.
  const auto cx = w / T(2);
  const auto cy = h / T(2);

  Eigen::AlignedBox<T, 3> bbox_eye;
  for (const auto& p_world : points) {
    bbox_eye.extend(eyeFromWorld_ * p_world);
  }

  // Step 1: re-center the camera laterally on the bounding-box midpoint and place its near plane
  // at the closest point. After this transform the points are roughly centered in eye space.
  CameraT<T> camera_recentered = *this;
  Eigen::Transform<T, 3, Eigen::Affine> newTransform =
      Eigen::Translation<T, 3>(
          -bbox_eye.center().x(), -bbox_eye.center().y(), -bbox_eye.min().z()) *
      eyeFromWorld_;
  camera_recentered.setEyeFromWorld(newTransform);

  // Step 2: dolly the camera back so all points fit within the image rect with edge padding.
  // The half-width of the usable rect (after padding) constrains how much depth each point needs.
  const T max_x_pixel_diff = (T(1) - T(2) * edgePadding) * std::max(cx, T(w - 1) - cx);
  const T max_y_pixel_diff = (T(1) - T(2) * edgePadding) * std::max(cy, T(h - 1) - cy);

  T max_dz = std::numeric_limits<T>::lowest();
  for (const auto& p_world : points) {
    const Eigen::Vector3<T> p_eye = newTransform * p_world;

    // Each point imposes a min-z constraint (clip plane) and two FOV constraints (one per axis).
    if (p_eye.z() < minZ) {
      max_dz = std::max(max_dz, minZ - p_eye.z());
    }
    max_dz = std::max(max_dz, (fx * std::abs(p_eye.x())) / max_x_pixel_diff - p_eye.z());
    max_dz = std::max(max_dz, (fy * std::abs(p_eye.y())) / max_y_pixel_diff - p_eye.z());
  }

  if (max_dz == std::numeric_limits<T>::lowest()) {
    return camera_recentered;
  }

  CameraT<T> camera_final = camera_recentered;
  camera_final.setEyeFromWorld(
      Eigen::Translation<T, 3>(T(0), T(0), max_dz) * camera_recentered.eyeFromWorld());

  return camera_final;
}

template <typename T>
Vector3P<T> CameraT<T>::transformWorldToEye(const Vector3P<T>& worldPoints) const {
  // Hand-unrolled R * p + t to keep the multiplications inside the SIMD lane width; Eigen's
  // operator* doesn't compose with Vector3P directly.
  const auto& R = eyeFromWorld_.linear();
  const auto& t = eyeFromWorld_.translation();

  Vector3P<T> eyePoints;
  eyePoints.x() = R(0, 0) * worldPoints.x() + R(0, 1) * worldPoints.y() + R(0, 2) * worldPoints.z();
  eyePoints.y() = R(1, 0) * worldPoints.x() + R(1, 1) * worldPoints.y() + R(1, 2) * worldPoints.z();
  eyePoints.z() = R(2, 0) * worldPoints.x() + R(2, 1) * worldPoints.y() + R(2, 2) * worldPoints.z();

  eyePoints.x() += t(0);
  eyePoints.y() += t(1);
  eyePoints.z() += t(2);

  return eyePoints;
}

template <typename T>
Eigen::Vector3<T> CameraT<T>::transformWorldToEye(const Eigen::Vector3<T>& worldPoint) const {
  return eyeFromWorld_ * worldPoint;
}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType> CameraT<T>::project(
    const Vector3P<T>& worldPoints) const {
  const Vector3P<T> eyePoints = transformWorldToEye(worldPoints);
  return intrinsicsModel_->project(eyePoints);
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool> CameraT<T>::project(const Eigen::Vector3<T>& worldPoint) const {
  const Eigen::Vector3<T> eyePoint = transformWorldToEye(worldPoint);
  return intrinsicsModel_->project(eyePoint);
}

template <typename T>
std::tuple<Eigen::Vector3<T>, Eigen::Matrix<T, 2, 3>, bool> CameraT<T>::projectJacobian(
    const Eigen::Vector3<T>& worldPoint) const {
  const Eigen::Vector3<T> eyePoint = transformWorldToEye(worldPoint);

  const auto [projectedPoint, jacobian_eye, isValid] = intrinsicsModel_->projectJacobian(eyePoint);

  if (!isValid) {
    return {projectedPoint, Eigen::Matrix<T, 2, 3>::Zero(), false};
  }

  // The intrinsics Jacobian is 3x3 with a depth pass-through row; only the [du,dv]/d(eye) rows
  // matter for projecting back to image space.
  const Eigen::Matrix<T, 2, 3> J_proj_eye = jacobian_eye.template topRows<2>();

  // Chain rule: d(image)/d(world) = d(image)/d(eye) * d(eye)/d(world). The world->eye Jacobian
  // is just the rotation part of eyeFromWorld_ (translation drops out under differentiation).
  const Eigen::Matrix<T, 3, 3> R_eye_world = eyeFromWorld_.linear();
  const Eigen::Matrix<T, 2, 3> J_proj_world = J_proj_eye * R_eye_world;

  return {projectedPoint, J_proj_world, true};
}

template <typename T>
std::pair<Vector3P<T>, typename Packet<T>::MaskType>
CameraT<T>::unproject(const Vector3P<T>& imagePoints, int maxIterations, T tolerance) const {
  using PacketT = Packet<T>;

  // Per-lane scalar fallback: the intrinsics models' unproject() iterates with Newton, which
  // does not vectorize cleanly across packet lanes (different lanes converge in different steps).
  Vector3P<T> worldPoints;
  auto validMask = drjit::full<typename PacketT::MaskType>(true);

  for (size_t i = 0; i < PacketT::Size; ++i) {
    Eigen::Vector3<T> imagePoint(imagePoints.x()[i], imagePoints.y()[i], imagePoints.z()[i]);

    auto [eyePoint, isValid] = intrinsicsModel_->unproject(imagePoint, maxIterations, tolerance);

    if (!isValid) {
      validMask[i] = false;
      worldPoints.x()[i] = T(0);
      worldPoints.y()[i] = T(0);
      worldPoints.z()[i] = T(0);
      continue;
    }

    const Eigen::Vector3<T> worldPoint = worldFromEye() * eyePoint;

    worldPoints.x()[i] = worldPoint(0);
    worldPoints.y()[i] = worldPoint(1);
    worldPoints.z()[i] = worldPoint(2);
  }

  return {worldPoints, validMask};
}

template <typename T>
std::pair<Eigen::Vector3<T>, bool>
CameraT<T>::unproject(const Eigen::Vector3<T>& imagePoint, int maxIterations, T tolerance) const {
  auto [eyePoint, isValid] = intrinsicsModel_->unproject(imagePoint, maxIterations, tolerance);

  if (!isValid) {
    return {Eigen::Vector3<T>::Zero(), false};
  }

  const Eigen::Vector3<T> worldPoint = worldFromEye() * eyePoint;

  return {worldPoint, true};
}

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
