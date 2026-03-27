/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/camera/projection_utils.h"

#include "momentum/camera/camera.h"

#include <Eigen/Geometry>

namespace momentum {

template <typename T>
std::pair<Eigen::Matrix<T, 3, 4>, bool> projectionMatrixPixels(
    const CameraT<T>& camera,
    const Eigen::Vector2<T>& targetPixel) {
  // Unproject the target pixel to eye-space at z=1.
  const Eigen::Vector3<T> imagePoint(targetPixel.x(), targetPixel.y(), T(1));
  const auto [eyePoint, valid] = camera.intrinsicsModel()->unproject(imagePoint);
  if (!valid) {
    return {Eigen::Matrix<T, 3, 4>::Zero(), false};
  }

  // Get the projection Jacobian evaluated at the unprojected point.
  // This 3x3 matrix J has the property that for any eye-space point p near eyePoint,
  // (J * p).hnormalized() ~ intrinsics.project(p).head<2>() - targetPixel.
  // At p = eyePoint itself, the residual is exactly zero.
  const auto [projected, jacobian, validJ] = camera.intrinsicsModel()->projectJacobian(eyePoint);

  // Compose with the eye-from-world transform's upper 3x4 block:
  // M = J * eyeFromWorld.topRows<3>()
  const Eigen::Matrix<T, 3, 4> eyeFromWorld34 =
      camera.eyeFromWorld().matrix().template topRows<3>();
  return {jacobian * eyeFromWorld34, true};
}

template <typename T>
std::pair<Eigen::Matrix<T, 3, 4>, bool> projectionMatrixRadians(
    const CameraT<T>& camera,
    const Eigen::Vector2<T>& targetPixel) {
  // Unproject the target pixel to eye-space at z=1 to get the ray direction.
  const Eigen::Vector3<T> imagePoint(targetPixel.x(), targetPixel.y(), T(1));
  const auto [eyePoint, valid] = camera.intrinsicsModel()->unproject(imagePoint);
  if (!valid) {
    return {Eigen::Matrix<T, 3, 4>::Zero(), false};
  }

  // Normalize to get the ray direction in eye-space.
  const Eigen::Vector3<T> rayDir = eyePoint.normalized();

  // Compute a rotation that maps the target ray to the z-axis.
  // After applying this rotation, any point on the target ray will have x=0, y=0,
  // so (x/z, y/z) = (0, 0). For other rays, x/z and y/z give the tangent of the
  // angle from the target ray.
  const Eigen::Quaternion<T> rotation =
      Eigen::Quaternion<T>::FromTwoVectors(rayDir, Eigen::Vector3<T>::UnitZ());
  const Eigen::Matrix<T, 3, 3> R = rotation.toRotationMatrix();

  // Compose with the eye-from-world transform's upper 3x4 block.
  const Eigen::Matrix<T, 3, 4> eyeFromWorld34 =
      camera.eyeFromWorld().matrix().template topRows<3>();
  return {R * eyeFromWorld34, true};
}

// Explicit template instantiations
template std::pair<Eigen::Matrix<float, 3, 4>, bool> projectionMatrixPixels(
    const CameraT<float>& camera,
    const Eigen::Vector2<float>& targetPixel);
template std::pair<Eigen::Matrix<double, 3, 4>, bool> projectionMatrixPixels(
    const CameraT<double>& camera,
    const Eigen::Vector2<double>& targetPixel);

template std::pair<Eigen::Matrix<float, 3, 4>, bool> projectionMatrixRadians(
    const CameraT<float>& camera,
    const Eigen::Vector2<float>& targetPixel);
template std::pair<Eigen::Matrix<double, 3, 4>, bool> projectionMatrixRadians(
    const CameraT<double>& camera,
    const Eigen::Vector2<double>& targetPixel);

} // namespace momentum
