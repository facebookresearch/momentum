/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/fwd.h>

#include <Eigen/Core>

#include <utility>

namespace momentum {

/// Constructs a 3x4 projection matrix that linearizes the camera's projection around a target
/// pixel.
///
/// After perspective divide, the residual is in pixel units:
///   homogeneous = M * [p_world; 1]
///   residual = homogeneous.head<2>() / homogeneous.z()
///
/// A world point that projects exactly to @p targetPixel yields zero residual. For nearby points,
/// the residual closely approximates `camera.project(p) - targetPixel`. The key advantage over full
/// nonlinear projection is off-screen robustness: because the distortion model is linearized (but
/// the perspective divide remains exact), the residual stays finite and well-behaved even for
/// points behind the camera or far outside the image bounds.
///
/// The matrix is constructed by evaluating the intrinsics model's projection Jacobian at the
/// eye-space point corresponding to the target pixel (unprojected at z=1), then composing with the
/// camera's eye-from-world transform.
///
/// When using the returned matrix with ProjectionErrorFunctionT, set
/// ProjectionConstraintDataT::target to (0, 0) since the target is already baked into the matrix.
///
/// @param camera The camera (intrinsics + extrinsics) to linearize
/// @param targetPixel The pixel coordinates around which to linearize the projection
/// @return Pair of (3x4 matrix M, valid flag). If valid is false, the target pixel could not be
///   unprojected (e.g. outside the camera's valid region) and the matrix should not be used.
template <typename T>
std::pair<Eigen::Matrix<T, 3, 4>, bool> projectionMatrixPixels(
    const CameraT<T>& camera,
    const Eigen::Vector2<T>& targetPixel);

/// Constructs a 3x4 projection matrix where the residual (after perspective divide) is in radians
/// (tangent-of-angle).
///
/// After perspective divide, x/z and y/z give the tangent of the angle between the point's ray and
/// the target ray. For small angles, tangent ~ angle in radians.
///
/// This variant is constructed by computing a rotation that maps the target pixel's eye-space ray
/// direction to the z-axis, then composing with the camera's eye-from-world transform. No intrinsic
/// parameters appear in the result, so the residual is purely geometric (angular).
///
/// When using the returned matrix with ProjectionErrorFunctionT, set
/// ProjectionConstraintDataT::target to (0, 0) since the target is already baked into the matrix.
///
/// @param camera The camera (intrinsics + extrinsics) to use
/// @param targetPixel The pixel coordinates whose ray defines the zero-residual direction
/// @return Pair of (3x4 matrix M, valid flag). If valid is false, the target pixel could not be
///   unprojected and the matrix should not be used.
template <typename T>
std::pair<Eigen::Matrix<T, 3, 4>, bool> projectionMatrixRadians(
    const CameraT<T>& camera,
    const Eigen::Vector2<T>& targetPixel);

} // namespace momentum
