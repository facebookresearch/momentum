/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/fwd.h>

#include <drjit/array.h>
#include <drjit/matrix.h>

namespace momentum::rasterizer {

// Re-export camera types from momentum namespace for backward compatibility
using momentum::Camera;
using momentum::Camerad;
using momentum::IntrinsicsModel;
using momentum::IntrinsicsModeld;
using momentum::OpenCVDistortionParameters;
using momentum::OpenCVDistortionParametersd;
using momentum::OpenCVIntrinsicsModel;
using momentum::OpenCVIntrinsicsModeld;
using momentum::PinholeIntrinsicsModel;
using momentum::PinholeIntrinsicsModeld;

// Re-export SIMD constants for use in rasterizer
using momentum::kSimdAlignment;
using momentum::kSimdPacketSize;

using momentum::ByteP;
using momentum::DoubleP;
using momentum::FloatP;
using momentum::IntP;
using momentum::UintP;

// DrJit-based scalar vector/matrix types used internally by rasterizer.
// These are distinct from the Eigen-based types in momentum namespace.
using Vector2f = drjit::Array<float, 2>;
using Vector2d = drjit::Array<double, 2>;
using Vector3f = drjit::Array<float, 3>;
using Vector3d = drjit::Array<double, 3>;
using Vector3b = drjit::Array<uint8_t, 3>;
using Matrix3f = drjit::Matrix<float, 3>;
using Matrix3d = drjit::Matrix<double, 3>;
using Matrix4f = drjit::Matrix<float, 4>;
using Matrix4d = drjit::Matrix<double, 4>;

// Re-export SIMD packet types from momentum namespace
using momentum::Matrix3dP;
using momentum::Matrix3fP;
using momentum::Matrix3xP;
using momentum::Matrix4xP;

using momentum::Vector2dP;
using momentum::Vector2fP;
using momentum::Vector2iP;
using momentum::Vector2xP;

using momentum::Vector3bP;
using momentum::Vector3dP;
using momentum::Vector3fP;
using momentum::Vector3iP;
using momentum::Vector3xP;

using momentum::Vector4dP;
using momentum::Vector4fP;
using momentum::Vector4xP;

using momentum::PacketType_t;

// Additional type alias for backward compatibility
using Uint8P = momentum::Packet<uint8_t>;

} // namespace momentum::rasterizer
