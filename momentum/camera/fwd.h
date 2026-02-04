/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/simd/simd.h>

#include <drjit/fwd.h>
#include <drjit/matrix.h>

namespace momentum {

// Forward declarations
template <typename T>
class CameraT;

template <typename T>
class IntrinsicsModelT;

template <typename T>
class PinholeIntrinsicsModelT;

template <typename T>
class OpenCVIntrinsicsModelT;

template <typename T>
struct OpenCVDistortionParametersT;

using Camera = CameraT<float>;
using Camerad = CameraT<double>;
using IntrinsicsModel = IntrinsicsModelT<float>;
using IntrinsicsModeld = IntrinsicsModelT<double>;
using PinholeIntrinsicsModel = PinholeIntrinsicsModelT<float>;
using PinholeIntrinsicsModeld = PinholeIntrinsicsModelT<double>;
using OpenCVIntrinsicsModel = OpenCVIntrinsicsModelT<float>;
using OpenCVIntrinsicsModeld = OpenCVIntrinsicsModelT<double>;
using OpenCVDistortionParameters = OpenCVDistortionParametersT<float>;
using OpenCVDistortionParametersd = OpenCVDistortionParametersT<double>;

// Additional SIMD types used by camera that are not in simd/simd.h
using ByteP = Packet<uint8_t>;
using UintP = Packet<uint32_t>;

using Vector2iP = drjit::Array<IntP, 2>;
using Vector3bP = drjit::Array<ByteP, 3>;
using Vector3iP = drjit::Array<IntP, 3>;
using Vector4fP = drjit::Array<FloatP, 4>;
using Vector4dP = drjit::Array<DoubleP, 4>;
using Matrix3fP = drjit::Matrix<FloatP, 3>;
using Matrix3dP = drjit::Matrix<DoubleP, 3>;

/// Template metafunction to map scalar types to their packet equivalents.
template <typename T>
struct PacketType;

template <>
struct PacketType<float> {
  using type = FloatP;
};

template <>
struct PacketType<double> {
  using type = DoubleP;
};

template <>
struct PacketType<int32_t> {
  using type = IntP;
};

template <>
struct PacketType<uint32_t> {
  using type = UintP;
};

template <>
struct PacketType<uint8_t> {
  using type = ByteP;
};

template <typename T>
using PacketType_t = typename PacketType<T>::type;

/// Templatized SIMD vector types.
template <typename T>
using Vector2xP = drjit::Array<PacketType_t<T>, 2>;

template <typename T>
using Vector3xP = drjit::Array<PacketType_t<T>, 3>;

template <typename T>
using Vector4xP = drjit::Array<PacketType_t<T>, 4>;

/// Templatized SIMD matrix types.
template <typename T>
using Matrix3xP = drjit::Matrix<PacketType_t<T>, 3>;

template <typename T>
using Matrix4xP = drjit::Matrix<PacketType_t<T>, 4>;

} // namespace momentum
