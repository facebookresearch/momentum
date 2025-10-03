/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <drjit/fwd.h>

namespace momentum::rasterizer {

template <typename T>
class CameraT;

template <typename T>
class IntrinsicsModelT;

using Camera = CameraT<float>;
using IntrinsicsModel = IntrinsicsModelT<float>;

constexpr size_t kSimdPacketSize = drjit::DefaultSize;
constexpr size_t kSimdAlignment = 4 * kSimdPacketSize;

using IntP = drjit::Packet<int32_t, kSimdPacketSize>;
using UintP = drjit::Packet<uint32_t, kSimdPacketSize>;
using Uint8P = drjit::Packet<uint8_t, kSimdPacketSize>;

using FloatP = drjit::Packet<float, kSimdPacketSize>;
using DoubleP = drjit::Packet<double, kSimdPacketSize>;
using ByteP = drjit::Packet<uint8_t, kSimdPacketSize>;
using Vector3f = drjit::Array<float, 3>;
using Vector3d = drjit::Array<double, 3>;
using Vector3b = drjit::Array<uint8_t, 3>;
using Vector2f = drjit::Array<float, 2>;
using Vector2d = drjit::Array<double, 2>;
using Matrix3f = drjit::Matrix<float, 3>;
using Matrix3d = drjit::Matrix<double, 3>;
using Matrix4f = drjit::Matrix<float, 4>;
using Matrix4d = drjit::Matrix<double, 4>;
using Vector2fP = drjit::Array<FloatP, 2>;
using Vector2dP = drjit::Array<DoubleP, 2>;
using Vector2iP = drjit::Array<IntP, 2>;
using Vector3fP = drjit::Array<FloatP, 3>;
using Vector3dP = drjit::Array<DoubleP, 3>;
using Vector3bP = drjit::Array<ByteP, 3>;
using Vector3iP = drjit::Array<IntP, 3>;
using Vector4fP = drjit::Array<FloatP, 4>;
using Vector4dP = drjit::Array<DoubleP, 4>;
using Matrix3fP = drjit::Matrix<FloatP, 3>;
using Matrix3dP = drjit::Matrix<DoubleP, 3>;

// Template metafunction to map scalar types to their packet equivalents
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

// Templatized vector and matrix definitions
template <typename T>
using Vector2xP = drjit::Array<PacketType_t<T>, 2>;

template <typename T>
using Vector3xP = drjit::Array<PacketType_t<T>, 3>;

template <typename T>
using Vector4xP = drjit::Array<PacketType_t<T>, 4>;

template <typename T>
using Matrix3xP = drjit::Matrix<PacketType_t<T>, 3>;

template <typename T>
using Matrix4xP = drjit::Matrix<PacketType_t<T>, 4>;

} // namespace momentum::rasterizer
