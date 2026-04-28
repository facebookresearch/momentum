/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/exception.h>
#include <momentum/math/transform.h>

#include <drjit/array.h>
#include <drjit/array_router.h>
#include <drjit/matrix.h>
#include <drjit/packet.h>
#include <drjit/util.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

/// Checks if DrJit version is greater than or equal to the specified version.
/// @param major Major version number
/// @param minor Minor version number
/// @param patch Patch version number
/// @return True if current DrJit version >= major.minor.patch
#define DRJIT_VERSION_GE(major, minor, patch)                           \
  ((DRJIT_VERSION_MAJOR > (major)) ||                                   \
   (DRJIT_VERSION_MAJOR == (major) && DRJIT_VERSION_MINOR > (minor)) || \
   (DRJIT_VERSION_MAJOR == (major) && DRJIT_VERSION_MINOR == (minor) && \
    DRJIT_VERSION_PATCH >= (patch)))

/// @file
/// Cross-platform SIMD foundation built on the DrJit library.
///
/// Provides fixed-width SIMD packet types and overloaded operators for
/// interoperating between Eigen vectors/matrices/transforms and DrJit SIMD
/// packets.
///
/// Packet sizes by target instruction set:
/// - x86_64 with AVX2: packets of 8 floats (256-bit)
/// - x86_64 without AVX2 (SSE): packets of 4 floats (128-bit)
/// - ARM (NEON): packets of 4 floats (128-bit)
///
/// Key types:
/// - `FloatP`: a packet of `kSimdPacketSize` floats
/// - `Vector3fP`: a 3-vector whose components are each a `FloatP` (i.e., N
///   3-vectors processed in parallel as SoA)
/// - `kSimdPacketSize`: 4 or 8, platform-dependent (chosen by DrJit)
///
/// Used by the SIMD error functions in `character_solver_simd/` and by the
/// rasterizer.

namespace momentum {

/// Number of floats in the platform's default SIMD register, chosen at compile
/// time by DrJit based on available instruction sets (4 on SSE/NEON, 8 on AVX).
///
/// Used as the inner-loop stride in SIMD error functions.
inline constexpr size_t kSimdPacketSize = drjit::DefaultSize;

/// Byte alignment required for the platform's default SIMD loads/stores.
///
/// `drjit::load<FloatP>(ptr)` requires `ptr` to be aligned to this many bytes;
/// misaligned loads cause segfaults (not exceptions). Use
/// `makeAlignedUnique<float, kSimdAlignment>()` from `common/aligned.h` for all
/// SoA data arrays consumed by SIMD error functions.
inline constexpr size_t kSimdAlignment = kSimdPacketSize * sizeof(float);

/// A fixed-size SIMD packet of kSimdPacketSize elements of type T.
template <typename T>
using Packet = drjit::Packet<T, kSimdPacketSize>;

/// @name Scalar packet aliases
/// @{
using FloatP = Packet<float>;
using DoubleP = Packet<double>;
using IntP = Packet<int>;
/// @}

/// A Dim-dimensional vector where each component is a SIMD packet.
template <typename T, int Dim>
using VectorP = drjit::Array<Packet<T>, Dim>;

template <typename T>
using Vector2P = VectorP<T, 2>;

template <typename T>
using Vector3P = VectorP<T, 3>;

/// @name Float and double 2D/3D vector packet aliases
/// @{
using Vector2fP = Vector2P<float>;
using Vector3fP = Vector3P<float>;

using Vector2dP = Vector2P<double>;
using Vector3dP = Vector3P<double>;
/// @}
using ByteP = Packet<uint8_t>;
using UintP = Packet<uint32_t>;

/// Integer and byte vector types.
using Vector2iP = drjit::Array<IntP, 2>;
using Vector3iP = drjit::Array<IntP, 3>;
using Vector3bP = drjit::Array<ByteP, 3>;

/// 4D vector types.
template <typename T>
using Vector4P = VectorP<T, 4>;

using Vector4fP = Vector4P<float>;
using Vector4dP = Vector4P<double>;

/// Matrix types for SIMD packets.
template <typename T, int Dim>
using MatrixP = drjit::Matrix<Packet<T>, Dim>;

template <typename T>
using Matrix3P = MatrixP<T, 3>;

template <typename T>
using Matrix4P = MatrixP<T, 4>;

using Matrix3fP = Matrix3P<float>;
using Matrix3dP = Matrix3P<double>;
using Matrix4fP = Matrix4P<float>;
using Matrix4dP = Matrix4P<double>;

/// Computes the offset required for the matrix data to meet the alignment requirement.
///
/// Alignment is critical for SIMD correctness: misaligned packet loads segfault
/// rather than throw. Returned value is the number of `float` elements to skip
/// before the first aligned address.
///
/// @param mat The matrix whose alignment offset to compute.
/// @return The number of elements to skip to achieve proper alignment.
template <size_t Alignment>
[[nodiscard]] size_t computeOffset(const Eigen::Ref<Eigen::MatrixXf>& mat) {
  constexpr size_t sizeOfScalar = sizeof(typename Eigen::MatrixXf::Scalar);
  const size_t addressOffset =
      Alignment / sizeOfScalar - (((size_t)mat.data() % Alignment) / sizeOfScalar);

  // If the current alignment already meets the requirement, no offset is needed.
  if (addressOffset == Alignment / sizeOfScalar) {
    return 0;
  }

  return addressOffset;
}

/// Checks if the data of the matrix is aligned correctly.
///
/// @param mat The matrix to check for alignment.
/// @param offset Offset into the matrix data in elements.
/// @throws if `mat.data() + offset` is not aligned to `Alignment` bytes.
template <size_t Alignment>
void checkAlignment(const Eigen::Ref<Eigen::MatrixXf>& mat, size_t offset = 0) {
  MT_THROW_IF(
      (uintptr_t(mat.data() + offset)) % Alignment != 0,
      "Matrix ({}x{}, ptr: {}) is not aligned ({}) correctly.",
      mat.rows(),
      mat.cols(),
      uintptr_t(mat.data()),
      Alignment);
}

/// @name Eigen / DrJit interop operators
/// Mixed-type arithmetic between scalar Eigen types (`Vector3f`, `Matrix3f`,
/// `Quaternionf`, `Transform`) and DrJit packet types (`Vector3fP`).
///
/// The core pattern for SIMD error function evaluation: a single joint
/// transform (scalar Eigen) is applied to N constraint positions (DrJit
/// packet) simultaneously. For example, computing N position residuals:
///
/// @code
/// Vector3f jointPos = state.translation(); // scalar
/// Vector3fP targets = {drjit::load<FloatP>(targetX),
///                      drjit::load<FloatP>(targetY),
///                      drjit::load<FloatP>(targetZ)}; // N packed targets
/// Vector3fP residuals = targets - jointPos; // N residuals at once
/// @endcode
/// @{

/// Computes dot product of Eigen::Vector3f and 3-vector of packets.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3D vector of Eigen.
/// @param v2 A 3-vector of packets.
/// @return Dot product as a packet of floats.
template <typename Derived, typename S>
FloatP dot(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return drjit::fmadd(v1.x(), v2.x(), drjit::fmadd(v1.y(), v2.y(), v1.z() * v2.z()));
}

/// Computes dot product of Vector3P and Eigen::Vector3f.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3-vector of packets.
/// @param v2 A 3D vector of Eigen.
/// @return Dot product as a packet of floats.
template <typename Derived, typename S>
FloatP dot(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return drjit::fmadd(v1.x(), v2.x(), drjit::fmadd(v1.y(), v2.y(), v1.z() * v2.z()));
}

/// Adds Eigen::Vector3f and Vector3P element-wise.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3D vector of Eigen.
/// @param v2 A 3-vector of packets.
/// @return Element-wise sum as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator+(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}

/// Adds Vector3P and Eigen::Vector3f element-wise.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3-vector of packets.
/// @param v2 A 3D vector of Eigen.
/// @return Element-wise sum as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator+(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}

/// Subtracts Vector3P from Eigen::Vector3f element-wise.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3D vector of Eigen.
/// @param v2 A 3-vector of packets.
/// @return Element-wise difference as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator-(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}

/// Subtracts Eigen::Vector3f from Vector3P element-wise.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3-vector of packets.
/// @param v2 A 3D vector of Eigen.
/// @return Element-wise difference as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator-(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}

/// Applies quaternion rotation to each 3x1 vector in packet.
///
/// Delegates to DrJit's quaternion-vector multiplication via ADL.
///
/// @param q A quaternion.
/// @param vec A 3-vector of packets.
/// @return Rotated vectors as a 3-vector of packets.
template <typename S>
Vector3P<S> operator*(const Eigen::Quaternion<S>& q, const Vector3P<S>& vec) {
  // DrJit provides quaternion-packet multiplication via overload resolution
  return q * vec;
}

/// Applies 3x3 matrix transformation to each 3x1 vector in packet.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Matrix3f.
/// @param xf A 3x3 matrix.
/// @param vec A 3-vector of packets.
/// @return Transformed vectors as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator*(const Eigen::MatrixBase<Derived>& xf, const Vector3P<S>& vec) {
  return Vector3P<S>{
      momentum::dot(xf.row(0), vec), momentum::dot(xf.row(1), vec), momentum::dot(xf.row(2), vec)};
}

/// Applies affine transformation to each 3x1 vector in packet.
///
/// @param xf An affine transformation matrix.
/// @param vec A 3-vector of packets.
/// @return Transformed vectors as a 3-vector of packets.
template <typename S>
Vector3P<S> operator*(const Eigen::Transform<S, 3, Eigen::Affine>& xf, const Vector3P<S>& vec) {
  return momentum::operator+(momentum::operator*(xf.linear(), vec), xf.translation());
}

/// Applies momentum::Transform to each 3x1 vector in packet.
///
/// @param xf A momentum transform.
/// @param vec A 3-vector of packets.
/// @return Transformed vectors as a 3-vector of packets.
template <typename S>
Vector3P<S> operator*(const TransformT<S>& xf, const Vector3P<S>& vec) {
  return momentum::operator+(momentum::operator*(xf.toLinear(), vec), xf.translation);
}

/// Computes cross product of Eigen::Vector3f and Vector3P.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3D vector of Eigen.
/// @param v2 A 3-vector of packets.
/// @return Cross product as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> cross(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {
      drjit::fmsub(v1.y(), v2.z(), v1.z() * v2.y()),
      drjit::fmsub(v1.z(), v2.x(), v1.x() * v2.z()),
      drjit::fmsub(v1.x(), v2.y(), v1.y() * v2.x()),
  };
}

/// Computes cross product of Vector3P and Eigen::Vector3f.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param v1 A 3-vector of packets.
/// @param v2 A 3D vector of Eigen.
/// @return Cross product as a 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> cross(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {
      drjit::fmsub(v1.y(), v2.z(), v1.z() * v2.y()),
      drjit::fmsub(v1.z(), v2.x(), v1.x() * v2.z()),
      drjit::fmsub(v1.x(), v2.y(), v1.y() * v2.x()),
  };
}
/// @}

} // namespace momentum
