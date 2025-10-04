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
#include <mdspan/mdspan.hpp>
#include <momentum/rasterizer/fwd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <memory>
#include <sstream>
#include <utility>

// Forward declarations to avoid circular dependencies
namespace momentum::rasterizer {

using index_t = std::ptrdiff_t;

// Actual implementations

template <std::size_t R>
std::string formatTensorSizes(const std::array<index_t, R>& extents) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < extents.size(); ++i) {
    if (i != 0) {
      oss << " x ";
    }
    oss << extents[i];
  }
  oss << "]";
  return oss.str();
}

// Overload for mdspan extents using parameter pack
template <class IndexType, size_t... Extents>
std::string formatTensorSizes(const Kokkos::extents<IndexType, Extents...>& extents) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < extents.rank(); ++i) {
    if (i != 0) {
      oss << " x ";
    }
    oss << extents.extent(i);
  }
  oss << "]";
  return oss.str();
}

// Validates that an mdspan buffer is suitable for rasterization operations.
// This function checks:
// 1. Data pointer is properly aligned for SIMD operations
// 2. All dimensions except the first are contiguous (rows are packed tightly)
// 3. The stride for the first dimension is a multiple of kSimdPacketSize
template <typename T, typename Extents>
bool isValidBuffer(const Kokkos::mdspan<T, Extents>& buffer) {
  if (buffer.empty()) {
    return true; // Empty buffers are always valid
  }

  constexpr auto rank = Kokkos::mdspan<T, Extents>::rank();

  // Check alignment of data pointer
  if (reinterpret_cast<uintptr_t>(buffer.data_handle()) % kSimdAlignment != 0) {
    return false;
  }

  // For 1D buffers, just check that the stride is compatible with SIMD
  if (rank == 1) {
    return buffer.stride(0) % kSimdPacketSize == 0;
  }

  // For multi-dimensional buffers, check contiguity of all dimensions except the first
  // Start from the rightmost dimension and work backwards
  index_t expected_stride = 1;
  for (size_t dim = rank; dim > 1; --dim) {
    size_t i = dim - 1;
    if (buffer.stride(i) != expected_stride) {
      return false; // Not contiguous in this dimension
    }
    expected_stride *= buffer.extent(i);
  }

  // Check that the first dimension stride is a multiple of kSimdPacketSize
  // This ensures proper SIMD alignment for accessing rows
  return buffer.stride(0) % kSimdPacketSize == 0;
}

// Helper function to get the row stride for accessing buffer rows properly
// This ensures we use the actual mdspan stride instead of assuming contiguity
template <typename T, typename Extents>
index_t getRowStride(const Kokkos::mdspan<T, Extents>& buffer) {
  static_assert(
      Kokkos::mdspan<T, Extents>::rank() >= 2, "getRowStride requires at least 2D buffer");
  return buffer.stride(0);
}

// Checks if an mdspan has a standard row-major contiguous layout
// This is useful for optimization decisions where contiguous access patterns can be used
template <typename T, typename Extents>
bool isContiguous(const Kokkos::mdspan<T, Extents>& buffer) {
  if (buffer.empty()) {
    return true; // Empty buffers are trivially contiguous
  }

  constexpr auto rank = Kokkos::mdspan<T, Extents>::rank();

  // Check contiguity from the rightmost dimension working backwards
  index_t expected_stride = 1;
  for (size_t dim = rank; dim > 0; --dim) {
    size_t i = dim - 1;
    if (buffer.stride(i) != expected_stride) {
      return false; // Not contiguous in this dimension
    }
    expected_stride *= buffer.extent(i);
  }

  return true;
}

inline Vector3f toEnokiVec(const Eigen::Vector3f& v) {
  return {v.x(), v.y(), v.z()};
}

inline Matrix3f toEnokiMat(const Eigen::Matrix3f& m) {
  return Matrix3f(m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2), m(2, 0), m(2, 1), m(2, 2));
}

inline auto extractSingleElement(const Matrix3dP& mat, int index) {
  return Matrix3d{
      mat(0, 0)[index],
      mat(0, 1)[index],
      mat(0, 2)[index],
      mat(1, 0)[index],
      mat(1, 1)[index],
      mat(1, 2)[index],
      mat(2, 0)[index],
      mat(2, 1)[index],
      mat(2, 2)[index]};
}
inline auto extractSingleElement(const Matrix3fP& mat, int index) {
  return Matrix3f{
      mat(0, 0)[index],
      mat(0, 1)[index],
      mat(0, 2)[index],
      mat(1, 0)[index],
      mat(1, 1)[index],
      mat(1, 2)[index],
      mat(2, 0)[index],
      mat(2, 1)[index],
      mat(2, 2)[index]};
}

inline auto extractSingleElement(const Vector3dP& vec, int index) {
  return Vector3d{vec.x()[index], vec.y()[index], vec.z()[index]};
}

inline auto extractSingleElement(const Vector3fP& vec, int index) {
  return Vector3f{vec.x()[index], vec.y()[index], vec.z()[index]};
}

inline auto extractSingleElement(const Vector2fP& vec, int index) {
  return Vector2f(vec.x()[index], vec.y()[index]);
}

inline auto extractSingleElement(const FloatP& vec, int index) {
  return vec[index];
}

class SimdCamera {
 public:
  // Camera that operates on drjit::Vector3fP.
  SimdCamera(const Camera& camera, Eigen::Matrix4f modelMatrix, Eigen::Vector2f imageOffset)
      : _modelMatrix(modelMatrix),
        _imageOffset(std::move(imageOffset)),
        _intrinsics(camera.intrinsicsModel()),
        _modelToWorld_rotation(toEnokiMat(modelMatrix.topLeftCorner<3, 3>())),
        _modelToWorld_translation(toEnokiVec(modelMatrix.block<3, 1>(0, 3))),
        _modelToWorld_row3(toEnokiVec(modelMatrix.block<1, 3>(3, 0))),
        _modelToWorld_33(modelMatrix(3, 3)),
        _normalMatrix(toEnokiMat(
            (camera.eyeFromWorld() * Eigen::Transform<float, 3, Eigen::Affine>(modelMatrix))
                .linear()
                .inverse()
                .transpose())),
        _worldToEye_translation(toEnokiVec(Eigen::Vector3f(camera.eyeFromWorld().translation()))),
        _worldToEye_rotation(toEnokiMat(camera.eyeFromWorld().linear().matrix())),
        _fx(camera.fx()),
        _fy(camera.fy()),
        _imageWidth(camera.imageWidth()),
        _imageHeight(camera.imageHeight()) {}

  [[nodiscard]] auto fx() const {
    return _fx;
  }

  [[nodiscard]] auto fy() const {
    return _fy;
  }

  [[nodiscard]] auto imageWidth() const {
    return _imageWidth;
  }

  [[nodiscard]] auto imageHeight() const {
    return _imageHeight;
  }

  [[nodiscard]] Vector3fP worldToEye(const Vector3fP& p_world) const {
    // Because we support projection in the model-to-world matrix, we can't just combine it
    // with the world-to-eye matrix, so we'll keep them separate.
    // Model-to-world matrix looks like this:
    //    [ mR   mT ]
    //    [ m3x m33 ]
    // Blockwise multiplication gives:
    //    [ mR   mT ] [ v ] = [ mR * v + mT    ]
    //    [ m3x m33 ] [ 1 ]   [ m3x^T * v + m33]
    const Vector3fP p_world_unnormalized =
        _modelToWorld_rotation * p_world + _modelToWorld_translation;
    const FloatP p_world_w = drjit::dot(_modelToWorld_row3, p_world) + _modelToWorld_33;
    const Vector3fP p_world_normalized = p_world_unnormalized / p_world_w;

    return _worldToEye_rotation * p_world_normalized + _worldToEye_translation;
  }

  [[nodiscard]] Vector3fP worldToEyeNormal(const Vector3fP& n_world) const {
    return drjit::normalize(_normalMatrix * n_world);
  }

  [[nodiscard]] auto eyeToWindow(const Vector3fP& p_eye) const {
    return _intrinsics->project(p_eye);
  }

  [[nodiscard]] auto worldToWindow(const Vector3fP& p_world) const {
    return eyeToWindow(worldToEye(p_world));
  }

 private:
  const Eigen::Matrix4f _modelMatrix;
  const Eigen::Vector2f _imageOffset;
  std::shared_ptr<const IntrinsicsModelT<float>> _intrinsics;

  const Matrix3f _modelToWorld_rotation;
  const Vector3f _modelToWorld_translation;
  const Vector3f _modelToWorld_row3;
  const float _modelToWorld_33;

  const Matrix3f _normalMatrix;

  const Vector3f _worldToEye_translation;
  const Matrix3f _worldToEye_rotation;

  const float _fx;
  const float _fy;

  const int32_t _imageWidth;
  const int32_t _imageHeight;
};

} // namespace momentum::rasterizer
