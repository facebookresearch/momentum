/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Internal header for shared rasterizer implementation utilities.
// Not intended for external use.

#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/utility.h>
#include <Eigen/Core>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

namespace momentum::rasterizer {

/// Returns the data pointer from a span, or nullptr if the span is empty.
template <typename T, typename Extents, typename Layout, typename Accessor>
inline T* dataOrNull(Kokkos::mdspan<T, Extents, Layout, Accessor> buffer) {
  return buffer.empty() ? nullptr : buffer.data_handle();
}

/// Maps a span of Eigen vectors to a flat Eigen::Map for interop with VectorXf APIs.
template <typename T, int N>
inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> mapVector(
    std::span<const Eigen::Matrix<T, N, 1>> vec) {
  if (vec.empty()) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(nullptr, 0);
  } else {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        vec.front().data(), N * vec.size());
  }
}

/// Validates buffer layout and alignment for all rasterizer buffers.
/// All non-empty buffers must be properly aligned and have valid strides for SIMD operations.
inline void validateBufferLayouts(
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer) {
  validateRasterizerBuffer(zBuffer, "Z buffer");

  if (!rgbBuffer.empty()) {
    validateRasterizerBuffer(rgbBuffer, "RGB buffer");
  }

  if (!surfaceNormalsBuffer.empty()) {
    validateRasterizerBuffer(surfaceNormalsBuffer, "Surface normals buffer");
  }

  if (!vertexIndexBuffer.empty()) {
    validateRasterizerBuffer(vertexIndexBuffer, "Vertex index buffer");
  }

  if (!triangleIndexBuffer.empty()) {
    validateRasterizerBuffer(triangleIndexBuffer, "Triangle index buffer");
  }
}

/// Validates that buffer dimensions match the camera and each other.
/// Z buffer dimensions must match camera image size, and all other buffers must match Z buffer.
inline void validateBufferDimensions(
    const SimdCamera& camera,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer) {
  // Validate that Z buffer dimensions match camera image size
  if (zBuffer.extent(0) != camera.imageHeight()) {
    throw std::runtime_error(
        "Invalid z buffer height " + std::to_string(zBuffer.extent(0)) + "; expected " +
        std::to_string(camera.imageHeight()));
  }

  // Check that Z buffer width is at least as wide as camera image width
  if (zBuffer.extent(1) < camera.imageWidth()) {
    throw std::runtime_error(
        "Z buffer width " + std::to_string(zBuffer.extent(1)) +
        " is less than camera image width " + std::to_string(camera.imageWidth()));
  }

  // For multi-dimensional buffers, validate that all buffers have the same extents as Z buffer
  if (!rgbBuffer.empty() &&
      (rgbBuffer.extent(0) != zBuffer.extent(0) || rgbBuffer.extent(1) != zBuffer.extent(1) ||
       rgbBuffer.extent(2) != 3)) {
    throw std::runtime_error(
        "Invalid RGB buffer " + formatTensorSizes(rgbBuffer.extents()) + "; expected " +
        formatTensorSizes(Extents<3>{zBuffer.extent(0), zBuffer.extent(1), 3}));
  }

  if (!surfaceNormalsBuffer.empty() &&
      (surfaceNormalsBuffer.extent(0) != zBuffer.extent(0) ||
       surfaceNormalsBuffer.extent(1) != zBuffer.extent(1) ||
       surfaceNormalsBuffer.extent(2) != 3)) {
    throw std::runtime_error(
        "Invalid surface normals buffer " + formatTensorSizes(surfaceNormalsBuffer.extents()) +
        "; expected " + formatTensorSizes(Extents<3>{zBuffer.extent(0), zBuffer.extent(1), 3}));
  }

  if (!vertexIndexBuffer.empty() &&
      (vertexIndexBuffer.extent(0) != zBuffer.extent(0) ||
       vertexIndexBuffer.extent(1) != zBuffer.extent(1))) {
    throw std::runtime_error(
        "Invalid vertex index buffer " + formatTensorSizes(vertexIndexBuffer.extents()) +
        "; expected " + formatTensorSizes(Extents<2>{zBuffer.extent(0), zBuffer.extent(1)}));
  }

  if (!triangleIndexBuffer.empty() &&
      (triangleIndexBuffer.extent(0) != zBuffer.extent(0) ||
       triangleIndexBuffer.extent(1) != zBuffer.extent(1))) {
    throw std::runtime_error(
        "Invalid triangle index buffer " + formatTensorSizes(triangleIndexBuffer.extents()) +
        "; expected " + formatTensorSizes(Extents<2>{zBuffer.extent(0), zBuffer.extent(1)}));
  }
}

/// Validates that all non-empty buffers have matching row strides.
/// This is required because the rasterizer uses a single stride value for all buffer accesses.
inline void validateBufferStrides(
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer) {
  const index_t zBufferRowStride = getRowStride(zBuffer);

  if (!rgbBuffer.empty()) {
    // For 3D buffers, the row stride in elements accounts for all channels
    // So rgbBuffer.stride(0) should equal zBufferRowStride * 3
    const index_t rgbBufferRowStride = rgbBuffer.stride(0) / 3;
    if (rgbBufferRowStride != zBufferRowStride) {
      throw std::runtime_error(
          "RGB buffer row stride (" + std::to_string(rgbBufferRowStride) +
          ") does not match Z buffer row stride (" + std::to_string(zBufferRowStride) + ")");
    }
  }

  if (!surfaceNormalsBuffer.empty()) {
    const index_t normalsBufferRowStride = surfaceNormalsBuffer.stride(0) / 3;
    if (normalsBufferRowStride != zBufferRowStride) {
      throw std::runtime_error(
          "Surface normals buffer row stride (" + std::to_string(normalsBufferRowStride) +
          ") does not match Z buffer row stride (" + std::to_string(zBufferRowStride) + ")");
    }
  }

  if (!vertexIndexBuffer.empty()) {
    const index_t vertexIndexBufferRowStride = getRowStride(vertexIndexBuffer);
    if (vertexIndexBufferRowStride != zBufferRowStride) {
      throw std::runtime_error(
          "Vertex index buffer row stride (" + std::to_string(vertexIndexBufferRowStride) +
          ") does not match Z buffer row stride (" + std::to_string(zBufferRowStride) + ")");
    }
  }

  if (!triangleIndexBuffer.empty()) {
    const index_t triangleIndexBufferRowStride = getRowStride(triangleIndexBuffer);
    if (triangleIndexBufferRowStride != zBufferRowStride) {
      throw std::runtime_error(
          "Triangle index buffer row stride (" + std::to_string(triangleIndexBufferRowStride) +
          ") does not match Z buffer row stride (" + std::to_string(zBufferRowStride) + ")");
    }
  }
}

/// Validates all buffer constraints: layout, dimensions, and strides.
inline void checkBuffers(
    const SimdCamera& camera,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer) {
  validateBufferLayouts(
      zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);
  validateBufferDimensions(
      camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);
  validateBufferStrides(
      zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);
}

/// Validates that triangle indices are within bounds and the array size is a multiple of 3.
template <typename TriangleT>
inline void validateTriangleIndices(
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    size_t nVerts,
    const char* vertexType = "vertex",
    const char* triangleType = "triangles") {
  const auto nTri = triangles.size() / 3;
  if (nTri * 3 != triangles.size()) {
    std::ostringstream oss;
    oss << triangleType << " array of size " << triangles.size() << " is not a multiple of three.";
    throw std::runtime_error(oss.str());
  }

  const int maxVertIdx = triangles.maxCoeff();
  if (maxVertIdx >= nVerts) {
    std::ostringstream oss;
    oss << "Invalid " << vertexType << " index " << maxVertIdx << " found in " << triangleType
        << " array.";
    throw std::runtime_error(oss.str());
  }

  const int minVertIdx = triangles.minCoeff();
  if (minVertIdx < 0) {
    std::ostringstream oss;
    oss << "Invalid " << vertexType << " index " << minVertIdx << " found in " << triangleType
        << " array.";
    throw std::runtime_error(oss.str());
  }
}

} // namespace momentum::rasterizer
