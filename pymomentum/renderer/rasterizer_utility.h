/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/common/aligned.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/simd/simd.h>

#include <ATen/ATen.h>
#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <tuple>

namespace pymomentum::rasterizer_detail {

namespace py = pybind11;

using momentum::rasterizer::index_t;
template <typename T, size_t R>
using Span = momentum::rasterizer::Span<T, R>;

template <typename T, size_t R>
Span<T, R> make_mdspan(const at::Tensor& t) {
  using ExtentsType = Kokkos::dextents<index_t, R>;
  using MappingType = Kokkos::layout_stride::mapping<ExtentsType>;

  if (t.scalar_type() != toScalarType<T>()) {
    throw std::runtime_error("Scalar type incorrect in mdspan conversion.");
  }

  // Handle tensors with a leading batch dimension of size 1
  at::Tensor tensor = t;
  if (t.dim() == R + 1) {
    if (t.size(0) != 1) {
      throw std::runtime_error(
          fmt::format(
              "Tensor has a batch dimension with size {} (shape={}). "
              "Batch rendering is not supported - batch size must be 1. "
              "Please ensure batch_size=1 before calling renderer functions.",
              t.size(0),
              formatTensorSizes(t)));
    }
    tensor = t.squeeze(0);
  }

  if (tensor.dim() != R) {
    throw std::runtime_error(
        fmt::format(
            "Incorrect dimension in mdspan conversion; expected ndim={} but got {}",
            R,
            formatTensorSizes(t)));
  }

  if (!tensor.device().is_cpu()) {
    throw std::runtime_error("Expected CPU tensor in mdspan conversion");
  }

  std::array<index_t, R> extents{};
  std::array<index_t, R> strides{};
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    extents[i] = tensor.size(i);
    strides[i] = tensor.stride(i);
  }

  return Span<T, R>(tensor.data_ptr<T>(), MappingType(ExtentsType(extents), strides));
}

template <typename T, size_t R>
Span<T, R> make_mdspan(std::optional<at::Tensor> t) {
  if (!t.has_value()) {
    return Span<T, R>();
  }

  return make_mdspan<T, R>(t.value());
}

template <typename T, size_t R>
Span<T, R> make_mdspan(const py::buffer_info& info) {
  using ExtentsType = Kokkos::dextents<index_t, R>;
  using MappingType = Kokkos::layout_stride::mapping<ExtentsType>;

  if (info.format != py::format_descriptor<T>::format()) {
    throw std::runtime_error("Incorrect scalar format in Numpy array.");
  }

  if (info.ndim != R) {
    throw std::runtime_error(
        fmt::format(
            "Incorrect dimension in mdspan conversion; expected ndim={} but got {}",
            R,
            formatTensorSizes(info.shape)));
  }

  std::array<index_t, R> extents{};
  std::array<index_t, R> strides{};
  for (size_t i = 0; i < R; ++i) {
    extents.at(i) = info.shape.at(i);
    // Convert byte strides to element strides
    strides.at(i) = info.strides.at(i) / sizeof(T);
  }

  return Span<T, R>(static_cast<T*>(info.ptr), MappingType(ExtentsType(extents), strides));
}

// make_mdspan from pybind11::buffer (calls request() internally - requires GIL)
template <typename T, size_t R>
Span<T, R> make_mdspan(const pybind11::buffer& buf) {
  auto info = buf.request();
  return make_mdspan<T, R>(info);
}

// make_mdspan from optional pybind11::buffer
template <typename T, size_t R>
Span<T, R> make_mdspan(const std::optional<pybind11::buffer>& buf) {
  if (!buf.has_value()) {
    return Span<T, R>();
  }
  return make_mdspan<T, R>(*buf);
}

// Create a flat Eigen::Map from a contiguous buffer.
// The buffer must be row-major (C-contiguous) for correct element ordering.
// This is the buffer-based equivalent of toEigenMap<T>(at::Tensor).
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> bufferToEigenMap(const py::buffer_info& info) {
  // Verify the innermost dimension is contiguous (stride == 1 element)
  if (info.ndim >= 1) {
    const auto innermostStride = info.strides[info.ndim - 1] / static_cast<py::ssize_t>(sizeof(T));
    if (innermostStride != 1) {
      throw std::runtime_error(
          "Buffer must be contiguous in the innermost dimension for rasterization");
    }
  }

  // For 2D buffers, verify row-major layout
  if (info.ndim == 2) {
    const auto rowStride = info.strides[0] / static_cast<py::ssize_t>(sizeof(T));
    const auto expectedRowStride = info.shape[1];
    if (rowStride != expectedRowStride) {
      throw std::runtime_error("Buffer must be row-major (C-contiguous) for rasterization");
    }
  }

  // Total number of elements
  py::ssize_t numel = 1;
  for (py::ssize_t i = 0; i < info.ndim; ++i) {
    numel *= info.shape[i];
  }

  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(static_cast<T*>(info.ptr), numel);
}

// bufferToEigenMap from optional buffer_info, returning an empty fallback if absent.
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> bufferToEigenMap(
    const std::optional<py::buffer_info>& info,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& emptyFallback) {
  if (!info.has_value()) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(emptyFallback.data(), 0);
  }
  return bufferToEigenMap<T>(*info);
}

// Create an aligned numpy array for SIMD operations with proper stride handling.
// Shape contains the logical dimensions (actual width).
// Row strides are padded internally for SIMD alignment.
template <typename T>
pybind11::array_t<T> create_aligned_array(const std::vector<py::ssize_t>& shape) {
  constexpr size_t alignment = momentum::kSimdAlignment;

  if (shape.empty()) {
    throw std::runtime_error("Shape cannot be empty");
  }

  // For images, shape is either [height, width] or [height, width, channels]
  // The row stride should be based on paddedWidth, not the logical width
  const py::ssize_t height = shape[0];
  const py::ssize_t width = shape.size() > 1 ? shape[1] : 1;
  const py::ssize_t channels = shape.size() > 2 ? shape[2] : 1;

  // Compute padded width for SIMD alignment
  const py::ssize_t paddedWidth =
      momentum::rasterizer::padImageWidthForRasterizer(static_cast<int32_t>(width));

  // Build strides in element counts
  std::vector<py::ssize_t> strides_in_elements;
  if (shape.size() == 2) {
    // 2D array: [height, width]
    strides_in_elements = {paddedWidth, 1};
  } else if (shape.size() == 3) {
    // 3D array: [height, width, channels]
    strides_in_elements = {paddedWidth * channels, channels, 1};
  } else {
    throw std::runtime_error("Only 2D and 3D arrays are supported");
  }

  // Convert strides to bytes for pybind11
  std::vector<py::ssize_t> strides_in_bytes;
  strides_in_bytes.reserve(strides_in_elements.size());
  for (auto stride : strides_in_elements) {
    strides_in_bytes.push_back(stride * static_cast<py::ssize_t>(sizeof(T)));
  }

  // Calculate total size: height * paddedWidth * channels
  const py::ssize_t total_size = height * paddedWidth * channels;

  // Allocate aligned memory using momentum's cross-platform aligned allocation
  T* ptr = momentum::alignedAlloc<T, alignment>(total_size);

  // Create a capsule to manage the memory - will be called when array is destroyed
  py::capsule deallocator(ptr, [](void* p) { momentum::aligned_free(p); });

  // Create numpy array with aligned memory and custom strides
  return pybind11::array_t<T>(
      shape, // logical shape (actual width, not padded)
      strides_in_bytes, // strides in bytes (row stride uses paddedWidth)
      ptr, // data pointer
      deallocator // capsule for cleanup
  );
}

// Validate a rasterizer buffer's shape, dtype, and strides against expected dimensions.
inline void validateBuffer(
    const pybind11::buffer& buffer,
    const char* name,
    int expectedRank,
    const std::string& expectedFormat,
    const std::vector<int64_t>& expectedShape) {
  auto info = buffer.request();

  if (info.format != expectedFormat) {
    throw std::runtime_error(
        fmt::format(
            "{} has incorrect data type. Expected {} but got {}",
            name,
            expectedFormat,
            info.format));
  }

  if (info.ndim != expectedRank) {
    throw std::runtime_error(
        fmt::format(
            "{} has incorrect dimensions. Expected {} dimensions but got {}",
            name,
            expectedRank,
            info.ndim));
  }

  for (int i = 0; i < expectedRank; ++i) {
    if (info.shape[i] != expectedShape[i]) {
      throw std::runtime_error(
          fmt::format(
              "{} has incorrect shape at dimension {}. Expected {} but got {}",
              name,
              i,
              expectedShape[i],
              info.shape[i]));
    }
  }

  // Compute padded width for SIMD alignment from the logical width
  const int64_t width = expectedRank >= 2 ? expectedShape[1] : 1;
  const int64_t paddedWidth =
      momentum::rasterizer::padImageWidthForRasterizer(static_cast<int32_t>(width));

  // Compute expected strides based on paddedWidth
  std::vector<int64_t> expectedStridesInElements;
  if (expectedRank == 2) {
    expectedStridesInElements = {paddedWidth, 1};
  } else if (expectedRank == 3) {
    int64_t channels = expectedShape[2];
    expectedStridesInElements = {paddedWidth * channels, channels, 1};
  } else {
    throw std::runtime_error("Only 2D and 3D arrays are supported");
  }

  // Validate strides
  for (int i = 0; i < expectedRank; ++i) {
    int64_t expectedStrideInBytes = expectedStridesInElements[i] * info.itemsize;
    if (info.strides[i] != expectedStrideInBytes) {
      throw std::runtime_error(
          fmt::format(
              "{} has incorrect stride at dimension {}. Expected {} bytes but got {}",
              name,
              i,
              expectedStrideInBytes,
              info.strides[i]));
    }
  }
}

// Buffer-based validation for rasterizer buffers (z, rgb, surface normals, etc.)
inline void validateRasterizerBuffers(
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer = {},
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer = {},
    const std::optional<pybind11::buffer>& vertexIndexBuffer = {},
    const std::optional<pybind11::buffer>& triangleIndexBuffer = {}) {
  const int32_t imageHeight = camera.imageHeight();
  const int32_t imageWidth = camera.imageWidth();

  validateBuffer(
      zBuffer, "zBuffer", 2, py::format_descriptor<float>::format(), {imageHeight, imageWidth});

  if (rgbBuffer.has_value()) {
    validateBuffer(
        *rgbBuffer,
        "rgbBuffer",
        3,
        py::format_descriptor<float>::format(),
        {imageHeight, imageWidth, 3});
  }

  if (surfaceNormalsBuffer.has_value()) {
    validateBuffer(
        *surfaceNormalsBuffer,
        "surfaceNormalsBuffer",
        3,
        py::format_descriptor<float>::format(),
        {imageHeight, imageWidth, 3});
  }

  if (vertexIndexBuffer.has_value()) {
    validateBuffer(
        *vertexIndexBuffer,
        "vertexIndexBuffer",
        2,
        py::format_descriptor<int32_t>::format(),
        {imageHeight, imageWidth});
  }

  if (triangleIndexBuffer.has_value()) {
    validateBuffer(
        *triangleIndexBuffer,
        "triangleIndexBuffer",
        2,
        py::format_descriptor<int32_t>::format(),
        {imageHeight, imageWidth});
  }
}

// Tensor-based validation (legacy - kept for functions not yet migrated to buffer API)
inline std::tuple<
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
validateRasterizerBuffers(
    TensorChecker& checker,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer = {},
    std::optional<at::Tensor> vertexIndexBuffer = {},
    std::optional<at::Tensor> triangleIndexBuffer = {}) {
  // Values chosen to not conflict with any other bound values:
  const int widthBindingID = -2003;
  const int heightBindingID = -2004;

  const int32_t imageWidth = camera.imageWidth();
  const int32_t imageHeight = camera.imageHeight();
  const int32_t paddedWidth = momentum::rasterizer::padImageWidthForRasterizer(imageWidth);

  zBuffer = checker.validateAndFixTensor(
      zBuffer,
      "zBuffer",
      {heightBindingID, widthBindingID},
      {"imageHeight", "imageWidth"},
      at::kFloat,
      true,
      false);

  if (rgbBuffer) {
    *rgbBuffer = checker.validateAndFixTensor(
        *rgbBuffer,
        "rgbBuffer",
        {heightBindingID, widthBindingID, 3},
        {"imageHeight", "imageWidth", "rgb"},
        at::kFloat,
        true,
        false);
  }

  if (surfaceNormalsBuffer) {
    *surfaceNormalsBuffer = checker.validateAndFixTensor(
        *surfaceNormalsBuffer,
        "surfaceNormalsBuffer",
        {heightBindingID, widthBindingID, 3},
        {"imageHeight", "imageWidth", "rgb"},
        at::kFloat,
        true,
        false);
  }

  if (vertexIndexBuffer) {
    *vertexIndexBuffer = checker.validateAndFixTensor(
        *vertexIndexBuffer,
        "vertexIndexBuffer",
        {heightBindingID, widthBindingID},
        {"imageHeight", "imageWidth"},
        at::kInt,
        true,
        false);
  }

  if (triangleIndexBuffer) {
    *triangleIndexBuffer = checker.validateAndFixTensor(
        *triangleIndexBuffer,
        "triangleIndexBuffer",
        {heightBindingID, widthBindingID},
        {"imageHeight", "imageWidth"},
        at::kInt,
        true,
        false);
  }

  const auto zBufferWidth = checker.getBoundValue(widthBindingID);
  const auto zBufferHeight = checker.getBoundValue(heightBindingID);

  // Write a custom error message rather than relying on the tensor checker so
  // we can warn people about the padding requirement.
  if (zBufferHeight != imageHeight || zBufferWidth != paddedWidth) {
    throw std::runtime_error(
        fmt::format(
            "Expected z buffer of size {} x {} but got {} x {}.  "
            "Note that the width must be padded out to the nearest {} to ensure fast SIMD code.  ",
            imageHeight,
            paddedWidth,
            zBufferHeight,
            zBufferWidth,
            momentum::kSimdPacketSize));
  }

  return {zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer};
}

inline std::vector<momentum::rasterizer::Light> convertLightsToEyeSpace(
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const momentum::rasterizer::Camera& camera) {
  if (!lights.has_value()) {
    return {};
  }

  const Eigen::Affine3f worldToEyeXF = camera.worldFromEye();
  std::vector<momentum::rasterizer::Light> result;
  result.reserve(lights->size());
  for (const auto& l : *lights) {
    result.push_back(momentum::rasterizer::transformLight(l, worldToEyeXF));
  }

  return result;
}

} // namespace pymomentum::rasterizer_detail
