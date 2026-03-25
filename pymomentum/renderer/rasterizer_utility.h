/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/simd/simd.h>

#include <ATen/ATen.h>
#include <fmt/format.h>
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
