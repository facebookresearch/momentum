/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/rasterizer_primitives.h"

#include "pymomentum/array_utility/array_utility.h"
#include "pymomentum/renderer/rasterizer_utility.h"

#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>

#include <drjit/packet.h>
#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace pymomentum {

using rasterizer_detail::bufferToEigenMap;
using rasterizer_detail::convertLightsToEyeSpace;
using rasterizer_detail::make_mdspan;
using rasterizer_detail::validateRasterizerBuffers;

pybind11::array_t<float> createZBuffer(const momentum::rasterizer::Camera& camera, float farClip) {
  const int32_t height = camera.imageHeight();
  const int32_t width = camera.imageWidth();

  pybind11::array_t<float> result = rasterizer_detail::create_aligned_array<float>({height, width});
  float* data = result.mutable_data();

  // Fill the entire allocated buffer (including padding) with farClip
  const auto paddedWidth = momentum::rasterizer::padImageWidthForRasterizer(width);
  const size_t total = static_cast<size_t>(height) * static_cast<size_t>(paddedWidth);
  std::fill(data, data + total, farClip);

  return result;
}

pybind11::array_t<float> createRGBBuffer(
    const momentum::rasterizer::Camera& camera,
    std::optional<Eigen::Vector3f> bgColor) {
  const int32_t height = camera.imageHeight();
  const int32_t width = camera.imageWidth();

  pybind11::array_t<float> result =
      rasterizer_detail::create_aligned_array<float>({height, width, 3});

  const float r_val = bgColor.has_value() ? bgColor->x() : 0.0f;
  const float g_val = bgColor.has_value() ? bgColor->y() : 0.0f;
  const float b_val = bgColor.has_value() ? bgColor->z() : 0.0f;

  // Fill the entire buffer (including padding) with background color
  const auto paddedWidth = momentum::rasterizer::padImageWidthForRasterizer(width);
  float* data = result.mutable_data();
  for (int32_t i = 0; i < height; ++i) {
    for (int32_t j = 0; j < paddedWidth; ++j) {
      const size_t idx = (static_cast<size_t>(i) * paddedWidth + j) * 3;
      data[idx + 0] = r_val;
      data[idx + 1] = g_val;
      data[idx + 2] = b_val;
    }
  }

  return result;
}

pybind11::array_t<int32_t> createIndexBuffer(const momentum::rasterizer::Camera& camera) {
  const int32_t height = camera.imageHeight();
  const int32_t width = camera.imageWidth();

  pybind11::array_t<int32_t> result =
      rasterizer_detail::create_aligned_array<int32_t>({height, width});
  int32_t* data = result.mutable_data();

  // Fill the entire allocated buffer (including padding) with -1
  const auto paddedWidth = momentum::rasterizer::padImageWidthForRasterizer(width);
  const size_t total = static_cast<size_t>(height) * static_cast<size_t>(paddedWidth);
  std::fill(data, data + total, -1);

  return result;
}

namespace {

momentum::rasterizer::Tensor<float, 3> toFloatImage(const pybind11::array_t<const float>& img) {
  if (img.ndim() != 3 || img.shape(2) != 3) {
    throw std::runtime_error("Expected tensor shape height x width x 3 for RGB image.");
  }

  momentum::rasterizer::Tensor<float, 3> result(
      {static_cast<std::ptrdiff_t>(img.shape(0)),
       static_cast<std::ptrdiff_t>(img.shape(1)),
       static_cast<std::ptrdiff_t>(img.shape(2))});
  auto r = img.unchecked<3>(); // x must have ndim = 3; can be non-writable
  for (py::ssize_t i = 0; i < r.shape(0); i++) {
    for (py::ssize_t j = 0; j < r.shape(1); j++) {
      for (py::ssize_t k = 0; k < r.shape(2); k++) {
        result(i, j, k) = r(i, j, k);
      }
    }
  }
  return result;
}

} // namespace

momentum::rasterizer::PhongMaterial createPhongMaterial(
    const std::optional<Eigen::Vector3f>& diffuseColor,
    const std::optional<Eigen::Vector3f>& specularColor,
    const std::optional<float>& specularExponent,
    const std::optional<Eigen::Vector3f>& emissiveColor,
    const std::optional<pybind11::array_t<const float>>& diffuseTexture,
    const std::optional<pybind11::array_t<const float>>& emissiveTexture) {
  momentum::rasterizer::PhongMaterial result;
  result.diffuseColor = diffuseColor.value_or(Eigen::Vector3f::Ones());
  result.specularColor = specularColor.value_or(Eigen::Vector3f::Zero());
  result.specularExponent = specularExponent.value_or(10.0f);
  result.emissiveColor = emissiveColor.value_or(Eigen::Vector3f::Zero());

  if (diffuseTexture) {
    result.diffuseTextureMap = toFloatImage(diffuseTexture.value());
  }
  if (emissiveTexture) {
    result.emissiveTextureMap = toFloatImage(emissiveTexture.value());
  }

  return result;
}

void alphaMatte(
    const pybind11::buffer& zBuffer,
    const pybind11::buffer& rgbBuffer,
    const pybind11::array& tgtRgbImage,
    float alpha) {
  // Validate Z buffer dimensions
  auto zInfo = zBuffer.request();
  if (zInfo.ndim != 2) {
    throw std::runtime_error("z_buffer must be a 2D array");
  }
  const int height = static_cast<int>(zInfo.shape[0]);
  const int width = static_cast<int>(zInfo.shape[1]);

  // Validate RGB buffer dimensions
  auto rgbInfo = rgbBuffer.request();
  if (rgbInfo.ndim != 3) {
    throw std::runtime_error("rgb_buffer must be a 3D array");
  }
  if (rgbInfo.shape[0] != height || rgbInfo.shape[1] != width) {
    throw std::runtime_error("rgb_buffer dimensions must match z_buffer");
  }
  if (rgbInfo.shape[2] != 3) {
    throw std::runtime_error("rgb_buffer must have 3 channels");
  }

  const auto tgtRgbImage_info = tgtRgbImage.request();
  if (tgtRgbImage_info.format == py::format_descriptor<float>::format()) {
    momentum::rasterizer::alphaMatte(
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<float, 3>(tgtRgbImage_info),
        alpha);
  } else if (tgtRgbImage_info.format == py::format_descriptor<uint8_t>::format()) {
    momentum::rasterizer::alphaMatte(
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<uint8_t, 3>(tgtRgbImage_info),
        alpha);
  } else {
    throw std::runtime_error("tgt_rgb_image must be a uint8 or float32 RGB array.");
  }
}

void rasterizeLines(
    pybind11::buffer positions,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nLinesBindingId = -1;

  ArrayChecker checker("rasterize_lines");
  checker.validateBuffer(
      positions, "positions", {nLinesBindingId, 2, 3}, {"nLines", "start_end", "xyz"}, false);

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, {}, {}, {});

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeLines(
      positionsMap,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      width,
      zBufferMdspan,
      rgbBufferMdspan,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeCircles(
    pybind11::buffer positions,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nCirclesBindingId = -1;

  ArrayChecker checker("rasterize_circles");
  checker.validateBuffer(
      positions, "positions", {nCirclesBindingId, 3}, {"nCircles", "xyz"}, false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, {}, {}, {});

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeCircles(
      positionsMap,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      zBufferMdspan,
      rgbBufferMdspan,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeCameraFrustum(
    const momentum::rasterizer::Camera& frustumCamera,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float lineWidth,
    float distance,
    size_t numSamples,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (lineWidth <= 0) {
    throw std::runtime_error("line_width should be positive.");
  }

  if (numSamples <= 2) {
    throw std::runtime_error("num_samples should be at least 3.");
  }

  if (distance <= 0) {
    throw std::runtime_error("distance should be positive.");
  }

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, {}, {}, {});

  const std::vector<Eigen::Vector3f> frustumLinesEye =
      momentum::rasterizer::makeCameraFrustumLines(frustumCamera, distance, numSamples);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeLines(
      frustumLinesEye,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()) * frustumCamera.worldFromEye().matrix(),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      lineWidth,
      zBufferMdspan,
      rgbBufferMdspan,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeGrid(
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numLines) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (width <= 0) {
    throw std::runtime_error("width should be positive.");
  }

  if (numLines <= 0) {
    throw std::runtime_error("num_lines should be positive.");
  }

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, {}, {}, {});

  // Generate grid line segments in the XZ plane (y=0).
  // numLines cells per axis => numLines+1 lines in each direction.
  const int linesPerAxis = numLines + 1;
  const float halfWidth = width / 2.0f;
  const float step = width / static_cast<float>(numLines);

  std::vector<Eigen::Vector3f> gridLines;
  gridLines.reserve(linesPerAxis * 4);

  for (int i = 0; i < linesPerAxis; ++i) {
    const float t = -halfWidth + static_cast<float>(i) * step;

    // Line parallel to X axis (varying Z)
    gridLines.emplace_back(-halfWidth, 0.0f, t);
    gridLines.emplace_back(halfWidth, 0.0f, t);

    // Line parallel to Z axis (varying X)
    gridLines.emplace_back(t, 0.0f, -halfWidth);
    gridLines.emplace_back(t, 0.0f, halfWidth);
  }

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeLines(
      gridLines,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      thickness,
      zBufferMdspan,
      rgbBufferMdspan,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeLines2D(
    pybind11::buffer positions,
    const pybind11::buffer& rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  const int nLinesBindingId = -1;

  ArrayChecker checker("rasterize_lines_2d");
  checker.validateBuffer(
      positions, "positions", {nLinesBindingId, 2, 2}, {"nLines", "start_end", "xy"}, false);

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  // Create mdspans while holding GIL
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeLines2D(
      positionsMap,
      color.value_or(Eigen::Vector3f::Ones()),
      thickness,
      rgbBufferMdspan,
      zBufferMdspan,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeText(
    pybind11::buffer positions,
    const std::vector<std::string>& texts,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  const int nTextBindingId = -1;

  ArrayChecker checker("rasterize_text");
  checker.validateBuffer(positions, "positions", {nTextBindingId, 3}, {"nTexts", "xyz"}, false);

  const auto numTexts = checker.getBoundValue(nTextBindingId);
  if (numTexts != static_cast<int64_t>(texts.size())) {
    throw std::runtime_error(
        fmt::format(
            "Mismatch between number of positions ({}) and texts ({})", numTexts, texts.size()));
  }

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  std::vector<Eigen::Vector3f> positionsVec;
  positionsVec.reserve(numTexts);
  for (int i = 0; i < numTexts; ++i) {
    positionsVec.emplace_back(positionsMap.segment<3>(3 * i));
  }

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeText(
      positionsVec,
      texts,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      textScale,
      zBufferMdspan,
      rgbBufferMdspan,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()),
      horizontalAlignment,
      verticalAlignment);
}

void rasterizeText2D(
    pybind11::buffer positions,
    const std::vector<std::string>& texts,
    const pybind11::buffer& rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  const int nTextBindingId = -1;

  ArrayChecker checker("rasterize_text_2d");
  checker.validateBuffer(positions, "positions", {nTextBindingId, 2}, {"nTexts", "xy"}, false);

  const auto numTexts = checker.getBoundValue(nTextBindingId);
  if (numTexts != static_cast<int64_t>(texts.size())) {
    throw std::runtime_error(
        fmt::format(
            "Mismatch between number of positions ({}) and texts ({})", numTexts, texts.size()));
  }

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  std::vector<Eigen::Vector2f> positionsVec;
  positionsVec.reserve(numTexts);
  for (int i = 0; i < numTexts; ++i) {
    positionsVec.emplace_back(positionsMap.segment<2>(2 * i));
  }

  // Create mdspans while holding GIL
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeText2D(
      positionsVec,
      texts,
      color.value_or(Eigen::Vector3f::Ones()),
      textScale,
      rgbBufferMdspan,
      zBufferMdspan,
      imageOffset.value_or(Eigen::Vector2f::Zero()),
      horizontalAlignment,
      verticalAlignment);
}

void rasterizeCircles2D(
    pybind11::buffer positions,
    const pybind11::buffer& rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  const int nCirclesBindingId = -1;

  ArrayChecker checker("rasterize_circles_2d");
  checker.validateBuffer(positions, "positions", {nCirclesBindingId, 2}, {"nCircles", "xy"}, false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  // Get buffer info for creating Eigen map (requires GIL)
  auto positionsInfo = positions.request();
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);

  // Create mdspans while holding GIL
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeCircles2D(
      positionsMap,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      rgbBufferMdspan,
      zBufferMdspan,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeTransforms(
    pybind11::buffer transforms,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    float scale,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int lengthSubdivisions,
    int radiusSubdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nTransformsBindingId = -1;

  ArrayChecker checker("rasterize_transforms");
  checker.validateBuffer(
      transforms,
      "transforms",
      {nTransformsBindingId, 4, 4},
      {"nTransforms", "rows", "cols"},
      false);

  const float c1 = 1.0f;
  const float c2 = 0.3f;
  const bool backfaceCulling = true;

  std::array<momentum::rasterizer::PhongMaterial, 3> materials = {
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c1, c2, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c1, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c2, c1)))};

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nTransforms = checker.getBoundValue(nTransformsBindingId);

  const auto arrowMesh = momentum::rasterizer::makeArrow(
      radiusSubdivisions, lengthSubdivisions, 0.05f, 0.2f, 0.4f, 0.6f);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // Get buffer info for raw pointer access (requires GIL)
  // The transforms buffer has shape (nTransforms, 4, 4) and is row-major (C-contiguous).
  // Element [i][r][c] is at offset i*16 + r*4 + c.
  auto transformsInfo = transforms.request();
  const auto* transformsPtr = static_cast<const float*>(transformsInfo.ptr);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  for (int i = 0; i < nTransforms; ++i) {
    // Access element [i][row][col] = transformsPtr[i*16 + row*4 + col]
    const float* m = transformsPtr + i * 16;
    const Eigen::Vector3f origin(m[0 * 4 + 3], m[1 * 4 + 3], m[2 * 4 + 3]);
    for (int j = 0; j < 3; ++j) {
      const Eigen::Vector3f col_j(m[0 * 4 + j], m[1 * 4 + j], m[2 * 4 + j]);
      const float length = col_j.norm();
      if (length == 0) {
        continue;
      }

      const auto& materialCur = materials[j];

      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translate(origin);
      transform.rotate(
          Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), col_j.normalized()));
      transform.scale(scale * Eigen::Vector3f(length, length, length));

      momentum::rasterizer::rasterizeMesh(
          arrowMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          materialCur,
          zBufferMdspan,
          rgbBufferMdspan,
          surfaceNormalsBufferMdspan,
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

} // namespace pymomentum
