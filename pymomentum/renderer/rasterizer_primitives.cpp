/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/rasterizer_primitives.h"

#include "pymomentum/renderer/rasterizer_utility.h"
#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

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

using rasterizer_detail::convertLightsToEyeSpace;
using rasterizer_detail::make_mdspan;
using rasterizer_detail::validateRasterizerBuffers;

at::Tensor createZBuffer(const momentum::rasterizer::Camera& camera, float farClip) {
  return at::full(
      {camera.imageHeight(), momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth())},
      farClip,
      at::kFloat);
}

at::Tensor createRGBBuffer(
    const momentum::rasterizer::Camera& camera,
    std::optional<Eigen::Vector3f> bgColor) {
  auto result = at::zeros(
      {camera.imageHeight(),
       momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth()),
       3},
      at::kFloat);

  if (bgColor.has_value()) {
    result.select(-1, 0) = bgColor->x();
    result.select(-1, 1) = bgColor->y();
    result.select(-1, 2) = bgColor->z();
  }

  return result;
}

at::Tensor createIndexBuffer(const momentum::rasterizer::Camera& camera) {
  return at::full(
      {camera.imageHeight(), momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth())},
      -1,
      at::kInt);
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
    at::Tensor zBuffer,
    at::Tensor rgbBuffer,
    const pybind11::array& tgtRgbImage,
    float alpha) {
  TensorChecker checker("alpha_matte");

  const int widthBindingID = -2003;
  const int heightBindingID = -2004;

  zBuffer = checker.validateAndFixTensor(
      zBuffer,
      "zBuffer",
      {heightBindingID, widthBindingID},
      {"imageHeight", "imageWidth"},
      at::kFloat,
      true,
      false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgbBuffer",
      {heightBindingID, widthBindingID, 3},
      {"imageHeight", "imageWidth", "rgb"},
      at::kFloat,
      true,
      false);
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
    at::Tensor positions,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nLinesBindingId = -1;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions,
      "positions",
      {nLinesBindingId, 2, 3},
      {"nLines", "start_end", "xyz"},
      at::kFloat,
      true,
      false);

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  momentum::rasterizer::rasterizeLines(
      toEigenMap<float>(positions),
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      width,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeCircles(
    at::Tensor positions,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nCirclesBindingId = -1;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nCirclesBindingId, 3}, {"nCircles", "xyz"}, at::kFloat, true, false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  momentum::rasterizer::rasterizeCircles(
      toEigenMap<float>(positions),
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeCameraFrustum(
    const momentum::rasterizer::Camera& frustumCamera,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float lineWidth,
    float distance,
    size_t numSamples,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

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

  TensorChecker checker("rasterize_camera_frustum");

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  const std::vector<Eigen::Vector3f> frustumLinesEye =
      momentum::rasterizer::makeCameraFrustumLines(frustumCamera, distance, numSamples);

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  momentum::rasterizer::rasterizeLines(
      frustumLinesEye,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()) * frustumCamera.worldFromEye().matrix(),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      lineWidth,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeGrid(
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numLines) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (width <= 0) {
    throw std::runtime_error("width should be positive.");
  }

  if (numLines <= 0) {
    throw std::runtime_error("num_lines should be positive.");
  }

  TensorChecker checker("rasterize_grid");

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

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

  momentum::rasterizer::rasterizeLines(
      gridLines,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      thickness,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeLines2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nLinesBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_lines_2d");
  positions = checker.validateAndFixTensor(
      positions,
      "positions",
      {nLinesBindingId, 2, 2},
      {"nLines", "start_end", "xy"},
      at::kFloat,
      true,
      false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgb_buffer",
      {heightBindingId, widthBindingId, 3},
      {"height", "width", "rgb"},
      at::kFloat,
      true,
      false);

  if (zBuffer.has_value()) {
    zBuffer = checker.validateAndFixTensor(
        *zBuffer,
        "z_buffer",
        {heightBindingId, widthBindingId},
        {"height", "width"},
        at::kFloat,
        true,
        false);
  }

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  zBuffer.has_value() ? zBuffer : std::optional<at::Tensor>{};

  momentum::rasterizer::rasterizeLines2D(
      toEigenMap<float>(positions),
      color.value_or(Eigen::Vector3f::Ones()),
      thickness,
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 2>(zBuffer),
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeText(
    at::Tensor positions,
    const std::vector<std::string>& texts,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nTextBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_text");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nTextBindingId, 3}, {"nTexts", "xyz"}, at::kFloat, true, false);

  zBuffer = checker.validateAndFixTensor(
      zBuffer,
      "z_buffer",
      {heightBindingId, widthBindingId},
      {"height", "width"},
      at::kFloat,
      true,
      false);

  if (rgbBuffer.has_value()) {
    rgbBuffer = checker.validateAndFixTensor(
        *rgbBuffer,
        "rgb_buffer",
        {heightBindingId, widthBindingId, 3},
        {"height", "width", "rgb"},
        at::kFloat,
        true,
        false);
  }

  const int64_t numTexts = checker.getBoundValue(nTextBindingId);
  if (numTexts != static_cast<int64_t>(texts.size())) {
    throw std::runtime_error(
        fmt::format(
            "Mismatch between number of positions ({}) and texts ({})", numTexts, texts.size()));
  }

  rgbBuffer.has_value() ? rgbBuffer : std::optional<at::Tensor>{};

  const Eigen::Ref<const Eigen::VectorXf> positionsFlat = toEigenMap<float>(positions);
  std::vector<Eigen::Vector3f> positionsVec;
  positionsVec.reserve(numTexts);
  for (int i = 0; i < numTexts; ++i) {
    positionsVec.emplace_back(positionsFlat.segment<3>(3 * i));
  }

  momentum::rasterizer::rasterizeText(
      positionsVec,
      texts,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f::Ones()),
      textScale,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()),
      horizontalAlignment,
      verticalAlignment);
}

void rasterizeText2D(
    at::Tensor positions,
    const std::vector<std::string>& texts,
    at::Tensor rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nTextBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_text_2d");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nTextBindingId, 2}, {"nTexts", "xy"}, at::kFloat, true, false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgb_buffer",
      {heightBindingId, widthBindingId, 3},
      {"height", "width", "rgb"},
      at::kFloat,
      true,
      false);

  if (zBuffer.has_value()) {
    zBuffer = checker.validateAndFixTensor(
        *zBuffer,
        "z_buffer",
        {heightBindingId, widthBindingId},
        {"height", "width"},
        at::kFloat,
        true,
        false);
  }

  const int64_t numTexts = checker.getBoundValue(nTextBindingId);
  if (numTexts != static_cast<int64_t>(texts.size())) {
    throw std::runtime_error(
        fmt::format(
            "Mismatch between number of positions ({}) and texts ({})", numTexts, texts.size()));
  }

  zBuffer.has_value() ? zBuffer : std::optional<at::Tensor>{};

  const Eigen::Ref<const Eigen::MatrixXf> positionsMat = toEigenMap<float>(positions);
  std::vector<Eigen::Vector2f> positionsVec;
  positionsVec.reserve(positionsMat.rows());
  for (int i = 0; i < positionsMat.rows(); ++i) {
    positionsVec.emplace_back(positionsMat.row(i));
  }

  momentum::rasterizer::rasterizeText2D(
      positionsVec,
      texts,
      color.value_or(Eigen::Vector3f::Ones()),
      textScale,
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 2>(zBuffer),
      imageOffset.value_or(Eigen::Vector2f::Zero()),
      horizontalAlignment,
      verticalAlignment);
}

void rasterizeCircles2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nCirclesBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_circles_2d");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nCirclesBindingId, 2}, {"nCircles", "xy"}, at::kFloat, true, false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgb_buffer",
      {heightBindingId, widthBindingId, 3},
      {"height", "width", "rgb"},
      at::kFloat,
      true,
      false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  if (zBuffer.has_value()) {
    zBuffer = checker.validateAndFixTensor(
        *zBuffer,
        "z_buffer",
        {heightBindingId, widthBindingId},
        {"height", "width"},
        at::kFloat,
        true,
        false);
  }

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  zBuffer.has_value() ? zBuffer : std::optional<at::Tensor>{};

  momentum::rasterizer::rasterizeCircles2D(
      toEigenMap<float>(positions),
      lineColor,
      fillColor,
      lineThickness,
      radius,
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 2>(zBuffer),
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeTransforms(
    at::Tensor transforms,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
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
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nTransformsBindingId = -1;

  TensorChecker checker("rasterize");

  transforms = checker.validateAndFixTensor(
      transforms,
      "transforms",
      {nTransformsBindingId, 4, 4},
      {"nTransforms", "rows", "cols"},
      at::kFloat,
      true,
      false);

  const float c1 = 1.0f;
  const float c2 = 0.3f;
  const bool backfaceCulling = true;

  std::array<momentum::rasterizer::PhongMaterial, 3> materials = {
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c1, c2, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c1, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c2, c1)))};

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nTransforms = checker.getBoundValue(nTransformsBindingId);

  const auto arrowMesh = momentum::rasterizer::makeArrow(
      radiusSubdivisions, lengthSubdivisions, 0.05f, 0.2f, 0.4f, 0.6f);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  const auto transformsCur = transforms.select(0, 0);
  const auto a = transformsCur.accessor<float, 3>();

  for (int i = 0; i < nTransforms; ++i) {
    const Eigen::Vector3f origin(a[i][0][3], a[i][1][3], a[i][2][3]);
    for (int j = 0; j < 3; ++j) {
      const Eigen::Vector3f col_j(a[i][0][j], a[i][1][j], a[i][2][j]);
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
          make_mdspan<float, 2>(zBuffer),
          make_mdspan<float, 3>(rgbBuffer),
          make_mdspan<float, 3>(surfaceNormalsBuffer),
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
