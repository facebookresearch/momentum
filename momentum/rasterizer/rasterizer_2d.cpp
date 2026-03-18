/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/rasterizer.h"

#include "momentum/rasterizer/rasterizer_internal.h"
#include "momentum/rasterizer/utility.h"

#include "momentum/common/exception.h"

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/math.h>

#include <cmath>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

namespace momentum::rasterizer {

namespace {

inline void rasterizeOneLine2D(
    const Vector2f& p_line,
    const Vector2f& d_line,
    const float d_line_sqrLen,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Vector3f& color,
    float halfThickness,
    index_t rowStride,
    float* rgbBufferPtr,
    float* zBufferPtr = nullptr) {
  const int startX_block = startX / kSimdPacketSize;
  const int endX_block = endX / kSimdPacketSize;

  for (int32_t y = startY; y <= endY; ++y) {
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      const FloatP p_window_x = (drjit::arange<FloatP>() + (float)xOffset);
      const float p_window_y = y;

      const Vector2fP p_cur(p_window_x, p_window_y);

      const FloatP t = drjit::dot(p_cur - p_line, d_line) / d_line_sqrLen;
      const auto t_mask = (t >= 0.0f && t <= 1.0f);
      if (!drjit::any(t_mask)) {
        continue;
      }

      const Vector2fP closestPoint = p_line + t * Vector2fP(d_line);
      const FloatP dist = drjit::norm(p_cur - closestPoint);
      const auto dist_mask = (dist < halfThickness);
      if (!drjit::any(dist_mask)) {
        continue;
      }

      const auto finalMask = dist_mask && t_mask;
      if (!drjit::any(finalMask)) {
        continue;
      }

      if (rgbBufferPtr != nullptr) {
        float* rgbBufferOffset = rgbBufferPtr + 3 * (y * rowStride + xOffset);
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        const Vector3fP rgbValuesFinal = drjit::select(finalMask, Vector3fP(color), rgbValuesOrig);
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }

      // Write zeros to z-buffer for alpha matting
      if (zBufferPtr != nullptr) {
        const auto blockOffset = y * rowStride + xOffset;
        const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
        const FloatP zBufferFinal = drjit::select(finalMask, FloatP(0.0f), zBufferOrig);
        drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);
      }
    }
  }
}

inline void rasterizeOneCircle2D(
    const Vector2f& center,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Vector3f& lineColor,
    const Vector3f& fillColor,
    bool filled,
    float halfLineThickness,
    float radius,
    index_t rowStride,
    float* rgbBufferPtr,
    float* zBufferPtr = nullptr) {
  const int startX_block = startX / kSimdPacketSize;
  const int endX_block = endX / kSimdPacketSize;

  const float outerRadius = radius + halfLineThickness;
  const float innerRadius = radius - halfLineThickness;

  for (int32_t y = startY; y <= endY; ++y) {
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      const FloatP p_window_x = (drjit::arange<FloatP>() + (float)xOffset);
      const float p_window_y = y;

      const Vector2fP p_cur(p_window_x, p_window_y);

      const FloatP dist = drjit::norm(p_cur - center);
      auto combinedMask = dist <= outerRadius;
      auto lineMask = combinedMask && (dist >= innerRadius);
      if (!filled) {
        combinedMask &= lineMask;
      }

      if (!drjit::any(combinedMask)) {
        continue;
      }

      if (rgbBufferPtr != nullptr) {
        float* rgbBufferOffset = rgbBufferPtr + 3 * (y * rowStride + xOffset);
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        Vector3fP rgbValuesFinal = drjit::select(lineMask, Vector3fP(lineColor), rgbValuesOrig);
        if (filled) {
          const auto fillMask = combinedMask && ~lineMask;
          rgbValuesFinal = drjit::select(fillMask, Vector3fP(fillColor), rgbValuesFinal);
        }
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }

      // Write zeros to z-buffer for alpha matting
      if (zBufferPtr != nullptr) {
        const auto blockOffset = y * rowStride + xOffset;
        const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
        const FloatP zBufferFinal = drjit::select(combinedMask, FloatP(0.0f), zBufferOrig);
        drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);
      }
    }
  }
}

void rasterizeLines2DImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  if (positions_image.size() == 0) {
    return;
  }

  if (!isValidBuffer(rgbBuffer)) {
    throw std::runtime_error(
        "RGB buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  if (positions_image.size() % 4 != 0) {
    throw std::runtime_error("Positions array should be an even number of points [start, end]");
  }

  const auto nVerts = positions_image.size() / 2;
  const auto nLines = nVerts / 2;
  if (nLines * 4 != positions_image.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_image.size()
        << " is not a multiple of four (start_x, start_y, end_x, end_y).";
    throw std::runtime_error(oss.str());
  }

  const auto imageHeight = static_cast<int32_t>(rgbBuffer.extent(0));
  const auto imageWidth = static_cast<int32_t>(rgbBuffer.extent(1));

  float* rgbBufferPtr = rgbBuffer.data_handle();
  const float halfThickness = 0.5f * thickness;
  const int halfThicknessPixels =
      std::clamp<int>(static_cast<int>(std::ceil(halfThickness)), 1, imageHeight - 1);

  const Vector3f color_drjit = toEnokiVec(color);

  for (int lineIdx = 0; lineIdx < nLines; ++lineIdx) {
    const Vector2f lineStart(
        positions_image[4 * lineIdx + 0] + imageOffset.x(),
        positions_image[4 * lineIdx + 1] + imageOffset.y());
    const Vector2f lineEnd(
        positions_image[4 * lineIdx + 2] + imageOffset.x(),
        positions_image[4 * lineIdx + 3] + imageOffset.y());

    Vector2f p_line = lineStart;
    Vector2f d_line = lineEnd - lineStart;
    float d_line_sqrLen = drjit::squared_norm(d_line);

    // Compute the bounds of the line in screen space:
    const int32_t startX = std::clamp<int32_t>(
        static_cast<int32_t>(
            std::floor(std::min(lineStart.x(), lineEnd.x()) - halfThicknessPixels)),
        0,
        imageWidth - 1);
    const int32_t endX = std::clamp<int32_t>(
        static_cast<int32_t>(std::ceil(std::max(lineStart.x(), lineEnd.x()) + halfThicknessPixels)),
        0,
        imageWidth - 1);

    const int32_t startY = std::clamp<int32_t>(
        static_cast<int32_t>(
            std::floor(std::min(lineStart.y(), lineEnd.y()) - halfThicknessPixels)),
        0,
        imageHeight - 1);
    const int32_t endY = std::clamp<int32_t>(
        static_cast<int32_t>(std::ceil(std::max(lineStart.y(), lineEnd.y()) + halfThicknessPixels)),
        0,
        imageHeight - 1);

    float* zBufferPtr = dataOrNull(zBuffer);

    rasterizeOneLine2D(
        p_line,
        d_line,
        d_line_sqrLen,
        startX,
        endX,
        startY,
        endY,
        color_drjit,
        halfThickness,
        getPixelRowStride(rgbBuffer),
        rgbBufferPtr,
        zBufferPtr);
  }
}

void rasterizeCircles2DImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  if (positions_image.size() == 0) {
    return;
  }

  if (!isValidBuffer(rgbBuffer)) {
    throw std::runtime_error(
        "RGB buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  MT_THROW_IF(radius <= 0, "Radius must be greater than zero.");
  MT_THROW_IF(lineThickness < 0, "Line thickness must be non-negative.");

  const auto nCircles = positions_image.size() / 2;
  if (nCircles * 2 != positions_image.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_image.size()
        << " is not a multiple of two (center_x, center_y).";
    throw std::runtime_error(oss.str());
  }

  const auto imageHeight = static_cast<int32_t>(rgbBuffer.extent(0));
  const auto imageWidth = static_cast<int32_t>(rgbBuffer.extent(1));

  float* rgbBufferPtr = rgbBuffer.data_handle();

  const float halfLineThickness = lineColor.has_value() ? lineThickness / 2.f : 0.0f;

  const bool filled = fillColor.has_value();
  const Vector3f lineColor_drjit = toEnokiVec(lineColor.value_or(Eigen::Vector3f::Ones()));
  const Vector3f fillColor_drjit = toEnokiVec(fillColor.value_or(Eigen::Vector3f::Zero()));

  const float combinedRadius = radius + halfLineThickness;
  const int combinedRadiusPixels =
      std::clamp<int>(static_cast<int>(std::ceil(combinedRadius)), 1, imageHeight - 1);

  for (int circleIdx = 0; circleIdx < nCircles; ++circleIdx) {
    const Vector2f center(
        positions_image[2 * circleIdx + 0] + imageOffset.x(),
        positions_image[2 * circleIdx + 1] + imageOffset.y());

    // Compute the bounds of the circle in screen space:
    const int32_t startX = std::clamp<int32_t>(
        static_cast<int32_t>(std::floor(center.x() - combinedRadiusPixels)), 0, imageWidth - 1);
    const int32_t endX = std::clamp<int32_t>(
        static_cast<int32_t>(std::ceil(center.x() + combinedRadiusPixels)), 0, imageWidth - 1);

    const int32_t startY = std::clamp<int32_t>(
        static_cast<int32_t>(std::floor(center.y() - combinedRadiusPixels)), 0, imageHeight - 1);
    const int32_t endY = std::clamp<int32_t>(
        static_cast<int32_t>(std::ceil(center.y() + combinedRadiusPixels)), 0, imageHeight - 1);

    float* zBufferPtr = dataOrNull(zBuffer);

    rasterizeOneCircle2D(
        center,
        startX,
        endX,
        startY,
        endY,
        lineColor_drjit,
        fillColor_drjit,
        filled,
        halfLineThickness,
        radius,
        getPixelRowStride(rgbBuffer),
        rgbBufferPtr,
        zBufferPtr);
  }
}

} // namespace

void rasterizeLines2D(
    std::span<const Eigen::Vector2f> positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  rasterizeLines2DImp(
      mapVector<float, 2>(positions_image), color, thickness, rgbBuffer, zBuffer, imageOffset);
}

void rasterizeLines2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  rasterizeLines2DImp(positions_image, color, thickness, rgbBuffer, zBuffer, imageOffset);
}

void rasterizeCircles2D(
    std::span<const Eigen::Vector2f> positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  rasterizeCircles2DImp(
      mapVector<float, 2>(positions_image),
      lineColor,
      fillColor,
      lineThickness,
      radius,
      rgbBuffer,
      zBuffer,
      imageOffset);
}

void rasterizeCircles2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer,
    const Eigen::Vector2f& imageOffset) {
  rasterizeCircles2DImp(
      positions_image,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      rgbBuffer,
      zBuffer,
      imageOffset);
}

} // namespace momentum::rasterizer
