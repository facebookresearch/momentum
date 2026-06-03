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
#include <drjit/matrix.h>
#include <drjit/util.h>

#include <cfloat>
#include <cmath>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace momentum::rasterizer {

Light createDirectionalLight(const Eigen::Vector3f& dir, const Eigen::Vector3f& color) {
  return {dir.normalized(), color, LightType::Directional};
}

Light createPointLight(const Eigen::Vector3f& pos, const Eigen::Vector3f& color) {
  return {pos, color, LightType::Point};
}

Light createAmbientLight(const Eigen::Vector3f& color) {
  return {Eigen::Vector3f::Zero(), color, LightType::Ambient};
}

Light transformLight(const Light& light, const Eigen::Affine3f& xf) {
  switch (light.type) {
    case LightType::Ambient:
      return light;
    case LightType::Directional:
      return {xf.linear() * light.position, light.color, LightType::Directional};
    case LightType::Point:
      return {xf * light.position, light.color, LightType::Point};
    default:
      MT_THROW("Unknown light type: {}", static_cast<int>(light.type));
  }
}

namespace {

inline void rasterizeOneLine(
    const Vector2f& p_line,
    const Vector2f& d_line,
    const float d_line_sqrLen,
    float recipW_start,
    float recipW_end,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Vector3f& color,
    float halfThickness,
    float nearClip,
    index_t rowStride,
    float depthOffset,
    float* zBufferPtr,
    float* rgbBufferPtr) {
  MT_THROW_IF(nearClip <= 0.0f, "Near clip must be positive");

  const int startX_block = startX / kSimdPacketSize;
  const int endX_block = endX / kSimdPacketSize;

  for (int32_t y = startY; y <= endY; ++y) {
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      // See notes in createVertexMatrix for how this interpolator gets used:
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

      const FloatP recipW = drjit::lerp(recipW_start, recipW_end, t);
      const FloatP w = 1.0f / recipW;
      const auto inFrontOfCameraMask = (w > nearClip);

      // Points behind the camera (with negative z) shouldn't get rendered:
      if (!drjit::any(inFrontOfCameraMask)) {
        continue;
      }

      const auto blockOffset = y * rowStride + xOffset;
      const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
      const FloatP zOffset = w + depthOffset;

      const auto zBufferMask = zOffset < zBufferOrig;
      const auto finalMask = zBufferMask && dist_mask && t_mask && inFrontOfCameraMask;
      if (!drjit::any(finalMask)) {
        continue;
      }

      const FloatP zBufferFinal = drjit::select(finalMask, zOffset, zBufferOrig);
      drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);

      if (rgbBufferPtr != nullptr) {
        float* rgbBufferOffset = rgbBufferPtr + 3 * blockOffset;
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        const Vector3fP rgbValuesFinal = drjit::select(finalMask, Vector3fP(color), rgbValuesOrig);
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }
    }
  }
}

inline void rasterizeOneCircle(
    const Vector3f& center_window,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Vector3f& lineColor,
    const Vector3f& fillColor,
    bool filled,
    float halfLineThickness,
    float radius,
    float nearClip,
    index_t rowStride,
    float depthOffset,
    float* zBufferPtr,
    float* rgbBufferPtr) {
  const int startX_block = startX / kSimdPacketSize;
  const int endX_block = endX / kSimdPacketSize;

  const float outerRadius = radius + halfLineThickness;
  const float innerRadius = radius - halfLineThickness;

  const float z_value = center_window.z();
  if (z_value < nearClip) {
    return;
  }

  for (int32_t y = startY; y <= endY; ++y) {
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      // See notes in createVertexMatrix for how this interpolator gets used:
      const FloatP p_window_x = (drjit::arange<FloatP>() + (float)xOffset);
      const float p_window_y = y;

      const Vector2fP p_cur(p_window_x, p_window_y);

      const FloatP dist = drjit::norm(p_cur - drjit::head<2>(center_window));
      auto combinedMask = dist <= outerRadius;
      auto lineMask = combinedMask && (dist >= innerRadius);
      if (!filled) {
        combinedMask &= lineMask;
      }

      if (!drjit::any(combinedMask)) {
        continue;
      }

      const auto blockOffset = y * rowStride + xOffset;
      const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
      const float zOffset = z_value + depthOffset;

      const auto zBufferMask = zOffset < zBufferOrig;
      combinedMask &= zBufferMask;
      lineMask &= zBufferMask;
      if (!drjit::any(combinedMask)) {
        continue;
      }

      const FloatP zBufferFinal = drjit::select(combinedMask, zOffset, zBufferOrig);
      drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);

      if (rgbBufferPtr != nullptr) {
        float* rgbBufferOffset = rgbBufferPtr + 3 * blockOffset;
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        Vector3fP rgbValuesFinal = drjit::select(lineMask, Vector3fP(lineColor), rgbValuesOrig);
        if (filled) {
          const auto fillMask = combinedMask && ~lineMask;
          rgbValuesFinal = drjit::select(fillMask, Vector3fP(fillColor), rgbValuesFinal);
        }
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }
    }
  }
}

void rasterizeOneLineP(
    const Vector3fP& lineStart_window,
    const Vector3fP& lineEnd_window,
    const SimdCamera& camera,
    const drjit::mask_t<IntP>& lineMask,
    float nearClip,
    const Vector3f& color,
    float thickness,
    index_t rowStride,
    float* zBufferPtr,
    float* rgbBufferPtr,
    float depthOffset) {
  const float halfThickness = 0.5f * thickness;
  const int halfThicknessPixels =
      std::clamp<int>(std::ceil(halfThickness), 1, camera.imageHeight() - 1);

  FloatP recipW_start = 1.0f / lineStart_window.z();
  FloatP recipW_end = 1.0f / lineEnd_window.z();

  Vector2fP p_line(lineStart_window.x(), lineStart_window.y());
  Vector2fP d_line(
      lineEnd_window.x() - lineStart_window.x(), lineEnd_window.y() - lineStart_window.y());
  FloatP d_line_sqrLen = drjit::squared_norm(d_line);

  auto validLines = lineMask && drjit::isfinite(recipW_start) && drjit::isfinite(recipW_end);

  const auto behindCamera = (lineStart_window.z() < nearClip) && (lineEnd_window.z() < nearClip);
  validLines = validLines && ~behindCamera;

  // Compute the bounds of the triangle in screen space:
  const IntP startX = drjit::floor2int<IntP, FloatP>(drjit::clip(
      drjit::minimum(lineStart_window.x(), lineEnd_window.x()) - halfThicknessPixels,
      0,
      camera.imageWidth() - 1));
  const IntP endX = drjit::ceil2int<IntP, FloatP>(drjit::clip(
      drjit::maximum(lineStart_window.x(), lineEnd_window.x()) + halfThicknessPixels,
      0,
      camera.imageWidth() - 1));

  const IntP startY = drjit::floor2int<IntP, FloatP>(drjit::clip(
      drjit::minimum(lineStart_window.y(), lineEnd_window.y()) - halfThicknessPixels,
      0,
      camera.imageHeight() - 1));
  const IntP endY = drjit::ceil2int<IntP, FloatP>(drjit::clip(
      drjit::maximum(lineStart_window.y(), lineEnd_window.y()) + halfThicknessPixels,
      0,
      camera.imageHeight() - 1));

  for (int lineOffset = 0; lineOffset < kSimdPacketSize; ++lineOffset) {
    if (!validLines[lineOffset]) {
      continue;
    }

    rasterizeOneLine(
        extractSingleElement(p_line, lineOffset),
        extractSingleElement(d_line, lineOffset),
        extractSingleElement(d_line_sqrLen, lineOffset),
        extractSingleElement(recipW_start, lineOffset),
        extractSingleElement(recipW_end, lineOffset),
        startX[lineOffset],
        endX[lineOffset],
        startY[lineOffset],
        endY[lineOffset],
        color,
        halfThickness,
        nearClip,
        rowStride,
        depthOffset,
        zBufferPtr,
        rgbBufferPtr);
  }
}

void rasterizeLinesImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  MT_THROW_IF(nearClip <= 0.0f, "Near clip must be positive");

  if (positions_world.size() == 0) {
    return;
  }

  const SimdCamera cameraSimd(camera, modelMatrix, imageOffset);
  checkBuffers(cameraSimd, zBuffer, rgbBuffer, {}, {}, {});

  if (positions_world.size() % 6 != 0) {
    throw std::runtime_error("Positions array should be an even number of points [start, end]");
  }

  const auto nVerts = positions_world.size() / 3;
  const auto nLines = nVerts / 2;
  if (nLines * 6 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_world.size()
        << " is not a multiple of six (start_x, start_y, start_z, end_x, end_y, end_z).";
    throw std::runtime_error(oss.str());
  }

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);

  const Vector3f color_drjit = toEnokiVec(color);

  for (auto [lineIndices, lineMask] : drjit::range<IntP>(nLines)) {
    const auto lineStart_world =
        drjit::gather<Vector3fP>(positions_world.data(), 2 * lineIndices + 0, lineMask);
    const auto lineEnd_world =
        drjit::gather<Vector3fP>(positions_world.data(), 2 * lineIndices + 1, lineMask);

    auto lineStart_eye = cameraSimd.worldToEye(lineStart_world);
    auto lineEnd_eye = cameraSimd.worldToEye(lineEnd_world);

    auto startBehindMask = lineStart_eye.z() < nearClip;
    auto endBehindMask = lineEnd_eye.z() < nearClip;
    auto crossesNearPlaneMask = startBehindMask ^ endBehindMask; // XOR for "not equal"

    // Clip lines that cross near plane: interpolate intersection point
    // Only update the endpoint that's behind the camera
    if (drjit::any(crossesNearPlaneMask)) {
      // Calculate intersection parameter (t) for all lines in batch
      auto t = (nearClip - lineStart_eye.z()) / (lineEnd_eye.z() - lineStart_eye.z());
      auto intersection = lineStart_eye + t * (lineEnd_eye - lineStart_eye);

      // Only update points that need it
      drjit::masked(lineStart_eye, crossesNearPlaneMask & startBehindMask) = intersection;
      drjit::masked(lineEnd_eye, crossesNearPlaneMask & ~startBehindMask) = intersection;
    }

    auto [lineStart_window, validProj1] = cameraSimd.eyeToWindow(lineStart_eye);
    auto [lineEnd_window, validProj2] = cameraSimd.eyeToWindow(lineEnd_eye);

    rasterizeOneLineP(
        lineStart_window,
        lineEnd_window,
        cameraSimd,
        IntP::MaskType(lineMask) && validProj1 && validProj2,
        nearClip,
        color_drjit,
        thickness,
        getRowStride(zBuffer),
        zBufferPtr,
        rgbBufferPtr,
        depthOffset);
  }
}

void rasterizeCirclesImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  if (positions_world.size() == 0) {
    return;
  }

  MT_THROW_IF(radius <= 0, "Radius must be greater than zero.");
  MT_THROW_IF(lineThickness < 0, "Line thickness must be non-negative.");

  const SimdCamera cameraSimd(camera, modelMatrix, imageOffset);
  checkBuffers(cameraSimd, zBuffer, rgbBuffer, {}, {}, {});

  const auto nCircles = positions_world.size() / 3;
  if (nCircles * 3 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_world.size()
        << " is not a multiple of three (center_x, center_y, center_z).";
    throw std::runtime_error(oss.str());
  }

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);

  const float halfLineThickness = lineColor.has_value() ? lineThickness / 2.f : 0.0f;

  const bool filled = fillColor.has_value();
  const Vector3f lineColor_drjit = toEnokiVec(lineColor.value_or(Eigen::Vector3f::Ones()));
  const Vector3f fillColor_drjit = toEnokiVec(fillColor.value_or(Eigen::Vector3f::Zero()));

  const float combinedRadius = radius + halfLineThickness;
  const int combinedRadiusPixels =
      std::clamp<int>(static_cast<int>(std::ceil(combinedRadius)), 1, cameraSimd.imageHeight() - 1);

  for (auto [circleIndices, circleMask] : drjit::range<IntP>(nCircles)) {
    const auto center_world =
        drjit::gather<Vector3fP>(positions_world.data(), circleIndices, circleMask);
    const auto [center_window, validProj] = cameraSimd.worldToWindow(center_world);

    const IntP::MaskType behindCamera = center_window.z() < nearClip;
    const auto validCircles = IntP::MaskType(circleMask) && ~behindCamera && validProj;

    // Compute the bounds of the triangle in screen space:
    const IntP startX = drjit::floor2int<IntP, FloatP>(
        drjit::clip(center_window.x() - combinedRadiusPixels, 0, cameraSimd.imageWidth() - 1));
    const IntP endX = drjit::ceil2int<IntP, FloatP>(
        drjit::clip(center_window.x() + combinedRadiusPixels, 0, cameraSimd.imageWidth() - 1));

    const IntP startY = drjit::floor2int<IntP, FloatP>(
        drjit::clip(center_window.y() - combinedRadiusPixels, 0, cameraSimd.imageHeight() - 1));
    const IntP endY = drjit::ceil2int<IntP, FloatP>(
        drjit::clip(center_window.y() + combinedRadiusPixels, 0, cameraSimd.imageHeight() - 1));

    for (int circleOffset = 0; circleOffset < kSimdPacketSize; ++circleOffset) {
      if (!validCircles[circleOffset]) {
        continue;
      }

      rasterizeOneCircle(
          extractSingleElement(center_window, circleOffset),
          startX[circleOffset],
          endX[circleOffset],
          startY[circleOffset],
          endY[circleOffset],
          lineColor_drjit,
          fillColor_drjit,
          filled,
          halfLineThickness,
          radius,
          nearClip,
          getRowStride(zBuffer),
          depthOffset,
          zBufferPtr,
          rgbBufferPtr);
    }
  }
}

template <typename TriangleT>
void rasterizeWireframeImp(
    Eigen::Ref<const Eigen::VectorXf> positions_world,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  const SimdCamera cameraSimd(camera, modelMatrix, imageOffset);
  MT_THROW_IF(nearClip <= 0, "near_clip must be greater than 0.");
  MT_THROW_IF(thickness <= 0, "thickness must be greater than 0.");

  checkBuffers(cameraSimd, zBuffer, rgbBuffer, {}, {}, {});

  if (triangles.size() == 0) {
    return;
  }

  const auto nVerts = positions_world.size() / 3;
  if (nVerts * 3 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_world.size() << " is not a multiple of three.";
    throw std::runtime_error(oss.str());
  }

  const auto nTriangles = triangles.size() / 3;
  validateTriangleIndices<TriangleT>(triangles, nVerts);

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);

  const Vector3f color_drjit = toEnokiVec(color);

  for (auto [triangleIndices, triangleMask] : drjit::range<IntP>(nTriangles)) {
    auto triangles_cur = drjit::gather<Vector3iP>(triangles.data(), triangleIndices, triangleMask);
    std::array<Vector3fP, 3> p_tri_window;
    IntP::MaskType validTriangles = triangleMask;
    for (int i = 0; i < 3; ++i) {
      auto p_world =
          drjit::gather<Vector3fP>(positions_world.data(), triangles_cur[i], triangleMask);
      auto [p_window, validProj] = cameraSimd.worldToWindow(p_world);
      p_tri_window[i] = p_window;
      validTriangles = validTriangles && validProj;
    }

    if (backfaceCulling) {
      // To do backface culling, we'll compute the signed area of the
      // triangle in window coordinates; this will tell us if it's wound
      // clockwise or counter-clockwise wrt the camera.
      const Vector2fP edge1 = drjit::head<2>(p_tri_window[1]) - drjit::head<2>(p_tri_window[0]);
      const Vector2fP edge2 = drjit::head<2>(p_tri_window[2]) - drjit::head<2>(p_tri_window[0]);

      const FloatP signedArea = edge1.y() * edge2.x() - edge1.x() * edge2.y();
      validTriangles = validTriangles && (signedArea > 0);
    }

    const auto behindCamera = p_tri_window[0].z() < nearClip && p_tri_window[1].z() < nearClip &&
        p_tri_window[2].z() < nearClip;
    validTriangles = validTriangles && ~behindCamera;
    if (!drjit::any(validTriangles)) {
      continue;
    }

    for (int kEdge = 0; kEdge < 3; ++kEdge) {
      rasterizeOneLineP(
          p_tri_window[kEdge],
          p_tri_window[(kEdge + 1) % 3],
          cameraSimd,
          validTriangles,
          nearClip,
          color_drjit,
          thickness,
          getRowStride(zBuffer),
          zBufferPtr,
          rgbBufferPtr,
          depthOffset);
    }
  }
}

} // namespace

index_t padImageWidthForRasterizer(index_t width) {
  return kSimdPacketSize * ((width + (kSimdPacketSize - 1)) / kSimdPacketSize);
}

Tensor2f makeRasterizerZBuffer(const Camera& camera) {
  return Tensor2f{
      {static_cast<index_t>(camera.imageHeight()), padImageWidthForRasterizer(camera.imageWidth())},
      FLT_MAX};
}

Tensor3f makeRasterizerRGBBuffer(const Camera& camera) {
  return Tensor3f{
      {static_cast<index_t>(camera.imageHeight()),
       padImageWidthForRasterizer(camera.imageWidth()),
       static_cast<index_t>(3)},
      0};
}

Tensor2i makeRasterizerIndexBuffer(const Camera& camera) {
  return Tensor2i(
      {static_cast<index_t>(camera.imageHeight()), padImageWidthForRasterizer(camera.imageWidth())},
      -1);
}

void rasterizeLines(
    std::span<const Eigen::Vector3f> positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeLinesImp(
      mapVector<float, 3>(positions_world),
      camera,
      modelMatrix,
      nearClip,
      color,
      thickness,
      zBuffer,
      rgbBuffer,
      depthOffset,
      imageOffset);
}

void rasterizeLines(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeLinesImp(
      positions_world,
      camera,
      modelMatrix,
      nearClip,
      color,
      thickness,
      zBuffer,
      rgbBuffer,
      depthOffset,
      imageOffset);
}

void rasterizeCircles(
    std::span<const Eigen::Vector3f> positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeCirclesImp(
      mapVector<float, 3>(positions_world),
      camera,
      modelMatrix,
      nearClip,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      zBuffer,
      rgbBuffer,
      depthOffset,
      imageOffset);
}

void rasterizeCircles(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeCirclesImp(
      positions_world,
      camera,
      modelMatrix,
      nearClip,
      lineColor,
      fillColor,
      lineThickness,
      radius,
      zBuffer,
      rgbBuffer,
      depthOffset,
      imageOffset);
}

void rasterizeWireframe(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3i> triangles,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeWireframeImp<int>(
      mapVector<float, 3>(positions_world),
      mapVector<int, 3>(triangles),
      camera,
      modelMatrix,
      nearClip,
      color,
      thickness,
      zBuffer,
      rgbBuffer,
      backfaceCulling,
      depthOffset,
      imageOffset);
}

void rasterizeWireframe(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXi>& triangles,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeWireframeImp<int>(
      positions_world,
      triangles,
      camera,
      modelMatrix,
      nearClip,
      color,
      thickness,
      zBuffer,
      rgbBuffer,
      backfaceCulling,
      depthOffset,
      imageOffset);
}

} // namespace momentum::rasterizer
