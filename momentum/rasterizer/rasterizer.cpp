/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstdlib>

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/math.h>
#include <drjit/matrix.h>
#include <drjit/util.h>
#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/utility.h>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

#include <momentum/common/exception.h>

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

template <typename T, int N>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> mapVector(
    std::span<const Eigen::Matrix<T, N, 1>> vec) {
  if (vec.empty()) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(nullptr, 0);
  } else {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        vec.front().data(), N * vec.size());
  }
}

// This is the vertex interpolation matrix, where we map from
// homgeneous coordinates to interpolate texture coordinates, etc.
// https://redirect.cs.umbc.edu/~olano/papers/2dh-tri/2dh-tri.pdf
// https://tayfunkayhan.wordpress.com/2018/12/30/rasterization-in-one-weekend-part-iii/
// In our old rasterizer code, we assumed that the homogeneous
// coordinates corresponded to points in eye space, since these are
// before the perspective divide.  However, with the full distortion
// camera model, going from window coordinates to eye coordinates is
// very expensive.  Now, the only reason we care about the specific
// homogeneous coordinates is to get perspective-correct
// interpolation.  We're therefore going to solve the issue by
// computing new pseudo-eye coordinates that ignore distortion,
// using only the focal length:
inline Matrix3fP createVertexMatrix(
    const std::array<Vector3fP, 3>& p_tri_window,
    const Camera& camera) {
  Matrix3fP result;
  for (int i = 0; i < 3; ++i) {
    // Pseudo-eye coordinates constructed by just multiplying with fx and
    // ignoring distortion:
    result(i, 0) = p_tri_window[i].x() / camera.fx() * p_tri_window[i].z();
    result(i, 1) = p_tri_window[i].y() / camera.fy() * p_tri_window[i].z();
    result(i, 2) = p_tri_window[i].z();
  }
  return result;
}

inline Vector3fP reflect(const Vector3fP& v, const Vector3fP& n) {
  const FloatP dotProd = drjit::dot(v, n);
  return (2.0f * dotProd) * n - v;
}

template <int N>
Matrix3f extractTriangleAttributes(
    const Eigen::Ref<const Eigen::VectorXf>& vec,
    const Eigen::Vector3i& triangle) {
  auto result = drjit::zeros<Matrix3f>();
  if (vec.size() == 0) {
    return result;
  }

  for (int jCol = 0; jCol < 3; ++jCol) {
    const Eigen::Matrix<float, N, 1> vAttr = vec.segment<N>(N * triangle[jCol]);
    for (int kRow = 0; kRow < N; ++kRow) {
      result(kRow, jCol) = vAttr(kRow);
    }
  }

  return result;
}

// Build a matrix from 3 column vectors:
template <typename T>
drjit::Matrix<T, 3> fromColumns(
    const drjit::Array<T, 3>& col0,
    const drjit::Array<T, 3>& col1,
    const drjit::Array<T, 3>& col2) {
  return drjit::Matrix<T, 3>(
      col0.x(), col1.x(), col2.x(), col0.y(), col1.y(), col2.y(), col0.z(), col1.z(), col2.z());
}

template <typename T>
drjit::Array<T, 3> extractColumn(const drjit::Matrix<T, 3>& mat, int col) {
  return drjit::Array<T, 3>(mat(0, col), mat(1, col), mat(2, col));
}

template <typename T>
drjit::Matrix<T, 3> toUniformMat(const drjit::Array<T, 3>& col) {
  return fromColumns(col, col, col);
}

Matrix3f toUniformMat(const Eigen::Vector3f& v) {
  return {v.x(), v.x(), v.x(), v.y(), v.y(), v.y(), v.z(), v.z(), v.z()};
}

inline Vector3fP sampleRGBTextureMap(
    const IntP& coord_x,
    const IntP& coord_y,
    const ConstSpan<float, 3>& textureMap,
    const FloatP::MaskType& mask) {
  const IntP offset = static_cast<int32_t>(textureMap.extent(1)) * coord_y + coord_x;
  return drjit::gather<Vector3fP>(textureMap.data_handle(), offset, mask);
}

Vector3fP interpolateRGBTextureMap(
    Vector2fP textureCoord,
    const ConstSpan<float, 3>& textureMap,
    const FloatP::MaskType& mask) {
  MT_THROW_IF(textureMap.extent(2) != 3, "Texture map must have 3 color channels");
  const int32_t textureWidth = textureMap.extent(1);
  const int32_t textureHeight = textureMap.extent(0);

  textureCoord = drjit::clip(textureCoord, 0.0f, 1.0f);

  const Vector2fP pixelCoord(
      (textureWidth - 1) * textureCoord.x(), (textureHeight - 1) * textureCoord.y());

  const Vector2fP offset = pixelCoord - drjit::floor(pixelCoord);
  const Vector2fP offset_inv = Vector2f(1.0f, 1.0f) - offset;

  const IntP pixelCoord_low_x = drjit::floor2int<IntP>(pixelCoord.x());
  const IntP pixelCoord_low_y = drjit::floor2int<IntP>(pixelCoord.y());
  const IntP pixelCoord_high_x(drjit::clip(pixelCoord_low_x + 1, 0, textureWidth - 1));
  const IntP pixelCoord_high_y(drjit::clip(pixelCoord_low_y + 1, 0, textureHeight - 1));

  return offset_inv.x() * offset_inv.y() *
      sampleRGBTextureMap(pixelCoord_low_x, pixelCoord_low_y, textureMap, mask) +
      offset_inv.x() * offset.y() *
      sampleRGBTextureMap(pixelCoord_low_x, pixelCoord_high_y, textureMap, mask) +
      offset.x() * offset_inv.y() *
      sampleRGBTextureMap(pixelCoord_high_x, pixelCoord_low_y, textureMap, mask) +
      offset.x() * offset.y() *
      sampleRGBTextureMap(pixelCoord_high_x, pixelCoord_high_y, textureMap, mask);
}

inline Vector3fP shade(
    const Light& l,
    const PhongMaterial& material,
    const Vector3fP& diffuseColor,
    const Vector3fP& p_eye,
    const Vector3fP& n_eye,
    bool hasSpecular) {
  auto result = drjit::zeros<Vector3fP>();
  // Vector pointing from surface toward light:
  if (l.type == LightType::Ambient) {
    result += Vector3fP(
        diffuseColor.x() * l.color.x(),
        diffuseColor.y() * l.color.y(),
        diffuseColor.z() * l.color.z());
  } else {
    const Vector3fP light_vec = l.type == LightType::Directional
        ? Vector3fP(-toEnokiVec(l.position.head<3>()))
        : drjit::normalize(toEnokiVec(l.position) - p_eye);
    // Vector pointing from surface toward camera:
    const Vector3fP view_vec = -drjit::normalize(p_eye);

    const FloatP intensity = drjit::clip(drjit::dot(light_vec, n_eye), 0, 1);
    result += Vector3fP(
        intensity * diffuseColor.x() * l.color.x(),
        intensity * diffuseColor.y() * l.color.y(),
        intensity * diffuseColor.z() * l.color.z());
    if (hasSpecular) {
      const Vector3fP reflected_vec = reflect(light_vec, n_eye);
      const FloatP specularIntensity = drjit::pow(
          drjit::clip(drjit::dot(reflected_vec, view_vec), 0.0f, 1.0f), material.specularExponent);
      result += Vector3fP(
          specularIntensity * material.specularColor.x() * l.color.x(),
          specularIntensity * material.specularColor.y() * l.color.y(),
          specularIntensity * material.specularColor.z() * l.color.z());
    }
  }

  return result;
}

inline void rasterizeOneTriangle(
    const int32_t triangleIndex,
    const Eigen::Vector3i& triangle,
    const SimdCamera& camera,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Vector3f& recipWInterp,
    const Matrix3f& vertexMatrixInverse,
    const Matrix3f& p_eye,
    const Matrix3f& n_eye,
    const Matrix3f& uv,
    const Matrix3f& perVertexDiffuseColor,
    const PhongMaterial& material,
    std::span<const Light> lights_eye,
    float nearClip,
    index_t zBufferRowStride,
    float depthOffset,
    float* zBufferPtr,
    float* rgbBufferPtr,
    float* surfaceNormalsBufferPtr,
    int* vertexIndexBufferPtr,
    int* triangleIndexBufferPtr,
    bool filterBySplatRadius) {
  const int startX_block = startX / kSimdPacketSize;
  const int endX_block = endX / kSimdPacketSize;

  const bool hasDiffuseMap = !material.diffuseTextureMap.empty();
  const bool hasEmissiveMap = !material.emissiveTextureMap.empty();
  const bool hasDiffuse = !material.diffuseColor.isZero() || hasDiffuseMap;
  const bool hasSpecular = !material.specularColor.isZero() && material.specularExponent != 0;

  const float fx = camera.fx();
  const float fy = camera.fy();

  for (int32_t y = startY; y <= endY; ++y) {
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      // See notes in createVertexMatrix for how this interpolator gets used:
      const FloatP p_ndc_x = (drjit::arange<FloatP>() + (float)xOffset) / fx;
      const float p_ndc_y = y / fy;

      const FloatP recipW =
          recipWInterp.x() * p_ndc_x + recipWInterp.y() * p_ndc_y + recipWInterp.z();

      // Need to maximize accuracy here so it's not safe to use the fast reciprocal:
      const FloatP w = 1.0f / recipW;

      // Points behind the camera (with negative z) shouldn't get rendered:
      const auto inFrontOfCameraMask = drjit::isfinite(w) && (w > nearClip);
      if (!drjit::any(inFrontOfCameraMask)) {
        continue;
      }

      // convert screen xyw to barycentrics
      //    bary_x = edge0.dot(p_ndc) * w
      //           = w * (edge0.x() * p_ndc.x() + edge0.y() * p_ndc.y() +
      //           edge0.z() * 1)
      const Vector3fP bary = w * (vertexMatrixInverse * Vector3fP(p_ndc_x, p_ndc_y, 1.0f));

      // Set the bits that are less than 0:
      const auto baryMask = (bary.x() >= 0 && bary.y() >= 0 && bary.z() >= 0);

      const auto blockOffset = y * zBufferRowStride + xOffset;
      const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
      const FloatP zOffset = w + depthOffset;

      const auto zBufferMask = zOffset < zBufferOrig;

      auto finalMask = zBufferMask && baryMask && inFrontOfCameraMask;
      if (filterBySplatRadius) {
        const Vector2fP textureCoord(
            uv(0, 0) * bary.x() + uv(0, 1) * bary.y() + uv(0, 2) * bary.z(),
            uv(1, 0) * bary.x() + uv(1, 1) * bary.y() + uv(1, 2) * bary.z());
        finalMask &= (drjit::square(textureCoord.x()) + drjit::square(textureCoord.y())) < 1.0f;
      }

      if (!drjit::any(finalMask)) {
        continue;
      }

      const FloatP zBufferFinal = drjit::select(finalMask, zOffset, zBufferOrig);
      drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);

      if (rgbBufferPtr != nullptr) {
        Vector3fP shaded = toEnokiVec(material.emissiveColor);

        const Vector2fP textureCoord(
            uv(0, 0) * bary.x() + uv(0, 1) * bary.y() + uv(0, 2) * bary.z(),
            uv(1, 0) * bary.x() + uv(1, 1) * bary.y() + uv(1, 2) * bary.z());

        if (hasEmissiveMap) {
          shaded +=
              interpolateRGBTextureMap(textureCoord, material.emissiveTextureMap.view(), finalMask);
        }

        if (hasDiffuse || hasSpecular) {
          Vector3fP diffuseColor = perVertexDiffuseColor * bary;
          if (hasDiffuseMap) {
            diffuseColor = interpolateRGBTextureMap(
                textureCoord, material.diffuseTextureMap.view(), finalMask);
          }

          for (const auto& l : lights_eye) {
            const Vector3fP p_eye_interp = p_eye * bary;
            const Vector3fP n_eye_interp = drjit::normalize(n_eye * bary);
            shaded += shade(l, material, diffuseColor, p_eye_interp, n_eye_interp, hasSpecular);
          }
        }

        shaded = drjit::clip(shaded, 0.0f, 1.0f);

        float* rgbBufferOffset = rgbBufferPtr + 3 * blockOffset;
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        const Vector3fP rgbValuesFinal = drjit::select(finalMask, shaded, rgbValuesOrig);
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }

      if (vertexIndexBufferPtr != nullptr) {
        int* vertexIndexBufferOffset = vertexIndexBufferPtr + blockOffset;
        const IntP vertexValuesOrig = drjit::load_aligned<IntP>(vertexIndexBufferOffset);
        // Select the vertex based on the largest barycentric coordinates:
        const IntP vertexValuesNew = drjit::select(
            bary.x() > bary.y() && bary.x() > bary.z(),
            IntP(triangle.x()),
            drjit::select(bary.y() > bary.z(), IntP(triangle.y()), IntP(triangle.z())));

        const IntP vertexValuesFinal = drjit::select(finalMask, vertexValuesNew, vertexValuesOrig);
        drjit::store_aligned<IntP>(vertexIndexBufferOffset, vertexValuesFinal);
      }

      if (triangleIndexBufferPtr != nullptr) {
        int* triangleIndexBufferOffset = triangleIndexBufferPtr + blockOffset;
        const IntP triangleValuesOrig = drjit::load_aligned<IntP>(triangleIndexBufferOffset);
        const IntP triangleValuesNew(triangleIndex);
        const IntP triangleValuesFinal =
            drjit::select(finalMask, triangleValuesNew, triangleValuesOrig);
        drjit::store_aligned<IntP>(triangleIndexBufferOffset, triangleValuesFinal);
      }

      if (surfaceNormalsBufferPtr != nullptr) {
        const Vector3fP n_eye_interp = drjit::normalize(n_eye * bary);

        float* surfaceNormalsBufferOffset = surfaceNormalsBufferPtr + 3 * blockOffset;
        const auto scatterIndices = drjit::arange<UintP>();
        const auto surfaceNormalValuesOrig =
            drjit::gather<Vector3fP>(surfaceNormalsBufferOffset, scatterIndices);
        const Vector3fP surfaceNormalValuesFinal =
            drjit::select(finalMask, n_eye_interp, surfaceNormalValuesOrig);
        drjit::scatter(surfaceNormalsBufferOffset, surfaceNormalValuesFinal, scatterIndices);
      }
    }
  }
}

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
    index_t zBufferWidth,
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

      const uint32_t blockOffset = y * zBufferWidth + xOffset;
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
    index_t zBufferWidth,
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

      const uint32_t blockOffset = y * zBufferWidth + xOffset;
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
    index_t bufferWidth,
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
        float* rgbBufferOffset = rgbBufferPtr + 3 * (y * bufferWidth + xOffset);
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
        const uint32_t blockOffset = y * bufferWidth + xOffset;
        const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
        const FloatP zBufferFinal = drjit::select(combinedMask, FloatP(0.0f), zBufferOrig);
        drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);
      }
    }
  }
}

void checkBuffers(
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

  // Validate buffer layout and alignment using isValidBuffer function
  if (!isValidBuffer(zBuffer)) {
    throw std::runtime_error(
        "Z buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  if (!rgbBuffer.empty() && !isValidBuffer(rgbBuffer)) {
    throw std::runtime_error(
        "RGB buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  if (!surfaceNormalsBuffer.empty() && !isValidBuffer(surfaceNormalsBuffer)) {
    throw std::runtime_error(
        "Surface normals buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  if (!vertexIndexBuffer.empty() && !isValidBuffer(vertexIndexBuffer)) {
    throw std::runtime_error(
        "Vertex index buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }

  if (!triangleIndexBuffer.empty() && !isValidBuffer(triangleIndexBuffer)) {
    throw std::runtime_error(
        "Triangle index buffer has invalid layout or alignment. All dimensions except the first must be "
        "contiguous, and the first dimension stride must be a multiple of " +
        std::to_string(kSimdPacketSize) + " for SIMD operations.");
  }
}

template <typename TriangleT>
void validateTriangleIndices(
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

// Support both signed and unsigned triangles for backward compatibility:
template <typename TriangleT>
void rasterizeMeshImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer,
    std::vector<Light> lights_eye,
    const PhongMaterial& material,
    Eigen::Matrix4f modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    Eigen::Vector2f imageOffset) {
  MT_THROW_IF(nearClip <= 0.0f, "Near clip must be positive");

  if (triangles.size() == 0) {
    return;
  }

  if (lights_eye.empty()) {
    // Default lighting setup: light co-located with camera:
    lights_eye.emplace_back();
  }

  const SimdCamera cameraSimd(camera, std::move(modelMatrix), std::move(imageOffset));
  checkBuffers(
      cameraSimd, zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);

  const int32_t imageWidth = cameraSimd.imageWidth();
  const int32_t imageHeight = cameraSimd.imageHeight();

  if (normals_world.size() != 0 && positions_world.size() != normals_world.size()) {
    throw std::runtime_error("positions size doesn't match normals size");
  }

  if (perVertexDiffuseColor.size() != 0 && positions_world.size() != perVertexDiffuseColor.size()) {
    throw std::runtime_error("per-vertex color size doesn't match normals size");
  }

  const bool usePerVertexColor = (perVertexDiffuseColor.size() != 0);

  const auto nTriangles = triangles.size() / 3;
  const auto nVerts = positions_world.size() / 3;
  if (nVerts * 3 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Positions array of size " << positions_world.size() << " is not a multiple of three.";
    throw std::runtime_error(oss.str());
  }

  validateTriangleIndices<TriangleT>(triangles, nVerts);

  const bool hasTextureMap = (textureCoords.size() != 0 && material.hasTextureMap());
  if (hasTextureMap) {
    // Check texture map invariants:
    if (textureTriangles.size() != 0) {
      const auto nTextureVerts = textureCoords.size() / 2;
      if (textureTriangles.size() != triangles.size()) {
        throw std::runtime_error("texture_triangles size should match triangles size");
      }

      validateTriangleIndices<TriangleT>(
          textureTriangles, nTextureVerts, "texture_coordinate", "triangles");
    } else {
      // Assume regular triangles are also texture triangles:
      if (textureCoords.size() != 2 * nVerts) {
        throw std::runtime_error(
            "texture_triangles not provided so texture_coordinates size should match vertices size");
      }
    }
  }

  if (perVertexDiffuseColor.size() != 0 && !material.diffuseTextureMap.empty()) {
    throw std::runtime_error("Can't provide both per-vertex color and diffuse texture map.");
  }

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = rgbBuffer.empty() ? nullptr : rgbBuffer.data_handle();
  float* surfaceNormalsBufferPtr =
      surfaceNormalsBuffer.empty() ? nullptr : surfaceNormalsBuffer.data_handle();
  int* vertexIndexBufferPtr = vertexIndexBuffer.empty() ? nullptr : vertexIndexBuffer.data_handle();
  int* triangleIndexBufferPtr =
      triangleIndexBuffer.empty() ? nullptr : triangleIndexBuffer.data_handle();

  for (auto [triangleIndices, triangleMask] : drjit::range<IntP>(nTriangles)) {
    auto triangles_cur = drjit::gather<Vector3iP>(triangles.data(), triangleIndices, triangleMask);
    std::array<Vector3fP, 3> p_tri_window;
    Matrix3fP p_tri_eye;

    IntP::MaskType validTriangles = triangleMask;
    for (int i = 0; i < 3; ++i) {
      auto p_world =
          drjit::gather<Vector3fP>(positions_world.data(), triangles_cur[i], triangleMask);
      Vector3fP p_eye = cameraSimd.worldToEye(p_world);
      auto [p_window, validProj] = cameraSimd.eyeToWindow(p_eye);

      p_tri_window[i] = p_window;
      for (int j = 0; j < 3; ++j) {
        p_tri_eye(j, i) = p_eye[j];
      }
      validTriangles = validTriangles && validProj;
    }

    if (backfaceCulling) {
      // To do backface culling, we'll compute the signed area of the
      // triangle in window coordinates; this will tell us if it's wound
      // clockwise or counter-clockwise wrt the camera.
      const Vector2fP edge1 = drjit::head<2>(p_tri_window[1]) - drjit::head<2>(p_tri_window[0]);
      const Vector2fP edge2 = drjit::head<2>(p_tri_window[2]) - drjit::head<2>(p_tri_window[0]);

      const FloatP signedArea = edge1.y() * edge2.x() - edge1.x() * edge2.y();
      validTriangles = validTriangles && IntP::MaskType(signedArea > 0);
      if (!drjit::any(validTriangles)) {
        continue;
      }
    }

    // This is the vertex interpolation matrix, where we map from
    // homgeneous coordinates to interpolate texture coordinates, etc.  See
    // note above about its construction:
    const Matrix3fP vertexMatrix = createVertexMatrix(p_tri_window, camera);
    // Use double precision here for extra precision, otherwise we get holes in
    // the mesh since the inverse function uses a relatively numerically
    // unstable algorithm using determinants since it's easy to SIMD.
    const Matrix3dP vertexMatrixInverse = drjit::inverse(Matrix3dP(vertexMatrix));

    const Vector3dP recipWInterp(
        vertexMatrixInverse(0, 0) + vertexMatrixInverse(0, 1) + vertexMatrixInverse(0, 2),
        vertexMatrixInverse(1, 0) + vertexMatrixInverse(1, 1) + vertexMatrixInverse(1, 2),
        vertexMatrixInverse(2, 0) + vertexMatrixInverse(2, 1) + vertexMatrixInverse(2, 2));
    validTriangles = validTriangles && drjit::isfinite(recipWInterp.x()) &&
        drjit::isfinite(recipWInterp.y()) && drjit::isfinite(recipWInterp.z());
    if (!drjit::any(validTriangles)) {
      continue;
    }

    const auto behindCamera = p_tri_window[0].z() < nearClip && p_tri_window[1].z() < nearClip &&
        p_tri_window[2].z() < nearClip;
    validTriangles = validTriangles && ~behindCamera;

    // Compute the bounds of the triangle in screen space:
    const IntP startX = drjit::floor2int<IntP, FloatP>(drjit::clip(
        drjit::minimum(
            drjit::minimum(p_tri_window[0].x(), p_tri_window[1].x()), p_tri_window[2].x()),
        0,
        imageWidth - 1));
    const IntP endX = drjit::ceil2int<IntP, FloatP>(drjit::clip(
        drjit::maximum(
            drjit::maximum(p_tri_window[0].x(), p_tri_window[1].x()), p_tri_window[2].x()),
        0,
        imageWidth - 1));

    const IntP startY = drjit::floor2int<IntP, FloatP>(drjit::clip(
        drjit::minimum(
            drjit::minimum(p_tri_window[0].y(), p_tri_window[1].y()), p_tri_window[2].y()),
        0,
        imageHeight - 1));
    const IntP endY = drjit::ceil2int<IntP, FloatP>(drjit::clip(
        drjit::maximum(
            drjit::maximum(p_tri_window[0].y(), p_tri_window[1].y()), p_tri_window[2].y()),
        0,
        imageHeight - 1));

    Matrix3fP n_tri_eye;
    if (normals_world.size() == 0) {
      const Vector3fP n_eye = drjit::normalize(
          drjit::cross(
              extractColumn(p_tri_eye, 1) - extractColumn(p_tri_eye, 0),
              extractColumn(p_tri_eye, 2) - extractColumn(p_tri_eye, 0)));
      n_tri_eye = toUniformMat(n_eye);
    } else {
      for (int i = 0; i < 3; ++i) {
        auto n_world =
            drjit::gather<Vector3fP>(normals_world.data(), triangles_cur[i], triangleMask);
        const Vector3fP n_eye = cameraSimd.worldToEyeNormal(n_world);
        for (int j = 0; j < 3; ++j) {
          n_tri_eye(j, i) = n_eye[j];
        }
      }
    }

    const Matrix3f diffuseColor = toUniformMat(material.diffuseColor);

    for (int triOffset = 0; triOffset < kSimdPacketSize; ++triOffset) {
      if (!validTriangles[triOffset]) {
        continue;
      }

      const auto iTriangle = triangleIndices[triOffset];
      const Eigen::Vector3i triangle =
          triangles.template segment<3>(3 * iTriangle).template cast<int32_t>();
      const Eigen::Vector3i textureTriangle = textureTriangles.size() == 0
          ? triangle
          : textureTriangles.template segment<3>(3 * iTriangle).template cast<int32_t>();

      rasterizeOneTriangle(
          iTriangle,
          triangle,
          cameraSimd,
          startX[triOffset],
          endX[triOffset],
          startY[triOffset],
          endY[triOffset],
          extractSingleElement(recipWInterp, triOffset),
          drjit::transpose(extractSingleElement(vertexMatrixInverse, triOffset)),
          extractSingleElement(p_tri_eye, triOffset),
          extractSingleElement(n_tri_eye, triOffset),
          extractTriangleAttributes<2>(textureCoords, textureTriangle),
          usePerVertexColor ? extractTriangleAttributes<3>(perVertexDiffuseColor, triangle)
                            : diffuseColor,
          material,
          lights_eye,
          nearClip,
          getRowStride(zBuffer),
          depthOffset,
          zBufferPtr,
          rgbBufferPtr,
          surfaceNormalsBufferPtr,
          vertexIndexBufferPtr,
          triangleIndexBufferPtr,
          false);
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
        camera.imageWidth(), // Use the logical image width since this is a temporary workaround
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
  float* rgbBufferPtr = rgbBuffer.empty() ? nullptr : rgbBuffer.data_handle();

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

    // Only process if any lines cross the near plane
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
  float* rgbBufferPtr = rgbBuffer.empty() ? nullptr : rgbBuffer.data_handle();

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

// Support both signed and unsigned triangles for backward compatibility:
void rasterizeSplatsImp(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Camera& camera,
    Span2f zBuffer,
    Span3f rgbBuffer,
    std::vector<Light> lights_eye,
    const PhongMaterial& frontMaterial,
    const PhongMaterial& backMaterial,
    float radius,
    Eigen::Matrix4f modelMatrix,
    float nearClip,
    float depthOffset,
    Eigen::Vector2f imageOffset) {
  MT_THROW_IF(nearClip <= 0.0f, "Near clip must be positive");
  MT_THROW_IF(radius <= 0.0f, "radius must be positive");

  if (positions_world.size() == 0) {
    return;
  }

  if (lights_eye.empty()) {
    // Default lighting setup: light co-located with camera:
    lights_eye.emplace_back();
  }

  if (lights_eye.empty()) {
    // Default lighting setup: light co-located with camera:
    lights_eye.emplace_back();
  }

  const SimdCamera cameraSimd(camera, std::move(modelMatrix), std::move(imageOffset));
  checkBuffers(cameraSimd, zBuffer, rgbBuffer, {}, {}, {});

  const int32_t imageWidth = cameraSimd.imageWidth();
  const int32_t imageHeight = cameraSimd.imageHeight();

  if (positions_world.size() != normals_world.size()) {
    throw std::runtime_error("positions size doesn't match normals size");
  }

  const auto nSplats = positions_world.size() / 3;
  if (nSplats * 3 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Vertex positions size must be a multiple of 3.";
    throw std::runtime_error(oss.str());
  }

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = rgbBuffer.empty() ? nullptr : rgbBuffer.data_handle();

  const Matrix3f diffuseColor_back = toUniformMat(backMaterial.diffuseColor);
  const Matrix3f diffuseColor_front = toUniformMat(frontMaterial.diffuseColor);

  const std::array<Vector3f, 4> quadTextureCoords = {
      Vector3f(-1, -1, 0), Vector3f(1, -1, 0), Vector3f(1, 1, 0), Vector3f(-1, 1, 0)};

  for (auto [splatIndices, splatMask] : drjit::range<IntP>(nSplats)) {
    auto position_world = drjit::gather<Vector3fP>(positions_world.data(), splatIndices, splatMask);
    auto normal_world = drjit::gather<Vector3fP>(normals_world.data(), splatIndices, splatMask);

    Vector3fP dir1 = drjit::normalize(drjit::cross(normal_world, Vector3f(1, 1, 1)));
    const Vector3fP dir2 = drjit::normalize(drjit::cross(normal_world, dir1));
    dir1 = drjit::normalize(drjit::cross(dir2, normal_world));

    const std::array<Vector3fP, 4> p_quad_world = {
        position_world - radius * dir1 - radius * dir2,
        position_world + radius * dir1 - radius * dir2,
        position_world + radius * dir1 + radius * dir2,
        position_world - radius * dir1 + radius * dir2,
    };

    auto validSplats = IntP::MaskType(splatMask) && drjit::isfinite(normal_world.x());

    std::array<Vector3fP, 4> p_quad_eye;
    std::array<Vector3fP, 4> p_quad_window;
    for (int i = 0; i < 4; ++i) {
      p_quad_eye[i] = cameraSimd.worldToEye(p_quad_world[i]);
      auto [p_window, validProj] = cameraSimd.eyeToWindow(p_quad_eye[i]);
      p_quad_window[i] = p_window;
      validSplats = validSplats && validProj;
    }

    // To do backface culling, we'll compute the signed area of the
    // triangle in window coordinates; this will tell us if it's wound
    // clockwise or counter-clockwise wrt the camera.
    const Vector2fP edge1 = drjit::head<2>(p_quad_window[1]) - drjit::head<2>(p_quad_window[0]);
    const Vector2fP edge2 = drjit::head<2>(p_quad_window[2]) - drjit::head<2>(p_quad_window[0]);
    const FloatP signedArea = edge1.y() * edge2.x() - edge1.x() * edge2.y();
    const auto backFace = (signedArea < 0);

    const Vector3fP normal_eye = cameraSimd.worldToEyeNormal(normal_world) *
        drjit::select(backFace, FloatP(-1.0f), FloatP(1.0f));

    const auto behindCamera = p_quad_window[0].z() < nearClip && p_quad_window[1].z() < nearClip &&
        p_quad_window[2].z() < nearClip && p_quad_window[3].z() < nearClip;
    validSplats = validSplats && ~behindCamera;
    if (!drjit::any(validSplats)) {
      continue;
    }

    // Divide the quad into two triangles:
    for (int iTriangle = 0; iTriangle < 2; ++iTriangle) {
      const Eigen::Vector3i triangle =
          (iTriangle == 0) ? Eigen::Vector3i(0, 1, 2) : Eigen::Vector3i(0, 2, 3);

      const std::array<Vector3fP, 3> p_tri_window = {
          p_quad_window[triangle.x()], p_quad_window[triangle.y()], p_quad_window[triangle.z()]};

      auto validTriangles = validSplats;
      // See above for explanation of the vertex interpolation matrix.
      const Matrix3fP vertexMatrix = createVertexMatrix(p_tri_window, camera);
      const Matrix3dP vertexMatrixInverse = drjit::inverse(Matrix3dP(vertexMatrix));
      const Vector3dP recipWInterp(
          vertexMatrixInverse(0, 0) + vertexMatrixInverse(0, 1) + vertexMatrixInverse(0, 2),
          vertexMatrixInverse(1, 0) + vertexMatrixInverse(1, 1) + vertexMatrixInverse(1, 2),
          vertexMatrixInverse(2, 0) + vertexMatrixInverse(2, 1) + vertexMatrixInverse(2, 2));

      validTriangles = validTriangles && drjit::isfinite(recipWInterp.x()) &&
          drjit::isfinite(recipWInterp.y()) && drjit::isfinite(recipWInterp.z());
      if (!drjit::any(validTriangles)) {
        continue;
      }

      // Compute the bounds of the triangle in screen space:
      const IntP startX = drjit::floor2int<IntP, FloatP>(drjit::clip(
          drjit::minimum(
              drjit::minimum(p_tri_window[0].x(), p_tri_window[1].x()), p_tri_window[2].x()),
          0,
          imageWidth - 1));
      const IntP endX = drjit::ceil2int<IntP, FloatP>(drjit::clip(
          drjit::maximum(
              drjit::maximum(p_tri_window[0].x(), p_tri_window[1].x()), p_tri_window[2].x()),
          0,
          imageWidth - 1));

      const IntP startY = drjit::floor2int<IntP, FloatP>(drjit::clip(
          drjit::minimum(
              drjit::minimum(p_tri_window[0].y(), p_tri_window[1].y()), p_tri_window[2].y()),
          0,
          imageHeight - 1));
      const IntP endY = drjit::ceil2int<IntP, FloatP>(drjit::clip(
          drjit::maximum(
              drjit::maximum(p_tri_window[0].y(), p_tri_window[1].y()), p_tri_window[2].y()),
          0,
          imageHeight - 1));

      const Matrix3fP pos_eye_tri =
          fromColumns(p_quad_eye[triangle.x()], p_quad_eye[triangle.y()], p_quad_eye[triangle.z()]);

      const Matrix3fP n_eye_tri = toUniformMat(normal_eye);

      const Matrix3f texCoords_tri = fromColumns(
          quadTextureCoords[triangle.x()],
          quadTextureCoords[triangle.y()],
          quadTextureCoords[triangle.z()]);

      for (int triOffset = 0; triOffset < kSimdPacketSize; ++triOffset) {
        if (!validTriangles[triOffset]) {
          continue;
        }

        rasterizeOneTriangle(
            iTriangle,
            triangle,
            cameraSimd,
            startX[triOffset],
            endX[triOffset],
            startY[triOffset],
            endY[triOffset],
            extractSingleElement(recipWInterp, triOffset),
            drjit::transpose(extractSingleElement(vertexMatrixInverse, triOffset)),
            extractSingleElement(pos_eye_tri, triOffset),
            extractSingleElement(n_eye_tri, triOffset),
            texCoords_tri,
            backFace[triOffset] ? diffuseColor_back : diffuseColor_front,
            backFace[triOffset] ? backMaterial : frontMaterial,
            lights_eye,
            nearClip,
            getRowStride(zBuffer),
            depthOffset,
            zBufferPtr,
            rgbBufferPtr,
            nullptr,
            nullptr,
            nullptr,
            true);
      }
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
  float* rgbBufferPtr = rgbBuffer.empty() ? nullptr : rgbBuffer.data_handle();

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

void rasterizeMesh(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Ref<const Eigen::VectorXi>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::VectorXi>& textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer,
    const std::vector<Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeMeshImp<int32_t>(
      positions_world,
      normals_world,
      triangles,
      textureCoords,
      textureTriangles,
      perVertexDiffuseColor,
      camera,
      zBuffer,
      rgbBuffer,
      surfaceNormalsBuffer,
      vertexIndexBuffer,
      triangleIndexBuffer,
      lights_eye,
      material,
      modelMatrix,
      backfaceCulling,
      nearClip,
      depthOffset,
      imageOffset);
}

void rasterizeMesh(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
    std::span<const Eigen::Matrix<uint32_t, 3, 1>> triangles,
    std::span<const Eigen::Vector2f> textureCoords,
    std::span<const Eigen::Matrix<uint32_t, 3, 1>> textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer,
    const std::vector<Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeMeshImp<uint32_t>(
      mapVector<float>(positions_world),
      mapVector<float>(normals_world),
      mapVector<uint32_t>(triangles),
      mapVector<float>(textureCoords),
      mapVector<uint32_t>(textureTriangles),
      perVertexDiffuseColor,
      camera,
      zBuffer,
      rgbBuffer,
      surfaceNormalsBuffer,
      vertexIndexBuffer,
      triangleIndexBuffer,
      lights_eye,
      material,
      modelMatrix,
      backfaceCulling,
      nearClip,
      depthOffset,
      imageOffset);
}

void rasterizeMesh(
    const Mesh& mesh,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer,
    const std::vector<Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeMeshImp<int>(
      mapVector<float, 3>(mesh.vertices),
      mapVector<float, 3>(mesh.normals),
      mapVector<int, 3>(mesh.faces),
      mapVector<float, 2>(mesh.texcoords),
      mapVector<int, 3>(mesh.texcoord_faces),
      Eigen::VectorXf{},
      camera,
      zBuffer,
      rgbBuffer,
      surfaceNormalsBuffer,
      vertexIndexBuffer,
      triangleIndexBuffer,
      lights_eye,
      material,
      modelMatrix,
      backfaceCulling,
      nearClip,
      depthOffset,
      imageOffset);
}

void rasterizeMesh(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
    std::span<const Eigen::Vector3i> triangles,
    std::span<const Eigen::Vector2f> textureCoords,
    std::span<const Eigen::Vector3i> textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer,
    Span3f surfaceNormalsBuffer,
    Span2i vertexIndexBuffer,
    Span2i triangleIndexBuffer,
    const std::vector<Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeMeshImp<int32_t>(
      mapVector<float, 3>(positions_world),
      mapVector<float, 3>(normals_world),
      mapVector<int32_t, 3>(triangles),
      mapVector<float, 2>(textureCoords),
      mapVector<int32_t, 3>(textureTriangles),
      perVertexDiffuseColor,
      camera,
      zBuffer,
      rgbBuffer,
      surfaceNormalsBuffer,
      vertexIndexBuffer,
      triangleIndexBuffer,
      lights_eye,
      material,
      modelMatrix,
      backfaceCulling,
      nearClip,
      depthOffset,
      imageOffset);
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

void rasterizeSplats(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& frontMaterial,
    const PhongMaterial& backMaterial,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer,
    const std::vector<Light>& lights_eye,
    float depthOffset,
    const Eigen::Vector2f& imageOffset) {
  rasterizeSplatsImp(
      mapVector<float, 3>(positions_world),
      mapVector<float, 3>(normals_world),
      camera,
      zBuffer,
      rgbBuffer,
      lights_eye,
      frontMaterial,
      backMaterial,
      radius,
      modelMatrix,
      nearClip,
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

// 2D rasterization functions that operate directly in image space
// without camera projection or z-buffer

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
    index_t bufferWidth,
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
        float* rgbBufferOffset = rgbBufferPtr + 3 * (y * bufferWidth + xOffset);
        const auto scatterIndices = drjit::arange<UintP>();
        const auto rgbValuesOrig = drjit::gather<Vector3fP>(rgbBufferOffset, scatterIndices);
        const Vector3fP rgbValuesFinal = drjit::select(finalMask, Vector3fP(color), rgbValuesOrig);
        drjit::scatter(rgbBufferOffset, rgbValuesFinal, scatterIndices);
      }

      // Write zeros to z-buffer for alpha matting
      if (zBufferPtr != nullptr) {
        const uint32_t blockOffset = y * bufferWidth + xOffset;
        const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
        const FloatP zBufferFinal = drjit::select(finalMask, FloatP(0.0f), zBufferOrig);
        drjit::store_aligned<FloatP>(zBufferPtr + blockOffset, zBufferFinal);
      }
    }
  }
}

// rasterizeOneCircle2D is now handled by rasterizeOneCircle with zBufferPtr=nullptr

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

  const int32_t imageHeight = rgbBuffer.extent(0);
  const int32_t imageWidth = rgbBuffer.extent(1);

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

    float* zBufferPtr = zBuffer.empty() ? nullptr : zBuffer.data_handle();

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
        getRowStride(rgbBuffer),
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

  const int32_t imageHeight = rgbBuffer.extent(0);
  const int32_t imageWidth = rgbBuffer.extent(1);

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

    float* zBufferPtr = zBuffer.empty() ? nullptr : zBuffer.data_handle();

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
        getRowStride(rgbBuffer),
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
