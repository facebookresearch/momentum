/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/rasterizer.h"

#include "momentum/rasterizer/geometry.h"
#include "momentum/rasterizer/image.h"
#include "momentum/rasterizer/rasterizer_internal.h"
#include "momentum/rasterizer/utility.h"

#include "momentum/common/exception.h"

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/math.h>
#include <drjit/matrix.h>
#include <drjit/util.h>

#include <array>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace momentum::rasterizer {

namespace {

// Vertex interpolation matrix for perspective-correct texture coordinates
// References:
// - https://redirect.cs.umbc.edu/~olano/papers/2dh-tri/2dh-tri.pdf
// - https://tayfunkayhan.wordpress.com/2018/12/30/rasterization-in-one-weekend-part-iii/
//
// We compute pseudo-eye coordinates that ignore distortion, using only focal length.
// This enables perspective-correct interpolation without expensive undistortion operations.
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

inline auto isAllFinite(const Vector3dP& v) {
  return drjit::isfinite(v.x()) && drjit::isfinite(v.y()) && drjit::isfinite(v.z());
}

inline auto allBehindNearClip(const std::array<Vector3fP, 4>& pts, float nearClip) {
  return pts[0].z() < nearClip && pts[1].z() < nearClip && pts[2].z() < nearClip &&
      pts[3].z() < nearClip;
}

inline auto sampleRGBTextureMap(
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
  const auto textureWidth = static_cast<int32_t>(textureMap.extent(1));
  const auto textureHeight = static_cast<int32_t>(textureMap.extent(0));

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

inline void shadeAndWriteRgb(
    const Vector3fP& bary,
    const FloatP::MaskType& finalMask,
    const Matrix3f& uv,
    const Matrix3f& perVertexDiffuseColor,
    const Matrix3f& p_eye,
    const Matrix3f& n_eye,
    const PhongMaterial& material,
    std::span<const Light> lights_eye,
    bool hasDiffuseMap,
    bool hasEmissiveMap,
    bool hasDiffuse,
    bool hasSpecular,
    float* rgbBufferPtr,
    index_t blockOffset) {
  Vector3fP shaded = toEnokiVec(material.emissiveColor);

  const Vector2fP textureCoord(
      uv(0, 0) * bary.x() + uv(0, 1) * bary.y() + uv(0, 2) * bary.z(),
      uv(1, 0) * bary.x() + uv(1, 1) * bary.y() + uv(1, 2) * bary.z());

  if (hasEmissiveMap) {
    shaded += interpolateRGBTextureMap(textureCoord, material.emissiveTextureMap.view(), finalMask);
  }

  if (hasDiffuse || hasSpecular) {
    Vector3fP diffuseColor = perVertexDiffuseColor * bary;
    if (hasDiffuseMap) {
      diffuseColor =
          interpolateRGBTextureMap(textureCoord, material.diffuseTextureMap.view(), finalMask);
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
        shadeAndWriteRgb(
            bary,
            finalMask,
            uv,
            perVertexDiffuseColor,
            p_eye,
            n_eye,
            material,
            lights_eye,
            hasDiffuseMap,
            hasEmissiveMap,
            hasDiffuse,
            hasSpecular,
            rgbBufferPtr,
            blockOffset);
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

// Support both signed and unsigned triangles for backward compatibility:
template <typename TriangleT>
void validateMeshInputs(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const PhongMaterial& material) {
  const auto nVerts = positions_world.size() / 3;

  if (normals_world.size() != 0 && positions_world.size() != normals_world.size()) {
    throw std::runtime_error("positions size doesn't match normals size");
  }

  if (perVertexDiffuseColor.size() != 0 && positions_world.size() != perVertexDiffuseColor.size()) {
    throw std::runtime_error("per-vertex color size doesn't match normals size");
  }

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
}

inline Matrix3fP computeTriangleEyeNormals(
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const SimdCamera& cameraSimd,
    const Matrix3fP& p_tri_eye,
    const Vector3iP& triangles_cur,
    const IntP::MaskType& triangleMask) {
  Matrix3fP n_tri_eye;
  if (normals_world.size() == 0) {
    const Vector3fP n_eye = drjit::normalize(
        drjit::cross(
            extractColumn(p_tri_eye, 1) - extractColumn(p_tri_eye, 0),
            extractColumn(p_tri_eye, 2) - extractColumn(p_tri_eye, 0)));
    n_tri_eye = toUniformMat(n_eye);
  } else {
    for (int i = 0; i < 3; ++i) {
      auto n_world = drjit::gather<Vector3fP>(normals_world.data(), triangles_cur[i], triangleMask);
      const Vector3fP n_eye = cameraSimd.worldToEyeNormal(n_world);
      for (int j = 0; j < 3; ++j) {
        n_tri_eye(j, i) = n_eye[j];
      }
    }
  }
  return n_tri_eye;
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

  validateMeshInputs<TriangleT>(
      positions_world,
      normals_world,
      triangles,
      textureCoords,
      textureTriangles,
      perVertexDiffuseColor,
      material);

  const bool usePerVertexColor = (perVertexDiffuseColor.size() != 0);

  const auto nTriangles = triangles.size() / 3;

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);
  float* surfaceNormalsBufferPtr = dataOrNull(surfaceNormalsBuffer);
  int* vertexIndexBufferPtr = dataOrNull(vertexIndexBuffer);
  int* triangleIndexBufferPtr = dataOrNull(triangleIndexBuffer);

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
    validTriangles = validTriangles && isAllFinite(recipWInterp);
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

    const Matrix3fP n_tri_eye = computeTriangleEyeNormals(
        normals_world, cameraSimd, p_tri_eye, triangles_cur, triangleMask);

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

void validateSplatInputs(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world) {
  if (positions_world.size() != normals_world.size()) {
    throw std::runtime_error("positions size doesn't match normals size");
  }

  const auto nSplats = positions_world.size() / 3;
  if (nSplats * 3 != positions_world.size()) {
    std::ostringstream oss;
    oss << "Vertex positions size must be a multiple of 3.";
    throw std::runtime_error(oss.str());
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

  const SimdCamera cameraSimd(camera, std::move(modelMatrix), std::move(imageOffset));
  checkBuffers(cameraSimd, zBuffer, rgbBuffer, {}, {}, {});

  const int32_t imageWidth = cameraSimd.imageWidth();
  const int32_t imageHeight = cameraSimd.imageHeight();

  validateSplatInputs(positions_world, normals_world);

  const auto nSplats = positions_world.size() / 3;

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);

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

    const auto behindCamera = allBehindNearClip(p_quad_window, nearClip);
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

      validTriangles = validTriangles && isAllFinite(recipWInterp);
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

} // namespace

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

} // namespace momentum::rasterizer
