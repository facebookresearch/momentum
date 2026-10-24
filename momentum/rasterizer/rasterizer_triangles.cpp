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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace momentum::rasterizer {

namespace {

#if (defined(_MSC_VER) && defined(MOMENTUM_ENABLE_SIMD) && MOMENTUM_ENABLE_SIMD) || \
    defined(MOMENTUM_RASTERIZER_FORCE_SCALAR_OUTER)
#define MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP 1
#else
#define MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP 0
#endif

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
// computing new pseudo-eye coordinates that ignore distortion.
//
// Any invertible affine change of window coordinates applied to both the
// vertices and the pixel samples leaves the barycentrics and the interpolated
// depth unchanged, so the pseudo-eye coordinates are built from window
// coordinates relative to an integer pixel origin near the triangle (the
// corner of its clipped bounding box).  Relative to the image origin, the
// per-pixel FP32 edge-function evaluation suffers cancellation proportional to
// the triangle's distance from that origin; relative to the triangle, it is
// proportional only to the triangle's own extent.  The window coordinates are
// widened before subtracting the origin so that the rows are exact, and shared
// vertices describe the same point in adjacent triangles with different
// origins.  Scaling by the focal length would cancel out of the barycentrics
// and depth, so window coordinates are used directly.
inline Matrix3dP createVertexMatrix(
    const std::array<Vector3fP, 3>& p_tri_window,
    const IntP& originX,
    const IntP& originY) {
  const DoubleP x0(originX);
  const DoubleP y0(originY);
  Matrix3dP result;
  for (int i = 0; i < 3; ++i) {
    const DoubleP z(p_tri_window[i].z());
    result(i, 0) = (DoubleP(p_tri_window[i].x()) - x0) * z;
    result(i, 1) = (DoubleP(p_tri_window[i].y()) - y0) * z;
    result(i, 2) = z;
  }
  return result;
}

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
inline Matrix3d
createVertexMatrix(const std::array<Vector3f, 3>& p_tri_window, int32_t originX, int32_t originY) {
  Matrix3d result;
  for (int i = 0; i < 3; ++i) {
    const double z = p_tri_window[i].z();
    result(i, 0) = (double(p_tri_window[i].x()) - originX) * z;
    result(i, 1) = (double(p_tri_window[i].y()) - originY) * z;
    result(i, 2) = z;
  }
  return result;
}
#endif

// Whether drjit::fmadd on FP32 packets rounds once.
#if defined(DRJIT_X86_FMA) || defined(DRJIT_ARM_FMA)
constexpr bool kHasFusedMultiplyAdd = true;
#else
constexpr bool kHasFusedMultiplyAdd = false;
#endif

// Per-triangle edge-function setup; see createEdgeFunctions.
template <typename Double>
struct EdgeFunctions {
  drjit::Matrix<Double, 3> coefficients;
  drjit::Array<Double, 3> thresholds;
  drjit::mask_t<Double> valid;
};

// Row i of the coefficient matrix maps a pixel sample relative to the vertex
// matrix origin, (x - originX, y - originY, 1), to the i-th perspective-correct
// barycentric scaled by 1/w.  Since the barycentrics sum to one, 1/w is the
// sum of the three edge values; computing it that way keeps w and the
// barycentrics consistent, so a sample whose edge values all vanish gets an
// infinite w and is rejected instead of producing zero barycentrics.  The rows
// are the cross products of pairs of vertex-matrix rows divided by the
// determinant (i.e. the inverse transpose), computed from the exact FP64 vertex
// matrix.
//
// The thresholds are minus a bound on the error of each edge value, as
// evaluated by rasterizeOneTriangle, over the scanned pixel range (including
// SIMD padding).  Adjacent triangles evaluate a shared edge with independent
// rounding, so with a zero threshold a pixel center on the edge can be
// rejected by both; with these thresholds, the triangle whose exact edge value
// is nonnegative always accepts it.  The bound covers the FP64 cross products
// and the FP32 evaluation; the latter is dominated by eps^2 terms when fmadd
// is fused and by eps terms otherwise.  The thresholds are thus far smaller
// than the edge values' typical magnitude, and don't measurably dilate even
// nearly degenerate triangles.
//
// A triangle is valid if its determinant is distinguishable from zero and the
// coefficients and the largest edge values over the scanned range are finite
// in FP32.  This rejects degenerate triangles before rasterization rather
// than relying on the per-pixel w test to catch infinite or NaN coefficients.
template <typename Double, typename Int>
EdgeFunctions<Double> createEdgeFunctions(
    const drjit::Matrix<Double, 3>& vertexMatrix,
    const Int& startX,
    const Int& endX,
    const Int& startY,
    const Int& endY) {
  using Vector = drjit::Array<Double, 3>;
  using Float = drjit::float32_array_t<Double>;
  constexpr double floatEps = std::numeric_limits<float>::epsilon();
  constexpr double doubleEps = std::numeric_limits<double>::epsilon();
  constexpr double evaluationError =
      kHasFusedMultiplyAdd ? 4.0 * floatEps * floatEps + 4.0 * doubleEps : 4.0 * floatEps;

  const std::array<Vector, 3> rows = {
      vertexMatrix.entry(0), vertexMatrix.entry(1), vertexMatrix.entry(2)};
  const std::array<Vector, 3> absRows = {
      drjit::abs(rows[0]), drjit::abs(rows[1]), drjit::abs(rows[2])};
  // Bounds on the local pixel coordinates.  The y bound is at least one so
  // that the bound on the edge values also bounds each coefficient.
  const Double maxLocalX(endX - startX + static_cast<int32_t>(kSimdPacketSize));
  const Double maxLocalY(endY - startY + 1);

  // The cross products, and bounds on the magnitudes of their products:
  std::array<Vector, 3> cross;
  std::array<Vector, 3> crossMagnitude;
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;
    cross[i] = drjit::cross(rows[j], rows[k]);
    const Vector& a = absRows[j];
    const Vector& b = absRows[k];
    crossMagnitude[i] = Vector(
        drjit::fmadd(a.y(), b.z(), a.z() * b.y()),
        drjit::fmadd(a.z(), b.x(), a.x() * b.z()),
        drjit::fmadd(a.x(), b.y(), a.y() * b.x()));
  }
  const Double det = drjit::dot(rows[0], cross[0]);
  const Double detError = 4.0 * doubleEps * drjit::dot(absRows[0], crossMagnitude[0]);
  const Double invDet = Double(1.0) / det;
  const Double absInvDet = drjit::abs(invDet);

  EdgeFunctions<Double> result;
  // The determinant is twice the projected triangle's area times the product
  // of the vertex depths; a triangle whose determinant is within rounding
  // error of zero is (numerically) degenerate.
  result.valid = drjit::abs(det) > detError;
  for (int i = 0; i < 3; ++i) {
    const Vector coefficients = cross[i] * invDet;
    result.coefficients.entry(i) = coefficients;

    const Double crossError = 4.0 * doubleEps * absInvDet *
        (crossMagnitude[i].x() * maxLocalX + crossMagnitude[i].y() * maxLocalY +
         crossMagnitude[i].z());
    const Double maxEdgeValue = drjit::abs(coefficients.x()) * maxLocalX +
        drjit::abs(coefficients.y()) * maxLocalY + drjit::abs(coefficients.z());
    result.thresholds[i] = -(evaluationError * maxEdgeValue + crossError);
    result.valid = result.valid && drjit::isfinite(Float(maxEdgeValue));
  }
  return result;
}

inline Vector3fP reflect(const Vector3fP& v, const Vector3fP& n) {
  const FloatP dotProd = drjit::dot(v, n);
  return (2.0f * dotProd) * n - v;
}

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
inline Eigen::Vector3f transformModelToEye(
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    const Eigen::Vector3f& p_model) {
  const Eigen::Vector4f p_model_h(p_model.x(), p_model.y(), p_model.z(), 1.0f);
  const Eigen::Vector4f p_world_h = modelMatrix * p_model_h;
  const Eigen::Vector3f p_world = p_world_h.head<3>() / p_world_h.w();
  return camera.eyeFromWorld() * p_world;
}

inline Eigen::Matrix3f createNormalMatrix(
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix) {
  return (camera.eyeFromWorld() * Eigen::Transform<float, 3, Eigen::Affine>(modelMatrix))
      .linear()
      .inverse()
      .transpose();
}
#endif

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

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
inline int32_t floorClippedToImage(float value, int32_t maxValue) {
  return static_cast<int32_t>(std::floor(std::clamp(value, 0.0f, static_cast<float>(maxValue))));
}

inline int32_t ceilClippedToImage(float value, int32_t maxValue) {
  return static_cast<int32_t>(std::ceil(std::clamp(value, 0.0f, static_cast<float>(maxValue))));
}
#endif

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

// Force inlining: otherwise the compiler may outline the shading, which costs
// ~20% on shaded renders.
DRJIT_INLINE void shadeAndWriteRgb(
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
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    const Matrix3d& edgeFunctions,
    const Vector3f& edgeThresholds,
    const Vector3f& vertexDepths,
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

  // Each edge function is evaluated per pixel as slope * x + intercept, where
  // the x slope and the scanline intercept (computed in FP64 once per row) are
  // split into FP32 high and low parts.  The FMA computes slope * x + intercept
  // exactly before rounding, so the cancellation between the two terms costs
  // no precision and the edge values are accurate relative to their own
  // magnitude rather than to the magnitude of the terms.
  const Vector3d slope(edgeFunctions(0, 0), edgeFunctions(1, 0), edgeFunctions(2, 0));
  const Vector3d rowSlope(edgeFunctions(0, 1), edgeFunctions(1, 1), edgeFunctions(2, 1));
  const Vector3d rowConstant(edgeFunctions(0, 2), edgeFunctions(1, 2), edgeFunctions(2, 2));
  const Vector3f slopeHi(slope);
  const Vector3f slopeLo(slope - Vector3d(slopeHi));

  const Vector2f depthDelta(
      vertexDepths.y() - vertexDepths.x(), vertexDepths.z() - vertexDepths.x());

  const bool hasDiffuseMap = !material.diffuseTextureMap.empty();
  const bool hasEmissiveMap = !material.emissiveTextureMap.empty();
  const bool hasDiffuse = !material.diffuseColor.isZero() || hasDiffuseMap;
  const bool hasSpecular = !material.specularColor.isZero() && material.specularExponent != 0;

  for (int32_t y = startY; y <= endY; ++y) {
    const Vector3d intercept = drjit::fmadd(rowSlope, Vector3d(double(y - startY)), rowConstant);
    const Vector3f interceptHi(intercept);
    const Vector3f interceptLo(intercept - Vector3d(interceptHi));
    for (int32_t xBlock = startX_block; xBlock <= endX_block; ++xBlock) {
      const int32_t xOffset = xBlock * kSimdPacketSize;

      // See notes in createVertexMatrix and createEdgeFunctions for how these
      // interpolators get used.  Pixel coordinates relative to the bounding
      // box corner are small integers and hence exact in FP32:
      const FloatP p_local_x = drjit::arange<FloatP>() + (float)(xOffset - startX);
      const Vector3fP edges = drjit::fmadd(Vector3fP(slopeHi), p_local_x, Vector3fP(interceptHi)) +
          drjit::fmadd(Vector3fP(slopeLo), p_local_x, Vector3fP(interceptLo));

      // The barycentrics have the signs of the edge values wherever w > 0, so
      // coverage can be tested before the division:
      const auto baryMask =
          (edges.x() >= edgeThresholds.x() && edges.y() >= edgeThresholds.y() &&
           edges.z() >= edgeThresholds.z());
      if (!drjit::any(baryMask)) {
        continue;
      }

      // Need to maximize accuracy here so it's not safe to use the fast reciprocal:
      const FloatP w = 1.0f / (edges.x() + edges.y() + edges.z());

      // Points behind the camera (with negative z) shouldn't get rendered:
      const auto inFrontOfCameraMask = drjit::isfinite(w) && (w > nearClip);
      if (!drjit::any(inFrontOfCameraMask && baryMask)) {
        continue;
      }

      const Vector3fP bary = w * edges;

      // The depth is equal to w in exact arithmetic, but interpolating it from
      // the vertex depths cancels most of the rounding error that the edge
      // values share for thin triangles.  The barycentrics can be negative only
      // down to the coverage thresholds, so the depth stays within the
      // triangle's depth range up to that tolerance.  The offsets from the
      // first vertex depth are accumulated before adding it, to round only
      // once at its magnitude:
      const FloatP depth = vertexDepths.x() +
          drjit::fmadd(bary.y(), FloatP(depthDelta.x()), bary.z() * depthDelta.y());

      const auto blockOffset = y * zBufferRowStride + xOffset;
      const auto zBufferOrig = drjit::load_aligned<FloatP>(zBufferPtr + blockOffset);
      const FloatP zOffset = depth + depthOffset;

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

template <typename TriangleT>
void validateTextureInputs(
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& textureTriangles,
    size_t nVerts) {
  const bool hasTextureCoords = textureCoords.size() != 0;
  const bool hasTextureTriangles = textureTriangles.size() != 0;
  if (hasTextureCoords) {
    const auto nTextureVerts = textureCoords.size() / 2;
    MT_THROW_IF(
        nTextureVerts * 2 != textureCoords.size(),
        "Texture coordinates array of size {} is not a multiple of two.",
        textureCoords.size());
  }

  if (hasTextureTriangles) {
    MT_THROW_IF(
        !hasTextureCoords,
        "texture_triangles provided with {} entries, but texture_coordinates is empty",
        textureTriangles.size());
    MT_THROW_IF(
        textureTriangles.size() != triangles.size(),
        "texture_triangles size ({}) should match triangles size ({})",
        textureTriangles.size(),
        triangles.size());
    validateTriangleIndices<TriangleT>(
        textureTriangles, textureCoords.size() / 2, "texture_coordinate", "triangles");
  } else if (hasTextureCoords) {
    // Assume regular triangles are also texture triangles when a mesh does not
    // provide separate texture-triangle topology.
    MT_THROW_IF(
        textureCoords.size() != 2 * nVerts,
        "texture_triangles not provided, so texture_coordinates size ({}) should be 2 * "
        "vertex count ({})",
        textureCoords.size(),
        nVerts);
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

  MT_THROW_IF(
      normals_world.size() != 0 && positions_world.size() != normals_world.size(),
      "positions size ({}) doesn't match normals size ({})",
      positions_world.size(),
      normals_world.size());
  MT_THROW_IF(
      perVertexDiffuseColor.size() != 0 && positions_world.size() != perVertexDiffuseColor.size(),
      "per-vertex color size ({}) doesn't match positions size ({})",
      perVertexDiffuseColor.size(),
      positions_world.size());
  MT_THROW_IF(
      nVerts * 3 != positions_world.size(),
      "Positions array of size {} is not a multiple of three.",
      positions_world.size());

  validateTriangleIndices<TriangleT>(triangles, nVerts);
  validateTextureInputs<TriangleT>(triangles, textureCoords, textureTriangles, nVerts);

  MT_THROW_IF(
      perVertexDiffuseColor.size() != 0 && !material.diffuseTextureMap.empty(),
      "Can't provide both per-vertex color and diffuse texture map.");
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

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
inline Matrix3f computeTriangleEyeNormals(
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Matrix3f& normalMatrix,
    const Matrix3f& p_tri_eye,
    const Eigen::Vector3i& triangle) {
  Matrix3f n_tri_eye;
  if (normals_world.size() == 0) {
    const Vector3f n_eye = drjit::normalize(
        drjit::cross(
            extractColumn(p_tri_eye, 1) - extractColumn(p_tri_eye, 0),
            extractColumn(p_tri_eye, 2) - extractColumn(p_tri_eye, 0)));
    n_tri_eye = toUniformMat(n_eye);
  } else {
    for (int i = 0; i < 3; ++i) {
      const Eigen::Vector3f n_world = normals_world.segment<3>(3 * triangle[i]);
      const Eigen::Vector3f n_eye = (normalMatrix * n_world).normalized();
      for (int j = 0; j < 3; ++j) {
        n_tri_eye(j, i) = n_eye[j];
      }
    }
  }
  return n_tri_eye;
}
#endif

// Support both signed and unsigned triangles for backward compatibility:
template <typename TriangleT>
Matrix3f getTextureAttributes(
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& textureTriangles,
    bool useTextureCoords,
    int iTriangle,
    const Eigen::Vector3i& triangle) {
  if (!useTextureCoords) {
    return drjit::zeros<Matrix3f>();
  }

  const Eigen::Vector3i textureTriangle = textureTriangles.size() == 0
      ? triangle
      : textureTriangles.template segment<3>(3 * iTriangle).template cast<int32_t>();
  return extractTriangleAttributes<2>(textureCoords, textureTriangle);
}

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
// Support both signed and unsigned triangles for backward compatibility:
template <typename TriangleT>
void rasterizeMeshImpScalarTriangleLoop(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::Matrix<TriangleT, Eigen::Dynamic, 1>>& textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const SimdCamera& cameraSimd,
    int32_t imageWidth,
    int32_t imageHeight,
    bool usePerVertexColor,
    bool useTextureCoords,
    float* zBufferPtr,
    float* rgbBufferPtr,
    float* surfaceNormalsBufferPtr,
    int* vertexIndexBufferPtr,
    int* triangleIndexBufferPtr,
    index_t zBufferRowStride,
    const std::vector<Light>& lights_eye,
    const PhongMaterial& material,
    const Eigen::Matrix4f& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  const auto nTriangles = triangles.size() / 3;
  const Eigen::Matrix3f normalMatrix = createNormalMatrix(camera, modelMatrix);
  const Matrix3f diffuseColor = toUniformMat(material.diffuseColor);

  for (index_t iTriangle = 0; iTriangle < nTriangles; ++iTriangle) {
    const Eigen::Vector3i triangle =
        triangles.template segment<3>(3 * iTriangle).template cast<int32_t>();

    std::array<Vector3f, 3> p_tri_window;
    Matrix3f p_tri_eye;

    bool validTriangle = true;
    for (int i = 0; i < 3; ++i) {
      const Eigen::Vector3f p_model = positions_world.segment<3>(3 * triangle[i]);
      const Eigen::Vector3f p_eye = transformModelToEye(camera, modelMatrix, p_model);
      const auto [p_window, validProj] = camera.intrinsicsModel()->project(p_eye);

      p_tri_window[i] = toEnokiVec(p_window);
      for (int j = 0; j < 3; ++j) {
        p_tri_eye(j, i) = p_eye[j];
      }
      validTriangle = validTriangle && validProj;
    }

    if (backfaceCulling) {
      const Eigen::Vector2f edge1(
          p_tri_window[1].x() - p_tri_window[0].x(), p_tri_window[1].y() - p_tri_window[0].y());
      const Eigen::Vector2f edge2(
          p_tri_window[2].x() - p_tri_window[0].x(), p_tri_window[2].y() - p_tri_window[0].y());

      const float signedArea = edge1.y() * edge2.x() - edge1.x() * edge2.y();
      validTriangle = validTriangle && signedArea > 0;
    }
    if (!validTriangle) {
      continue;
    }

    if (p_tri_window[0].z() < nearClip && p_tri_window[1].z() < nearClip &&
        p_tri_window[2].z() < nearClip) {
      continue;
    }

    const int32_t startX = floorClippedToImage(
        std::min({p_tri_window[0].x(), p_tri_window[1].x(), p_tri_window[2].x()}), imageWidth - 1);
    const int32_t endX = ceilClippedToImage(
        std::max({p_tri_window[0].x(), p_tri_window[1].x(), p_tri_window[2].x()}), imageWidth - 1);
    const int32_t startY = floorClippedToImage(
        std::min({p_tri_window[0].y(), p_tri_window[1].y(), p_tri_window[2].y()}), imageHeight - 1);
    const int32_t endY = ceilClippedToImage(
        std::max({p_tri_window[0].y(), p_tri_window[1].y(), p_tri_window[2].y()}), imageHeight - 1);

    const auto edgeFunctions = createEdgeFunctions(
        createVertexMatrix(p_tri_window, startX, startY), startX, endX, startY, endY);
    if (!edgeFunctions.valid) {
      continue;
    }

    const Matrix3f n_tri_eye =
        computeTriangleEyeNormals(normals_world, normalMatrix, p_tri_eye, triangle);

    rasterizeOneTriangle(
        static_cast<int32_t>(iTriangle),
        triangle,
        startX,
        endX,
        startY,
        endY,
        edgeFunctions.coefficients,
        Vector3f(edgeFunctions.thresholds),
        Vector3f(p_tri_window[0].z(), p_tri_window[1].z(), p_tri_window[2].z()),
        p_tri_eye,
        n_tri_eye,
        getTextureAttributes(
            textureCoords,
            textureTriangles,
            useTextureCoords,
            static_cast<int>(iTriangle),
            triangle),
        usePerVertexColor ? extractTriangleAttributes<3>(perVertexDiffuseColor, triangle)
                          : diffuseColor,
        material,
        lights_eye,
        nearClip,
        zBufferRowStride,
        depthOffset,
        zBufferPtr,
        rgbBufferPtr,
        surfaceNormalsBufferPtr,
        vertexIndexBufferPtr,
        triangleIndexBufferPtr,
        false);
  }
}
#endif

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

  const SimdCamera cameraSimd(camera, modelMatrix, imageOffset);
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
  const bool useTextureCoords = (textureCoords.size() != 0 && material.hasTextureMap());

  const auto nTriangles = triangles.size() / 3;

  float* zBufferPtr = zBuffer.data_handle();
  float* rgbBufferPtr = dataOrNull(rgbBuffer);
  float* surfaceNormalsBufferPtr = dataOrNull(surfaceNormalsBuffer);
  int* vertexIndexBufferPtr = dataOrNull(vertexIndexBuffer);
  int* triangleIndexBufferPtr = dataOrNull(triangleIndexBuffer);
  const index_t zBufferRowStride = getRowStride(zBuffer);

#if MOMENTUM_RASTERIZER_USE_SCALAR_TRIANGLE_SETUP
  // MSVC's codegen for the DrJit packetized triangle setup has very large stack
  // frames in the Python renderer call chain. Keep pixel SIMD, but process the
  // per-triangle setup scalarly on Windows so the renderer fits the default
  // Python thread stack.
  rasterizeMeshImpScalarTriangleLoop(
      positions_world,
      normals_world,
      triangles,
      textureCoords,
      textureTriangles,
      perVertexDiffuseColor,
      camera,
      cameraSimd,
      imageWidth,
      imageHeight,
      usePerVertexColor,
      useTextureCoords,
      zBufferPtr,
      rgbBufferPtr,
      surfaceNormalsBufferPtr,
      vertexIndexBufferPtr,
      triangleIndexBufferPtr,
      zBufferRowStride,
      lights_eye,
      material,
      modelMatrix,
      backfaceCulling,
      nearClip,
      depthOffset);
#else
  for (auto [triangleIndices, triangleMask] : intPacketRange(nTriangles)) {
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

    // This is the vertex interpolation matrix, where we map from
    // homgeneous coordinates to interpolate texture coordinates, etc.  See
    // note above about its construction.  Use double precision for the
    // inverse, otherwise we get holes in the mesh since the inverse function
    // uses a relatively numerically unstable algorithm using determinants
    // since it's easy to SIMD.
    const auto edgeFunctions = createEdgeFunctions(
        createVertexMatrix(p_tri_window, startX, startY), startX, endX, startY, endY);
    validTriangles = validTriangles && IntP::MaskType(edgeFunctions.valid);
    if (!drjit::any(validTriangles)) {
      continue;
    }
    const Vector3fP edgeThresholds(edgeFunctions.thresholds);
    const Vector3fP vertexDepths(p_tri_window[0].z(), p_tri_window[1].z(), p_tri_window[2].z());

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

      rasterizeOneTriangle(
          iTriangle,
          triangle,
          startX[triOffset],
          endX[triOffset],
          startY[triOffset],
          endY[triOffset],
          extractSingleElement(edgeFunctions.coefficients, triOffset),
          extractSingleElement(edgeThresholds, triOffset),
          extractSingleElement(vertexDepths, triOffset),
          extractSingleElement(p_tri_eye, triOffset),
          extractSingleElement(n_tri_eye, triOffset),
          getTextureAttributes(
              textureCoords, textureTriangles, useTextureCoords, iTriangle, triangle),
          usePerVertexColor ? extractTriangleAttributes<3>(perVertexDiffuseColor, triangle)
                            : diffuseColor,
          material,
          lights_eye,
          nearClip,
          zBufferRowStride,
          depthOffset,
          zBufferPtr,
          rgbBufferPtr,
          surfaceNormalsBufferPtr,
          vertexIndexBufferPtr,
          triangleIndexBufferPtr,
          false);
    }
  }
#endif
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

  const SimdCamera cameraSimd(camera, modelMatrix, imageOffset);
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

  for (auto [splatIndices, splatMask] : intPacketRange(nSplats)) {
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

      // See above for explanation of the vertex interpolation matrix.
      const auto edgeFunctions = createEdgeFunctions(
          createVertexMatrix(p_tri_window, startX, startY), startX, endX, startY, endY);
      const auto validTriangles = validSplats && IntP::MaskType(edgeFunctions.valid);
      if (!drjit::any(validTriangles)) {
        continue;
      }
      const Vector3fP edgeThresholds(edgeFunctions.thresholds);
      const Vector3fP vertexDepths(p_tri_window[0].z(), p_tri_window[1].z(), p_tri_window[2].z());

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
            startX[triOffset],
            endX[triOffset],
            startY[triOffset],
            endY[triOffset],
            extractSingleElement(edgeFunctions.coefficients, triOffset),
            extractSingleElement(edgeThresholds, triOffset),
            extractSingleElement(vertexDepths, triOffset),
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
