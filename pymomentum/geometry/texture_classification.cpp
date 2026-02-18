/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/texture_classification.h"

#include <momentum/common/exception.h>

#include <dispenso/parallel_for.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <mutex>

namespace pymomentum {

namespace {

/// Barycentric coordinate for a sample point within a triangle.
struct BarycentricCoord {
  float u;
  float v;
  float w;

  constexpr BarycentricCoord(float u_, float v_, float w_) : u(u_), v(v_), w(w_) {}
};

/// Get the barycentric sample points for a given num_samples value.
std::vector<BarycentricCoord> getSamplePoints(int32_t numSamples) {
  // Vertex coordinates
  constexpr BarycentricCoord v0(1.0f, 0.0f, 0.0f);
  constexpr BarycentricCoord v1(0.0f, 1.0f, 0.0f);
  constexpr BarycentricCoord v2(0.0f, 0.0f, 1.0f);

  // Centroid
  constexpr float third = 1.0f / 3.0f;
  constexpr BarycentricCoord centroid(third, third, third);

  // Edge midpoints
  constexpr BarycentricCoord mid01(0.5f, 0.5f, 0.0f);
  constexpr BarycentricCoord mid12(0.0f, 0.5f, 0.5f);
  constexpr BarycentricCoord mid02(0.5f, 0.0f, 0.5f);

  // Interior points (at 1/4 from each vertex toward centroid)
  constexpr float quarter = 0.25f;
  constexpr float threeQuarters = 0.75f;
  constexpr BarycentricCoord int0(threeQuarters, quarter * 0.5f, quarter * 0.5f);
  constexpr BarycentricCoord int1(quarter * 0.5f, threeQuarters, quarter * 0.5f);
  constexpr BarycentricCoord int2(quarter * 0.5f, quarter * 0.5f, threeQuarters);

  switch (numSamples) {
    case 1:
      return {centroid};
    case 3:
      // Vertices only - backward compatible with Python implementation
      return {v0, v1, v2};
    case 4:
      return {centroid, v0, v1, v2};
    case 6:
      return {v0, v1, v2, mid01, mid12, mid02};
    case 7:
      return {centroid, v0, v1, v2, mid01, mid12, mid02};
    case 10:
      return {centroid, v0, v1, v2, mid01, mid12, mid02, int0, int1, int2};
    default:
      MT_THROW("Invalid num_samples value {}. Must be one of: 1, 3, 4, 6, 7, 10.", numSamples);
  }
}

/// Hash function for packing RGB into a single size_t value for fast lookup.
inline size_t packRgb(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<size_t>(r) | (static_cast<size_t>(g) << 8) | (static_cast<size_t>(b) << 16);
}

/// Build a lookup table from packed RGB to region index using the regionColors array.
std::unordered_map<size_t, int32_t> buildColorLookup(
    const uint8_t* regionColorsData,
    py::ssize_t stride0,
    py::ssize_t stride1,
    int32_t nRegions) {
  std::unordered_map<size_t, int32_t> lookup;
  lookup.reserve(nRegions);

  for (int32_t regionIdx = 0; regionIdx < nRegions; ++regionIdx) {
    const uint8_t* colorPtr = regionColorsData + regionIdx * stride0;
    const uint8_t r = colorPtr[0 * stride1];
    const uint8_t g = colorPtr[1 * stride1];
    const uint8_t b = colorPtr[2 * stride1];
    lookup[packRgb(r, g, b)] = regionIdx;
  }
  return lookup;
}

} // namespace

std::vector<std::vector<int32_t>> classifyTrianglesByTexture(
    const momentum::Mesh& mesh,
    const py::array_t<uint8_t>& texture,
    const py::array_t<uint8_t>& regionColors,
    float threshold,
    int32_t numSamples) {
  // Validate inputs
  MT_THROW_IF(mesh.texcoords.empty(), "Mesh must have texture coordinates");
  MT_THROW_IF(mesh.texcoord_faces.empty(), "Mesh must have texture coordinate faces");
  MT_THROW_IF(threshold < 0.0f || threshold > 1.0f, "threshold must be in range [0, 1]");

  // Validate texture array
  py::buffer_info texInfo = texture.request();
  MT_THROW_IF(texInfo.ndim != 3, "texture must be a 3D array [height, width, 3]");
  MT_THROW_IF(texInfo.shape[2] != 3, "texture must have 3 channels (RGB)");

  const auto height = static_cast<py::ssize_t>(texInfo.shape[0]);
  const auto width = static_cast<py::ssize_t>(texInfo.shape[1]);
  MT_THROW_IF(height <= 0 || width <= 0, "texture dimensions must be positive");

  const auto* textureData = static_cast<const uint8_t*>(texInfo.ptr);
  const auto texStride0 = static_cast<py::ssize_t>(texInfo.strides[0]);
  const auto texStride1 = static_cast<py::ssize_t>(texInfo.strides[1]);
  const auto texStride2 = static_cast<py::ssize_t>(texInfo.strides[2]);

  // Validate regionColors array
  py::buffer_info colorsInfo = regionColors.request();
  MT_THROW_IF(colorsInfo.ndim != 2, "region_colors must be a 2D array [n_regions, 3]");
  MT_THROW_IF(colorsInfo.shape[1] != 3, "region_colors must have 3 columns (RGB)");

  const auto nRegions = static_cast<int32_t>(colorsInfo.shape[0]);
  MT_THROW_IF(nRegions == 0, "region_colors must not be empty");

  const auto* regionColorsData = static_cast<const uint8_t*>(colorsInfo.ptr);
  const auto colorsStride0 = static_cast<py::ssize_t>(colorsInfo.strides[0]);
  const auto colorsStride1 = static_cast<py::ssize_t>(colorsInfo.strides[1]);

  // Get sample points
  const auto samplePoints = getSamplePoints(numSamples);
  const auto nSamples = static_cast<int32_t>(samplePoints.size());

  // Build fast color lookup from region colors array
  const auto colorLookup =
      buildColorLookup(regionColorsData, colorsStride0, colorsStride1, nRegions);

  const auto nTriangles = static_cast<int64_t>(mesh.faces.size());

  // Per-region triangle lists
  std::vector<std::vector<int32_t>> regionTriangles(nRegions);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;

    // Use thread-local accumulation to avoid mutex contention
    std::mutex mergeMutex;
    dispenso::parallel_for(
        dispenso::makeChunkedRange(
            static_cast<int64_t>(0), nTriangles, dispenso::ParForChunking::kAuto),
        [&](int64_t rangeBegin, int64_t rangeEnd) {
          // Thread-local storage for region counts and per-region triangle lists
          std::vector<int32_t> regionCounts(nRegions, 0);
          std::vector<std::vector<int32_t>> localRegionTriangles(nRegions);

          for (int64_t triIdx = rangeBegin; triIdx != rangeEnd; ++triIdx) {
            // Reset counts
            std::fill(regionCounts.begin(), regionCounts.end(), 0);

            // Get texture coordinate indices for this triangle
            const auto& texFace = mesh.texcoord_faces[triIdx];
            const auto& uv0 = mesh.texcoords[texFace.x()];
            const auto& uv1 = mesh.texcoords[texFace.y()];
            const auto& uv2 = mesh.texcoords[texFace.z()];

            // Sample at each barycentric coordinate
            for (const auto& bary : samplePoints) {
              // Interpolate UV coordinates
              const float u = bary.u * uv0.x() + bary.v * uv1.x() + bary.w * uv2.x();
              const float v = bary.u * uv0.y() + bary.v * uv1.y() + bary.w * uv2.y();

              // Convert UV to pixel coordinates (matching Python implementation)
              const std::ptrdiff_t col = std::clamp(
                  static_cast<std::ptrdiff_t>(u * static_cast<float>(width - 1)),
                  std::ptrdiff_t{0},
                  width - 1);
              const std::ptrdiff_t row = std::clamp(
                  static_cast<std::ptrdiff_t>(v * static_cast<float>(height - 1)),
                  std::ptrdiff_t{0},
                  height - 1);

              // Sample texture color
              const auto* pixel = textureData + row * texStride0 + col * texStride1;
              const uint8_t r = pixel[0 * texStride2];
              const uint8_t g = pixel[1 * texStride2];
              const uint8_t b = pixel[2 * texStride2];

              // Look up region
              const size_t packedColor = packRgb(r, g, b);
              auto it = colorLookup.find(packedColor);
              if (it != colorLookup.end()) {
                const int32_t regionIdx = it->second;
                if (regionIdx >= 0 && regionIdx < nRegions) {
                  ++regionCounts[regionIdx];
                }
              }
            }

            // Determine which regions this triangle belongs to based on threshold
            const int32_t minSamplesNeeded = (threshold == 0.0f)
                ? 1
                : static_cast<int32_t>(std::ceil(threshold * static_cast<float>(nSamples)));

            for (int32_t regionIdx = 0; regionIdx < nRegions; ++regionIdx) {
              if (regionCounts[regionIdx] >= minSamplesNeeded) {
                localRegionTriangles[regionIdx].push_back(static_cast<int32_t>(triIdx));
              }
            }
          }

          // Merge thread-local results into shared vectors
          std::lock_guard<std::mutex> lock(mergeMutex);
          for (int32_t regionIdx = 0; regionIdx < nRegions; ++regionIdx) {
            auto& dst = regionTriangles[regionIdx];
            auto& src = localRegionTriangles[regionIdx];
            dst.insert(dst.end(), src.begin(), src.end());
          }
        });
  }

  // Sort triangle lists for determinism
  for (auto& triangles : regionTriangles) {
    std::sort(triangles.begin(), triangles.end());
  }

  return regionTriangles;
}

} // namespace pymomentum
