/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/texture_classification.h"

#include <momentum/character/character_utility.h>
#include <momentum/common/exception.h>

#include <dispenso/parallel_for.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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

/// Validated texture array information.
struct TextureInfo {
  const uint8_t* data;
  py::ssize_t stride0;
  py::ssize_t stride1;
  py::ssize_t stride2;
  py::ssize_t height;
  py::ssize_t width;
};

/// Validate a texture numpy array and extract sampling info.
TextureInfo validateTexture(const py::array_t<uint8_t>& texture) {
  py::buffer_info texInfo = texture.request();
  MT_THROW_IF(texInfo.ndim != 3, "texture must be a 3D array [height, width, 3]");
  MT_THROW_IF(texInfo.shape[2] != 3, "texture must have 3 channels (RGB)");

  const auto height = static_cast<py::ssize_t>(texInfo.shape[0]);
  const auto width = static_cast<py::ssize_t>(texInfo.shape[1]);
  MT_THROW_IF(height <= 0 || width <= 0, "texture dimensions must be positive");

  return TextureInfo{
      static_cast<const uint8_t*>(texInfo.ptr),
      static_cast<py::ssize_t>(texInfo.strides[0]),
      static_cast<py::ssize_t>(texInfo.strides[1]),
      static_cast<py::ssize_t>(texInfo.strides[2]),
      height,
      width};
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
  const auto tex = validateTexture(texture);
  const auto* textureData = tex.data;
  const auto height = tex.height;
  const auto width = tex.width;
  const auto texStride0 = tex.stride0;
  const auto texStride1 = tex.stride1;
  const auto texStride2 = tex.stride2;

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

namespace {

// Canonical edge representation for caching split vertices.
using Edge = std::pair<int, int>;

Edge makeEdge(int v1, int v2) {
  if (v1 > v2) {
    std::swap(v1, v2);
  }
  return {v1, v2};
}

struct HashEdge {
  std::size_t operator()(const Edge& e) const {
    std::size_t h1 = std::hash<int>{}(e.first);
    std::size_t h2 = std::hash<int>{}(e.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

// Cached result from splitting a geometry edge.
struct GeomCacheEntry {
  int vertexIndex;
  float t;
};

// Build a set of "inside" colors from the regionColors array.
std::unordered_set<size_t> buildInsideColorSet(
    const uint8_t* regionColorsData,
    py::ssize_t stride0,
    py::ssize_t stride1,
    int32_t nColors) {
  std::unordered_set<size_t> result;
  result.reserve(nColors);
  for (int32_t i = 0; i < nColors; ++i) {
    const uint8_t* colorPtr = regionColorsData + i * stride0;
    const uint8_t r = colorPtr[0 * stride1];
    const uint8_t g = colorPtr[1 * stride1];
    const uint8_t b = colorPtr[2 * stride1];
    result.insert(packRgb(r, g, b));
  }
  return result;
}

// Bundles texture data and inside-color set for sampling operations.
struct TextureSampler {
  TextureInfo tex;
  std::unordered_set<size_t> insideColors;

  // Sample texture at a UV coordinate and check if the color is "inside".
  bool isInside(const Eigen::Vector2f& uv) const {
    const float colF = uv.x() * static_cast<float>(tex.width - 1);
    const auto col = std::clamp(static_cast<py::ssize_t>(colF), py::ssize_t{0}, tex.width - 1);
    const float rowF = uv.y() * static_cast<float>(tex.height - 1);
    const auto row = std::clamp(static_cast<py::ssize_t>(rowF), py::ssize_t{0}, tex.height - 1);

    const auto* pixel = tex.data + row * tex.stride0 + col * tex.stride1;
    const uint8_t r = pixel[0 * tex.stride2];
    const uint8_t g = pixel[1 * tex.stride2];
    const uint8_t b = pixel[2 * tex.stride2];

    return insideColors.count(packRgb(r, g, b)) > 0;
  }

  // Binary search for the crossing point on an edge from inside vertex A to outside vertex B
  // in UV space. Returns the parameter t such that the crossing is at A + t*(B - A).
  float findCrossingT(const Eigen::Vector2f& uvA, const Eigen::Vector2f& uvB, int32_t numSteps)
      const {
    float lo = 0.0f;
    float hi = 1.0f;
    for (int32_t i = 0; i < numSteps; ++i) {
      const float mid = 0.5f * (lo + hi);
      const Eigen::Vector2f uvMid = (1.0f - mid) * uvA + mid * uvB;
      if (isInside(uvMid)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return 0.5f * (lo + hi);
  }
};

// Find the crossing t-value on the edge from V0 to V1, checking the geometry cache first.
// Returns t in the V0->V1 direction. v0IsInside indicates whether V0 is inside the region.
float findEdgeCrossingT(
    const TextureSampler& sampler,
    const Eigen::Vector2f& uv0,
    const Eigen::Vector2f& uv1,
    int geomV0,
    int geomV1,
    bool v0IsInside,
    const std::unordered_map<Edge, GeomCacheEntry, HashEdge>& geomCache,
    int32_t numSteps) {
  // Check geometry cache for a previously computed t-value on this edge
  const auto edge = makeEdge(geomV0, geomV1);
  auto it = geomCache.find(edge);
  if (it != geomCache.end()) {
    // Convert cached t (in min->max direction) to V0->V1 direction
    float tOriented = it->second.t;
    return (geomV0 < geomV1) ? tOriented : (1.0f - tOriented);
  }

  // Binary search: always orient from inside to outside
  if (v0IsInside) {
    return sampler.findCrossingT(uv0, uv1, numSteps);
  } else {
    return 1.0f - sampler.findCrossingT(uv1, uv0, numSteps);
  }
}

// Process a mixed face (1 or 2 vertices inside) by splitting it along the texture
// region boundary. Appends the resulting inside triangle(s) to newFaces/newTexcoordFaces.
template <typename GetOrCreateGeomVertexFn, typename GetOrCreateTexVertexFn>
void processMixedFace(
    const Eigen::Vector3i& face,
    const Eigen::Vector3i& texFace,
    const std::array<bool, 3>& inside,
    int nInside,
    const std::vector<Eigen::Vector2f>& meshTexcoords,
    const TextureSampler& sampler,
    int32_t numBinarySearchSteps,
    const std::unordered_map<Edge, GeomCacheEntry, HashEdge>& geomCache,
    GetOrCreateGeomVertexFn& getOrCreateGeomVertex,
    GetOrCreateTexVertexFn& getOrCreateTexVertex,
    std::vector<Eigen::Vector3i>& newFaces,
    std::vector<Eigen::Vector3i>& newTexcoordFaces) {
  // Find the solo vertex: for "1 in, 2 out" the lone inside vertex;
  // for "2 in, 1 out" the lone outside vertex.
  int solo = -1;
  for (int k = 0; k < 3; ++k) {
    if ((nInside == 1 && inside[k]) || (nInside == 2 && !inside[k])) {
      solo = k;
      break;
    }
  }

  // The two edges that cross are (solo, (solo+1)%3) and (solo, (solo+2)%3)
  const int k0 = solo;
  const int k1 = (solo + 1) % 3;
  const int k2 = (solo + 2) % 3;

  const int gV0 = face[k0], gV1 = face[k1], gV2 = face[k2];
  const int tV0 = texFace[k0], tV1 = texFace[k1], tV2 = texFace[k2];

  // Find crossing points on the two edges from V0 to V1 and V0 to V2.
  const bool v0IsInside = (nInside == 1);
  const float t01 = findEdgeCrossingT(
      sampler,
      meshTexcoords[tV0],
      meshTexcoords[tV1],
      gV0,
      gV1,
      v0IsInside,
      geomCache,
      numBinarySearchSteps);
  const float t02 = findEdgeCrossingT(
      sampler,
      meshTexcoords[tV0],
      meshTexcoords[tV2],
      gV0,
      gV2,
      v0IsInside,
      geomCache,
      numBinarySearchSteps);

  // Create split vertices on edge V0-V1 and V0-V2
  const int gM01 = getOrCreateGeomVertex(gV0, gV1, t01);
  const int gM02 = getOrCreateGeomVertex(gV0, gV2, t02);
  const int tM01 = getOrCreateTexVertex(tV0, tV1, t01);
  const int tM02 = getOrCreateTexVertex(tV0, tV2, t02);

  if (nInside == 1) {
    // 1 inside (V0), 2 outside: emit 1 triangle (V0, M01, M02)
    newFaces.emplace_back(gV0, gM01, gM02);
    newTexcoordFaces.emplace_back(tV0, tM01, tM02);
  } else {
    // 2 inside (V1, V2), 1 outside (V0): emit 2 triangles forming the inside quad
    newFaces.emplace_back(gV1, gV2, gM01);
    newTexcoordFaces.emplace_back(tV1, tV2, tM01);
    newFaces.emplace_back(gV2, gM02, gM01);
    newTexcoordFaces.emplace_back(tV2, tM02, tM01);
  }
}

} // namespace

momentum::Mesh splitMeshByTextureRegion(
    const momentum::Mesh& mesh,
    const py::array_t<uint8_t>& texture,
    const py::array_t<uint8_t>& regionColors,
    int32_t numBinarySearchSteps) {
  // Validate inputs
  MT_THROW_IF(mesh.texcoords.empty(), "Mesh must have texture coordinates");
  MT_THROW_IF(mesh.texcoord_faces.empty(), "Mesh must have texture coordinate faces");
  MT_THROW_IF(
      mesh.faces.size() != mesh.texcoord_faces.size(),
      "faces and texcoord_faces must have the same size");

  // Validate and build texture sampler
  const auto tex = validateTexture(texture);

  py::buffer_info colorsInfo = regionColors.request();
  MT_THROW_IF(colorsInfo.ndim != 2, "region_colors must be a 2D array [n_colors, 3]");
  MT_THROW_IF(colorsInfo.shape[1] != 3, "region_colors must have 3 columns (RGB)");

  const auto nColors = static_cast<int32_t>(colorsInfo.shape[0]);
  MT_THROW_IF(nColors == 0, "region_colors must not be empty");

  const auto* regionColorsData = static_cast<const uint8_t*>(colorsInfo.ptr);
  const auto colorsStride0 = static_cast<py::ssize_t>(colorsInfo.strides[0]);
  const auto colorsStride1 = static_cast<py::ssize_t>(colorsInfo.strides[1]);

  const TextureSampler sampler{
      tex, buildInsideColorSet(regionColorsData, colorsStride0, colorsStride1, nColors)};

  // Classify each texcoord vertex as inside or outside
  const auto nTexVerts = static_cast<int>(mesh.texcoords.size());
  std::vector<bool> texVertInside(nTexVerts);
  for (int i = 0; i < nTexVerts; ++i) {
    texVertInside[i] = sampler.isInside(mesh.texcoords[i]);
  }

  // Initialize output: start with copies of all original vertices and texcoords.
  // We'll append new split vertices/texcoords for boundary edges.
  auto vertices = mesh.vertices;
  auto normals = mesh.normals;
  auto colors = mesh.colors;
  auto confidence = mesh.confidence;
  auto texcoords = mesh.texcoords;

  std::vector<Eigen::Vector3i> newFaces;
  std::vector<Eigen::Vector3i> newTexcoordFaces;
  newFaces.reserve(mesh.faces.size());
  newTexcoordFaces.reserve(mesh.faces.size());

  // Edge caches for vertex reuse
  std::unordered_map<Edge, GeomCacheEntry, HashEdge> geomCache;
  std::unordered_map<Edge, int, HashEdge> texCache;

  const bool hasNormals = !normals.empty();
  const bool hasColors = !colors.empty();
  const bool hasConfidence = !confidence.empty();

  // Helper: get or create a split vertex on a geometry edge
  auto getOrCreateGeomVertex = [&](int geomA, int geomB, float t) -> int {
    const auto edge = makeEdge(geomA, geomB);
    auto it = geomCache.find(edge);
    if (it != geomCache.end()) {
      return it->second.vertexIndex;
    }
    // Ensure t is oriented: from min to max vertex index
    float tOriented = (geomA < geomB) ? t : (1.0f - t);
    const int newIdx = static_cast<int>(vertices.size());
    vertices.emplace_back(
        (1.0f - tOriented) * vertices[edge.first] + tOriented * vertices[edge.second]);
    if (hasNormals) {
      Eigen::Vector3f n =
          (1.0f - tOriented) * normals[edge.first] + tOriented * normals[edge.second];
      n.normalize();
      normals.emplace_back(n);
    }
    if (hasColors) {
      // Linearly interpolate colors (cast to float for interpolation)
      Eigen::Vector3f cf = (1.0f - tOriented) * colors[edge.first].cast<float>() +
          tOriented * colors[edge.second].cast<float>();
      colors.emplace_back(
          static_cast<uint8_t>(std::clamp(cf.x(), 0.0f, 255.0f)),
          static_cast<uint8_t>(std::clamp(cf.y(), 0.0f, 255.0f)),
          static_cast<uint8_t>(std::clamp(cf.z(), 0.0f, 255.0f)));
    }
    if (hasConfidence) {
      confidence.emplace_back(
          (1.0f - tOriented) * confidence[edge.first] + tOriented * confidence[edge.second]);
    }
    geomCache[edge] = {newIdx, tOriented};
    return newIdx;
  };

  // Helper: get or create a split vertex on a texcoord edge
  auto getOrCreateTexVertex = [&](int texA, int texB, float t) -> int {
    const auto edge = makeEdge(texA, texB);
    auto it = texCache.find(edge);
    if (it != texCache.end()) {
      return it->second;
    }
    float tOriented = (texA < texB) ? t : (1.0f - t);
    const int newIdx = static_cast<int>(texcoords.size());
    texcoords.emplace_back(
        (1.0f - tOriented) * texcoords[edge.first] + tOriented * texcoords[edge.second]);
    texCache[edge] = newIdx;
    return newIdx;
  };

  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto& face = mesh.faces[fi];
    const auto& texFace = mesh.texcoord_faces[fi];

    // Classify: count how many texcoord vertices are inside
    std::array<bool, 3> inside = {
        texVertInside[texFace.x()], texVertInside[texFace.y()], texVertInside[texFace.z()]};
    const int nInside =
        static_cast<int>(inside[0]) + static_cast<int>(inside[1]) + static_cast<int>(inside[2]);

    if (nInside == 3) {
      // All inside: keep face unchanged
      newFaces.push_back(face);
      newTexcoordFaces.push_back(texFace);
    } else if (nInside > 0) {
      // Mixed face: split along the texture region boundary
      processMixedFace(
          face,
          texFace,
          inside,
          nInside,
          mesh.texcoords,
          sampler,
          numBinarySearchSteps,
          geomCache,
          getOrCreateGeomVertex,
          getOrCreateTexVertex,
          newFaces,
          newTexcoordFaces);
    }
    // nInside == 0: all outside, discard
  }

  // Build the output mesh with all vertices/texcoords (including newly created ones)
  momentum::Mesh result;
  result.vertices = std::move(vertices);
  result.normals = std::move(normals);
  result.colors = std::move(colors);
  result.confidence = std::move(confidence);
  result.texcoords = std::move(texcoords);
  result.faces = std::move(newFaces);
  result.texcoord_faces = std::move(newTexcoordFaces);
  // Lines are not preserved through triangle splitting

  // Compact the mesh: remove unreferenced vertices and texcoords
  if (result.faces.empty()) {
    return momentum::Mesh{};
  }

  std::vector<bool> activeFaces(result.faces.size(), true);
  return momentum::reduceMeshByFaces<float>(result, activeFaces);
}

} // namespace pymomentum
