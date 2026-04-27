/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/texture_classification.h"

#include "momentum/common/exception.h"

#include <momentum/character/character_utility.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace momentum {

namespace {

struct BarycentricCoord {
  float u;
  float v;
  float w;
};

std::vector<BarycentricCoord> getSamplePoints(int32_t numSamples) {
  constexpr BarycentricCoord v0{1.0f, 0.0f, 0.0f};
  constexpr BarycentricCoord v1{0.0f, 1.0f, 0.0f};
  constexpr BarycentricCoord v2{0.0f, 0.0f, 1.0f};

  constexpr float third = 1.0f / 3.0f;
  constexpr BarycentricCoord centroid{third, third, third};

  constexpr BarycentricCoord mid01{0.5f, 0.5f, 0.0f};
  constexpr BarycentricCoord mid12{0.0f, 0.5f, 0.5f};
  constexpr BarycentricCoord mid02{0.5f, 0.0f, 0.5f};

  constexpr float quarter = 0.25f;
  constexpr float threeQuarters = 0.75f;
  constexpr BarycentricCoord int0{threeQuarters, quarter * 0.5f, quarter * 0.5f};
  constexpr BarycentricCoord int1{quarter * 0.5f, threeQuarters, quarter * 0.5f};
  constexpr BarycentricCoord int2{quarter * 0.5f, quarter * 0.5f, threeQuarters};

  switch (numSamples) {
    case 1:
      return {centroid};
    case 3:
      return {v0, v1, v2};
    case 4:
      return {centroid, v0, v1, v2};
    case 6:
      return {v0, v1, v2, mid01, mid12, mid02};
    case 7:
      return {centroid, v0, v1, v2, mid01, mid12, mid02};
    case 10:
      return {centroid, v0, v1, v2, mid01, mid12, mid02, int0, int1, int2};
  }
  MT_THROW("Invalid num_samples value {}. Must be one of: 1, 3, 4, 6, 7, 10.", numSamples);
}

inline size_t packRgb(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<size_t>(r) | (static_cast<size_t>(g) << 8) | (static_cast<size_t>(b) << 16);
}

std::unordered_map<size_t, int32_t> buildColorLookup(const RegionColorsView& colors) {
  std::unordered_map<size_t, int32_t> lookup;
  lookup.reserve(colors.nRegions);
  for (int32_t i = 0; i < colors.nRegions; ++i) {
    const uint8_t* ptr = colors.data + i * colors.stride0;
    lookup[packRgb(ptr[0 * colors.stride1], ptr[1 * colors.stride1], ptr[2 * colors.stride1])] = i;
  }
  return lookup;
}

std::unordered_set<size_t> buildInsideColorSet(const RegionColorsView& colors) {
  std::unordered_set<size_t> result;
  result.reserve(colors.nRegions);
  for (int32_t i = 0; i < colors.nRegions; ++i) {
    const uint8_t* ptr = colors.data + i * colors.stride0;
    result.insert(
        packRgb(ptr[0 * colors.stride1], ptr[1 * colors.stride1], ptr[2 * colors.stride1]));
  }
  return result;
}

} // namespace

std::vector<std::vector<int32_t>> classifyTrianglesByTexture(
    const Mesh& mesh,
    const TextureView& texture,
    const RegionColorsView& regionColors,
    float threshold,
    int32_t numSamples) {
  MT_THROW_IF(mesh.texcoords.empty(), "Mesh must have texture coordinates");
  MT_THROW_IF(mesh.texcoord_faces.empty(), "Mesh must have texture coordinate faces");
  MT_THROW_IF(threshold < 0.0f || threshold > 1.0f, "threshold must be in range [0, 1]");

  const auto samplePoints = getSamplePoints(numSamples);
  const auto nSamples = static_cast<int32_t>(samplePoints.size());
  const auto colorLookup = buildColorLookup(regionColors);
  const auto nTriangles = static_cast<int64_t>(mesh.faces.size());
  const auto nRegions = regionColors.nRegions;

  const int32_t minSamplesNeeded = (threshold == 0.0f)
      ? 1
      : static_cast<int32_t>(std::ceil(threshold * static_cast<float>(nSamples)));

  std::vector<std::vector<int32_t>> regionTriangles(nRegions);
  std::vector<int32_t> regionCounts(nRegions, 0);

  for (int64_t triIdx = 0; triIdx < nTriangles; ++triIdx) {
    std::fill(regionCounts.begin(), regionCounts.end(), 0);

    const auto& texFace = mesh.texcoord_faces[triIdx];
    const auto& uv0 = mesh.texcoords[texFace.x()];
    const auto& uv1 = mesh.texcoords[texFace.y()];
    const auto& uv2 = mesh.texcoords[texFace.z()];

    for (const auto& bary : samplePoints) {
      const float u = bary.u * uv0.x() + bary.v * uv1.x() + bary.w * uv2.x();
      const float v = bary.u * uv0.y() + bary.v * uv1.y() + bary.w * uv2.y();

      const auto col = std::clamp(
          static_cast<int64_t>(std::lround(u * static_cast<float>(texture.width - 1))),
          int64_t{0},
          texture.width - 1);
      const auto row = std::clamp(
          static_cast<int64_t>(std::lround(v * static_cast<float>(texture.height - 1))),
          int64_t{0},
          texture.height - 1);

      const auto* pixel = texture.data + row * texture.stride0 + col * texture.stride1;
      const auto packedColor = packRgb(
          pixel[0 * texture.stride2], pixel[1 * texture.stride2], pixel[2 * texture.stride2]);

      auto it = colorLookup.find(packedColor);
      if (it != colorLookup.end() && it->second >= 0 && it->second < nRegions) {
        ++regionCounts[it->second];
      }
    }

    for (int32_t regionIdx = 0; regionIdx < nRegions; ++regionIdx) {
      if (regionCounts[regionIdx] >= minSamplesNeeded) {
        regionTriangles[regionIdx].push_back(static_cast<int32_t>(triIdx));
      }
    }
  }

  return regionTriangles;
}

namespace {

struct TextureSampler {
  const TextureView& tex;
  std::unordered_set<size_t> insideColors;

  bool isInside(const Eigen::Vector2f& uv) const {
    const auto col = std::clamp(
        static_cast<int64_t>(std::lround(uv.x() * static_cast<float>(tex.width - 1))),
        int64_t{0},
        tex.width - 1);
    const auto row = std::clamp(
        static_cast<int64_t>(std::lround(uv.y() * static_cast<float>(tex.height - 1))),
        int64_t{0},
        tex.height - 1);
    const auto* pixel = tex.data + row * tex.stride0 + col * tex.stride1;
    return insideColors.count(
               packRgb(pixel[0 * tex.stride2], pixel[1 * tex.stride2], pixel[2 * tex.stride2])) > 0;
  }

  float findCrossingT(const Eigen::Vector2f& uvA, const Eigen::Vector2f& uvB, int32_t numSteps)
      const {
    float lo = 0.0f, hi = 1.0f;
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

using Edge = std::pair<int, int>;

Edge makeEdge(int v1, int v2) {
  return v1 < v2 ? Edge{v1, v2} : Edge{v2, v1};
}

struct HashEdge {
  std::size_t operator()(const Edge& e) const {
    std::size_t h1 = std::hash<int>{}(e.first);
    std::size_t h2 = std::hash<int>{}(e.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

struct GeomCacheEntry {
  int vertexIndex;
  float t;
};

float findEdgeCrossingT(
    const TextureSampler& sampler,
    const Eigen::Vector2f& uv0,
    const Eigen::Vector2f& uv1,
    int geomV0,
    int geomV1,
    bool v0IsInside,
    const std::unordered_map<Edge, GeomCacheEntry, HashEdge>& geomCache,
    int32_t numSteps) {
  const auto edge = makeEdge(geomV0, geomV1);
  auto it = geomCache.find(edge);
  if (it != geomCache.end()) {
    float tOriented = it->second.t;
    return (geomV0 < geomV1) ? tOriented : (1.0f - tOriented);
  }
  if (v0IsInside) {
    return sampler.findCrossingT(uv0, uv1, numSteps);
  }
  return 1.0f - sampler.findCrossingT(uv1, uv0, numSteps);
}

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
  int solo = -1;
  for (int k = 0; k < 3; ++k) {
    if ((nInside == 1 && inside[k]) || (nInside == 2 && !inside[k])) {
      solo = k;
      break;
    }
  }

  const int k0 = solo, k1 = (solo + 1) % 3, k2 = (solo + 2) % 3;
  const int gV0 = face[k0], gV1 = face[k1], gV2 = face[k2];
  const int tV0 = texFace[k0], tV1 = texFace[k1], tV2 = texFace[k2];

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

  const int gM01 = getOrCreateGeomVertex(gV0, gV1, t01);
  const int gM02 = getOrCreateGeomVertex(gV0, gV2, t02);
  const int tM01 = getOrCreateTexVertex(tV0, tV1, t01);
  const int tM02 = getOrCreateTexVertex(tV0, tV2, t02);

  if (nInside == 1) {
    newFaces.emplace_back(gV0, gM01, gM02);
    newTexcoordFaces.emplace_back(tV0, tM01, tM02);
  } else {
    newFaces.emplace_back(gV1, gV2, gM01);
    newTexcoordFaces.emplace_back(tV1, tV2, tM01);
    newFaces.emplace_back(gV2, gM02, gM01);
    newTexcoordFaces.emplace_back(tV2, tM02, tM01);
  }
}

} // namespace

Mesh splitMeshByTextureRegion(
    const Mesh& mesh,
    const TextureView& texture,
    const RegionColorsView& regionColors,
    int32_t numBinarySearchSteps) {
  MT_THROW_IF(mesh.texcoords.empty(), "Mesh must have texture coordinates");
  MT_THROW_IF(mesh.texcoord_faces.empty(), "Mesh must have texture coordinate faces");
  MT_THROW_IF(
      mesh.faces.size() != mesh.texcoord_faces.size(),
      "faces and texcoord_faces must have the same size");

  TextureSampler sampler{texture, buildInsideColorSet(regionColors)};

  const auto nTexVerts = static_cast<int>(mesh.texcoords.size());
  std::vector<bool> texVertInside(nTexVerts);
  for (int i = 0; i < nTexVerts; ++i) {
    texVertInside[i] = sampler.isInside(mesh.texcoords[i]);
  }

  auto vertices = mesh.vertices;
  auto normals = mesh.normals;
  auto colors = mesh.colors;
  auto confidence = mesh.confidence;
  auto texcoords = mesh.texcoords;

  std::vector<Eigen::Vector3i> newFaces;
  std::vector<Eigen::Vector3i> newTexcoordFaces;
  newFaces.reserve(mesh.faces.size());
  newTexcoordFaces.reserve(mesh.faces.size());

  std::unordered_map<Edge, GeomCacheEntry, HashEdge> geomCache;
  std::unordered_map<Edge, int, HashEdge> texCache;

  const bool hasNormals = !normals.empty();
  const bool hasColors = !colors.empty();
  const bool hasConfidence = !confidence.empty();

  auto getOrCreateGeomVertex = [&](int geomA, int geomB, float t) -> int {
    const auto edge = makeEdge(geomA, geomB);
    auto it = geomCache.find(edge);
    if (it != geomCache.end()) {
      return it->second.vertexIndex;
    }
    float tOriented = (geomA < geomB) ? t : (1.0f - t);
    const int newIdx = static_cast<int>(vertices.size());
    vertices.emplace_back(
        (1.0f - tOriented) * vertices[edge.first] + tOriented * vertices[edge.second]);
    if (hasNormals) {
      Eigen::Vector3f n =
          ((1.0f - tOriented) * normals[edge.first] + tOriented * normals[edge.second])
              .normalized();
      normals.emplace_back(n);
    }
    if (hasColors) {
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

    std::array<bool, 3> inside = {
        texVertInside[texFace.x()], texVertInside[texFace.y()], texVertInside[texFace.z()]};
    const int nInside =
        static_cast<int>(inside[0]) + static_cast<int>(inside[1]) + static_cast<int>(inside[2]);

    if (nInside == 3) {
      newFaces.push_back(face);
      newTexcoordFaces.push_back(texFace);
    } else if (nInside > 0) {
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
  }

  Mesh result;
  result.vertices = std::move(vertices);
  result.normals = std::move(normals);
  result.colors = std::move(colors);
  result.confidence = std::move(confidence);
  result.texcoords = std::move(texcoords);
  result.faces = std::move(newFaces);
  result.texcoord_faces = std::move(newTexcoordFaces);

  if (result.faces.empty()) {
    return Mesh{};
  }

  std::vector<bool> activeFaces(result.faces.size(), true);
  return reduceMeshByFaces<float>(result, activeFaces);
}

} // namespace momentum
