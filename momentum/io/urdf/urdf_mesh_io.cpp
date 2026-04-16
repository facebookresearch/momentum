/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_mesh_io.h"

#include <cmath>
#include "momentum/common/log.h"
#include "momentum/math/constants.h"

#include <array>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace momentum {

namespace {

// Hash for Eigen::Vector3f to support vertex deduplication in STL loading.
struct Vector3fHash {
  size_t operator()(const Eigen::Vector3f& v) const {
    size_t h = 0;
    for (int i = 0; i < 3; ++i) {
      // Combine bits of each float component
      uint32_t bits = 0;
      std::memcpy(&bits, &v[i], sizeof(float));
      h ^= std::hash<uint32_t>{}(bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

// Approximate equality for vertex deduplication.
struct Vector3fEqual {
  bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
    return a.isApprox(b, 1e-7f);
  }
};

// Returns the deduplicated vertex index, inserting a new vertex if needed.
int32_t deduplicateVertex(
    const Eigen::Vector3f& v,
    std::unordered_map<Eigen::Vector3f, int32_t, Vector3fHash, Vector3fEqual>& vertexMap,
    std::vector<Eigen::Vector3f>& vertices) {
  auto it = vertexMap.find(v);
  if (it != vertexMap.end()) {
    return it->second;
  }
  const auto idx = static_cast<int32_t>(vertices.size());
  vertexMap[v] = idx;
  vertices.push_back(v);
  return idx;
}

// Try to detect if an STL file is binary (as opposed to ASCII).
// A binary STL has an 80-byte header + 4-byte triangle count, then
// 50 bytes per triangle. If the file size matches, it's binary.
bool isBinaryStl(const filesystem::path& filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  const auto fileSize = file.tellg();
  if (fileSize < 84) {
    return false; // Too small for binary STL
  }

  // Read triangle count from bytes 80-83
  file.seekg(80);
  uint32_t numTriangles = 0;
  file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));

  // Binary STL: 80 (header) + 4 (count) + numTriangles * 50
  const auto expectedSize =
      static_cast<std::streamoff>(84) + static_cast<std::streamoff>(numTriangles) * 50;
  return fileSize == expectedSize;
}

std::optional<Mesh> loadBinaryStl(const filesystem::path& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return std::nullopt;
  }

  // Skip 80-byte header
  file.seekg(80);

  uint32_t numTriangles = 0;
  file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
  if (!file.good() || numTriangles == 0) {
    return std::nullopt;
  }

  Mesh mesh;
  std::unordered_map<Eigen::Vector3f, int32_t, Vector3fHash, Vector3fEqual> vertexMap;

  for (uint32_t i = 0; i < numTriangles; ++i) {
    // Read normal (3 floats) — we'll recompute normals later
    std::array<float, 3> normal{};
    file.read(reinterpret_cast<char*>(normal.data()), sizeof(float) * 3);

    // Read 3 vertices (9 floats)
    std::array<Eigen::Vector3f, 3> v;
    for (auto& j : v) {
      std::array<float, 3> coords{};
      file.read(reinterpret_cast<char*>(coords.data()), sizeof(float) * 3);
      j = Eigen::Vector3f(coords[0], coords[1], coords[2]);
    }

    // Skip attribute byte count (2 bytes)
    uint16_t attrByteCount = 0;
    file.read(reinterpret_cast<char*>(&attrByteCount), sizeof(uint16_t));

    if (!file.good()) {
      return std::nullopt;
    }

    // Deduplicate vertices and add face
    Eigen::Vector3i face;
    for (int j = 0; j < 3; ++j) {
      face[j] = deduplicateVertex(v[j], vertexMap, mesh.vertices);
    }
    mesh.faces.push_back(face);
  }

  mesh.updateNormals();
  return mesh;
}

std::optional<Mesh> loadAsciiStl(const filesystem::path& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return std::nullopt;
  }

  Mesh mesh;
  std::unordered_map<Eigen::Vector3f, int32_t, Vector3fHash, Vector3fEqual> vertexMap;
  std::string line;
  std::string token;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> token;

    if (token == "vertex") {
      // Read three vertex lines for one facet
      std::array<Eigen::Vector3f, 3> v;
      float x = NAN, y = NAN, z = NAN;
      iss >> x >> y >> z;
      v[0] = Eigen::Vector3f(x, y, z);

      for (int j = 1; j < 3; ++j) {
        if (!std::getline(file, line)) {
          return std::nullopt;
        }
        std::istringstream viss(line);
        viss >> token; // "vertex"
        viss >> x >> y >> z;
        v[j] = Eigen::Vector3f(x, y, z);
      }

      Eigen::Vector3i face;
      for (int j = 0; j < 3; ++j) {
        face[j] = deduplicateVertex(v[j], vertexMap, mesh.vertices);
      }
      mesh.faces.push_back(face);
    }
  }

  if (mesh.faces.empty()) {
    return std::nullopt;
  }

  mesh.updateNormals();
  return mesh;
}

} // namespace

Mesh generateBoxMesh(float dimX, float dimY, float dimZ) {
  const float hx = dimX * 0.5f;
  const float hy = dimY * 0.5f;
  const float hz = dimZ * 0.5f;

  Mesh mesh;
  mesh.vertices = {
      {-hx, -hy, -hz},
      {hx, -hy, -hz},
      {hx, hy, -hz},
      {-hx, hy, -hz},
      {-hx, -hy, hz},
      {hx, -hy, hz},
      {hx, hy, hz},
      {-hx, hy, hz},
  };

  // Two triangles per face, 6 faces = 12 triangles.
  // Winding order: counter-clockwise when viewed from outside.
  mesh.faces = {
      // -Z face
      {0, 2, 1},
      {0, 3, 2},
      // +Z face
      {4, 5, 6},
      {4, 6, 7},
      // -Y face
      {0, 1, 5},
      {0, 5, 4},
      // +Y face
      {2, 3, 7},
      {2, 7, 6},
      // -X face
      {0, 4, 7},
      {0, 7, 3},
      // +X face
      {1, 2, 6},
      {1, 6, 5},
  };

  mesh.updateNormals();
  return mesh;
}

Mesh generateCylinderMesh(float radius, float length, int segments) {
  Mesh mesh;

  const float halfLen = length * 0.5f;

  // Bottom center vertex (index 0)
  mesh.vertices.emplace_back(0.0f, 0.0f, -halfLen);
  // Top center vertex (index 1)
  mesh.vertices.emplace_back(0.0f, 0.0f, halfLen);

  // Ring vertices: bottom ring starts at index 2, top ring at index 2 + segments
  for (int i = 0; i < segments; ++i) {
    const float angle = twopi<float>() * static_cast<float>(i) / static_cast<float>(segments);
    const float x = radius * std::cos(angle);
    const float y = radius * std::sin(angle);
    mesh.vertices.emplace_back(x, y, -halfLen); // bottom ring
  }
  for (int i = 0; i < segments; ++i) {
    const float angle = twopi<float>() * static_cast<float>(i) / static_cast<float>(segments);
    const float x = radius * std::cos(angle);
    const float y = radius * std::sin(angle);
    mesh.vertices.emplace_back(x, y, halfLen); // top ring
  }

  const int bottomRingStart = 2;
  const int topRingStart = 2 + segments;

  for (int i = 0; i < segments; ++i) {
    const int next = (i + 1) % segments;

    // Bottom cap (winding: center, next, current — looking from -Z)
    mesh.faces.emplace_back(0, bottomRingStart + next, bottomRingStart + i);

    // Top cap (winding: center, current, next — looking from +Z)
    mesh.faces.emplace_back(1, topRingStart + i, topRingStart + next);

    // Side quads (two triangles each)
    const int b0 = bottomRingStart + i;
    const int b1 = bottomRingStart + next;
    const int t0 = topRingStart + i;
    const int t1 = topRingStart + next;
    mesh.faces.emplace_back(b0, b1, t1);
    mesh.faces.emplace_back(b0, t1, t0);
  }

  mesh.updateNormals();
  return mesh;
}

Mesh generateSphereMesh(float radius, int latSegments, int lonSegments) {
  Mesh mesh;

  // Top pole (index 0)
  mesh.vertices.emplace_back(0.0f, 0.0f, radius);

  // Latitude rings (from top to bottom, excluding poles)
  for (int lat = 1; lat < latSegments; ++lat) {
    const float phi = pi<float>() * static_cast<float>(lat) / static_cast<float>(latSegments);
    const float sinPhi = std::sin(phi);
    const float cosPhi = std::cos(phi);

    for (int lon = 0; lon < lonSegments; ++lon) {
      const float theta =
          twopi<float>() * static_cast<float>(lon) / static_cast<float>(lonSegments);
      const float x = radius * sinPhi * std::cos(theta);
      const float y = radius * sinPhi * std::sin(theta);
      const float z = radius * cosPhi;
      mesh.vertices.emplace_back(x, y, z);
    }
  }

  // Bottom pole (last vertex)
  const auto bottomPole = static_cast<int32_t>(mesh.vertices.size());
  mesh.vertices.emplace_back(0.0f, 0.0f, -radius);

  // Top cap triangles (pole to first ring)
  for (int lon = 0; lon < lonSegments; ++lon) {
    const int next = (lon + 1) % lonSegments;
    mesh.faces.emplace_back(0, 1 + lon, 1 + next);
  }

  // Middle quad strips between rings
  for (int lat = 0; lat < latSegments - 2; ++lat) {
    const int ringStart = 1 + lat * lonSegments;
    const int nextRingStart = 1 + (lat + 1) * lonSegments;
    for (int lon = 0; lon < lonSegments; ++lon) {
      const int next = (lon + 1) % lonSegments;
      const int a = ringStart + lon;
      const int b = ringStart + next;
      const int c = nextRingStart + next;
      const int d = nextRingStart + lon;
      mesh.faces.emplace_back(a, d, c);
      mesh.faces.emplace_back(a, c, b);
    }
  }

  // Bottom cap triangles (last ring to pole)
  const int lastRingStart = 1 + (latSegments - 2) * lonSegments;
  for (int lon = 0; lon < lonSegments; ++lon) {
    const int next = (lon + 1) % lonSegments;
    mesh.faces.emplace_back(lastRingStart + lon, bottomPole, lastRingStart + next);
  }

  mesh.updateNormals();
  return mesh;
}

std::optional<Mesh> loadStlMesh(const filesystem::path& filepath) {
  if (!filesystem::exists(filepath)) {
    MT_LOGW("STL file not found: {}", filepath.string());
    return std::nullopt;
  }

  if (isBinaryStl(filepath)) {
    return loadBinaryStl(filepath);
  }
  return loadAsciiStl(filepath);
}

std::optional<Mesh> loadObjMesh(const filesystem::path& filepath) {
  if (!filesystem::exists(filepath)) {
    MT_LOGW("OBJ file not found: {}", filepath.string());
    return std::nullopt;
  }

  std::ifstream file(filepath);
  if (!file.is_open()) {
    MT_LOGW("Failed to open OBJ file: {}", filepath.string());
    return std::nullopt;
  }

  Mesh mesh;
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "v") {
      float x = NAN, y = NAN, z = NAN;
      iss >> x >> y >> z;
      mesh.vertices.emplace_back(x, y, z);
    } else if (token == "f") {
      // Parse face indices. Supports formats:
      //   f v1 v2 v3 ...
      //   f v1/vt1 v2/vt2 v3/vt3 ...
      //   f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
      //   f v1//vn1 v2//vn2 v3//vn3 ...
      std::vector<int32_t> faceIndices;
      std::string vertexToken;
      while (iss >> vertexToken) {
        // Extract vertex index (everything before the first '/')
        const auto slashPos = vertexToken.find('/');
        const std::string indexStr =
            (slashPos != std::string::npos) ? vertexToken.substr(0, slashPos) : vertexToken;
        const int idx = std::stoi(indexStr);
        // OBJ indices are 1-based; negative indices are relative to current vertex count
        const int32_t resolvedIdx =
            (idx > 0) ? (idx - 1) : static_cast<int32_t>(mesh.vertices.size()) + idx;
        faceIndices.push_back(resolvedIdx);
      }

      // Triangulate polygon faces using fan triangulation
      if (faceIndices.size() >= 3) {
        for (size_t i = 2; i < faceIndices.size(); ++i) {
          mesh.faces.emplace_back(
              faceIndices[0], faceIndices[static_cast<int32_t>(i) - 1], faceIndices[i]);
        }
      }
    }
  }

  if (mesh.vertices.empty() || mesh.faces.empty()) {
    MT_LOGW("OBJ file contains no geometry: {}", filepath.string());
    return std::nullopt;
  }

  mesh.updateNormals();
  return mesh;
}

std::optional<filesystem::path> resolveMeshPath(
    const std::string& filename,
    const filesystem::path& urdfDir) {
  // Handle file:// URI
  const std::string fileScheme = "file://";
  if (filename.substr(0, fileScheme.size()) == fileScheme) {
    const filesystem::path resolved(filename.substr(fileScheme.size()));
    if (filesystem::exists(resolved)) {
      return resolved;
    }
    MT_LOGW("Mesh file not found: {}", resolved.string());
    return std::nullopt;
  }

  // Handle package:// URI — strip scheme and package name, resolve relative to URDF dir
  const std::string packageScheme = "package://";
  if (filename.substr(0, packageScheme.size()) == packageScheme) {
    // Find the end of the package name (next '/' after "package://")
    const auto pathStart = filename.find('/', packageScheme.size());
    if (pathStart == std::string::npos) {
      MT_LOGW("Invalid package:// URI: {}", filename);
      return std::nullopt;
    }
    const std::string relativePath = filename.substr(pathStart + 1);

    // Try resolving relative to URDF directory
    filesystem::path resolved = urdfDir / relativePath;
    if (filesystem::exists(resolved)) {
      return resolved;
    }

    // Try resolving relative to parent of URDF directory
    // (common layout: pkg/urdf/robot.urdf and pkg/meshes/part.stl)
    resolved = urdfDir.parent_path() / relativePath;
    if (filesystem::exists(resolved)) {
      return resolved;
    }

    MT_LOGW("Could not resolve package:// mesh path: {}", filename);
    return std::nullopt;
  }

  // Handle absolute paths
  const filesystem::path path(filename);
  if (path.is_absolute()) {
    if (filesystem::exists(path)) {
      return path;
    }
    MT_LOGW("Mesh file not found: {}", path.string());
    return std::nullopt;
  }

  // Handle relative paths — resolve relative to URDF directory
  const filesystem::path resolved = urdfDir / path;
  if (filesystem::exists(resolved)) {
    return resolved;
  }

  MT_LOGW("Mesh file not found: {}", resolved.string());
  return std::nullopt;
}

} // namespace momentum
