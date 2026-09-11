/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/obj/obj_io.h"

#include "momentum/common/exception.h"
#include "momentum/common/log.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace momentum {

namespace {

/// One face corner. ``texcoord`` is -1 when the corner carries no `vt` index.
struct Corner {
  int32_t vertex = -1;
  int32_t texcoord = -1;
};

/// Parse an OBJ index, reporting the offending token and line the way the rest of the
/// parser does. `std::stoll` alone throws a bare std exception naming neither.
///
/// `int64_t` rather than `long`, which is 32 bits on LLP64 (Windows/MSVC) and would
/// truncate the `count` cast in resolveIndex below.
int64_t parseIndex(const std::string& field, const char* what, size_t lineNumber) {
  size_t consumed = 0;
  int64_t value = 0;
  try {
    value = std::stoll(field, &consumed);
  } catch (const std::exception&) {
    MT_THROW("OBJ line {}: {} index '{}' is not a valid integer", lineNumber, what, field);
  }
  MT_THROW_IF(
      consumed != field.size(),
      "OBJ line {}: {} index '{}' has trailing characters",
      lineNumber,
      what,
      field);
  return value;
}

/// Resolve a 1-based OBJ index, where a negative value counts back from the end of what has
/// been read so far -- which is why this can only be done while parsing, not afterwards.
int32_t resolveIndex(int64_t raw, size_t count, const char* what, size_t lineNumber) {
  MT_THROW_IF(
      raw == 0, "OBJ line {}: {} index 0 is invalid, OBJ indices are 1-based", lineNumber, what);
  const int64_t resolved = (raw > 0) ? (raw - 1) : (static_cast<int64_t>(count) + raw);
  MT_THROW_IF(
      resolved < 0 || resolved >= static_cast<int64_t>(count),
      "OBJ line {}: {} index {} is out of range, {} have been read",
      lineNumber,
      what,
      raw,
      count);
  return static_cast<int32_t>(resolved);
}

/// Parse one whitespace-delimited face corner: `v`, `v/vt`, `v//vn` or `v/vt/vn`.
Corner
parseCorner(const std::string& token, size_t numVertices, size_t numTexcoords, size_t lineNumber) {
  const size_t firstSlash = token.find('/');
  const std::string vertexField = token.substr(0, firstSlash);
  MT_THROW_IF(
      vertexField.empty(), "OBJ line {}: face corner '{}' has no vertex index", lineNumber, token);

  Corner corner;
  corner.vertex = resolveIndex(
      parseIndex(vertexField, "vertex", lineNumber), numVertices, "vertex", lineNumber);

  if (firstSlash != std::string::npos) {
    const size_t secondSlash = token.find('/', firstSlash + 1);
    const std::string texcoordField = (secondSlash == std::string::npos)
        ? token.substr(firstSlash + 1)
        : token.substr(firstSlash + 1, secondSlash - firstSlash - 1);
    // Empty means the `v//vn` form: a normal index but no texture coordinate.
    if (!texcoordField.empty()) {
      corner.texcoord = resolveIndex(
          parseIndex(texcoordField, "texture coordinate", lineNumber),
          numTexcoords,
          "texture coordinate",
          lineNumber);
    }
  }
  return corner;
}

uint8_t toByteColor(float v) {
  return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}

Mesh parseObj(std::istream& stream) {
  Mesh mesh;
  std::vector<Eigen::Vector2f> texcoords;
  std::vector<Eigen::Vector3i> texcoordFaces;
  std::vector<Corner> corners;

  // Texture coordinates are kept only if *every* face has them, since texcoord_faces has to
  // stay index-aligned with faces.
  bool allFacesTextured = true;
  bool anyFaceTextured = false;

  std::string line;
  size_t lineNumber = 0;
  while (std::getline(stream, line)) {
    ++lineNumber;
    if (const size_t comment = line.find('#'); comment != std::string::npos) {
      line.resize(comment);
    }

    std::istringstream fields(line);
    std::string token;
    if (!(fields >> token)) {
      continue;
    }

    if (token == "v") {
      std::vector<float> values;
      float value = 0.0f;
      while (fields >> value) {
        values.push_back(value);
      }
      MT_THROW_IF(
          values.size() < 3,
          "OBJ line {}: vertex needs at least 3 coordinates, got {}",
          lineNumber,
          values.size());
      mesh.vertices.emplace_back(values[0], values[1], values[2]);
      // `v x y z r g b` -- the six-float form carries a vertex colour. Four floats is the
      // rational `w` form instead, whose weight momentum has nowhere to put.
      if (values.size() >= 6) {
        mesh.colors.emplace_back(
            toByteColor(values[3]), toByteColor(values[4]), toByteColor(values[5]));
      }
    } else if (token == "vt") {
      float u = 0.0f;
      float v = 0.0f;
      MT_THROW_IF(
          !(fields >> u >> v),
          "OBJ line {}: texture coordinate needs at least two floats",
          lineNumber);
      texcoords.emplace_back(u, v);
    } else if (token == "f") {
      corners.clear();
      std::string cornerToken;
      while (fields >> cornerToken) {
        corners.push_back(
            parseCorner(cornerToken, mesh.vertices.size(), texcoords.size(), lineNumber));
      }
      if (corners.size() < 3) {
        MT_LOGW("OBJ line {}: skipping face with {} corners", lineNumber, corners.size());
        continue;
      }

      const bool textured =
          std::ranges::all_of(corners, [](const Corner& c) { return c.texcoord >= 0; });
      anyFaceTextured |= textured;
      allFacesTextured &= textured;

      mesh.polyFaceSizes.push_back(static_cast<uint32_t>(corners.size()));
      for (const Corner& c : corners) {
        mesh.polyFaces.push_back(static_cast<uint32_t>(c.vertex));
        if (textured) {
          mesh.polyTexcoordFaces.push_back(static_cast<uint32_t>(c.texcoord));
        }
      }

      for (size_t i = 2; i < corners.size(); ++i) {
        mesh.faces.emplace_back(corners[0].vertex, corners[i - 1].vertex, corners[i].vertex);
        if (textured) {
          texcoordFaces.emplace_back(
              corners[0].texcoord, corners[i - 1].texcoord, corners[i].texcoord);
        }
      }
    }
    // `vn`, `g`, `o`, `s`, `usemtl` and `mtllib` are deliberately ignored; see the header.
  }

  MT_THROW_IF(mesh.vertices.empty(), "OBJ file contains no vertices");
  MT_THROW_IF(mesh.faces.empty(), "OBJ file contains no faces");

  if (anyFaceTextured && allFacesTextured) {
    mesh.texcoords = std::move(texcoords);
    mesh.texcoord_faces = std::move(texcoordFaces);
  } else {
    if (anyFaceTextured) {
      MT_LOGW(
          "OBJ file texture-maps only some of its {} faces; loading it untextured",
          mesh.faces.size());
    }
    mesh.polyTexcoordFaces.clear();
  }

  if (mesh.colors.size() != mesh.vertices.size()) {
    if (!mesh.colors.empty()) {
      MT_LOGW(
          "OBJ file gives colours for {} of {} vertices; discarding them",
          mesh.colors.size(),
          mesh.vertices.size());
    }
    mesh.colors.assign(mesh.vertices.size(), Vector3b::Constant(255));
  }
  mesh.confidence.assign(mesh.vertices.size(), 1.0f);
  mesh.updateNormals();
  return mesh;
}

} // namespace

Mesh loadObj(const filesystem::path& filepath) {
  std::ifstream file(filepath);
  MT_THROW_IF(!file.is_open(), "Failed to open OBJ file: {}", filepath.string());
  return parseObj(file);
}

Mesh loadObj(std::span<const std::byte> bytes) {
  std::istringstream stream(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  return parseObj(stream);
}

} // namespace momentum
