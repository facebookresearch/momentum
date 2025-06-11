/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/polygon_data.h"

#include "momentum/common/exception.h"
#include "momentum/common/log.h"

namespace momentum {

inline constexpr uint32_t kInvalidTexCoord = static_cast<uint32_t>(-1);

std::vector<Eigen::Vector3i> triangulate(
    gsl::span<const uint32_t> indices,
    gsl::span<const uint32_t> offsets) {
  if (indices.empty()) {
    return {};
  }

  const auto nFaces = offsets.size() - 1;

  std::vector<Eigen::Vector3i> result;
  result.reserve(2 * nFaces);

  for (size_t iFace = 0; iFace < nFaces; ++iFace) {
    const auto faceStart = offsets[iFace];
    const auto faceEnd = offsets[iFace + 1];
    const auto nv = faceEnd - faceStart;
    MT_THROW_IF(nv < 3, "Invalid face with {} indices; expected at least 3.", nv);

    for (size_t j = 1; j < (nv - 1); ++j) {
      result.emplace_back(
          indices[faceStart + 0], indices[faceStart + j], indices[faceStart + j + 1]);
    }
  }

  return result;
}

size_t PolygonData::numPolygons() const {
  MT_CHECK(!offsets.empty());
  return offsets.size() - 1;
}

std::string PolygonData::errorMessage(const size_t numVertices) const {
  if (offsets.empty()) {
    return "Empty offsets array";
  }

  if (offsets.front() != 0) {
    return "Invalid first offset (expected 0)";
  }

  if (!textureIndices.empty() && textureIndices.size() != indices.size()) {
    return "Mismatch between indices and textureIndices";
  }

  const auto nPoly = numPolygons();

  for (size_t iFace = 0; iFace < nPoly; ++iFace) {
    const auto faceBegin = offsets[iFace];
    const auto faceEnd = offsets[iFace + 1];

    if (faceBegin >= faceEnd) {
      return "Non-monotonic offsets array";
    }

    if (faceEnd > indices.size()) {
      return "Too-large offset";
    }

    const auto nv = faceEnd - faceBegin;
    if (nv < 3) {
      return "Too few vertices in face";
    }
  }

  for (const auto& i : indices) {
    if (i >= numVertices) {
      return "Invalid too-large vertex";
    }
  }

  if (offsets.back() != indices.size()) {
    return "Unused trailing vertices";
  }

  return {};
}

std::string PolygonData::warnMessage(const size_t numTexVertices) const {
  for (const auto& i : textureIndices) {
    if (i != kInvalidTexCoord && i >= numTexVertices) {
      return "Invalid too-large texture vertex";
    }
  }

  return {};
}

} // namespace momentum
