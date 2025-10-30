/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <span>

#include <string>
#include <vector>

namespace momentum {

// Structure for holding arbitrary polygon data.  We will read the data as-is from the
// FBX file and then triangulate at the end for momentum.  Doing it this way allows us
// to split up the functionality (triangulating while you read complicates handling
// texture coordinates for example).
struct PolygonData {
  // We will pack polygons in an array like this:
  // The indices array will contain all the polygons indices packed back-to-back.
  //   1 2 3 4   2 3 4 6   ...
  // The offsets array will include an offset for the start and end of each polygon.
  //   0 4 8 ...
  // The advantage of doing it this way is that we have random access into the
  // polygons.  The disadvantage (compared to just providing an array of vertex counts)
  // is that the offset array should always have one more than the number of polygons,
  // which is a little confusing.
  std::vector<uint32_t> indices;
  std::vector<uint32_t> offsets;

  // The texture coordinate indices for each face.
  // Uses the same offsets array as the indices array.
  // Note that this is allowed to be empty, but if not empty it
  // must be the same size as the indices array.
  std::vector<uint32_t> textureIndices;

  PolygonData() : offsets{0} {}

  // If there's a problem with the mesh, returns an error message.
  // If everything is kosher, returns an empty string.
  // This function is used in a few places when reading the mesh from
  // disk to make sure it's valid.
  [[nodiscard]] std::string errorMessage(size_t numVertices) const;

  // Similar to errorMessage, but the issue is not severe enough to terminate.
  [[nodiscard]] std::string warnMessage(size_t numTexVertices) const;

  [[nodiscard]] size_t numPolygons() const;
};

std::vector<Eigen::Vector3i> triangulate(
    std::span<const uint32_t> indices,
    std::span<const uint32_t> offsets);

} // namespace momentum
