/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/mesh.h>

#include <cstdint>
#include <vector>

namespace momentum {

/// View into a contiguous RGB texture image (non-owning).
struct TextureView {
  const uint8_t* data;
  int64_t height;
  int64_t width;
  int64_t stride0; ///< Row stride in bytes
  int64_t stride1; ///< Column stride in bytes
  int64_t stride2; ///< Channel stride in bytes
};

/// View into a contiguous array of RGB region colors (non-owning).
struct RegionColorsView {
  const uint8_t* data;
  int32_t nRegions;
  int64_t stride0; ///< Stride between regions in bytes
  int64_t stride1; ///< Stride between channels in bytes
};

/// Classify mesh triangles into regions based on texture colors.
///
/// For each triangle, samples the texture at barycentric coordinates within the
/// triangle, matches the sampled color against the provided region colors, and
/// assigns the triangle to any matching regions.
///
/// @param mesh The mesh with texcoords and texcoord_faces.
/// @param texture RGB texture image data.
/// @param regionColors RGB colors defining each region.
/// @param threshold Fraction of samples that must match for a triangle to be
///     assigned to a region. 0.0 means at least one sample must match.
/// @param numSamples Number of barycentric sample points per triangle
///     (1, 3, 4, 6, 7, or 10).
/// @return Vector of sorted triangle index lists, one per region.
std::vector<std::vector<int32_t>> classifyTrianglesByTexture(
    const Mesh& mesh,
    const TextureView& texture,
    const RegionColorsView& regionColors,
    float threshold = 0.0f,
    int32_t numSamples = 3);

/// Split mesh triangles along texture region boundaries.
///
/// For each face, classifies its texcoord vertices as inside or outside the
/// specified region colors. Faces fully inside are kept, fully outside are
/// discarded, and boundary faces are split by binary-searching for the crossing
/// point along each edge in UV space.
///
/// @param mesh The mesh with vertices, faces, texcoords, and texcoord_faces.
/// @param texture RGB texture image data.
/// @param regionColors RGB colors defining the "inside" region.
/// @param numBinarySearchSteps Number of binary search iterations for edge
///     crossing precision (8 gives 1/256 sub-pixel accuracy).
/// @return A new Mesh containing only the inside portion, with boundary
///     triangles split along the texture region boundary.
Mesh splitMeshByTextureRegion(
    const Mesh& mesh,
    const TextureView& texture,
    const RegionColorsView& regionColors,
    int32_t numBinarySearchSteps = 8);

} // namespace momentum
