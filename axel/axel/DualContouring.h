/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <span>

#include "axel/BoundingBox.h"
#include "axel/SignedDistanceField.h"
#include "axel/common/Types.h"

namespace axel {

/**
 * Result of dual contouring operation.
 * Dual contouring naturally produces quads, so this always contains quads.
 * Use triangulateQuads() to convert to triangles if needed.
 */
template <typename S>
struct DualContouringResult {
  using Scalar = S;

  /// Generated vertices
  std::vector<Eigen::Vector3<S>> vertices;

  /// Generated quads (dual contouring naturally produces quads)
  std::vector<Eigen::Vector4i> quads;

  /// Success flag
  bool success = false;

  /// Number of cells processed
  size_t processedCells = 0;

  /// Number of vertices generated
  size_t generatedVertices = 0;
};

/**
 * Extract an isosurface from a signed distance field using dual contouring.
 *
 * Dual contouring places vertices inside grid cells (rather than on edges like marching cubes)
 * and uses both function values and gradients to determine optimal vertex placement.
 * This results in better preservation of sharp features and corners compared to marching cubes.
 *
 * The algorithm works by:
 * 1. Finding all cells that intersect the isosurface (sign changes across cell corners)
 * 2. Placing one vertex at each intersecting cell, positioned on the surface using gradient descent
 * 3. Generating quads for each edge crossing that connects 4 adjacent cells
 *
 * @param sdf The signed distance field to extract from
 * @param isovalue The isovalue to extract (typically 0.0 for zero level set)
 * @return Result containing extracted mesh
 */
template <typename ScalarType>
DualContouringResult<ScalarType> dualContouring(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue = ScalarType{0.0});

/**
 * Triangulate a quad mesh into triangles.
 * Each quad is split into two triangles using the diagonal (0,2).
 *
 * @param quads Vector of quads to triangulate
 * @return Vector of triangles
 */
std::vector<Eigen::Vector3i> triangulateQuads(const std::vector<Eigen::Vector4i>& quads);

} // namespace axel
