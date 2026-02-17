/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <span>

#include "axel/common/Types.h"

namespace axel {

/**
 * Method for filling mesh holes.
 */
enum class HoleFillingMethod {
  /// Add a centroid vertex and create fan triangles (default, best for SDF generation)
  Centroid,
  /// Use ear clipping without adding new vertices
  EarClipping,
  /// Fill with a smooth hemispherical cap (produces smoother SDF gradients near cut boundaries)
  SphericalCap,
  /// Automatically choose based on hole size (centroid for ≤8 vertices, ear clipping for larger)
  Auto
};

/**
 * Represents a hole boundary in the mesh.
 */
struct HoleBoundary {
  /// Ordered list of vertex indices forming the hole boundary
  std::vector<Index> vertices;

  /// Edge pairs (vertex_i, vertex_i+1) forming the boundary
  std::vector<std::pair<Index, Index>> edges;

  /// Center point of the hole (computed from boundary vertices)
  Eigen::Vector3f center;

  /// Approximate radius of the hole
  float radius{};
};

/**
 * Result of hole filling operation.
 */
struct HoleFillingResult {
  /// New vertices added during hole filling
  std::vector<Eigen::Vector3f> newVertices;

  /// New triangles added (indices refer to original + new vertices)
  std::vector<Eigen::Vector3i> newTriangles;

  /// Information about filled holes
  std::vector<HoleBoundary> filledHoles;

  /// Success flag
  bool success = false;

  /// Number of holes that were successfully filled
  size_t holesFilledCount = 0;
};

/**
 * Detect holes/boundaries in a triangle mesh.
 *
 * @param vertices Mesh vertices
 * @param triangles Mesh triangles
 * @return List of detected hole boundaries
 */
std::vector<HoleBoundary> detectMeshHoles(
    std::span<const Eigen::Vector3f> vertices,
    std::span<const Eigen::Vector3i> triangles);

/**
 * Fill holes in a triangle mesh.
 *
 * This function identifies holes in the mesh and fills them with new triangles
 * to create a watertight surface suitable for SDF generation.
 *
 * @param vertices Original mesh vertices
 * @param triangles Original mesh triangles
 * @param method Hole filling method (default: Centroid for best triangle quality)
 * @param capHeightRatio Cap height ratio for SphericalCap method, as a fraction of hole radius
 *   (0.0 = flat fill, 0.5 = half-hemisphere, 1.0 = full hemisphere). Ignored for other methods.
 * @return Result containing new vertices and triangles, or failure info
 */
template <typename ScalarType>
HoleFillingResult fillMeshHoles(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    HoleFillingMethod method = HoleFillingMethod::Centroid,
    float capHeightRatio = 0.5f);

/**
 * Convenience function that fills holes and returns complete mesh.
 *
 * @param vertices Original mesh vertices
 * @param triangles Original mesh triangles
 * @param method Hole filling method (default: Centroid for best triangle quality)
 * @param capHeightRatio Cap height ratio for SphericalCap method, as a fraction of hole radius
 *   (0.0 = flat fill, 0.5 = half-hemisphere, 1.0 = full hemisphere). Ignored for other methods.
 * @return Pair of (all_vertices, all_triangles) with holes filled
 */
template <typename ScalarType>
std::pair<std::vector<Eigen::Vector3<ScalarType>>, std::vector<Eigen::Vector3i>>
fillMeshHolesComplete(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    HoleFillingMethod method = HoleFillingMethod::Centroid,
    float capHeightRatio = 0.5f);

/**
 * Apply Laplacian smoothing to mesh vertices with optional masking.
 *
 * @param vertices Input mesh vertices
 * @param faces Mesh faces (triangles or quads)
 * @param vertex_mask Optional mask to specify which vertices to smooth (if empty, all vertices are
 * smoothed)
 * @param iterations Number of smoothing iterations
 * @param step Smoothing step size (0-1)
 * @return Smoothed vertices
 */
template <typename ScalarType, typename FaceType>
std::vector<Eigen::Vector3<ScalarType>> smoothMeshLaplacian(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const FaceType> faces,
    const std::vector<bool>& vertex_mask = {},
    Index iterations = 1,
    ScalarType step = ScalarType{0.5});

} // namespace axel
