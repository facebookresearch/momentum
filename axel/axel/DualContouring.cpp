/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/DualContouring.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace axel {

// ================================================================================================
// DETAIL IMPLEMENTATIONS
// ================================================================================================

namespace detail {

/**
 * Hash function for 3D grid indices.
 */
struct GridIndexHash {
  std::size_t operator()(const Eigen::Vector3<Index>& idx) const {
    return std::hash<Index>{}(idx.x()) ^ (std::hash<Index>{}(idx.y()) << 1) ^
        (std::hash<Index>{}(idx.z()) << 2);
  }
};

using CellVertexMap = std::unordered_map<Eigen::Vector3<Index>, Index, GridIndexHash>;

/**
 * Check if a value is inside the isosurface (negative side).
 *
 * @param val The SDF value
 * @param isovalue The isovalue threshold
 * @return True if the value is on the inside (negative) side
 */
template <typename ScalarType>
bool isInside(ScalarType val, ScalarType isovalue) {
  return val <= isovalue;
}

/**
 * Push a vertex toward the zero level set using gradient descent.
 * Uses Newton's method to iteratively move the vertex closer to the target isovalue.
 *
 * @param sdf The signed distance field
 * @param startPosition Initial vertex position (typically cell center)
 * @param isovalue The target isovalue to move toward
 * @return Optimized vertex position closer to the surface
 */
template <typename ScalarType>
Eigen::Vector3<ScalarType> pushVertexToSurface(
    const SignedDistanceField<ScalarType>& sdf,
    const Eigen::Vector3<ScalarType>& startPosition,
    ScalarType isovalue) {
  Eigen::Vector3<ScalarType> position = startPosition;

  const ScalarType targetIsovalue = isovalue;
  const int maxIterations = 10;
  const auto tolerance = ScalarType{1e-6};

  for (int iter = 0; iter < maxIterations; ++iter) {
    const ScalarType value = sdf.sample(position);
    const Eigen::Vector3<ScalarType> gradient = sdf.gradient(position);

    const ScalarType distance = value - targetIsovalue;
    if (std::abs(distance) < tolerance) {
      break;
    }

    const ScalarType gradientMagnitude = gradient.norm();
    if (gradientMagnitude < tolerance) {
      break;
    }

    const ScalarType stepSize = distance / gradientMagnitude;
    position -= stepSize * gradient.normalized();

    const auto voxelSize = sdf.voxelSize();
    const ScalarType maxOffset = voxelSize.maxCoeff() * ScalarType{2.0};
    const auto offset = position - startPosition;
    if (offset.norm() > maxOffset) {
      position = startPosition + offset.normalized() * maxOffset;
      break;
    }
  }

  return position;
}

/**
 * Check if a cell intersects the isosurface by looking for sign changes.
 *
 * @param cornerValues Array of 8 corner SDF values
 * @param isovalue The isovalue to check against
 * @return True if the cell intersects the isosurface
 */
template <typename ScalarType>
bool cellIntersectsIsosurface(const std::array<ScalarType, 8>& cornerValues, ScalarType isovalue) {
  const ScalarType firstValue = cornerValues[0];
  for (int c = 1; c < 8; ++c) {
    if (isInside(firstValue, isovalue) != isInside(cornerValues[c], isovalue)) {
      return true;
    }
  }
  return false;
}

/**
 * Find all cells that intersect the isosurface and create vertices.
 * Places one vertex per intersecting cell, positioned on the surface using gradient descent.
 *
 * @param sdf The signed distance field
 * @param isovalue The isovalue to extract
 * @return Pair of (vertices, cellToVertexIndex map)
 */
template <typename ScalarType>
std::pair<std::vector<Eigen::Vector3<ScalarType>>, CellVertexMap>
findIntersectingCellsAndCreateVertices(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue) {
  const auto& resolution = sdf.resolution();
  const auto& sdfData = sdf.data();

  const auto linearIndex = [&](Index i, Index j, Index k) -> size_t {
    return k * resolution.x() * resolution.y() + j * resolution.x() + i;
  };

  CellVertexMap cellToVertexIndex;
  std::vector<Eigen::Vector3<ScalarType>> vertices;
  Index vertexIndex = 0;

  for (Index k = 0; k < resolution.z() - 1; ++k) {
    for (Index j = 0; j < resolution.y() - 1; ++j) {
      for (Index i = 0; i < resolution.x() - 1; ++i) {
        // Get the 8 corner values of this cell using direct data access
        const std::array<ScalarType, 8> cornerValues = {
            {sdfData[linearIndex(i, j, k)], // 0: (0,0,0)
             sdfData[linearIndex(i + 1, j, k)], // 1: (1,0,0)
             sdfData[linearIndex(i + 1, j + 1, k)], // 2: (1,1,0)
             sdfData[linearIndex(i, j + 1, k)], // 3: (0,1,0)
             sdfData[linearIndex(i, j, k + 1)], // 4: (0,0,1)
             sdfData[linearIndex(i + 1, j, k + 1)], // 5: (1,0,1)
             sdfData[linearIndex(i + 1, j + 1, k + 1)], // 6: (1,1,1)
             sdfData[linearIndex(i, j + 1, k + 1)]}}; // 7: (0,1,1)

        if (cellIntersectsIsosurface(cornerValues, isovalue)) {
          // This cell intersects the isosurface, create a vertex positioned on the surface
          const Eigen::Vector3<Index> cellIdx(i, j, k);
          const auto cellCenter = sdf.gridToWorld(
              cellIdx.template cast<ScalarType>() + Eigen::Vector3<ScalarType>::Constant(0.5));

          // Push vertex toward the zero level set using gradient descent
          auto surfacePosition = pushVertexToSurface(sdf, cellCenter, isovalue);

          cellToVertexIndex[cellIdx] = vertexIndex++;
          vertices.push_back(surfacePosition);
        }
      }
    }
  }

  return {std::move(vertices), std::move(cellToVertexIndex)};
}

/**
 * Generate quads for edges in the X direction that cross the isosurface.
 *
 * @param sdf The signed distance field
 * @param isovalue The isovalue to check against
 * @param cellToVertexIndex Map from cell indices to vertex indices
 * @param quads Output vector to append generated quads
 */
template <typename ScalarType>
void generateQuadsForXEdges(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue,
    const CellVertexMap& cellToVertexIndex,
    std::vector<Eigen::Vector4i>& quads) {
  const auto& resolution = sdf.resolution();
  const auto& sdfData = sdf.data();

  const auto linearIndex = [&](Index i, Index j, Index k) -> size_t {
    return k * resolution.x() * resolution.y() + j * resolution.x() + i;
  };

  const auto hasVertex = [&](Index i, Index j, Index k) -> std::pair<bool, Index> {
    if (i < 0 || j < 0 || k < 0 || i >= resolution.x() - 1 || j >= resolution.y() - 1 ||
        k >= resolution.z() - 1) {
      return {false, static_cast<Index>(-1)};
    }
    auto it = cellToVertexIndex.find(Eigen::Vector3<Index>(i, j, k));
    if (it != cellToVertexIndex.end()) {
      return {true, it->second};
    }
    return {false, static_cast<Index>(-1)};
  };

  // X-direction edges: from grid point (i,j,k) to (i+1,j,k)
  for (Index k = 0; k < resolution.z(); ++k) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index i = 0; i < resolution.x() - 1; ++i) {
        // Check if this edge crosses the isosurface
        const ScalarType val1 = sdfData[linearIndex(i, j, k)];
        const ScalarType val2 = sdfData[linearIndex(i + 1, j, k)];

        if (isInside(val1, isovalue) != isInside(val2, isovalue)) {
          // This X-edge crosses the isosurface
          // Find the 4 cells that share this edge (if they exist)
          const std::array<std::pair<bool, Index>, 4> quadVertices = {
              {hasVertex(i, j - 1, k - 1), // Cell (i, j-1, k-1)
               hasVertex(i, j, k - 1), // Cell (i, j, k-1)
               hasVertex(i, j, k), // Cell (i, j, k)
               hasVertex(i, j - 1, k)}}; // Cell (i, j-1, k)

          // Check if all 4 vertices exist
          bool allExist = true;
          for (const auto& [exists, _] : quadVertices) {
            if (!exists) {
              allExist = false;
              break;
            }
          }

          if (allExist) {
            // Determine winding order based on which side is inside (negative)
            // Default winding (0,1,2,3) produces normal pointing in +X direction
            if (val1 < val2) {
              // val1 more negative (inside), val2 less negative (outside)
              // Normal should point +X (toward val2/outside) - use default winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[1].second,
                  quadVertices[2].second,
                  quadVertices[3].second);
            } else {
              // val2 more negative (inside), val1 less negative (outside)
              // Normal should point -X (toward val1/outside) - flip winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[3].second,
                  quadVertices[2].second,
                  quadVertices[1].second);
            }
          }
        }
      }
    }
  }
}

/**
 * Generate quads for edges in the Y direction that cross the isosurface.
 *
 * @param sdf The signed distance field
 * @param isovalue The isovalue to check against
 * @param cellToVertexIndex Map from cell indices to vertex indices
 * @param quads Output vector to append generated quads
 */
template <typename ScalarType>
void generateQuadsForYEdges(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue,
    const CellVertexMap& cellToVertexIndex,
    std::vector<Eigen::Vector4i>& quads) {
  const auto& resolution = sdf.resolution();
  const auto& sdfData = sdf.data();

  const auto linearIndex = [&](Index i, Index j, Index k) -> size_t {
    return k * resolution.x() * resolution.y() + j * resolution.x() + i;
  };

  const auto hasVertex = [&](Index i, Index j, Index k) -> std::pair<bool, Index> {
    if (i < 0 || j < 0 || k < 0 || i >= resolution.x() - 1 || j >= resolution.y() - 1 ||
        k >= resolution.z() - 1) {
      return {false, static_cast<Index>(-1)};
    }
    auto it = cellToVertexIndex.find(Eigen::Vector3<Index>(i, j, k));
    if (it != cellToVertexIndex.end()) {
      return {true, it->second};
    }
    return {false, static_cast<Index>(-1)};
  };

  // Y-direction edges: from grid point (i,j,k) to (i,j+1,k)
  for (Index k = 0; k < resolution.z(); ++k) {
    for (Index j = 0; j < resolution.y() - 1; ++j) {
      for (Index i = 0; i < resolution.x(); ++i) {
        // Check if this edge crosses the isosurface
        const ScalarType val1 = sdfData[linearIndex(i, j, k)];
        const ScalarType val2 = sdfData[linearIndex(i, j + 1, k)];

        if (isInside(val1, isovalue) != isInside(val2, isovalue)) {
          // This Y-edge crosses the isosurface
          // Find the 4 cells that share this edge
          // Reorder to maintain consistent winding: (back-left, back-right, front-right,
          // front-left)
          const std::array<std::pair<bool, Index>, 4> quadVertices = {
              {hasVertex(i - 1, j, k - 1), // Cell (i-1, j, k-1)
               hasVertex(i - 1, j, k), // Cell (i-1, j, k)
               hasVertex(i, j, k), // Cell (i, j, k)
               hasVertex(i, j, k - 1)}}; // Cell (i, j, k-1)

          // Check if all 4 vertices exist
          bool allExist = true;
          for (const auto& [exists, _] : quadVertices) {
            if (!exists) {
              allExist = false;
              break;
            }
          }

          if (allExist) {
            // Determine winding order based on which side is inside (negative)
            // Default winding (0,1,2,3) produces normal pointing in +Y direction
            if (val1 < val2) {
              // val1 more negative (inside), val2 less negative (outside)
              // Normal should point +Y (toward val2/outside) - use default winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[1].second,
                  quadVertices[2].second,
                  quadVertices[3].second);
            } else {
              // val2 more negative (inside), val1 less negative (outside)
              // Normal should point -Y (toward val1/outside) - flip winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[3].second,
                  quadVertices[2].second,
                  quadVertices[1].second);
            }
          }
        }
      }
    }
  }
}

/**
 * Generate quads for edges in the Z direction that cross the isosurface.
 *
 * @param sdf The signed distance field
 * @param isovalue The isovalue to check against
 * @param cellToVertexIndex Map from cell indices to vertex indices
 * @param quads Output vector to append generated quads
 */
template <typename ScalarType>
void generateQuadsForZEdges(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue,
    const CellVertexMap& cellToVertexIndex,
    std::vector<Eigen::Vector4i>& quads) {
  const auto& resolution = sdf.resolution();
  const auto& sdfData = sdf.data();

  const auto linearIndex = [&](Index i, Index j, Index k) -> size_t {
    return k * resolution.x() * resolution.y() + j * resolution.x() + i;
  };

  const auto hasVertex = [&](Index i, Index j, Index k) -> std::pair<bool, Index> {
    if (i < 0 || j < 0 || k < 0 || i >= resolution.x() - 1 || j >= resolution.y() - 1 ||
        k >= resolution.z() - 1) {
      return {false, static_cast<Index>(-1)};
    }
    auto it = cellToVertexIndex.find(Eigen::Vector3<Index>(i, j, k));
    if (it != cellToVertexIndex.end()) {
      return {true, it->second};
    }
    return {false, static_cast<Index>(-1)};
  };

  // Z-direction edges: from grid point (i,j,k) to (i,j,k+1)
  for (Index k = 0; k < resolution.z() - 1; ++k) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index i = 0; i < resolution.x(); ++i) {
        // Check if this edge crosses the isosurface
        const ScalarType val1 = sdfData[linearIndex(i, j, k)];
        const ScalarType val2 = sdfData[linearIndex(i, j, k + 1)];

        if (isInside(val1, isovalue) != isInside(val2, isovalue)) {
          // This Z-edge crosses the isosurface
          // Find the 4 cells that share this edge
          const std::array<std::pair<bool, Index>, 4> quadVertices = {
              {hasVertex(i - 1, j - 1, k), // Cell (i-1, j-1, k)
               hasVertex(i, j - 1, k), // Cell (i, j-1, k)
               hasVertex(i, j, k), // Cell (i, j, k)
               hasVertex(i - 1, j, k)}}; // Cell (i-1, j, k)

          // Check if all 4 vertices exist
          bool allExist = true;
          for (const auto& [exists, _] : quadVertices) {
            if (!exists) {
              allExist = false;
              break;
            }
          }

          if (allExist) {
            // Determine winding order based on which side is inside (negative)
            // Default winding (0,1,2,3) produces normal pointing in +Z direction
            if (val1 < val2) {
              // val1 more negative (inside), val2 less negative (outside)
              // Normal should point +Z (toward val2/outside) - use default winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[1].second,
                  quadVertices[2].second,
                  quadVertices[3].second);
            } else {
              // val2 more negative (inside), val1 less negative (outside)
              // Normal should point -Z (toward val1/outside) - flip winding
              quads.emplace_back(
                  quadVertices[0].second,
                  quadVertices[3].second,
                  quadVertices[2].second,
                  quadVertices[1].second);
            }
          }
        }
      }
    }
  }
}

} // namespace detail

// ================================================================================================
// MAIN DUAL CONTOURING FUNCTIONS
// ================================================================================================

template <typename ScalarType>
DualContouringResult<ScalarType> dualContouring(
    const SignedDistanceField<ScalarType>& sdf,
    ScalarType isovalue) {
  DualContouringResult<ScalarType> result;

  // Step 1: Find all cells that intersect the isosurface and create vertices
  auto [vertices, cellToVertexIndex] =
      detail::findIntersectingCellsAndCreateVertices(sdf, isovalue);

  if (vertices.empty()) {
    result.success = true;
    return result;
  }

  // Step 2: Generate quads by processing edges in each direction
  std::vector<Eigen::Vector4i> quads;

  detail::generateQuadsForXEdges(sdf, isovalue, cellToVertexIndex, quads);
  detail::generateQuadsForYEdges(sdf, isovalue, cellToVertexIndex, quads);
  detail::generateQuadsForZEdges(sdf, isovalue, cellToVertexIndex, quads);

  result.vertices = std::move(vertices);
  result.quads = std::move(quads);
  result.success = true;
  result.processedCells = cellToVertexIndex.size();
  result.generatedVertices = result.vertices.size();

  return result;
}

std::vector<Eigen::Vector3i> triangulateQuads(const std::vector<Eigen::Vector4i>& quads) {
  std::vector<Eigen::Vector3i> triangles;
  triangles.reserve(quads.size() * 2);

  for (const auto& quad : quads) {
    // Split each quad into two triangles using diagonal (0,2)
    // Triangle 1: vertices 0, 1, 2
    triangles.emplace_back(quad[0], quad[1], quad[2]);
    // Triangle 2: vertices 0, 2, 3
    triangles.emplace_back(quad[0], quad[2], quad[3]);
  }

  return triangles;
}

// ================================================================================================
// EXPLICIT INSTANTIATIONS
// =========================================================================================

template DualContouringResult<float> dualContouring<float>(
    const SignedDistanceField<float>&,
    float);

template DualContouringResult<double> dualContouring<double>(
    const SignedDistanceField<double>&,
    double);

// Helper functions are automatically instantiated when dualContouring is instantiated

} // namespace axel
