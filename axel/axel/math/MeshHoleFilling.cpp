/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/MeshHoleFilling.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace axel {

namespace {

using Edge = std::pair<Index, Index>;

// Simple hash for edge pairs - custom hash struct to avoid std::hash<std::pair> issues
struct EdgePairHash {
  std::size_t operator()(const std::pair<Index, Index>& edge) const {
    return std::hash<Index>{}(edge.first) ^ (std::hash<Index>{}(edge.second) << 1);
  }
};

using DirectedEdgeSet = std::unordered_set<std::pair<Index, Index>, EdgePairHash>;

/**
 * Build directed edge set for hole detection.
 * Tracks each directed edge separately to preserve winding information.
 */
DirectedEdgeSet buildDirectedEdgeSet(std::span<const Eigen::Vector3i> triangles) {
  DirectedEdgeSet directedEdgeSet;

  for (const auto& triangle : triangles) {
    // Add three directed edges of the triangle (preserving winding order)
    const std::array<Edge, 3> edges = {
        {{triangle[0], triangle[1]}, {triangle[1], triangle[2]}, {triangle[2], triangle[0]}}};

    for (const auto& edge : edges) {
      directedEdgeSet.insert(edge);
    }
  }

  return directedEdgeSet;
}

/**
 * Find boundary edges (directed edges that have no reverse edge).
 * These are edges from triangles at the mesh boundary.
 */
std::vector<std::pair<Index, Index>> findBoundaryEdges(const DirectedEdgeSet& directedEdgeSet) {
  std::vector<std::pair<Index, Index>> boundaryEdges;

  for (const auto& edge : directedEdgeSet) {
    // Check if the reverse edge exists
    const auto reverseEdge = std::make_pair(edge.second, edge.first);

    // If reverse edge doesn't exist, this is a boundary edge
    if (directedEdgeSet.find(reverseEdge) == directedEdgeSet.end()) {
      boundaryEdges.push_back(edge);
    }
  }

  return boundaryEdges;
}

/**
 * Group boundary edges into connected hole loops, preserving edge direction.
 */
std::vector<std::vector<std::pair<Index, Index>>> groupBoundaryEdgesIntoLoops(
    const std::vector<std::pair<Index, Index>>& boundaryEdges) {
  if (boundaryEdges.empty()) {
    return {};
  }

  // Build adjacency map: vertex -> list of edge indices where vertex is the start
  std::unordered_map<Index, std::vector<Index>> vertexToOutgoingEdges;
  for (size_t i = 0; i < boundaryEdges.size(); ++i) {
    const auto& edge = boundaryEdges[i];
    vertexToOutgoingEdges[edge.first].push_back(static_cast<Index>(i));
  }

  std::vector<std::vector<std::pair<Index, Index>>> loops;
  std::vector<bool> usedEdges(boundaryEdges.size(), false);

  for (size_t startEdgeIdx = 0; startEdgeIdx < boundaryEdges.size(); ++startEdgeIdx) {
    if (usedEdges[startEdgeIdx]) {
      continue;
    }

    std::vector<std::pair<Index, Index>> currentLoop;
    size_t currentEdgeIdx = startEdgeIdx;

    // Follow directed edges to form a loop
    do {
      usedEdges[currentEdgeIdx] = true;
      currentLoop.push_back(boundaryEdges[currentEdgeIdx]);

      // Move to the vertex at the end of current edge
      const Index nextVertex = boundaryEdges[currentEdgeIdx].second;

      // Find the next unused edge starting from nextVertex
      size_t nextEdgeIdx = SIZE_MAX;
      const auto it = vertexToOutgoingEdges.find(nextVertex);
      if (it != vertexToOutgoingEdges.end()) {
        for (const auto edgeIdx : it->second) {
          if (!usedEdges[edgeIdx]) {
            nextEdgeIdx = edgeIdx;
            break;
          }
        }
      }

      if (nextEdgeIdx == SIZE_MAX) {
        break; // End of loop
      }

      currentEdgeIdx = nextEdgeIdx;

    } while (!usedEdges[currentEdgeIdx]);

    if (!currentLoop.empty()) {
      loops.push_back(std::move(currentLoop));
    }
  }

  return loops;
}

/**
 * Test whether a point lies inside a triangle using barycentric coordinates.
 */
template <typename ScalarType>
bool isPointInsideTriangle(
    const Eigen::Vector3<ScalarType>& point,
    const Eigen::Vector3<ScalarType>& triA,
    const Eigen::Vector3<ScalarType>& triB,
    const Eigen::Vector3<ScalarType>& triC) {
  const Eigen::Vector3<ScalarType> v0 = triC - triA;
  const Eigen::Vector3<ScalarType> v1 = triB - triA;
  const Eigen::Vector3<ScalarType> v2 = point - triA;

  const ScalarType dot00 = v0.dot(v0);
  const ScalarType dot01 = v0.dot(v1);
  const ScalarType dot02 = v0.dot(v2);
  const ScalarType dot11 = v1.dot(v1);
  const ScalarType dot12 = v1.dot(v2);

  const ScalarType invDenom = ScalarType{1} / (dot00 * dot11 - dot01 * dot01);
  const ScalarType u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const ScalarType v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  return u >= 0 && v >= 0 && u + v <= 1;
}

/**
 * Fill a hole by adding a centroid vertex and creating fan triangles.
 */
template <typename ScalarType>
HoleFillingResult fillHoleWithCentroid(
    const HoleBoundary& hole,
    std::span<const Eigen::Vector3<ScalarType>> vertices) {
  HoleFillingResult result;

  // Compute centroid of boundary vertices
  Eigen::Vector3<ScalarType> centroid = Eigen::Vector3<ScalarType>::Zero();
  for (const auto vertexIdx : hole.vertices) {
    centroid += vertices[vertexIdx];
  }
  centroid /= static_cast<ScalarType>(hole.vertices.size());

  // Adjust centroid position to be slightly inside the hole for better triangle quality
  // Find the normal of the hole plane (average of cross products)
  Eigen::Vector3<ScalarType> holeNormal = Eigen::Vector3<ScalarType>::Zero();
  for (size_t i = 0; i < hole.vertices.size(); ++i) {
    const size_t nextI = (i + 1) % hole.vertices.size();
    const auto& p1 = vertices[hole.vertices[i]];
    const auto& p2 = vertices[hole.vertices[nextI]];
    const auto edge1 = p1 - centroid;
    const auto edge2 = p2 - centroid;
    holeNormal += edge1.cross(edge2).normalized();
  }

  if (holeNormal.norm() > ScalarType{1e-6}) {
    holeNormal.normalize();
    // Move centroid slightly along the normal to improve triangle quality
    const ScalarType offset = static_cast<ScalarType>(0.1) * hole.radius;
    centroid += offset * holeNormal;
  }

  // Add centroid as new vertex
  result.newVertices.push_back(centroid.template cast<float>());
  const auto centroidIdx = static_cast<Index>(vertices.size()); // Index in combined mesh

  // Create triangles from centroid to boundary edges with proper winding
  for (size_t i = 0; i < hole.vertices.size(); ++i) {
    const size_t nextI = (i + 1) % hole.vertices.size();

    const Index v1 = hole.vertices[i];
    const Index v2 = hole.vertices[nextI];

    // Create triangle with reversed winding to match mesh exterior
    result.newTriangles.emplace_back(v2, v1, centroidIdx);
  }

  result.success = true;
  return result;
}

/**
 * Fill a hole using ear clipping triangulation.
 */
template <typename ScalarType>
HoleFillingResult fillHoleWithEarClipping(
    const HoleBoundary& hole,
    std::span<const Eigen::Vector3<ScalarType>> vertices) {
  HoleFillingResult result;
  result.newTriangles.reserve(hole.vertices.size() - 2);

  std::vector<Index> remainingVertices = hole.vertices;

  while (remainingVertices.size() > 3) {
    bool foundEar = false;
    auto bestEarQuality = ScalarType{-1};
    size_t bestEarIndex = 0;

    // Find the best ear (most convex, best triangle quality)
    for (size_t i = 0; i < remainingVertices.size(); ++i) {
      const size_t prevI = (i + remainingVertices.size() - 1) % remainingVertices.size();
      const size_t nextI = (i + 1) % remainingVertices.size();

      const Index vi1 = remainingVertices[prevI];
      const Index vi2 = remainingVertices[i];
      const Index vi3 = remainingVertices[nextI];

      // Check if this forms a valid ear
      const auto& p1 = vertices[vi1];
      const auto& p2 = vertices[vi2];
      const auto& p3 = vertices[vi3];

      const Eigen::Vector3<ScalarType> edge1 = p2 - p1;
      const Eigen::Vector3<ScalarType> edge2 = p3 - p2;
      const Eigen::Vector3<ScalarType> crossProd = edge1.cross(edge2);

      // Check if triangle is valid and convex
      if (crossProd.norm() > ScalarType{1e-6}) {
        // Calculate triangle quality (avoid very thin triangles)
        const auto edge3 = p1 - p3;
        const ScalarType area = crossProd.norm() * ScalarType{0.5};
        const ScalarType perimeter = edge1.norm() + edge2.norm() + edge3.norm();
        const ScalarType quality = area / (perimeter * perimeter); // Higher is better

        // Check if no other vertices are inside this triangle
        bool isEar = true;
        for (size_t j = 0; j < remainingVertices.size(); ++j) {
          if (j == prevI || j == i || j == nextI) {
            continue;
          }

          if (isPointInsideTriangle(vertices[remainingVertices[j]], p1, p2, p3)) {
            isEar = false;
            break;
          }
        }

        if (isEar && quality > bestEarQuality) {
          bestEarQuality = quality;
          bestEarIndex = i;
          foundEar = true;
        }
      }
    }

    if (foundEar) {
      const size_t prevI = (bestEarIndex + remainingVertices.size() - 1) % remainingVertices.size();
      const size_t nextI = (bestEarIndex + 1) % remainingVertices.size();

      // Reverse winding to match mesh exterior
      result.newTriangles.emplace_back(
          remainingVertices[nextI], remainingVertices[bestEarIndex], remainingVertices[prevI]);

      remainingVertices.erase(remainingVertices.begin() + bestEarIndex);
    } else {
      // Fallback: just create a triangle from first three vertices
      if (remainingVertices.size() >= 3) {
        // Reverse winding to match mesh exterior
        result.newTriangles.emplace_back(
            remainingVertices[2], remainingVertices[1], remainingVertices[0]);
        remainingVertices.erase(remainingVertices.begin() + 1);
      } else {
        break;
      }
    }
  }

  // Add final triangle
  if (remainingVertices.size() == 3) {
    // Reverse winding to match mesh exterior
    result.newTriangles.emplace_back(
        remainingVertices[2], remainingVertices[1], remainingVertices[0]);
  }

  result.success = !result.newTriangles.empty();
  return result;
}

/**
 * Fill a single hole using the specified method.
 */
template <typename ScalarType>
HoleFillingResult fillSingleHole(
    const HoleBoundary& hole,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    HoleFillingMethod method) {
  if (hole.vertices.size() < 3) {
    return {}; // Cannot fill holes with less than 3 vertices
  }

  // For very small holes (triangles), just create one triangle
  if (hole.vertices.size() == 3) {
    HoleFillingResult result;
    // Reverse winding to match mesh exterior
    result.newTriangles.emplace_back(hole.vertices[2], hole.vertices[1], hole.vertices[0]);
    result.success = true;
    return result;
  }

  // Determine actual method to use
  bool useCentroid = false;
  switch (method) {
    case HoleFillingMethod::Centroid:
      useCentroid = true;
      break;
    case HoleFillingMethod::EarClipping:
      useCentroid = false;
      break;
    case HoleFillingMethod::Auto:
      // Use centroid for small to medium holes, ear clipping for larger
      useCentroid = (hole.vertices.size() <= 8);
      break;
  }

  if (useCentroid) {
    return fillHoleWithCentroid(hole, vertices);
  }
  return fillHoleWithEarClipping(hole, vertices);
}

/**
 * Compute hole center and radius from boundary vertices.
 */
template <typename ScalarType>
std::pair<Eigen::Vector3<ScalarType>, ScalarType> computeHoleGeometry(
    const std::vector<Index>& boundaryVertices,
    std::span<const Eigen::Vector3<ScalarType>> vertices);

/**
 * Apply Laplacian smoothing to newly added vertices.
 */
template <typename ScalarType>
void smoothHoleFilledRegion(
    std::vector<Eigen::Vector3<ScalarType>>& vertices,
    std::span<const Eigen::Vector3i> triangles,
    const std::unordered_set<Index>& newVertexIndices,
    Index iterations,
    ScalarType factor);

} // anonymous namespace

// ================================================================================================
// HOLE DETECTION
// ================================================================================================

std::vector<HoleBoundary> detectMeshHoles(
    std::span<const Eigen::Vector3f> vertices,
    std::span<const Eigen::Vector3i> triangles) {
  const auto directedEdgeSet = buildDirectedEdgeSet(triangles);
  const auto boundaryEdges = findBoundaryEdges(directedEdgeSet);
  const auto edgeLoops = groupBoundaryEdgesIntoLoops(boundaryEdges);

  std::vector<HoleBoundary> holes;
  holes.reserve(edgeLoops.size());

  for (const auto& loop : edgeLoops) {
    if (loop.empty()) {
      continue;
    }

    HoleBoundary hole;
    hole.edges = loop;

    // Extract ordered vertex list from directed edge loop
    // Since edges are already directed and ordered in the loop, just extract the first vertex of
    // each edge
    hole.vertices.reserve(loop.size());
    for (const auto& edge : loop) {
      hole.vertices.push_back(edge.first);
    }

    // Compute hole geometry
    const auto [center, radius] = computeHoleGeometry(hole.vertices, vertices);
    hole.center = center;
    hole.radius = radius;

    holes.push_back(std::move(hole));
  }

  return holes;
}

// ================================================================================================
// HOLE FILLING IMPLEMENTATION
// ================================================================================================

// ================================================================================================
// DETAIL IMPLEMENTATIONS
// ================================================================================================

namespace {

template <typename ScalarType>
std::pair<Eigen::Vector3<ScalarType>, ScalarType> computeHoleGeometry(
    const std::vector<Index>& boundaryVertices,
    std::span<const Eigen::Vector3<ScalarType>> vertices) {
  if (boundaryVertices.empty()) {
    return {Eigen::Vector3<ScalarType>::Zero(), ScalarType{0}};
  }

  // Compute centroid
  Eigen::Vector3<ScalarType> center = Eigen::Vector3<ScalarType>::Zero();
  for (const auto vertexIdx : boundaryVertices) {
    center += vertices[vertexIdx];
  }
  center /= static_cast<ScalarType>(boundaryVertices.size());

  // Compute average distance to center (rough radius estimate)
  auto radius = ScalarType{0};
  for (const auto vertexIdx : boundaryVertices) {
    radius += (vertices[vertexIdx] - center).norm();
  }
  radius /= static_cast<ScalarType>(boundaryVertices.size());

  return {center, radius};
}

} // namespace

template <typename ScalarType, typename FaceType>
std::vector<Eigen::Vector3<ScalarType>> smoothMeshLaplacian(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const FaceType> faces,
    const std::vector<bool>& vertex_mask,
    Index iterations,
    ScalarType step) {
  std::vector<Eigen::Vector3<ScalarType>> result(vertices.begin(), vertices.end());

  // Validate vertex mask size if provided
  if (!vertex_mask.empty() && vertex_mask.size() != vertices.size()) {
    throw std::invalid_argument("Vertex mask size must match number of vertices");
  }

  // Create adjacency information
  std::unordered_map<Index, std::vector<Index>> adjacency;

  // Build adjacency from faces (works for both triangles and quads)
  for (const auto& face : faces) {
    // Get the number of vertices in this face type
    constexpr int faceVertexCount = FaceType::RowsAtCompileTime;

    // Add edges between consecutive vertices in the face
    for (int i = 0; i < faceVertexCount; ++i) {
      const Index v1 = face[i];
      const Index v2 = face[(i + 1) % faceVertexCount];

      adjacency[v1].push_back(v2);
      adjacency[v2].push_back(v1);
    }
  }

  // Perform smoothing iterations
  for (Index iter = 0; iter < iterations; ++iter) {
    std::vector<Eigen::Vector3<ScalarType>> newPositions = result;

    for (Index i = 0; i < static_cast<Index>(result.size()); ++i) {
      // Skip vertices not in mask (if mask is provided) or with no neighbors
      if (!vertex_mask.empty() && !vertex_mask[i]) {
        continue;
      }

      const auto& neighbors = adjacency[i];
      if (neighbors.empty()) {
        continue;
      }

      Eigen::Vector3<ScalarType> averagePos = Eigen::Vector3<ScalarType>::Zero();
      for (const auto neighborIdx : neighbors) {
        averagePos += result[neighborIdx];
      }
      averagePos /= static_cast<ScalarType>(neighbors.size());

      // Blend between original and average position
      newPositions[i] = (ScalarType{1} - step) * result[i] + step * averagePos;
    }

    result = std::move(newPositions);
  }

  return result;
}

template <typename ScalarType>
HoleFillingResult fillMeshHoles(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    HoleFillingMethod method) {
  HoleFillingResult result;

  // Convert vertices to float for hole detection (detectMeshHoles expects float)
  std::vector<Eigen::Vector3f> floatVertices;
  floatVertices.reserve(vertices.size());
  for (const auto& v : vertices) {
    floatVertices.emplace_back(v.template cast<float>());
  }

  // Detect holes
  const auto holes = detectMeshHoles(floatVertices, triangles);

  if (holes.empty()) {
    result.success = true;
    return result;
  }

  // Fill each hole
  size_t totalNewVertices = 0;

  for (const auto& hole : holes) {
    if (hole.vertices.size() < 3) {
      continue; // Invalid hole
    }

    const auto holeResult = fillSingleHole(hole, vertices, method);

    if (!holeResult.success) {
      continue;
    }

    // Adjust indices for accumulated vertices
    auto adjustedTriangles = holeResult.newTriangles;
    for (auto& triangle : adjustedTriangles) {
      // Indices >= original vertex count refer to new vertices
      for (int i = 0; i < 3; ++i) {
        if (triangle[i] >= static_cast<Index>(vertices.size())) {
          // need to shift by total number of new vertices added so far
          triangle[i] += totalNewVertices;
        }
      }
    }

    // Append results
    result.newVertices.insert(
        result.newVertices.end(), holeResult.newVertices.begin(), holeResult.newVertices.end());

    result.newTriangles.insert(
        result.newTriangles.end(), adjustedTriangles.begin(), adjustedTriangles.end());

    result.filledHoles.push_back(hole);
    result.holesFilledCount++;

    totalNewVertices += holeResult.newVertices.size();
  }

  result.success = result.holesFilledCount > 0;
  return result;
}

template <typename ScalarType>
std::pair<std::vector<Eigen::Vector3<ScalarType>>, std::vector<Eigen::Vector3i>>
fillMeshHolesComplete(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    HoleFillingMethod method) {
  const auto result = fillMeshHoles(vertices, triangles, method);

  // Combine original and new vertices
  std::vector<Eigen::Vector3<ScalarType>> allVertices;
  allVertices.reserve(vertices.size() + result.newVertices.size());

  for (const auto& v : vertices) {
    allVertices.push_back(v);
  }

  for (const auto& v : result.newVertices) {
    allVertices.push_back(v.template cast<ScalarType>());
  }

  // Combine original and new triangles
  std::vector<Eigen::Vector3i> allTriangles;
  allTriangles.reserve(triangles.size() + result.newTriangles.size());

  for (const auto& t : triangles) {
    allTriangles.push_back(t);
  }

  for (const auto& t : result.newTriangles) {
    allTriangles.push_back(t);
  }

  return {std::move(allVertices), std::move(allTriangles)};
}

// ================================================================================================
// EXPLICIT INSTANTIATIONS
// ================================================================================================

template HoleFillingResult fillMeshHoles<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    HoleFillingMethod);

template HoleFillingResult fillMeshHoles<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    HoleFillingMethod);

template std::pair<std::vector<Eigen::Vector3<float>>, std::vector<Eigen::Vector3i>>
    fillMeshHolesComplete<float>(
        std::span<const Eigen::Vector3<float>>,
        std::span<const Eigen::Vector3i>,
        HoleFillingMethod);

template std::pair<std::vector<Eigen::Vector3<double>>, std::vector<Eigen::Vector3i>>
    fillMeshHolesComplete<double>(
        std::span<const Eigen::Vector3<double>>,
        std::span<const Eigen::Vector3i>,
        HoleFillingMethod);

// Triangle mesh smoothing instantiations
template std::vector<Eigen::Vector3<float>> smoothMeshLaplacian<float, Eigen::Vector3i>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    const std::vector<bool>&,
    Index,
    float);

template std::vector<Eigen::Vector3<double>> smoothMeshLaplacian<double, Eigen::Vector3i>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    const std::vector<bool>&,
    Index,
    double);

// Quad mesh smoothing instantiations
template std::vector<Eigen::Vector3<float>> smoothMeshLaplacian<float, Eigen::Vector4i>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector4i>,
    const std::vector<bool>&,
    Index,
    float);

template std::vector<Eigen::Vector3<double>> smoothMeshLaplacian<double, Eigen::Vector4i>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector4i>,
    const std::vector<bool>&,
    Index,
    double);

} // namespace axel
