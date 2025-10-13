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

using EdgeCountMap = std::unordered_map<std::pair<Index, Index>, size_t, EdgePairHash>;

/**
 * Build edge-to-triangle count map for hole detection.
 * Much more efficient than storing full triangle lists - we only need counts.
 */
EdgeCountMap buildEdgeCountMap(gsl::span<const Eigen::Vector3i> triangles) {
  EdgeCountMap edgeCountMap;

  for (const auto& triangle : triangles) {
    // Add three edges of the triangle
    const std::array<Edge, 3> edges = {
        {{triangle[0], triangle[1]}, {triangle[1], triangle[2]}, {triangle[2], triangle[0]}}};

    for (const auto& edge : edges) {
      // Normalize edge orientation (smaller index first)
      const auto normalizedEdge =
          std::make_pair(std::min(edge.first, edge.second), std::max(edge.first, edge.second));

      // Simply increment the count - much more efficient than storing triangle IDs
      edgeCountMap[normalizedEdge]++;
    }
  }

  return edgeCountMap;
}

/**
 * Find boundary edges (edges that belong to only one triangle).
 */
std::vector<std::pair<Index, Index>> findBoundaryEdges(const EdgeCountMap& edgeCountMap) {
  std::vector<std::pair<Index, Index>> boundaryEdges;

  for (const auto& [edge, count] : edgeCountMap) {
    if (count == 1) {
      // This edge belongs to only one triangle, so it's a boundary edge
      boundaryEdges.push_back(edge);
    }
  }

  return boundaryEdges;
}

/**
 * Group boundary edges into connected hole loops.
 */
std::vector<std::vector<std::pair<Index, Index>>> groupBoundaryEdgesIntoLoops(
    const std::vector<std::pair<Index, Index>>& boundaryEdges) {
  if (boundaryEdges.empty()) {
    return {};
  }

  // Build adjacency map for vertices
  std::unordered_map<Index, std::vector<Index>> vertexEdgeMap;
  for (size_t i = 0; i < boundaryEdges.size(); ++i) {
    const auto& edge = boundaryEdges[i];
    vertexEdgeMap[edge.first].push_back(static_cast<Index>(i));
    vertexEdgeMap[edge.second].push_back(static_cast<Index>(i));
  }

  std::vector<std::vector<std::pair<Index, Index>>> loops;
  std::vector<bool> usedEdges(boundaryEdges.size(), false);

  for (size_t startEdgeIdx = 0; startEdgeIdx < boundaryEdges.size(); ++startEdgeIdx) {
    if (usedEdges[startEdgeIdx]) {
      continue;
    }

    std::vector<std::pair<Index, Index>> currentLoop;

    // Start new loop
    size_t currentEdgeIdx = startEdgeIdx;
    Index currentVertex = boundaryEdges[startEdgeIdx].first;

    do {
      usedEdges[currentEdgeIdx] = true;
      currentLoop.push_back(boundaryEdges[currentEdgeIdx]);

      // Find next edge
      const auto& currentEdge = boundaryEdges[currentEdgeIdx];
      currentVertex = (currentVertex == currentEdge.first) ? currentEdge.second : currentEdge.first;

      // Find next unused edge connected to currentVertex
      size_t nextEdgeIdx = SIZE_MAX;
      for (const auto edgeIdx : vertexEdgeMap[currentVertex]) {
        if (!usedEdges[edgeIdx] && edgeIdx != currentEdgeIdx) {
          nextEdgeIdx = edgeIdx;
          break;
        }
      }

      if (nextEdgeIdx == SIZE_MAX) {
        break; // End of loop
      }

      currentEdgeIdx = nextEdgeIdx;

    } while (currentEdgeIdx != startEdgeIdx && !usedEdges[currentEdgeIdx]);

    if (!currentLoop.empty()) {
      loops.push_back(std::move(currentLoop));
    }
  }

  return loops;
}

/**
 * Fill a single hole using advancing front method.
 */
template <typename ScalarType>
HoleFillingResult fillSingleHole(
    const HoleBoundary& hole,
    gsl::span<const Eigen::Vector3<ScalarType>> vertices) {
  HoleFillingResult result;

  if (hole.vertices.size() < 3) {
    return result; // Cannot fill holes with less than 3 vertices
  }

  // For very small holes (triangles), just create one triangle
  if (hole.vertices.size() == 3) {
    result.newTriangles.emplace_back(hole.vertices[0], hole.vertices[1], hole.vertices[2]);
    result.success = true;
    return result;
  }

  // For small to medium holes, use improved centroid-based triangulation
  if (hole.vertices.size() <= 8) {
    // Compute a better centroid position
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

      // Create triangle with consistent winding order
      result.newTriangles.emplace_back(v1, v2, centroidIdx);
    }

    result.success = true;
    return result;
  }

  // For larger holes, use improved ear clipping with better ear detection
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

          const auto& testPoint = vertices[remainingVertices[j]];
          // Simple point-in-triangle test using barycentric coordinates
          const Eigen::Vector3<ScalarType> v0 = p3 - p1;
          const Eigen::Vector3<ScalarType> v1 = p2 - p1;
          const Eigen::Vector3<ScalarType> v2 = testPoint - p1;

          const ScalarType dot00 = v0.dot(v0);
          const ScalarType dot01 = v0.dot(v1);
          const ScalarType dot02 = v0.dot(v2);
          const ScalarType dot11 = v1.dot(v1);
          const ScalarType dot12 = v1.dot(v2);

          const ScalarType invDenom = ScalarType{1} / (dot00 * dot11 - dot01 * dot01);
          const ScalarType u = (dot11 * dot02 - dot01 * dot12) * invDenom;
          const ScalarType v = (dot00 * dot12 - dot01 * dot02) * invDenom;

          if (u >= 0 && v >= 0 && u + v <= 1) {
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

      result.newTriangles.emplace_back(
          remainingVertices[prevI], remainingVertices[bestEarIndex], remainingVertices[nextI]);

      remainingVertices.erase(remainingVertices.begin() + bestEarIndex);
    } else {
      // Fallback: just create a triangle from first three vertices
      if (remainingVertices.size() >= 3) {
        result.newTriangles.emplace_back(
            remainingVertices[0], remainingVertices[1], remainingVertices[2]);
        remainingVertices.erase(remainingVertices.begin() + 1);
      } else {
        break;
      }
    }
  }

  // Add final triangle
  if (remainingVertices.size() == 3) {
    result.newTriangles.emplace_back(
        remainingVertices[0], remainingVertices[1], remainingVertices[2]);
  }

  result.success = !result.newTriangles.empty();
  return result;
}

/**
 * Compute hole center and radius from boundary vertices.
 */
template <typename ScalarType>
std::pair<Eigen::Vector3<ScalarType>, ScalarType> computeHoleGeometry(
    const std::vector<Index>& boundaryVertices,
    gsl::span<const Eigen::Vector3<ScalarType>> vertices);

/**
 * Apply Laplacian smoothing to newly added vertices.
 */
template <typename ScalarType>
void smoothHoleFilledRegion(
    std::vector<Eigen::Vector3<ScalarType>>& vertices,
    gsl::span<const Eigen::Vector3i> triangles,
    const std::unordered_set<Index>& newVertexIndices,
    Index iterations,
    ScalarType factor);

} // anonymous namespace

// ================================================================================================
// HOLE DETECTION
// ================================================================================================

std::vector<HoleBoundary> detectMeshHoles(
    gsl::span<const Eigen::Vector3f> vertices,
    gsl::span<const Eigen::Vector3i> triangles) {
  const auto edgeCountMap = buildEdgeCountMap(triangles);
  const auto boundaryEdges = findBoundaryEdges(edgeCountMap);
  const auto edgeLoops = groupBoundaryEdgesIntoLoops(boundaryEdges);

  std::vector<HoleBoundary> holes;
  holes.reserve(edgeLoops.size());

  for (const auto& loop : edgeLoops) {
    if (loop.empty()) {
      continue;
    }

    HoleBoundary hole;
    hole.edges = loop;

    // Extract ordered vertex list from edge loop
    std::unordered_set<Index> visitedVertices;
    hole.vertices.reserve(loop.size());

    // Start with first edge
    hole.vertices.push_back(loop[0].first);
    hole.vertices.push_back(loop[0].second);
    visitedVertices.insert(loop[0].first);
    visitedVertices.insert(loop[0].second);

    // Follow the loop
    Index currentVertex = loop[0].second;
    for (size_t i = 1; i < loop.size(); ++i) {
      // Find next edge that starts with currentVertex
      for (const auto& edge : loop) {
        if (edge.first == currentVertex &&
            visitedVertices.find(edge.second) == visitedVertices.end()) {
          hole.vertices.push_back(edge.second);
          visitedVertices.insert(edge.second);
          currentVertex = edge.second;
          break;
        } else if (
            edge.second == currentVertex &&
            visitedVertices.find(edge.first) == visitedVertices.end()) {
          hole.vertices.push_back(edge.first);
          visitedVertices.insert(edge.first);
          currentVertex = edge.first;
          break;
        }
      }
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
    gsl::span<const Eigen::Vector3<ScalarType>> vertices) {
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

template <typename ScalarType>
void smoothHoleFilledRegion(
    std::vector<Eigen::Vector3<ScalarType>>& vertices,
    gsl::span<const Eigen::Vector3i> triangles,
    const std::unordered_set<Index>& newVertexIndices,
    Index iterations,
    ScalarType factor) {
  // Build vertex-vertex adjacency for smoothing
  std::unordered_map<Index, std::vector<Index>> adjacency;

  for (const auto& triangle : triangles) {
    for (int i = 0; i < 3; ++i) {
      const Index v1 = triangle[i];
      const Index v2 = triangle[(i + 1) % 3];

      adjacency[v1].push_back(v2);
      adjacency[v2].push_back(v1);
    }
  }

  // Apply Laplacian smoothing iterations
  for (Index iter = 0; iter < iterations; ++iter) {
    std::vector<Eigen::Vector3<ScalarType>> newPositions = vertices;

    for (const auto vertexIdx : newVertexIndices) {
      const auto& neighbors = adjacency[vertexIdx];
      if (neighbors.empty()) {
        continue;
      }

      Eigen::Vector3<ScalarType> averagePos = Eigen::Vector3<ScalarType>::Zero();
      for (const auto neighborIdx : neighbors) {
        averagePos += vertices[neighborIdx];
      }
      averagePos /= static_cast<ScalarType>(neighbors.size());

      // Blend between original and average position
      newPositions[vertexIdx] =
          (ScalarType{1} - factor) * vertices[vertexIdx] + factor * averagePos;
    }

    vertices = std::move(newPositions);
  }
}

} // namespace

template <typename ScalarType>
HoleFillingResult fillMeshHoles(
    gsl::span<const Eigen::Vector3<ScalarType>> vertices,
    gsl::span<const Eigen::Vector3i> triangles) {
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

    const auto holeResult = fillSingleHole(hole, vertices);

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
    gsl::span<const Eigen::Vector3<ScalarType>> vertices,
    gsl::span<const Eigen::Vector3i> triangles) {
  const auto result = fillMeshHoles(vertices, triangles);

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
    gsl::span<const Eigen::Vector3<float>>,
    gsl::span<const Eigen::Vector3i>);

template HoleFillingResult fillMeshHoles<double>(
    gsl::span<const Eigen::Vector3<double>>,
    gsl::span<const Eigen::Vector3i>);

template std::pair<std::vector<Eigen::Vector3<float>>, std::vector<Eigen::Vector3i>>
    fillMeshHolesComplete<float>(
        gsl::span<const Eigen::Vector3<float>>,
        gsl::span<const Eigen::Vector3i>);

template std::pair<std::vector<Eigen::Vector3<double>>, std::vector<Eigen::Vector3i>>
    fillMeshHolesComplete<double>(
        gsl::span<const Eigen::Vector3<double>>,
        gsl::span<const Eigen::Vector3i>);

} // namespace axel
