/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/MeshHoleFilling.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <span>

using namespace axel;

class MeshHoleFillingTest : public ::testing::Test {
 protected:
  /**
   * Directed-edge topology statistics for a triangle mesh.
   */
  struct TopologyStats {
    size_t totalUndirectedEdges = 0;
    size_t boundaryEdges = 0; // edges with only one direction
    size_t interiorEdges = 0; // edges with both directions appearing once each
    size_t nonWatertightEdges = 0; // edges with duplicate directions
    size_t inconsistentWindingEdges = 0; // edges where winding directions don't match
    bool isWatertight = false;
    bool hasConsistentWinding = false;
  };

  static TopologyStats getMeshTopologyStats(
      const std::vector<Eigen::Vector3f>& /*vertices*/,
      const std::vector<Eigen::Vector3i>& triangles) {
    using DirectedEdge = std::pair<Index, Index>;
    struct DirectedEdgeHash {
      std::size_t operator()(const DirectedEdge& edge) const {
        return std::hash<Index>{}(edge.first) ^ (std::hash<Index>{}(edge.second) << 1);
      }
    };

    TopologyStats stats;

    // Count occurrences of each directed edge
    std::unordered_map<DirectedEdge, size_t, DirectedEdgeHash> directedEdgeCount;
    for (const auto& triangle : triangles) {
      const std::array<DirectedEdge, 3> edges = {
          {{triangle[0], triangle[1]}, {triangle[1], triangle[2]}, {triangle[2], triangle[0]}}};
      for (const auto& edge : edges) {
        directedEdgeCount[edge]++;
      }
    }

    // Process each undirected edge by checking both directions
    std::unordered_set<DirectedEdge, DirectedEdgeHash> processedEdges;
    for (const auto& [directedEdge, forwardCount] : directedEdgeCount) {
      const auto v1 = directedEdge.first;
      const auto v2 = directedEdge.second;
      const auto normalizedEdge = std::make_pair(std::min(v1, v2), std::max(v1, v2));
      if (processedEdges.count(normalizedEdge) > 0) {
        continue;
      }
      processedEdges.insert(normalizedEdge);
      stats.totalUndirectedEdges++;

      const DirectedEdge reverseEdge = {v2, v1};
      const size_t reverseCount =
          directedEdgeCount.count(reverseEdge) ? directedEdgeCount[reverseEdge] : 0;

      if ((forwardCount == 0 && reverseCount == 1) || (forwardCount == 1 && reverseCount == 0)) {
        // Boundary edge: appears in only one direction (belongs to one triangle)
        stats.boundaryEdges++;
      } else if (forwardCount == 1 && reverseCount == 1) {
        // Interior edge: shared by exactly 2 triangles with opposite winding
        stats.interiorEdges++;
      } else {
        // Degenerate: edge shared by >2 triangles, or duplicated in the same direction
        stats.nonWatertightEdges++;
        const size_t totalCount = forwardCount + reverseCount;
        // Winding-specific issue: all triangles sharing this edge face the same way,
        // e.g. counts (2,0) or (0,3), rather than a general over-sharing like (1,2).
        if ((totalCount == 2 && (forwardCount == 2 || reverseCount == 2)) ||
            (totalCount > 2 && (forwardCount == 0 || reverseCount == 0))) {
          stats.inconsistentWindingEdges++;
        }
      }
    }

    stats.isWatertight = (stats.boundaryEdges == 0 && stats.nonWatertightEdges == 0);
    stats.hasConsistentWinding =
        (stats.inconsistentWindingEdges == 0 && stats.totalUndirectedEdges > 0);
    return stats;
  }

  /**
   * Verify that hole filling on the given mesh produces a watertight result with consistent
   * winding.
   */
  static void verifyWatertightAfterFilling(
      const std::vector<Eigen::Vector3f>& vertices,
      const std::vector<Eigen::Vector3i>& triangles,
      HoleFillingMethod method = HoleFillingMethod::Centroid,
      float capHeightRatio = 0.5f) {
    const auto originalStats = getMeshTopologyStats(vertices, triangles);
    ASSERT_GT(originalStats.boundaryEdges, 0) << "Input mesh should have holes for this test";
    ASSERT_TRUE(originalStats.hasConsistentWinding) << "Input mesh must have consistent winding";

    const auto result = fillMeshHoles(
        std::span<const Eigen::Vector3f>(vertices),
        std::span<const Eigen::Vector3i>(triangles),
        method,
        capHeightRatio);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.holesFilledCount, 0);

    const auto [filledVertices, filledTriangles] = fillMeshHolesComplete(
        std::span<const Eigen::Vector3f>(vertices),
        std::span<const Eigen::Vector3i>(triangles),
        method,
        capHeightRatio);
    const auto filledStats = getMeshTopologyStats(filledVertices, filledTriangles);

    EXPECT_EQ(filledStats.boundaryEdges, 0)
        << "Filled mesh should be watertight (no boundary edges)";
    EXPECT_EQ(filledStats.nonWatertightEdges, 0) << "Filled mesh should have no degenerate edges";
    EXPECT_TRUE(filledStats.hasConsistentWinding) << "Filled mesh should have consistent winding";
    EXPECT_TRUE(filledStats.isWatertight) << "Filled mesh should be watertight";
  }

  // Helper meshes

  static std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
  createTetrahedronWithHole() {
    std::vector<Eigen::Vector3f> vertices = {
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.5f, 1.0f, 0.0f),
        Eigen::Vector3f(0.5f, 0.5f, 1.0f)};

    // Three of four tetrahedron faces, consistent outward-facing winding.
    // Missing face (2, 3, 0) creates one triangular hole.
    std::vector<Eigen::Vector3i> triangles = {
        Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 3, 1), Eigen::Vector3i(1, 3, 2)};

    return {vertices, triangles};
  }

  /**
   * Create an open cylinder (tube) along the Z axis with two circular holes at the ends.
   * nSegments controls the number of vertices around each ring (≥ 3).
   */
  static std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
  createOpenCylinder(size_t nSegments = 16, float radius = 1.0f, float halfHeight = 2.0f) {
    // Bottom ring: vertices [0, nSegments)
    // Top ring:    vertices [nSegments, 2*nSegments)
    std::vector<Eigen::Vector3f> vertices;
    vertices.reserve(2 * nSegments);
    for (size_t i = 0; i < nSegments; ++i) {
      const float angle =
          2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(nSegments);
      const float x = radius * std::cos(angle);
      const float y = radius * std::sin(angle);
      vertices.emplace_back(x, y, -halfHeight); // bottom
    }
    for (size_t i = 0; i < nSegments; ++i) {
      const float angle =
          2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(nSegments);
      const float x = radius * std::cos(angle);
      const float y = radius * std::sin(angle);
      vertices.emplace_back(x, y, halfHeight); // top
    }

    // Side quads split into two triangles each, with consistent outward-facing winding.
    std::vector<Eigen::Vector3i> triangles;
    triangles.reserve(2 * nSegments);
    for (size_t i = 0; i < nSegments; ++i) {
      const auto i0 = static_cast<int>(i);
      const auto i1 = static_cast<int>((i + 1) % nSegments);
      const auto t0 = static_cast<int>(i + nSegments);
      const auto t1 = static_cast<int>(((i + 1) % nSegments) + nSegments);

      // Two triangles per quad; winding so that normals point outward (radially).
      triangles.emplace_back(i0, t0, i1);
      triangles.emplace_back(i1, t0, t1);
    }

    return {vertices, triangles};
  }

  /**
   * Create a flat rectangular frame with a rectangular hole in the center.
   *
   *    3-----------2
   *    |           |
   *    | 7-------6 |
   *    | |  HOLE | |
   *    | |       | |
   *    | 4-------5 |
   *    |           |
   *    0-----------1
   */
  static std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
  createRectangularFrameWithHole() {
    std::vector<Eigen::Vector3f> vertices = {
        // Outer rectangle
        Eigen::Vector3f(0.0f, 0.0f, 0.0f), // 0 - bottom-left
        Eigen::Vector3f(4.0f, 0.0f, 0.0f), // 1 - bottom-right
        Eigen::Vector3f(4.0f, 4.0f, 0.0f), // 2 - top-right
        Eigen::Vector3f(0.0f, 4.0f, 0.0f), // 3 - top-left
        // Inner rectangle (forms hole boundary)
        Eigen::Vector3f(1.0f, 1.0f, 0.0f), // 4 - inner bottom-left
        Eigen::Vector3f(3.0f, 1.0f, 0.0f), // 5 - inner bottom-right
        Eigen::Vector3f(3.0f, 3.0f, 0.0f), // 6 - inner top-right
        Eigen::Vector3f(1.0f, 3.0f, 0.0f), // 7 - inner top-left
    };

    // Triangles connecting outer and inner rectangles, all with consistent CCW winding.
    // The inner rectangle edges form a closed loop of boundary edges (the hole).
    std::vector<Eigen::Vector3i> triangles = {
        // Bottom strip
        Eigen::Vector3i(0, 1, 5),
        Eigen::Vector3i(0, 5, 4),
        // Right strip
        Eigen::Vector3i(1, 2, 6),
        Eigen::Vector3i(1, 6, 5),
        // Top strip
        Eigen::Vector3i(2, 3, 7),
        Eigen::Vector3i(2, 7, 6),
        // Left strip
        Eigen::Vector3i(3, 0, 4),
        Eigen::Vector3i(3, 4, 7),
    };

    return {vertices, triangles};
  }
};

// =============================================================================
// Sanity: verify topology checker on a known-good closed mesh
// =============================================================================

TEST_F(MeshHoleFillingTest, TopologyStatsOnClosedCube) {
  std::vector<Eigen::Vector3f> cubeVertices = {
      Eigen::Vector3f(-1, -1, -1),
      Eigen::Vector3f(1, -1, -1),
      Eigen::Vector3f(1, 1, -1),
      Eigen::Vector3f(-1, 1, -1),
      Eigen::Vector3f(-1, -1, 1),
      Eigen::Vector3f(1, -1, 1),
      Eigen::Vector3f(1, 1, 1),
      Eigen::Vector3f(-1, 1, 1),
  };

  std::vector<Eigen::Vector3i> cubeTriangles = {
      Eigen::Vector3i(0, 2, 1),
      Eigen::Vector3i(0, 3, 2), // bottom
      Eigen::Vector3i(4, 5, 6),
      Eigen::Vector3i(4, 6, 7), // top
      Eigen::Vector3i(0, 1, 5),
      Eigen::Vector3i(0, 5, 4), // front
      Eigen::Vector3i(2, 7, 6),
      Eigen::Vector3i(2, 3, 7), // back
      Eigen::Vector3i(0, 4, 7),
      Eigen::Vector3i(0, 7, 3), // left
      Eigen::Vector3i(1, 2, 6),
      Eigen::Vector3i(1, 6, 5), // right
  };

  const auto stats = getMeshTopologyStats(cubeVertices, cubeTriangles);
  EXPECT_TRUE(stats.isWatertight);
  EXPECT_EQ(stats.boundaryEdges, 0);
  EXPECT_EQ(stats.nonWatertightEdges, 0);
  EXPECT_TRUE(stats.hasConsistentWinding);
}

// =============================================================================
// Edge case: empty mesh
// =============================================================================

TEST_F(MeshHoleFillingTest, EmptyMesh) {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> triangles;

  const auto holes = detectMeshHoles(vertices, triangles);
  EXPECT_TRUE(holes.empty());

  const auto result = fillMeshHoles(
      std::span<const Eigen::Vector3f>(vertices), std::span<const Eigen::Vector3i>(triangles));
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.holesFilledCount, 0);
}

// =============================================================================
// Watertight after Centroid fill
// =============================================================================

TEST_F(MeshHoleFillingTest, WatertightAfterCentroidFill_Tetrahedron) {
  const auto [vertices, triangles] = createTetrahedronWithHole();
  verifyWatertightAfterFilling(vertices, triangles, HoleFillingMethod::Centroid);
}

TEST_F(MeshHoleFillingTest, WatertightAfterCentroidFill_RectangularFrame) {
  const auto [vertices, triangles] = createRectangularFrameWithHole();
  verifyWatertightAfterFilling(vertices, triangles, HoleFillingMethod::Centroid);
}

// =============================================================================
// Watertight after SphericalCap fill
// =============================================================================

TEST_F(MeshHoleFillingTest, WatertightAfterSphericalCapFill_Tetrahedron) {
  const auto [vertices, triangles] = createTetrahedronWithHole();
  verifyWatertightAfterFilling(vertices, triangles, HoleFillingMethod::SphericalCap, 0.5f);
}

TEST_F(MeshHoleFillingTest, WatertightAfterSphericalCapFill_RectangularFrame) {
  const auto [vertices, triangles] = createRectangularFrameWithHole();
  verifyWatertightAfterFilling(vertices, triangles, HoleFillingMethod::SphericalCap, 0.5f);
}

// =============================================================================
// SphericalCap: new vertices should lie on a spherical cap surface
// =============================================================================

TEST_F(MeshHoleFillingTest, SphericalCapVerticesOnSphere) {
  const auto [vertices, triangles] = createTetrahedronWithHole();

  const auto holes = detectMeshHoles(vertices, triangles);
  ASSERT_EQ(holes.size(), 1);
  const auto& hole = holes[0];

  // Compute centroid, radius, and normal of the hole boundary
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto idx : hole.vertices) {
    centroid += vertices[idx];
  }
  centroid /= static_cast<float>(hole.vertices.size());

  float avgRadius = 0.0f;
  for (const auto idx : hole.vertices) {
    avgRadius += (vertices[idx] - centroid).norm();
  }
  avgRadius /= static_cast<float>(hole.vertices.size());

  const float capHeightRatio = 0.5f;
  const auto result = fillMeshHoles(
      std::span<const Eigen::Vector3f>(vertices),
      std::span<const Eigen::Vector3i>(triangles),
      HoleFillingMethod::SphericalCap,
      capHeightRatio);

  ASSERT_TRUE(result.success);
  ASSERT_GT(result.newVertices.size(), 0);

  // The spherical cap creates ring vertices that interpolate between the boundary
  // and a pole at distance capHeightRatio * radius along the normal.  Each ring k
  // (out of nRings) has radius factor cos(theta) and offset capH * R * sin(theta)
  // where theta = k/nRings * pi/2.  This means each ring vertex, when projected
  // back, satisfies:   (r/R)^2 + (h / (capH * R))^2  ≈  1
  // where r = distance from centroid projected onto boundary plane,
  //       h = distance along normal from boundary plane.
  // We verify this invariant for every new vertex (excluding the pole, which
  // sits at (0, capH * R) and trivially satisfies it).

  // Compute hole normal
  const auto& v0 = vertices[hole.vertices[0]];
  const auto& v1 = vertices[hole.vertices[1]];
  const auto& v2 = vertices[hole.vertices[2]];
  const Eigen::Vector3f holeNormal = (v1 - v0).cross(v2 - v0).normalized();

  for (size_t i = 0; i < result.newVertices.size(); ++i) {
    const auto& v = result.newVertices[i];
    const Eigen::Vector3f diff = v - centroid;
    const float h = std::abs(diff.dot(holeNormal));
    const Eigen::Vector3f inPlane = diff - diff.dot(holeNormal) * holeNormal;
    const float r = inPlane.norm();

    // For the pole vertex (r ≈ 0), the "sphere" relation degenerates; just check
    // that it's at approximately capHeightRatio * radius distance along the normal.
    if (r < 1e-4f) {
      EXPECT_NEAR(h, capHeightRatio * avgRadius, avgRadius * 0.15f)
          << "Pole vertex should be at capHeightRatio * radius from centroid";
      continue;
    }

    // Check that (r/R)^2 + (h / (capH*R))^2 ≈ 1
    const float rNorm = r / avgRadius;
    const float hNorm = h / (capHeightRatio * avgRadius);
    const float sphereEq = rNorm * rNorm + hNorm * hNorm;
    EXPECT_NEAR(sphereEq, 1.0f, 0.15f)
        << "New vertex " << i << " should lie on the spherical cap surface";
  }
}

// =============================================================================
// SphericalCap: capHeightRatio=0 produces vertices in the hole plane
// =============================================================================

TEST_F(MeshHoleFillingTest, SphericalCapFlatWhenHeightZero) {
  const auto [vertices, triangles] = createTetrahedronWithHole();

  const auto [capVertices, capTriangles] = fillMeshHolesComplete(
      std::span<const Eigen::Vector3f>(vertices),
      std::span<const Eigen::Vector3i>(triangles),
      HoleFillingMethod::SphericalCap,
      0.0f);

  // Should still be watertight
  const auto stats = getMeshTopologyStats(capVertices, capTriangles);
  EXPECT_TRUE(stats.isWatertight);
  EXPECT_TRUE(stats.hasConsistentWinding);
  EXPECT_EQ(stats.boundaryEdges, 0);

  // All new vertices should lie in the plane of the hole boundary
  const auto holes = detectMeshHoles(vertices, triangles);
  ASSERT_EQ(holes.size(), 1);
  const auto& hole = holes[0];

  const auto& v0 = vertices[hole.vertices[0]];
  const auto& v1 = vertices[hole.vertices[1]];
  const auto& v2 = vertices[hole.vertices[2]];
  const Eigen::Vector3f holeNormal = (v1 - v0).cross(v2 - v0).normalized();
  const float planeDist = holeNormal.dot(v0);

  for (size_t i = vertices.size(); i < capVertices.size(); ++i) {
    const float dist = std::abs(holeNormal.dot(capVertices[i]) - planeDist);
    EXPECT_NEAR(dist, 0.0f, 0.01f)
        << "With capHeightRatio=0, new vertex " << i << " should be in the hole plane";
  }
}

// =============================================================================
// Cylinder capped with hemispheres: watertight + cap vertices on sphere
// =============================================================================

TEST_F(MeshHoleFillingTest, CylinderCappedWithHemispheres) {
  const float radius = 1.0f;
  const float halfHeight = 2.0f;
  const float capHeightRatio = 1.0f; // full hemisphere
  const auto [vertices, triangles] = createOpenCylinder(16, radius, halfHeight);

  // The open cylinder should have two holes (top and bottom rings)
  const auto holes = detectMeshHoles(vertices, triangles);
  EXPECT_EQ(holes.size(), 2);

  // Fill with spherical caps
  const auto [filledVertices, filledTriangles] = fillMeshHolesComplete(
      std::span<const Eigen::Vector3f>(vertices),
      std::span<const Eigen::Vector3i>(triangles),
      HoleFillingMethod::SphericalCap,
      capHeightRatio);

  // Should be watertight
  const auto stats = getMeshTopologyStats(filledVertices, filledTriangles);
  EXPECT_TRUE(stats.isWatertight) << "Capped cylinder should be watertight";
  EXPECT_EQ(stats.boundaryEdges, 0) << "Capped cylinder should have no boundary edges";
  EXPECT_EQ(stats.nonWatertightEdges, 0);
  EXPECT_TRUE(stats.hasConsistentWinding);

  // Verify that the new cap vertices lie on hemispheres.
  // Each cap is centered at the centroid of a circular ring (which is on the Z axis)
  // with radius equal to the cylinder radius.  For capHeightRatio=1, vertices should
  // satisfy  (r/R)^2 + (h/(capH*R))^2 ≈ 1  where r is the in-plane distance from
  // the ring centroid and h is the distance along the cap normal.
  // Since fillMeshHoles returns vertices for all caps combined, we check each new
  // vertex against the nearest cap centroid.
  std::vector<Eigen::Vector3f> capCentroids;
  std::vector<float> capRadii;
  for (const auto& hole : holes) {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto idx : hole.vertices) {
      centroid += vertices[idx];
    }
    centroid /= static_cast<float>(hole.vertices.size());
    capCentroids.push_back(centroid);

    float avgR = 0.0f;
    for (const auto idx : hole.vertices) {
      avgR += (vertices[idx] - centroid).norm();
    }
    avgR /= static_cast<float>(hole.vertices.size());
    capRadii.push_back(avgR);
  }

  const auto result = fillMeshHoles(
      std::span<const Eigen::Vector3f>(vertices),
      std::span<const Eigen::Vector3i>(triangles),
      HoleFillingMethod::SphericalCap,
      capHeightRatio);

  for (size_t vi = 0; vi < result.newVertices.size(); ++vi) {
    const auto& v = result.newVertices[vi];

    // Find the nearest cap centroid
    size_t nearest = 0;
    float nearestDist = (v - capCentroids[0]).squaredNorm();
    for (size_t ci = 1; ci < capCentroids.size(); ++ci) {
      const float d = (v - capCentroids[ci]).squaredNorm();
      if (d < nearestDist) {
        nearestDist = d;
        nearest = ci;
      }
    }

    const auto& centroid = capCentroids[nearest];
    const float avgR = capRadii[nearest];
    // Cap normal points outward along Z
    const Eigen::Vector3f normal = Eigen::Vector3f(0.0f, 0.0f, centroid.z() > 0 ? 1.0f : -1.0f);

    const Eigen::Vector3f diff = v - centroid;
    const float h = std::abs(diff.dot(normal));
    const Eigen::Vector3f inPlane = diff - diff.dot(normal) * normal;
    const float r = inPlane.norm();

    if (r < 1e-4f) {
      EXPECT_NEAR(h, capHeightRatio * avgR, avgR * 0.15f)
          << "Pole vertex should be at capHeightRatio * radius from cap centroid";
      continue;
    }

    const float rNorm = r / avgR;
    const float hNorm = h / (capHeightRatio * avgR);
    const float sphereEq = rNorm * rNorm + hNorm * hNorm;
    EXPECT_NEAR(sphereEq, 1.0f, 0.15f)
        << "Cap vertex " << vi << " should lie on hemisphere surface";
  }
}
