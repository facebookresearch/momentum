/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/DualContouring.h"

#include <gtest/gtest.h>

#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "axel/BoundingBox.h"
#include "axel/SignedDistanceField.h"

namespace axel {

class DualContouringTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  SignedDistanceFieldf createSphereSdf(
      float radius,
      const Eigen::Vector3f& center,
      const BoundingBoxf& bounds,
      const Eigen::Vector3<Index>& resolution) {
    SignedDistanceFieldf sdf(bounds, resolution);

    for (Index i = 0; i < resolution.x(); ++i) {
      for (Index j = 0; j < resolution.y(); ++j) {
        for (Index k = 0; k < resolution.z(); ++k) {
          const Eigen::Vector3f gridPos(
              static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
          const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);
          const float distance = (worldPos - center).norm() - radius;
          sdf.set(i, j, k, distance);
        }
      }
    }

    return sdf;
  }

  struct EdgeKey {
    Index v0, v1;

    EdgeKey(Index a, Index b) : v0(std::min(a, b)), v1(std::max(a, b)) {}

    bool operator==(const EdgeKey& other) const {
      return v0 == other.v0 && v1 == other.v1;
    }
  };

  struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const {
      return std::hash<Index>{}(key.v0) ^ (std::hash<Index>{}(key.v1) << 1);
    }
  };

  template <typename S>
  void validateQuadMesh(const DualContouringResult<S>& result) {
    for (const auto& quad : result.quads) {
      EXPECT_GE(quad[0], 0) << "Quad vertex index 0 is negative";
      EXPECT_LT(quad[0], static_cast<int>(result.vertices.size()))
          << "Quad vertex index 0 out of range";
      EXPECT_GE(quad[1], 0) << "Quad vertex index 1 is negative";
      EXPECT_LT(quad[1], static_cast<int>(result.vertices.size()))
          << "Quad vertex index 1 out of range";
      EXPECT_GE(quad[2], 0) << "Quad vertex index 2 is negative";
      EXPECT_LT(quad[2], static_cast<int>(result.vertices.size()))
          << "Quad vertex index 2 out of range";
      EXPECT_GE(quad[3], 0) << "Quad vertex index 3 is negative";
      EXPECT_LT(quad[3], static_cast<int>(result.vertices.size()))
          << "Quad vertex index 3 out of range";
    }
  }

  template <typename S>
  void validateTriangleMesh(
      const std::vector<Eigen::Vector3<S>>& vertices,
      const std::vector<Eigen::Vector3i>& triangles) {
    for (const auto& triangle : triangles) {
      EXPECT_GE(triangle[0], 0) << "Triangle vertex index 0 is negative";
      EXPECT_LT(triangle[0], static_cast<int>(vertices.size()))
          << "Triangle vertex index 0 out of range";
      EXPECT_GE(triangle[1], 0) << "Triangle vertex index 1 is negative";
      EXPECT_LT(triangle[1], static_cast<int>(vertices.size()))
          << "Triangle vertex index 1 out of range";
      EXPECT_GE(triangle[2], 0) << "Triangle vertex index 2 is negative";
      EXPECT_LT(triangle[2], static_cast<int>(vertices.size()))
          << "Triangle vertex index 2 out of range";
    }
  }
};

TEST_F(DualContouringTest, SphereVerticesOnSurface) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(32, 32, 32);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.vertices.size(), 0);
  EXPECT_GT(result.quads.size(), 0);

  validateQuadMesh(result);

  const float tolerance = sdf.voxelSize().maxCoeff() * 2.0f;

  for (const auto& vertex : result.vertices) {
    const float distanceToCenter = (vertex - sphereCenter).norm();
    const float distanceToSurface = std::abs(distanceToCenter - sphereRadius);

    EXPECT_LT(distanceToSurface, tolerance)
        << "Vertex [" << vertex.transpose() << "] is too far from sphere surface. "
        << "Distance to surface: " << distanceToSurface << ", tolerance: " << tolerance;
  }
}

TEST_F(DualContouringTest, SphereMeshManifoldness) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(32, 32, 32);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  // If you use isovalue == 0, then you end up with vertices exactly on corners which we don't
  // currently handle well.
  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.vertices.size(), 0);
  EXPECT_GT(result.quads.size(), 0);

  validateQuadMesh(result);

  struct DirectedEdge {
    Index from, to;

    bool operator==(const DirectedEdge& other) const {
      return from == other.from && to == other.to;
    }
  };

  struct DirectedEdgeHash {
    std::size_t operator()(const DirectedEdge& edge) const {
      return std::hash<Index>{}(edge.from) ^ (std::hash<Index>{}(edge.to) << 1);
    }
  };

  std::unordered_set<DirectedEdge, DirectedEdgeHash> directedEdges;

  for (const auto& quad : result.quads) {
    directedEdges.insert({quad[0], quad[1]});
    directedEdges.insert({quad[1], quad[2]});
    directedEdges.insert({quad[2], quad[3]});
    directedEdges.insert({quad[3], quad[0]});
  }

  int edgesWithOpposite = 0;
  int edgesWithoutOpposite = 0;

  for (const auto& edge : directedEdges) {
    DirectedEdge opposite = {edge.to, edge.from};
    if (directedEdges.find(opposite) != directedEdges.end()) {
      edgesWithOpposite++;
    } else {
      edgesWithoutOpposite++;
    }
  }

  EXPECT_EQ(edgesWithoutOpposite, 0)
      << "All directed edges should have their opposite edge for a manifold mesh. "
      << "Edges with opposite: " << edgesWithOpposite
      << ", Edges without opposite: " << edgesWithoutOpposite;

  EXPECT_GT(edgesWithOpposite, 0) << "Mesh should have edges";
}

TEST_F(DualContouringTest, SphereCorrectWinding) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(32, 32, 32);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.vertices.size(), 0);
  EXPECT_GT(result.quads.size(), 0);

  validateQuadMesh(result);

  int correctWindingCount = 0;
  int incorrectWindingCount = 0;

  for (const auto& quad : result.quads) {
    const Eigen::Vector3f v0 = result.vertices[quad[0]];
    const Eigen::Vector3f v1 = result.vertices[quad[1]];
    const Eigen::Vector3f v2 = result.vertices[quad[2]];
    const Eigen::Vector3f v3 = result.vertices[quad[3]];

    const Eigen::Vector3f edge1 = v1 - v0;
    const Eigen::Vector3f edge2 = v2 - v0;
    const Eigen::Vector3f normal = edge1.cross(edge2).normalized();

    const Eigen::Vector3f quadCenter = (v0 + v1 + v2 + v3) / 4.0f;
    const Eigen::Vector3f expectedNormal = (quadCenter - sphereCenter).normalized();

    const float dot = normal.dot(expectedNormal);

    if (dot > 0.0f) {
      correctWindingCount++;
    } else {
      incorrectWindingCount++;
    }
  }

  const float correctRatio =
      static_cast<float>(correctWindingCount) / static_cast<float>(result.quads.size());

  EXPECT_GT(correctRatio, 0.9f) << "Most quad normals should point outward from sphere center. "
                                << "Correct: " << correctWindingCount
                                << ", Incorrect: " << incorrectWindingCount
                                << ", Total: " << result.quads.size()
                                << ", Ratio: " << correctRatio;
}

TEST_F(DualContouringTest, EmptySDF) {
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(8, 8, 8);
  SignedDistanceFieldf sdf(bounds, resolution);

  sdf.fill(1.0f);

  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.vertices.size(), 0);
  EXPECT_EQ(result.quads.size(), 0);
}

TEST_F(DualContouringTest, TriangulateQuads) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(16, 16, 16);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.quads.size(), 0);

  validateQuadMesh(result);

  const auto triangles = triangulateQuads(result.quads);

  EXPECT_EQ(triangles.size(), result.quads.size() * 2);

  for (size_t i = 0; i < result.quads.size(); ++i) {
    const auto& quad = result.quads[i];
    const auto& tri1 = triangles[i * 2];
    const auto& tri2 = triangles[i * 2 + 1];

    EXPECT_EQ(tri1[0], quad[0]);
    EXPECT_EQ(tri1[1], quad[1]);
    EXPECT_EQ(tri1[2], quad[2]);

    EXPECT_EQ(tri2[0], quad[0]);
    EXPECT_EQ(tri2[1], quad[2]);
    EXPECT_EQ(tri2[2], quad[3]);
  }
}

TEST_F(DualContouringTest, NonZeroIsovalue) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(16, 16, 16);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  const float isovalue = 0.2f;
  const auto result = dualContouring(sdf, isovalue);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.vertices.size(), 0);

  validateQuadMesh(result);

  const float expectedRadius = sphereRadius + isovalue;
  const float tolerance = sdf.voxelSize().maxCoeff() * 2.0f;

  for (const auto& vertex : result.vertices) {
    const float distanceToCenter = (vertex - sphereCenter).norm();
    const float distanceToIsosurface = std::abs(distanceToCenter - expectedRadius);

    EXPECT_LT(distanceToIsosurface, tolerance)
        << "Vertex should be near isosurface at distance " << expectedRadius;
  }
}

TEST_F(DualContouringTest, TriangulatedQuadsCorrectWinding) {
  const float sphereRadius = 1.0f;
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(32, 32, 32);

  auto sdf = createSphereSdf(sphereRadius, sphereCenter, bounds, resolution);

  const auto result = dualContouring(sdf, 0.0f);

  EXPECT_TRUE(result.success);
  EXPECT_GT(result.vertices.size(), 0);
  EXPECT_GT(result.quads.size(), 0);

  validateQuadMesh(result);

  const auto triangles = triangulateQuads(result.quads);

  validateTriangleMesh(result.vertices, triangles);

  int correctWindingCount = 0;
  int incorrectWindingCount = 0;

  for (const auto& triangle : triangles) {
    const Eigen::Vector3f v0 = result.vertices[triangle[0]];
    const Eigen::Vector3f v1 = result.vertices[triangle[1]];
    const Eigen::Vector3f v2 = result.vertices[triangle[2]];

    const Eigen::Vector3f edge1 = v1 - v0;
    const Eigen::Vector3f edge2 = v2 - v0;
    const Eigen::Vector3f normal = edge1.cross(edge2).normalized();

    const Eigen::Vector3f triangleCenter = (v0 + v1 + v2) / 3.0f;
    const Eigen::Vector3f expectedNormal = (triangleCenter - sphereCenter).normalized();

    const float dot = normal.dot(expectedNormal);

    if (dot > 0.0f) {
      correctWindingCount++;
    } else {
      incorrectWindingCount++;
    }
  }

  const float correctRatio =
      static_cast<float>(correctWindingCount) / static_cast<float>(triangles.size());

  EXPECT_GT(correctRatio, 0.9f) << "Most triangle normals should point outward from sphere center. "
                                << "Correct: " << correctWindingCount
                                << ", Incorrect: " << incorrectWindingCount
                                << ", Total: " << triangles.size() << ", Ratio: " << correctRatio;
}

} // namespace axel
