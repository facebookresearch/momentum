/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/math/intersection.h"
#include "momentum/math/mesh.h"

using namespace momentum;

TEST(Momentum_Mesh_Intersection, TriangleIntersection) {
  // Construct non-empty mesh
  Mesh mesh = {
      {{0.0, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       {0.0, 1.0, 0.0},
       {0.0, 0.5, -0.5},
       {0.0, 0.5, 0.5},
       {1.0, 0.5, 0.5},
       {0.0, 1.5, -0.5},
       {0.0, 1.5, 0.5},
       {1.0, 1.5, 0.5}}, // vertices
      {}, // normals
      {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, // faces
  };

  std::vector<Vector3<float>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<float> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<float> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<float> v2 = mesh.vertices[faceVerts[2]];
    faceNormals[iFace] = (v1 - v0).cross(v2 - v0).normalized();
  }

  EXPECT_TRUE(intersectFace(mesh, faceNormals, 0, 1));
  EXPECT_FALSE(intersectFace(mesh, faceNormals, 0, 2));
  EXPECT_FALSE(intersectFace(mesh, faceNormals, 1, 2));
}

TEST(Momentum_Mesh_Intersection, MeshSelfIntersection) {
  // Construct non-empty mesh
  Mesh mesh = {
      {{0.0, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       {0.0, 1.0, 0.0},
       {0.0, 0.5, -0.5},
       {0.0, 0.5, 0.5},
       {1.0, 0.5, 0.5},
       {0.0, 1.5, -0.5},
       {0.0, 1.5, 0.5},
       {1.0, 1.5, 0.5}}, // vertices
      {}, // normals
      {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, // faces
  };

  std::vector<std::pair<int32_t, int32_t>> intersections = intersectMeshBruteForce(mesh);
  std::vector<std::pair<int32_t, int32_t>> intersections2 = intersectMesh(mesh);
  EXPECT_FALSE(intersections.empty());
  EXPECT_FALSE(intersections2.empty());
  EXPECT_TRUE(intersections.size() == intersections2.size());
  EXPECT_TRUE(intersections[0] == intersections2[0]);
}

TEST(Momentum_Mesh_Intersection, MeshSelfIntersectionRandom) {
  // Construct non-empty mesh
  constexpr size_t numVerts = 1000;
  constexpr size_t numFaces = 100;
  std::vector<Vector3<float>> vertices(numVerts);
  std::vector<Vector3<int32_t>> faces(numFaces);
  for (size_t iVert = 0; iVert < numVerts; iVert++) {
    vertices[iVert] = Vector3<float>::Random();
  }
  for (size_t iFace = 0; iFace < numFaces; iFace++) {
    faces[iFace] = Vector3<int32_t>::Random();
    for (size_t d = 0; d < 3; d++) {
      faces[iFace][d] = faces[iFace][d] % numVerts;
    }
  }
  Mesh mesh = {
      vertices,
      {}, // normals
      faces, // faces
  };

  // check that both brute force and optimized version return the same result
  std::vector<std::pair<int32_t, int32_t>> intersections = intersectMeshBruteForce(mesh);
  std::vector<std::pair<int32_t, int32_t>> intersections2 = intersectMesh(mesh);
  EXPECT_TRUE(intersections.size() == intersections2.size());
  for (auto& intersection : intersections) {
    const auto it1 = std::find(intersections2.begin(), intersections2.end(), intersection);
    const auto it2 = std::find(
        intersections2.begin(),
        intersections2.end(),
        std::pair<int32_t, int32_t>(intersection.second, intersection.first));
    EXPECT_TRUE(it1 != intersections2.end() || it2 != intersections2.end());
  }
}
