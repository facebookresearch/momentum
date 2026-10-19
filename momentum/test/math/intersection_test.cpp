/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "momentum/math/intersection.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"

using namespace momentum;

namespace {

template <typename T>
MeshT<T> makeMesh(std::vector<Vector3<T>> vertices, std::vector<Vector3<int32_t>> faces) {
  MeshT<T> mesh;
  mesh.vertices = std::move(vertices);
  mesh.faces = std::move(faces);
  return mesh;
}

std::vector<std::pair<int32_t, int32_t>> sortedIntersections(
    std::vector<std::pair<int32_t, int32_t>> intersections) {
  for (auto& [face0, face1] : intersections) {
    if (face0 > face1) {
      std::swap(face0, face1);
    }
  }
  std::ranges::sort(intersections);
  return intersections;
}

template <typename T>
void expectIntersections(
    const MeshT<T>& mesh,
    const std::vector<std::pair<int32_t, int32_t>>& expected) {
  EXPECT_EQ(sortedIntersections(intersectMeshBruteForce(mesh)), expected);
  EXPECT_EQ(sortedIntersections(intersectMesh(mesh)), expected);
}

template <typename T>
class MomentumMeshIntersectionTest : public testing::Test {};

using ScalarTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(MomentumMeshIntersectionTest, ScalarTypes);

} // namespace

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
      {}, // lines
      {}, // colors
      {}, // confidence
      {}, // texcoords
      {}, // texcoord_faces
      {}, // texcoord_lines
      {}, // polyFaces
      {}, // polyFaceSizes
      {}, // polyTexcoordFaces
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
      {}, // lines
      {}, // colors
      {}, // confidence
      {}, // texcoords
      {}, // texcoord_faces
      {}, // texcoord_lines
      {}, // polyFaces
      {}, // polyFaceSizes
      {}, // polyTexcoordFaces
  };

  std::vector<std::pair<int32_t, int32_t>> intersections = intersectMeshBruteForce(mesh);
  std::vector<std::pair<int32_t, int32_t>> intersections2 = intersectMesh(mesh);
  EXPECT_FALSE(intersections.empty());
  EXPECT_FALSE(intersections2.empty());
  EXPECT_TRUE(intersections.size() == intersections2.size());
  EXPECT_TRUE(intersections[0] == intersections2[0]);
}

TEST(Momentum_Mesh_Intersection, MeshSelfIntersectionRandom) {
  // Construct non-empty mesh with deterministic random data so the brute-force vs BVH
  // comparison below is reproducible across runs and platforms.
  Random<> rng{12345};
  constexpr size_t numVerts = 1000;
  constexpr size_t numFaces = 100;
  std::vector<Vector3<float>> vertices(numVerts);
  std::vector<Vector3<int32_t>> faces(numFaces);
  for (size_t iVert = 0; iVert < numVerts; iVert++) {
    vertices[iVert] = rng.uniform<Vector3<float>>(-1.0f, 1.0f);
  }
  for (size_t iFace = 0; iFace < numFaces; iFace++) {
    for (size_t d = 0; d < 3; d++) {
      faces[iFace][d] = static_cast<int32_t>(rng.uniform<int32_t>(0, numVerts - 1));
    }
  }
  Mesh mesh = {
      vertices,
      {}, // normals
      faces, // faces
      {}, // lines
      {}, // colors
      {}, // confidence
      {}, // texcoords
      {}, // texcoord_faces
      {}, // texcoord_lines
      {}, // polyFaces
      {}, // polyFaceSizes
      {}, // polyTexcoordFaces
  };

  EXPECT_EQ(
      sortedIntersections(intersectMeshBruteForce(mesh)), sortedIntersections(intersectMesh(mesh)));
}

TYPED_TEST(MomentumMeshIntersectionTest, RejectsPlaneCrossingsOutsideVerticalTriangle) {
  using T = TypeParam;
  const MeshT<T> mesh = makeMesh<T>(
      {{T(0), T(0), T(0)},
       {T(0), T(1), T(0)},
       {T(0), T(0), T(1)},
       {T(-1), T(0.5), T(2)},
       {T(1), T(0.5), T(2)},
       {T(1), T(0.5), T(0.5)}},
      {{0, 1, 2}, {3, 4, 5}});

  expectIntersections(mesh, {});
}

TYPED_TEST(MomentumMeshIntersectionTest, DetectsIntersectionsAtAnyScale) {
  using T = TypeParam;
  constexpr T kSmallScale = std::is_same_v<T, float> ? T(1e-8) : T(1e-18);
  const std::vector<std::pair<int32_t, int32_t>> expected{{0, 1}};

  for (const T scale : {T(1), kSmallScale}) {
    const MeshT<T> mesh = makeMesh<T>(
        {{T(0), T(0), T(0)},
         {scale, T(0), T(0)},
         {T(0), scale, T(0)},
         {T(0), T(0.5) * scale, T(-0.5) * scale},
         {T(0), T(0.5) * scale, T(0.5) * scale},
         {scale, T(0.5) * scale, T(0.5) * scale}},
        {{0, 1, 2}, {3, 4, 5}});

    expectIntersections(mesh, expected);
  }
}

TYPED_TEST(MomentumMeshIntersectionTest, DetectsInteriorIntersectionWithSharedVertex) {
  using T = TypeParam;
  const MeshT<T> mesh = makeMesh<T>(
      {{T(0), T(0), T(0)},
       {T(2), T(0), T(0)},
       {T(0), T(2), T(0)},
       {T(1), T(1), T(-1)},
       {T(1), T(1), T(1)}},
      {{0, 1, 2}, {0, 3, 4}});

  expectIntersections(mesh, {{0, 1}});
}

TYPED_TEST(MomentumMeshIntersectionTest, IgnoresDegenerateTriangles) {
  using T = TypeParam;
  const MeshT<T> mesh = makeMesh<T>(
      {{T(0), T(0), T(0)},
       {T(1), T(0), T(0)},
       {T(2), T(0), T(0)},
       {T(0.5), T(-1), T(-1)},
       {T(0.5), T(1), T(-1)},
       {T(0.5), T(0), T(1)}},
      {{0, 1, 2}, {3, 4, 5}});

  expectIntersections(mesh, {});
}
