/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/intersection.h"

#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <axel/Bvh.h>
#include <axel/math/RayTriangleIntersection.h>

namespace momentum {

namespace {

template <typename T>
Vector3<T> normalizedFaceNormal(const MeshT<T>& mesh, const size_t faceIndex) {
  const auto& face = mesh.faces[faceIndex];
  const Vector3<T> edge0 = mesh.vertices[face[1]] - mesh.vertices[face[0]];
  const Vector3<T> edge1 = mesh.vertices[face[2]] - mesh.vertices[face[0]];
  const Vector3<T> normal = edge0.cross(edge1);
  const T edgeScale = edge0.norm() * edge1.norm();
  const T normalNorm = normal.norm();
  if (edgeScale == T(0) || normalNorm <= momentum::Eps<T>(T(1e-7), T(1e-15)) * edgeScale) {
    return Vector3<T>::Zero();
  }
  return normal / normalNorm;
}

} // namespace

template <typename T>
bool intersectFace(
    const MeshT<T>& mesh,
    const std::vector<Vector3<T>>& faceNormals,
    const int32_t face0,
    const int32_t face1) {
  if (!faceNormals[face0].allFinite() || !faceNormals[face1].allFinite() ||
      faceNormals[face0].squaredNorm() == T(0) || faceNormals[face1].squaredNorm() == T(0)) {
    return false;
  }

  // Ignore edge-adjacent or duplicate faces, but retain pairs that share only one vertex:
  // their interiors can still cross away from the shared point.
  size_t sharedVertexCount = 0;
  for (size_t iVertex1 : mesh.faces[face0]) {
    for (size_t iVertex2 : mesh.faces[face1]) {
      if (iVertex1 == iVertex2) {
        ++sharedVertexCount;
      }
    }
  }
  if (sharedVertexCount >= 2) {
    return false;
  }

  // Test each edge as a finite ray against the opposite triangle. Running both orderings catches
  // intersections whose endpoints lie on either triangle's boundary.
  for (size_t pass = 0; pass < 2; ++pass) {
    const int32_t iFaceA = pass == 0 ? face0 : face1;
    const int32_t iFaceB = pass == 0 ? face1 : face0;
    const auto& faceVertsA = mesh.faces[iFaceA];
    const auto& faceVertsB = mesh.faces[iFaceB];

    for (size_t iVertex = 0; iVertex < 3; ++iVertex) {
      const Vector3<T>& edgeStart = mesh.vertices[faceVertsA[iVertex]];
      const Vector3<T> edge = mesh.vertices[faceVertsA[(iVertex + 1) % 3]] - edgeStart;
      Vector3<T> intersectionPoint;
      T edgeParameter = T(0);
      if (axel::rayTriangleIntersect(
              edgeStart,
              edge,
              mesh.vertices[faceVertsB[0]],
              mesh.vertices[faceVertsB[1]],
              mesh.vertices[faceVertsB[2]],
              intersectionPoint,
              edgeParameter) &&
          edgeParameter > T(0) && edgeParameter < T(1)) {
        return true;
      }
    }
  }
  return false;
}

template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce(const MeshT<T>& mesh) {
  std::vector<std::pair<int32_t, int32_t>> intersectingFaces;

  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    faceNormals[iFace] = normalizedFaceNormal(mesh, iFace);
  }
  // O(F^2) all-pairs check; iterate the upper-triangular pair set so each unordered
  // pair is tested exactly once.
  for (size_t iFace0 = 0; iFace0 < mesh.faces.size(); iFace0++) {
    for (size_t iFace1 = iFace0 + 1; iFace1 < mesh.faces.size(); iFace1++) {
      if (intersectFace(
              mesh, faceNormals, static_cast<int32_t>(iFace0), static_cast<int32_t>(iFace1))) {
        intersectingFaces.emplace_back(static_cast<int32_t>(iFace0), static_cast<int32_t>(iFace1));
      }
    }
  }

  return intersectingFaces;
}

template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMesh(const MeshT<T>& mesh) {
  std::vector<std::pair<int32_t, int32_t>> intersectingFaces;

  std::vector<axel::BoundingBox<T>> aabbs(mesh.faces.size());
  axel::Bvh<T> bvh;

  // Build per-face normals and AABBs in a single pass. Degenerate faces retain a zero normal and
  // are ignored by the narrow phase.
  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<T> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<T> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<T> v2 = mesh.vertices[faceVerts[2]];
    faceNormals[iFace] = normalizedFaceNormal(mesh, iFace);

    auto& aabb = aabbs[iFace];
    aabb.aabb.min() = v0;
    aabb.aabb.max() = v0;
    aabb.id = static_cast<axel::Index>(iFace);
    aabb.extend(v1);
    aabb.extend(v2);
  }
  bvh.setBoundingBoxes(aabbs);

  // Use the BVH to enumerate only AABB-overlapping face pairs (broad phase), then run
  // the exact triangle-triangle test (narrow phase) on each candidate. Returning true
  // from the callback tells axel to continue traversal.
  bvh.traverseOverlappingPairs([&](size_t iFace0, size_t iFace1) {
    if (intersectFace(
            mesh, faceNormals, static_cast<int32_t>(iFace0), static_cast<int32_t>(iFace1))) {
      intersectingFaces.emplace_back(static_cast<int32_t>(iFace0), static_cast<int32_t>(iFace1));
    }
    return true;
  });

  return intersectingFaces;
}

// explicit instantiations
template bool intersectFace<float>(
    const MeshT<float>& mesh,
    const std::vector<Vector3<float>>& faceNormals,
    const int32_t face0,
    const int32_t face1);
template bool intersectFace<double>(
    const MeshT<double>& mesh,
    const std::vector<Vector3<double>>& faceNormals,
    const int32_t face0,
    const int32_t face1);
template std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce<float>(
    const MeshT<float>& mesh);
template std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce<double>(
    const MeshT<double>& mesh);
template std::vector<std::pair<int32_t, int32_t>> intersectMesh<float>(const MeshT<float>& mesh);
template std::vector<std::pair<int32_t, int32_t>> intersectMesh<double>(const MeshT<double>& mesh);

} // namespace momentum
