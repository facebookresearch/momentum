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

namespace momentum {

namespace {

template <typename T>
T sign(const Eigen::Vector3<T>& p1, const Eigen::Vector3<T>& p2, const Eigen::Vector3<T>& p3) {
  return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

} // namespace

template <typename T>
bool intersectFace(
    const MeshT<T>& mesh,
    const std::vector<Vector3<T>>& faceNormals,
    const int32_t face0,
    const int32_t face1) {
  // if the faces share a vertex, they are not intersecting
  for (size_t iVertex1 : mesh.faces[face0]) {
    for (size_t iVertex2 : mesh.faces[face1]) {
      if (iVertex1 == iVertex2) {
        return false;
      }
    }
  }

  size_t iFaceA = 0, iFaceB = 0;
  for (size_t i = 0; i < 2; i++) {
    // check for overlap by casting edges of triangle A to plane of triangle B
    // and checking if the point of intersection is inside triangle B
    if (i == 0) {
      iFaceA = face0;
      iFaceB = face1;
    } else {
      iFaceA = face1;
      iFaceB = face0;
    }

    // calculate plane of triangle B
    const auto& faceVertsB = mesh.faces[iFaceB];
    T planeD = -(mesh.vertices[faceVertsB[0]].dot(faceNormals[iFaceB]));

    const auto& faceVertsA = mesh.faces[iFaceA];

    for (size_t iTriVertex = 0; iTriVertex < 3; iTriVertex++) {
      // iterate edges of triangleA and ray cast onto plane of triangle B
      const Eigen::Vector3<T>& e0 = mesh.vertices[faceVertsA[iTriVertex]];
      const Eigen::Vector3<T> eRay = mesh.vertices[faceVertsA[(iTriVertex + 1) % 3]] - e0;
      if (abs(eRay.dot(faceNormals[iFaceB])) - momentum::Eps<T>(1e-7, 1e-16) < 0) {
        // avoid divide by zero. If its almost 0, ray is almost parallel to plane
        continue;
      }
      const T rayUnitLength =
          -(e0.dot(faceNormals[iFaceB]) + planeD) / (eRay.dot(faceNormals[iFaceB]));
      if (rayUnitLength <= 0 || rayUnitLength >= 1) {
        // edge does not intersect with plane
        continue;
      }

      // point of intersection between edge of triangle A and plane of triangle B
      const Eigen::Vector3<T> pIntersect = e0 + rayUnitLength * eRay;

      // check if point is in triangle by projectio onto z=0 plane
      const T s1 = sign(pIntersect, mesh.vertices[faceVertsB[0]], mesh.vertices[faceVertsB[1]]);
      const T s2 = sign(pIntersect, mesh.vertices[faceVertsB[1]], mesh.vertices[faceVertsB[2]]);
      const T s3 = sign(pIntersect, mesh.vertices[faceVertsB[2]], mesh.vertices[faceVertsB[0]]);

      const bool has_neg = (s1 < 0) || (s2 < 0) || (s3 < 0);
      const bool has_pos = (s1 > 0) || (s2 > 0) || (s3 > 0);

      if (!(has_neg && has_pos)) {
        return true;
      }
    }
  }
  return false;
}

template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce(const MeshT<T>& mesh) {
  std::vector<std::pair<int32_t, int32_t>> intersectingFaces;

  // calculate face normals
  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<T> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<T> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<T> v2 = mesh.vertices[faceVerts[2]];
    faceNormals[iFace] = (v1 - v0).cross(v2 - v0).normalized();
  }
  // calculate all face intersections brute force
  for (size_t iFace0 = 0; iFace0 < mesh.faces.size(); iFace0++) {
    for (size_t iFace1 = iFace0 + 1; iFace1 < mesh.faces.size(); iFace1++) {
      if (intersectFace(mesh, faceNormals, iFace0, iFace1)) {
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

  // calculate face normals and bounding boxes
  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<T> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<T> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<T> v2 = mesh.vertices[faceVerts[2]];
    faceNormals[iFace] = (v1 - v0).cross(v2 - v0).normalized();

    auto& aabb = aabbs[iFace];
    aabb.aabb.min() = v0;
    aabb.aabb.max() = v0;
    aabb.id = iFace;
    aabb.extend(v1);
    aabb.extend(v2);
  }
  bvh.setBoundingBoxes(aabbs);

  // calculate all face intersections
  bvh.traverseOverlappingPairs([&](size_t iFace0, size_t iFace1) {
    if (intersectFace(mesh, faceNormals, iFace0, iFace1)) {
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
