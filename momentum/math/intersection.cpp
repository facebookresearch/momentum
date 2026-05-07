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

// Signed area (twice) of the 2D triangle (p1, p2, p3) projected onto the z=0 plane.
// Equivalent to the z-component of the cross product (p2-p1) x (p3-p1) using only
// the x,y coordinates. The sign indicates the orientation (CCW positive, CW negative)
// and is used to test whether p1 lies on a consistent side of the directed edge p2->p3.
//
// TODO: This always projects onto z=0, which is degenerate for triangles whose normal
// is close to the world Z axis (i.e., nearly horizontal in the xy plane). A more robust
// test should project onto the coordinate plane that drops the dominant component of
// the triangle's normal, or use barycentric/edge-cross tests in 3D.
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
  // Faces that share at least one vertex are treated as non-intersecting. This avoids
  // false positives on adjacent triangles that meet along an edge or at a vertex but
  // do not actually overlap in their interiors.
  // TODO: This O(9) loop also rejects pairs that share only a single vertex, which can
  // miss real interior intersections when two triangles fan out from a common vertex.
  for (size_t iVertex1 : mesh.faces[face0]) {
    for (size_t iVertex2 : mesh.faces[face1]) {
      if (iVertex1 == iVertex2) {
        return false;
      }
    }
  }

  // Two-pass edge-vs-plane test: for each ordering (A=face0,B=face1) and (A=face1,B=face0),
  // intersect each of triangle A's three edges against triangle B's supporting plane and
  // check whether the intersection point lies inside triangle B. If any edge-segment of
  // either triangle pierces the other triangle's interior, the triangles intersect.
  // Note: this is asymmetric and does not implement the full Möller separating-axis test;
  // it can miss coplanar intersections and edge-on-edge contacts.
  size_t iFaceA = 0, iFaceB = 0;
  for (size_t i = 0; i < 2; i++) {
    if (i == 0) {
      iFaceA = face0;
      iFaceB = face1;
    } else {
      iFaceA = face1;
      iFaceB = face0;
    }

    // Plane of triangle B in the form n . x + d = 0, with n = faceNormals[iFaceB] and
    // d = -n . v0. Any point x satisfies n . x + d = 0 iff x lies on the plane.
    const auto& faceVertsB = mesh.faces[iFaceB];
    T planeD = -(mesh.vertices[faceVertsB[0]].dot(faceNormals[iFaceB]));

    const auto& faceVertsA = mesh.faces[iFaceA];

    for (size_t iTriVertex = 0; iTriVertex < 3; iTriVertex++) {
      // Treat edge (e0 -> e0+eRay) of triangle A as a parametric ray and intersect it
      // with the plane of triangle B. Solve for t in: n . (e0 + t*eRay) + d = 0.
      const Eigen::Vector3<T>& e0 = mesh.vertices[faceVertsA[iTriVertex]];
      const Eigen::Vector3<T> eRay = mesh.vertices[faceVertsA[(iTriVertex + 1) % 3]] - e0;
      // If eRay is (nearly) perpendicular to the plane normal, the edge is (nearly)
      // parallel to the plane, so the denominator vanishes; skip to avoid division
      // by zero. Tolerance is 1e-7 (float) / 1e-16 (double) on |eRay . n|, which is
      // an absolute (not scale-invariant) threshold.
      // TODO: Tolerance on |eRay . n| is not scale-invariant; it should be relative
      // to ||eRay|| (and possibly ||n||, though n is unit-length here).
      if (abs(eRay.dot(faceNormals[iFaceB])) - momentum::Eps<T>(1e-7, 1e-16) < 0) {
        continue;
      }
      const T rayUnitLength =
          -(e0.dot(faceNormals[iFaceB]) + planeD) / (eRay.dot(faceNormals[iFaceB]));
      // Edge parametrization is t in [0,1]; values outside (0,1) mean the plane is
      // hit beyond the edge endpoints. Endpoints (t=0 or t=1) are excluded, which
      // matches the share-vertex early-out above and prevents double-counting.
      if (rayUnitLength <= 0 || rayUnitLength >= 1) {
        continue;
      }

      const Eigen::Vector3<T> pIntersect = e0 + rayUnitLength * eRay;

      // Point-in-triangle test using same-side-of-edges via 2D signed areas in the
      // xy plane (see sign() above). pIntersect is inside triangle B iff all three
      // signed areas have the same sign (or zero), i.e., NOT (some negative AND some
      // positive). This 2D projection is degenerate for triangles whose normal is
      // close to the world Z axis; see TODO on sign().
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

  // Per-face unit normals: n = (v1 - v0) x (v2 - v0) / ||...||. Degenerate (zero-area)
  // triangles produce a zero cross product; .normalized() then returns NaN/zero, which
  // can poison downstream tests in intersectFace.
  // TODO: Skip or specially handle degenerate triangles where (v1-v0) x (v2-v0) ~= 0.
  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<T> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<T> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<T> v2 = mesh.vertices[faceVerts[2]];
    const auto cross = (v1 - v0).cross(v2 - v0);
    const T crossNorm = cross.norm();
    faceNormals[iFace] =
        (crossNorm > T(0)) ? (cross / crossNorm).eval() : Eigen::Vector3<T>::Zero();
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

  // Build per-face unit normals (see intersectMeshBruteForce for caveats on degenerate
  // triangles) and per-face AABBs in a single pass. The AABB is seeded from v0 and then
  // extended to include v1 and v2 so that min/max bound all three vertices.
  std::vector<Vector3<T>> faceNormals(mesh.faces.size());
  for (size_t iFace = 0; iFace < mesh.faces.size(); iFace++) {
    const auto& faceVerts = mesh.faces[iFace];
    const Eigen::Vector3<T> v0 = mesh.vertices[faceVerts[0]];
    const Eigen::Vector3<T> v1 = mesh.vertices[faceVerts[1]];
    const Eigen::Vector3<T> v2 = mesh.vertices[faceVerts[2]];
    const auto cross = (v1 - v0).cross(v2 - v0);
    const T crossNorm = cross.norm();
    faceNormals[iFace] =
        (crossNorm > T(0)) ? (cross / crossNorm).eval() : Eigen::Vector3<T>::Zero();

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
