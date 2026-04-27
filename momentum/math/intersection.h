/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

/// Tests whether two given triangle faces of a mesh intersect with each other.
///
/// @param mesh Source mesh providing vertex positions and triangle face indices.
/// @param faceNormals Per-face normals, indexed by face id; must be sized to match
///   `mesh.faces` and computed from the same vertex positions as `mesh`.
/// @param face0 Index of the first face into `mesh.faces`.
/// @param face1 Index of the second face into `mesh.faces`.
/// @return True if the two triangles intersect, false otherwise. Faces that share
///   one or more vertex indices are treated as non-intersecting (adjacent faces are
///   ignored to avoid false positives along shared edges/vertices).
/// @pre `face0` and `face1` are valid indices into `mesh.faces`.
/// @pre `faceNormals.size() == mesh.faces.size()`.
// TODO: Document the behavior for degenerate triangles (zero-area faces) and for
// coplanar overlapping triangles, neither of which is explicitly specified here.
template <typename T>
bool intersectFace(
    const MeshT<T>& mesh,
    const std::vector<Vector3<T>>& faceNormals,
    int32_t face0,
    int32_t face1);

/// Tests if the mesh self-intersects anywhere and returns all intersecting face pairs
/// using a brute-force O(n^2) pairwise check over all faces.
///
/// @param mesh Source mesh to test for self-intersection.
/// @return All pairs `(i, j)` of face indices (with `i < j` by convention of the
///   pairwise loop) whose triangles intersect according to `intersectFace`. Pairs of
///   faces that share a vertex are excluded (see `intersectFace`). The returned
///   vector is empty if no intersections are found.
/// @note Intended primarily for correctness reference and small meshes; prefer
///   `intersectMesh` for large meshes.
template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce(const MeshT<T>& mesh);

/// Tests if the mesh self-intersects anywhere and returns all intersecting face pairs
/// using a BVH (bounding volume hierarchy) acceleration structure.
///
/// @param mesh Source mesh to test for self-intersection.
/// @return All unique pairs of face indices whose triangles intersect according to
///   `intersectFace`. Pairs of faces that share a vertex are excluded. The returned
///   vector is empty if no intersections are found. Order of pairs is implementation-
///   defined and may differ from `intersectMeshBruteForce`.
/// @note Functionally equivalent to `intersectMeshBruteForce` but asymptotically
///   faster for large meshes.
template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMesh(const MeshT<T>& mesh);

} // namespace momentum
