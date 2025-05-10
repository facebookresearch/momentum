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

/// test whether two given faces of a mesh intersect with each other
template <typename T>
bool intersectFace(
    const MeshT<T>& mesh,
    const std::vector<Vector3<T>>& faceNormals,
    int32_t face0,
    int32_t face1);

/// test if the mesh self intersects anywhere and return all intersecting face pairs using brute
/// force
template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMeshBruteForce(const MeshT<T>& mesh);
/// test if the mesh self intersects anywhere and return all intersecting face pairs using a bvh
/// tree
template <typename T>
std::vector<std::pair<int32_t, int32_t>> intersectMesh(const MeshT<T>& mesh);

} // namespace momentum
