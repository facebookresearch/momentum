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

// base mesh class
template <typename T>
struct MeshT {
  std::vector<Eigen::Vector3<T>> vertices; // list of mesh vertices
  std::vector<Eigen::Vector3<T>> normals; // list of normals
  std::vector<Eigen::Vector3i> faces; // list of vertex indices per face
  std::vector<std::vector<int32_t>> lines; // list of list of vertex indices per line

  std::vector<Eigen::Vector3b> colors; // list of per-vertex colors
  std::vector<T> confidence; // list of per-vertex confidences

  /// List of texture coordinates.
  ///
  /// The texture coordinates are format-agnostic, and it's the user's responsibility to ensure
  /// their consistent use.
  ///
  /// For GLTF, Momentum stores and saves the texture coordinates as they are represented by the
  /// underlying GLTF parser. When dealing with FBX, the Y-axis is flipped. This distinction is
  /// crucial to understand when working with different formats.
  std::vector<Eigen::Vector2f> texcoords;

  std::vector<Eigen::Vector3i> texcoord_faces; // list of texture coordinate indices per face
  std::vector<std::vector<int32_t>> texcoord_lines; // list of texture coordinate indices per line

  MeshT(
      const std::vector<Eigen::Vector3<T>>& vertices = {},
      const std::vector<Eigen::Vector3<T>>& normals = {},
      const std::vector<Eigen::Vector3i>& faces = {},
      const std::vector<std::vector<int32_t>>& lines = {},
      const std::vector<Eigen::Vector3b>& colors = {},
      const std::vector<T>& confidence = {},
      const std::vector<Eigen::Vector2f>& texcoords = {},
      const std::vector<Eigen::Vector3i>& texcoord_faces = {},
      const std::vector<std::vector<int32_t>>& texcoord_lines = {})
      : vertices(vertices),
        normals(normals),
        faces(faces),
        lines(lines),
        colors(colors),
        confidence(confidence),
        texcoords(texcoords),
        texcoord_faces(texcoord_faces),
        texcoord_lines(texcoord_lines) {
    // Empty
  }

  /// Compute vertex normals (by averaging connected face normals)
  void updateNormals();

  /// Compute vertex normals (by averaging connected face normals) in multi-threaded fashion
  void updateNormalsMt(size_t maxThreads = std::numeric_limits<uint32_t>::max());

  /// Cast data type of mesh vertices, normals and confidence
  template <typename T2>
  MeshT<T2> cast() const;

  /// Reset mesh
  void reset();

 private:
  // Cache for updateNormalsMt()
  std::vector<std::vector<size_t>> facePerVertex_; // for each vertex, list of faces it belongs to
  std::vector<Vector3<T>> faceNormals_;
};

} // namespace momentum
