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

/// A generic mesh representation with support for vertices, faces, lines, and texture coordinates.
///
/// @tparam T The scalar type used for vertex coordinates and confidence values (typically float or
/// double)
template <typename T>
struct MeshT {
  /// List of mesh vertices in 3D space
  std::vector<Eigen::Vector3<T>> vertices;

  /// List of vertex normals (same size as vertices when computed)
  std::vector<Eigen::Vector3<T>> normals;

  /// List of triangular faces defined by vertex indices
  std::vector<Eigen::Vector3i> faces;

  /// List of polylines defined by vertex indices
  /// Each inner vector represents a single polyline
  std::vector<std::vector<int32_t>> lines;

  /// List of per-vertex RGB colors (0-255 for each channel)
  std::vector<Eigen::Vector3b> colors;

  /// List of per-vertex confidence values
  /// Higher values typically indicate higher confidence in the vertex position
  std::vector<T> confidence;

  /// List of texture coordinates (UV coordinates).
  ///
  /// The texture coordinates are format-agnostic, and it's the user's responsibility to ensure
  /// their consistent use.
  ///
  /// For GLTF, Momentum stores and saves the texture coordinates as they are represented by the
  /// underlying GLTF parser. When dealing with FBX, the Y-axis is flipped. This distinction is
  /// crucial to understand when working with different formats.
  std::vector<Eigen::Vector2f> texcoords;

  /// List of texture coordinate indices per face
  /// Maps each face to its corresponding texture coordinates
  std::vector<Eigen::Vector3i> texcoord_faces;

  /// List of texture coordinate indices per line
  /// Maps each line to its corresponding texture coordinates
  std::vector<std::vector<int32_t>> texcoord_lines;

  /// Compute vertex normals by averaging connected face normals.
  ///
  /// This method calculates normals for each vertex by averaging the normals of all
  /// connected faces. The resulting normals are normalized to unit length.
  /// If a vertex is part of a degenerate face (e.g., colinear vertices), that face
  /// will not contribute to the vertex normal.
  void updateNormals();

  /// Cast the mesh to a different scalar type.
  ///
  /// This method creates a new mesh with all numeric data converted to the target type.
  /// Vertex positions, normals, and confidence values are cast to the new type.
  /// Non-numeric data like faces and lines remain unchanged.
  ///
  /// @tparam T2 The target scalar type
  /// @return A new mesh with the target scalar type
  template <typename T2>
  [[nodiscard]] MeshT<T2> cast() const;

  /// Reset the mesh by clearing all data.
  ///
  /// This method clears all vectors, effectively resetting the mesh to an empty state.
  void reset();
};

} // namespace momentum
