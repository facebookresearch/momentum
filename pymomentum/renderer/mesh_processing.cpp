/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/mesh_processing.h"

#include <momentum/common/exception.h>
#include <momentum/rasterizer/geometry.h>

namespace pymomentum {

std::tuple<RowMatrixf, RowMatrixf, RowMatrixi, RowMatrixf, RowMatrixi> subdivideMesh(
    RowMatrixf vertices,
    RowMatrixf normals,
    RowMatrixi triangles,
    std::optional<RowMatrixf> textureCoords,
    std::optional<RowMatrixi> textureTriangles,
    int levels,
    float max_edge_length) {
  MT_THROW_IF(levels <= 0, "Expected levels >= 1, got {}", levels);

  MT_THROW_IF(
      levels > 5,
      "Too many levels ({}), exiting to avoid excessive computation/memory usage",
      levels);

  MT_THROW_IF(vertices.cols() != 3, "Expected n x 3 vertices, got n x {}", vertices.cols());

  MT_THROW_IF(normals.cols() != 3, "Expected n x 3 normals, got n x {}", normals.cols());

  MT_THROW_IF(
      vertices.rows() != normals.rows(),
      "Number of vertices ({}) does not match number of normals ({})",
      vertices.rows(),
      normals.rows());

  MT_THROW_IF(triangles.cols() != 3, "Expected n x 3 triangles, got n x {}", triangles.cols());

  // Initialize optional parameters with default values if not provided
  if (!textureTriangles.has_value()) {
    textureTriangles = triangles;
  }

  MT_THROW_IF(
      textureTriangles->cols() != 3,
      "Expected n x 3 texture_triangles, got n x {}",
      textureTriangles->cols());

  if (!textureCoords.has_value()) {
    textureCoords = RowMatrixf::Zero(vertices.rows(), 2);
  }

  MT_THROW_IF(
      textureCoords->cols() != 2,
      "Expected n x 2 texture_coords, got n x {}",
      textureCoords->cols());

  // Validate triangle indices for the input (output of subdivideMeshNoSmoothing should be valid)
  MT_THROW_IF(
      triangles.minCoeff() < 0 || triangles.maxCoeff() >= vertices.rows(),
      "Invalid triangle index; expected all triangle indices to be within [0, {})",
      vertices.rows());

  MT_THROW_IF(
      textureTriangles->minCoeff() < 0 || textureTriangles->maxCoeff() >= textureCoords->rows(),
      "Invalid texture_triangles index; expected all texture_triangles indices to be within [0, {})",
      textureCoords->rows());

  for (int i = 0; i < levels; ++i) {
    const auto nTrianglesOrig = triangles.rows();

    float maxEdgeLengthCur = 0;
    for (int iTri = 0; iTri < nTrianglesOrig; ++iTri) {
      for (int j = 0; j < 3; ++j) {
        const int vi = triangles(iTri, j);
        const int vj = triangles(iTri, (j + 1) % 3);
        maxEdgeLengthCur = std::max(maxEdgeLengthCur, (vertices.row(vi) - vertices.row(vj)).norm());
      }
    }

    if (max_edge_length != 0 && maxEdgeLengthCur < max_edge_length) {
      break;
    }

    std::tie(vertices, normals, triangles, textureCoords, textureTriangles) =
        momentum::rasterizer::subdivideMeshNoSmoothing(
            vertices, normals, triangles, *textureCoords, *textureTriangles, max_edge_length);
  }

  return {vertices, normals, triangles, *textureCoords, *textureTriangles};
}

} // namespace pymomentum
