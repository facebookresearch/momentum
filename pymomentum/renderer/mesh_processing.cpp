/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/mesh_processing.h"

#include <momentum/rasterizer/geometry.h>

#include <stdexcept>

namespace pymomentum {

std::tuple<RowMatrixf, RowMatrixf, RowMatrixi, RowMatrixf, RowMatrixi> subdivideMesh(
    RowMatrixf vertices,
    RowMatrixf normals,
    RowMatrixi triangles,
    std::optional<RowMatrixf> textureCoords,
    std::optional<RowMatrixi> textureTriangles,
    int levels,
    float max_edge_length) {
  if (levels <= 0) {
    throw std::runtime_error("Expected levels >= 1");
  }

  if (levels > 5) {
    throw std::runtime_error(
        "Too many levels, exiting to avoid excessive computation/memory usage");
  }

  for (int i = 0; i < levels; ++i) {
    if (vertices.cols() != 3) {
      throw std::runtime_error("Expected n x 3 vertices");
    }

    if (normals.cols() != 3) {
      throw std::runtime_error("Expected n x 3 normals");
    }

    if (vertices.rows() != normals.rows()) {
      throw std::runtime_error("number of vertices does not match number of normals");
    }

    if (triangles.cols() != 3) {
      throw std::runtime_error("Expected n x 3 triangles");
    }

    if (!textureTriangles.has_value()) {
      textureTriangles = triangles;
    }

    if (textureTriangles->cols() != 3) {
      throw std::runtime_error("Expected n x 3 texture_triangles");
    }

    if (!textureCoords.has_value()) {
      textureCoords = RowMatrixf::Zero(vertices.rows(), 2);
    }

    if (textureCoords->cols() != 2) {
      throw std::runtime_error("Expected n x 2 texture_coords");
    }

    if (triangles.minCoeff() < 0 || triangles.maxCoeff() >= vertices.rows()) {
      throw std::runtime_error(
          "Invalid triangle index; expected all triangle indices to be within [0, nVerts)");
    }

    if (textureTriangles->minCoeff() < 0 || textureTriangles->maxCoeff() >= textureCoords->rows()) {
      throw std::runtime_error(
          "Invalid texture_triangles index; expected all texture_triangles indices to be within [0, nTextureCoords)");
    }

    const auto nVertOrig = vertices.rows();
    const auto nTrianglesOrig = triangles.rows();

    if (triangles.maxCoeff() >= nVertOrig || triangles.minCoeff() < 0) {
      throw std::runtime_error("Invalid triangle");
    }

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
