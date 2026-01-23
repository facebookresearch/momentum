/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_vertex_normals.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <momentum/common/exception.h>

#include <dispenso/parallel_for.h>

namespace pymomentum {

namespace {

// Validate that all triangle indices are within bounds [0, nVertices).
// This check is performed once before the parallel loop to provide
// a clear error message rather than undefined behavior.
void validateTriangleIndices(
    const IntVectorArrayAccessor<3>::ElementView& triView,
    py::ssize_t nVertices) {
  const auto [minIdx, maxIdx] = triView.minmax();
  MT_THROW_IF(
      minIdx < 0,
      "compute_vertex_normals: triangle has invalid negative vertex index {}. "
      "Expected index in range [0, {}).",
      minIdx,
      nVertices);
  MT_THROW_IF(
      maxIdx >= nVertices,
      "compute_vertex_normals: triangle has invalid vertex index {} >= nVertices. "
      "Expected index in range [0, {}).",
      maxIdx,
      nVertices);
}

template <typename T>
py::array_t<T> computeVertexNormalsImpl(
    const py::array& vertexPositions,
    const py::array& triangles,
    const LeadingDimensions& leadingDims) {
  const auto nVertices =
      static_cast<py::ssize_t>(vertexPositions.shape(vertexPositions.ndim() - 2));
  const auto nTriangles = static_cast<py::ssize_t>(triangles.shape(0));
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with same shape as input [..., nVertices, 3]
  auto result = createOutputArray<T>(leadingDims, {nVertices, static_cast<py::ssize_t>(3)});

  // Create accessors using the VectorArrayAccessor API
  VectorArrayAccessor<T, 3> positionsAcc(vertexPositions, leadingDims, nVertices);
  VectorArrayAccessor<T, 3> normalsAcc(result, leadingDims, nVertices);

  // Create triangle accessor with empty leading dimensions (triangles are not batched)
  // IntVectorArrayAccessor handles int32/int64 conversion automatically
  LeadingDimensions emptyLeadingDims;
  IntVectorArrayAccessor<3> trianglesAcc(triangles, emptyLeadingDims, nTriangles);
  auto triView = trianglesAcc.view({});

  // Validate triangle indices before processing
  validateTriangleIndices(triView, nVertices);

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get views for this batch element using the accessor API
      auto posView = positionsAcc.view(indices);
      auto normView = normalsAcc.view(indices);

      // Initialize normals to zero
      normView.setZero();

      // Accumulate face normals to vertices
      for (py::ssize_t iTri = 0; iTri < nTriangles; ++iTri) {
        // Get triangle vertex indices using the accessor
        Eigen::Vector3i tri = triView.get(iTri);
        const auto i0 = tri[0];
        const auto i1 = tri[1];
        const auto i2 = tri[2];

        // Get vertex positions using the ElementView API
        Eigen::Vector3<T> p0 = posView.get(i0);
        Eigen::Vector3<T> p1 = posView.get(i1);
        Eigen::Vector3<T> p2 = posView.get(i2);

        // Compute face normal (cross product of edges)
        // Note: We don't normalize here - larger triangles contribute more
        Eigen::Vector3<T> faceNormal = (p1 - p0).cross(p2 - p0);

        // Accumulate to each vertex using the add() method
        normView.add(i0, faceNormal);
        normView.add(i1, faceNormal);
        normView.add(i2, faceNormal);
      }

      // Normalize all vertex normals
      normView.normalize();
    });
  }

  return result;
}

} // namespace

py::array computeVertexNormalsArray(
    const py::buffer& vertexPositions,
    const py::buffer& triangles) {
  ArrayChecker checker("compute_vertex_normals");

  // Validate vertex positions: [..., nVertices, 3]
  checker.validateBuffer(vertexPositions, "vertex_positions", {-1, 3}, {"numVertices", "xyz"});

  // Validate triangles: [nTriangles, 3] - must be integer type
  // Note: IntVectorArrayAccessor will validate the dtype and convert to int on-the-fly
  auto triInfo = triangles.request();
  MT_THROW_IF(
      triInfo.ndim != 2 || triInfo.shape[1] != 3,
      "compute_vertex_normals: expected triangles to have shape [nTriangles, 3], got {}",
      formatArrayDims(std::vector<py::ssize_t>(triInfo.shape.begin(), triInfo.shape.end())));

  if (checker.isFloat64()) {
    return computeVertexNormalsImpl<double>(
        vertexPositions, triangles, checker.getLeadingDimensions());
  } else {
    return computeVertexNormalsImpl<float>(
        vertexPositions, triangles, checker.getLeadingDimensions());
  }
}

} // namespace pymomentum
