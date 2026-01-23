/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

namespace py = pybind11;

/// Computes smooth vertex normals for a triangle mesh given vertex positions.
///
/// For each vertex, the normal is computed as the normalized sum of the face
/// normals of all triangles that share that vertex. Face normals are computed
/// using the cross product of triangle edges.
///
/// Supports arbitrary leading dimensions with broadcasting and both float32/float64 dtypes.
///
/// @param vertexPositions Numpy array containing vertex positions.
///   Shape: [..., numVertices, 3]
/// @param triangles Numpy array containing triangle indices.
///   Shape: [numTriangles, 3]
/// @return Numpy array containing the computed vertex normals (normalized).
///   Shape: [..., numVertices, 3]
py::array computeVertexNormalsArray(const py::buffer& vertexPositions, const py::buffer& triangles);

} // namespace pymomentum
