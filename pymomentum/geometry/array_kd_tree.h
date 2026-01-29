/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tuple>

namespace pymomentum {

namespace py = pybind11;

/// Find closest points without normals (2D or 3D points).
///
/// For each point in points_source, finds the closest point in points_target.
/// Supports both 2D and 3D point sets.
///
/// @param points_source Source points with shape [..., nSrcPoints, dim] where dim is 2 or 3.
/// @param points_target Target points with shape [..., nTgtPoints, dim] where dim is 2 or 3.
/// @param maxDist Maximum distance to search. Points farther than this are marked invalid.
/// @return Tuple of (closest_points, indices, valid) where:
///         - closest_points: shape [..., nSrcPoints, dim]
///         - indices: shape [..., nSrcPoints] with int32 indices (-1 for invalid)
///         - valid: shape [..., nSrcPoints] with bool validity flags
std::tuple<py::array, py::array, py::array> findClosestPointsArray(
    py::buffer points_source,
    py::buffer points_target,
    float maxDist = std::numeric_limits<float>::max());

/// Find closest points with normal compatibility checking (3D only).
///
/// For each point in points_source, finds the closest point in points_target
/// whose normal is compatible (n_source · n_target > max_normal_dot).
///
/// @param points_source Source points with shape [..., nSrcPoints, 3].
/// @param normals_source Source normals with shape [..., nSrcPoints, 3] (must be normalized).
/// @param points_target Target points with shape [..., nTgtPoints, 3].
/// @param normals_target Target normals with shape [..., nTgtPoints, 3] (must be normalized).
/// @param maxDist Maximum distance to search.
/// @param maxNormalDot Minimum dot product for normal compatibility (typically 0.0).
/// @return Tuple of (closest_points, closest_normals, indices, valid) where:
///         - closest_points: shape [..., nSrcPoints, 3]
///         - closest_normals: shape [..., nSrcPoints, 3]
///         - indices: shape [..., nSrcPoints] with int32 indices (-1 for invalid)
///         - valid: shape [..., nSrcPoints] with bool validity flags
std::tuple<py::array, py::array, py::array, py::array> findClosestPointsWithNormalsArray(
    py::buffer points_source,
    py::buffer normals_source,
    py::buffer points_target,
    py::buffer normals_target,
    float maxDist = std::numeric_limits<float>::max(),
    float maxNormalDot = 0.0f);

/// Find closest points on a triangle mesh.
///
/// For each point in points_source, finds the closest point on the target mesh surface.
///
/// @param points_source Source points with shape [..., nSrcPoints, 3].
/// @param vertices_target Mesh vertices with shape [..., nVertices, 3].
/// @param faces_target Mesh faces with shape [..., nFaces, 3] (triangle vertex indices).
/// @return Tuple of (valid, closest_points, face_indices, barycentric) where:
///         - valid: shape [..., nSrcPoints] with bool validity flags
///         - closest_points: shape [..., nSrcPoints, 3]
///         - face_indices: shape [..., nSrcPoints] with int32 face indices (-1 for invalid)
///         - barycentric: shape [..., nSrcPoints, 3] with barycentric coordinates
std::tuple<py::array, py::array, py::array, py::array> findClosestPointsOnMeshArray(
    py::buffer points_source,
    py::buffer vertices_target,
    py::buffer faces_target);

} // namespace pymomentum
