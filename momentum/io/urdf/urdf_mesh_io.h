/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/filesystem.h>
#include <momentum/math/mesh.h>

#include <optional>

namespace momentum {

/// Generates a box mesh with the given dimensions centered at the origin.
///
/// @param dimX Full extent along the X axis (in the caller's units).
/// @param dimY Full extent along the Y axis.
/// @param dimZ Full extent along the Z axis.
/// @return A mesh with 8 vertices and 12 triangular faces.
Mesh generateBoxMesh(float dimX, float dimY, float dimZ);

/// Generates a cylinder mesh centered at the origin with the Z axis as its axis.
///
/// @param radius Cylinder radius.
/// @param length Cylinder length along the Z axis.
/// @param segments Number of segments around the circumference.
/// @return A triangulated cylinder mesh with top and bottom caps.
Mesh generateCylinderMesh(float radius, float length, int segments = 32);

/// Generates a UV sphere mesh centered at the origin.
///
/// @param radius Sphere radius.
/// @param latSegments Number of latitude bands (stacks).
/// @param lonSegments Number of longitude segments (slices).
/// @return A triangulated sphere mesh.
Mesh generateSphereMesh(float radius, int latSegments = 16, int lonSegments = 32);

/// Loads a mesh from an STL file (binary or ASCII).
///
/// Binary STL vertices are deduplicated to produce an indexed mesh.
///
/// @param filepath Path to the STL file.
/// @return The loaded mesh, or std::nullopt if the file cannot be read.
std::optional<Mesh> loadStlMesh(const filesystem::path& filepath);

/// Loads a mesh from a Wavefront OBJ file (vertices and triangular/polygon faces).
///
/// Only `v` (vertex) and `f` (face) records are parsed. Face vertex indices may use
/// `v`, `v/vt`, `v/vt/vn`, or `v//vn` notation; only vertex indices are used.
///
/// @param filepath Path to the OBJ file.
/// @return The loaded mesh, or std::nullopt if the file cannot be read.
std::optional<Mesh> loadObjMesh(const filesystem::path& filepath);

/// Resolves a mesh filename from a URDF visual element to a filesystem path.
///
/// Handles absolute paths, paths relative to the URDF directory,
/// `package://` URIs (strips scheme and package name), and `file://` URIs.
///
/// @param filename The mesh filename string from the URDF (e.g., "meshes/part.stl" or
///   "package://robot/meshes/part.stl").
/// @param urdfDir The directory containing the URDF file, used as the base for
///   relative path resolution.
/// @return The resolved filesystem path, or std::nullopt if the file cannot be found.
std::optional<filesystem::path> resolveMeshPath(
    const std::string& filename,
    const filesystem::path& urdfDir);

} // namespace momentum
