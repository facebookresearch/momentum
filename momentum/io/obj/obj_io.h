/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/filesystem.h>
#include <momentum/math/mesh.h>

#include <span>

namespace momentum {

/// Loads a mesh from a Wavefront OBJ file.
///
/// Reads positions (`v`), texture coordinates (`vt`) and faces (`f`), accepting all four
/// OBJ face-corner forms (`v`, `v/vt`, `v//vn`, `v/vt/vn`) and both positive and negative
/// (relative) indices. Polygons are fan-triangulated into @ref Mesh::faces, and the
/// original topology is preserved in @ref Mesh::polyFaces and @ref Mesh::polyFaceSizes.
///
/// Texture coordinates are indexed per face corner, which is OBJ's own layout, so they map
/// directly onto @ref Mesh::texcoords and @ref Mesh::texcoord_faces. They are kept only if
/// every face carries a complete set of `vt` indices; a file that texture-maps some faces
/// and not others is loaded untextured, because @ref Mesh::texcoord_faces has to align
/// one-to-one with @ref Mesh::faces.
///
/// Vertex normals are recomputed from the geometry rather than read from `vn`: @ref Mesh
/// stores one normal per vertex, while OBJ indexes them per face corner, so honouring `vn`
/// in general would mean splitting vertices. This matches how the FBX and glTF loaders
/// treat normals.
///
/// Vertex colours are read from the six-float `v x y z r g b` form some exporters emit, and
/// default to white otherwise.
///
/// Material, group, object and smoothing directives (`mtllib`, `usemtl`, `g`, `o`, `s`) are
/// ignored, so the whole file loads as one mesh.
///
/// @param[in] filepath Path to the OBJ file.
/// @return The loaded mesh.
/// @throws std::runtime_error if the file cannot be opened, an index is out of range, or the
///     file contains no geometry.
[[nodiscard]] Mesh loadObj(const filesystem::path& filepath);

/// Loads a mesh from an in-memory OBJ file.
///
/// @param[in] bytes Buffer holding the OBJ file contents.
/// @return The loaded mesh.
/// @throws std::runtime_error under the same conditions as the path overload.
[[nodiscard]] Mesh loadObj(std::span<const std::byte> bytes);

} // namespace momentum
