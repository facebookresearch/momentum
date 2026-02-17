/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/mesh.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace pymomentum {

namespace py = pybind11;

/// Classify mesh triangles into regions based on texture colors.
///
/// This function samples texture colors at multiple points within each triangle
/// and assigns triangles to regions based on the sampled colors. Uses parallel
/// processing for high performance on large meshes.
///
/// The sampling strategy is deterministic and uses barycentric coordinates:
/// - num_samples=1: Centroid only
/// - num_samples=3: 3 vertices (backward compatible with Python implementation)
/// - num_samples=4: Centroid + 3 vertices
/// - num_samples=6: 3 vertices + 3 edge midpoints
/// - num_samples=7: Centroid + 3 vertices + 3 edge midpoints
/// - num_samples=10: 7 + 3 interior points
///
/// @param mesh The mesh containing texture coordinates and texture coordinate faces.
///             Must have non-empty texcoords and texcoord_faces.
/// @param texture RGB texture image as a numpy array with shape [height, width, 3] and dtype uint8.
/// @param regionColors RGB colors for each region as a numpy array with shape [n_regions, 3].
///                     The array index corresponds to the region index.
/// @param threshold Fraction of samples that must match for a triangle to be assigned
///                  to a region. 0.0 means >= 1 sample must match (backward compatible),
///                  0.5 means >= 50% must match, 1.0 means all samples must match.
/// @param numSamples Number of sample points per triangle (1, 3, 4, 6, 7, or 10).
/// @return List of sorted triangle index lists, one per region. The outer list index
///         corresponds to the region index in regionColors.
std::vector<std::vector<int32_t>> classifyTrianglesByTexture(
    const momentum::Mesh& mesh,
    const py::array_t<uint8_t>& texture,
    const py::array_t<uint8_t>& regionColors,
    float threshold = 0.0f,
    int32_t numSamples = 3);

} // namespace pymomentum
