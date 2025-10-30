/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>
#include <queue>
#include <vector>

#include <Eigen/Core>
#include <span>

#include "axel/BoundingBox.h"
#include "axel/SignedDistanceField.h"
#include "axel/TriBvh.h"

namespace axel {

/**
 * Configuration parameters for mesh-to-SDF conversion.
 */
template <typename ScalarType>
struct MeshToSdfConfig {
  using Scalar = ScalarType;

  /// Narrow band width around triangles (in voxel units)
  Scalar narrowBandWidth = Scalar{1.5};

  /// Maximum distance to compute (distances beyond this are clamped)
  /// Set to 0 to disable clamping
  Scalar maxDistance = Scalar{0};

  /// Numerical tolerance for computations
  Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{1000};
};

/**
 * Convert a triangle mesh to a signed distance field using modern 3-step approach:
 * 1. Narrow band initialization with exact triangle distances
 * 2. Fast marching propagation using Eikonal equation
 * 3. Sign determination using ray casting
 *
 * @param vertices Vertex positions as span (works with std::vector, arrays, subranges)
 * @param triangles Triangle indices as span (indices must be valid within vertices)
 * @param bounds Spatial bounds for the SDF
 * @param resolution Grid resolution (nx, ny, nz)
 * @param config Configuration parameters
 * @return Generated signed distance field
 */
template <typename ScalarType>
SignedDistanceField<ScalarType> meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    const BoundingBox<ScalarType>& bounds,
    const Eigen::Vector3<Index>& resolution,
    const MeshToSdfConfig<ScalarType>& config = {});

/**
 * Convenience overload that computes bounds automatically from the mesh.
 *
 * @param vertices Vertex positions as span
 * @param triangles Triangle indices as span
 * @param resolution Grid resolution (nx, ny, nz)
 * @param padding Extra space around mesh bounds (as fraction of bounding box size)
 * @param config Configuration parameters
 * @return Generated signed distance field
 */
template <typename ScalarType>
SignedDistanceField<ScalarType> meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    const Eigen::Vector3<Index>& resolution,
    ScalarType padding = ScalarType{0.1},
    const MeshToSdfConfig<ScalarType>& config = {});

namespace detail {

// ================================================================================================
// FORWARD DECLARATIONS
// ================================================================================================

/**
 * Initialize narrow band with exact triangle distances.
 */
template <typename ScalarType>
void initializeNarrowBand(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignedDistanceField<ScalarType>& sdf,
    ScalarType bandWidth);

/**
 * Propagate distances from narrow band to entire grid using fast marching.
 */
template <typename ScalarType>
void fastMarchingPropagate(SignedDistanceField<ScalarType>& sdf);

/**
 * Apply correct signs to distance field based on inside/outside classification.
 */
template <typename ScalarType>
void applySignsToDistanceField(
    SignedDistanceField<ScalarType>& sdf,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles);

/**
 * Compute mesh bounding box from vertex spans.
 */
template <typename ScalarType>
BoundingBox<ScalarType> computeMeshBounds(std::span<const Eigen::Vector3<ScalarType>> vertices);

} // namespace detail

// Type aliases for convenience
using MeshToSdfConfigf = MeshToSdfConfig<float>;
using MeshToSdfConfigd = MeshToSdfConfig<double>;

} // namespace axel
