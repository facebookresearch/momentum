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
 * Method for determining inside/outside classification in SDF sign determination.
 */
enum class SignMethod {
  /// Ray casting with 6-direction majority vote. Fast but fails on non-watertight meshes.
  RayCasting,

  /// Winding number assuming consistent outward-facing normals (wn > 0.5).
  /// Will reject meshes with accidentally flipped normals rather than silently
  /// accepting them. Use when triangle orientation is known to be correct.
  WindingNumber,

  /// Winding number tolerant of either winding convention (abs(wn) > 0.5).
  /// Handles mesh soups, mixed orientations, and geometry from multiple sources
  /// where consistent winding cannot be guaranteed.
  WindingNumberPermissive,
};

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

  /// Print progress messages to stdout
  bool verbose = false;

  /// Method for inside/outside classification during sign determination.
  SignMethod signMethod = SignMethod::RayCasting;
};

/**
 * Padding configuration for mesh-to-SDF bounds computation.
 * Final per-axis padding = max(extent[i] * fractional, absolute).
 */
template <typename ScalarType>
struct MeshToSdfPadding {
  using Scalar = ScalarType;

  /// Fractional padding as a fraction of bounding box extent per axis.
  Scalar fractional = Scalar{0.1};

  /// Absolute minimum padding in world units (e.g. cm), applied per axis.
  Scalar absolute = Scalar{0};
};

/**
 * Convert a triangle mesh to a signed distance field using modern 3-step approach:
 * 1. Narrow band initialization with exact triangle distances
 * 2. Fast marching propagation using Eikonal equation
 * 3. Sign determination (configurable via `SignMethod`)
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
 * Convenience overload that computes bounds automatically from the mesh
 * using a uniform voxel size. Grid dimensions adapt to mesh shape.
 *
 * @param vertices Vertex positions as span
 * @param triangles Triangle indices as span
 * @param voxelSize Uniform voxel size in world units
 * @param padding Padding configuration for bounds expansion
 * @param config Configuration parameters
 * @return Generated signed distance field with isotropic voxels
 */
template <typename ScalarType>
SignedDistanceField<ScalarType> meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    ScalarType voxelSize,
    const MeshToSdfPadding<ScalarType>& padding = {},
    const MeshToSdfConfig<ScalarType>& config = {});

/**
 * In-place overload that writes into a pre-existing SDF grid.
 * Reuses the existing SDF's bounds and resolution.
 *
 * @param vertices Vertex positions as span
 * @param triangles Triangle indices as span
 * @param sdf Pre-existing signed distance field to write into
 * @param config Configuration parameters
 */
template <typename ScalarType>
void meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignedDistanceField<ScalarType>& sdf,
    const MeshToSdfConfig<ScalarType>& config = {});

/**
 * Apply signs to an existing (unsigned) distance field based on inside/outside classification.
 * This is the sign determination step from the meshToSdf pipeline, exposed for cases where
 * you need to re-sign an SDF with a different method without recomputing distances.
 *
 * @param sdf Distance field to apply signs to (modified in place)
 * @param vertices Mesh vertex positions
 * @param triangles Mesh triangle indices
 * @param signMethod Method for inside/outside classification
 */
template <typename ScalarType>
void applySignsToDistanceField(
    SignedDistanceField<ScalarType>& sdf,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignMethod signMethod = SignMethod::RayCasting);

/**
 * Fill interior voids in an SDF using boundary-seeded flood fill.
 *
 * Identifies connected exterior regions by flood-filling from boundary voxels (faces of the grid)
 * through 6-connected neighbors with value >= 0. Any voxel with value >= 0 not reached by the
 * flood fill is considered an interior void and has its value negated.
 *
 * @param sdf Signed distance field to modify in place
 */
template <typename ScalarType>
void floodFillExterior(SignedDistanceField<ScalarType>& sdf);

/**
 * Morphologically close the interior region of an SDF.
 *
 * Treats voxels with value < 0 as "interior" and performs a binary morphological
 * closing (`iterations` dilations followed by `iterations` erosions) of that
 * region using a 6-connected (face-adjacent) structuring element. Grid faces are
 * treated as exterior for both passes. Any voxel newly classified as interior by
 * the closing has its (positive) value negated, preserving distance magnitude.
 *
 * Unlike floodFillExterior, which only fixes fully enclosed voids, closing also
 * bridges thin interior regions that the sign step misclassified as exterior even
 * when they remain connected to the outside (e.g. an open grip cavity on a
 * controller). Closing is extensive, so it only ever adds interior voxels.
 *
 * @param sdf Signed distance field to modify in place
 * @param iterations Number of dilation/erosion passes (closing radius in voxels)
 */
template <typename ScalarType>
void closeInterior(SignedDistanceField<ScalarType>& sdf, int iterations = 1);

/**
 * Morphologically open the interior region of an SDF.
 *
 * Treats voxels with value < 0 as "interior" and performs a binary morphological
 * opening (`iterations` erosions followed by `iterations` dilations) using a
 * 6-connected structuring element. This removes isolated interior specks and
 * protrusions thinner than `iterations` voxels (e.g. spurious single voxels the
 * winding-number sign step marks as interior in free space). Grid faces are
 * treated as exterior. Any voxel removed from the interior has its (negative)
 * value flipped positive, preserving distance magnitude.
 *
 * Opening is anti-extensive, so it only ever removes interior voxels — apply it
 * after closeInterior / floodFillExterior to despeckle the result.
 *
 * @param sdf Signed distance field to modify in place
 * @param iterations Number of erosion/dilation passes (opening radius in voxels)
 */
template <typename ScalarType>
void openInterior(SignedDistanceField<ScalarType>& sdf, int iterations = 1);

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
    std::span<const Eigen::Vector3i> triangles,
    SignMethod signMethod = SignMethod::RayCasting);

/**
 * Compute mesh bounding box from vertex spans.
 */
template <typename ScalarType>
BoundingBox<ScalarType> computeMeshBounds(std::span<const Eigen::Vector3<ScalarType>> vertices);

} // namespace detail

// Type aliases for convenience
using MeshToSdfConfigf = MeshToSdfConfig<float>;
using MeshToSdfConfigd = MeshToSdfConfig<double>;
using MeshToSdfPaddingf = MeshToSdfPadding<float>;
using MeshToSdfPaddingd = MeshToSdfPadding<double>;

} // namespace axel
