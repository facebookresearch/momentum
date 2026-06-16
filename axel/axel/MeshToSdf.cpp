/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/MeshToSdf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <tuple>
#include <unordered_map>

#ifndef AXEL_NO_DISPENSO
#include <dispenso/parallel_for.h>
#endif

#include "axel/BoundingBox.h"
#include "axel/Checks.h"
#include "axel/Ray.h"
#include "axel/TriBvh.h"
#include "axel/common/Constants.h"
#include "axel/math/PointTriangleProjection.h"

namespace axel {

namespace {

// ================================================================================================
// DATA STRUCTURES
// ================================================================================================

/**
 * Voxel candidate for narrow band initialization.
 */
template <typename ScalarType>
struct VoxelCandidate {
  using Scalar = ScalarType;
  using Vector3 = Eigen::Vector3<Scalar>;
  using Vector3i = Eigen::Vector3<Index>;

  Vector3i index; ///< Voxel grid coordinates
  Scalar distance; ///< Distance to triangle
  Vector3 closestPoint; ///< Closest point on triangle
};

// ================================================================================================
// UTILITY TYPES
// ================================================================================================

/**
 * Voxel states for fast marching algorithm.
 */
enum class VoxelState : uint8_t {
  UNKNOWN = 0, ///< Distance not yet computed
  TRIAL = 1, ///< In priority queue, being updated
  KNOWN = 2 ///< Final distance computed
};

/**
 * Fast marching voxel for priority queue.
 */
template <typename ScalarType>
struct FastMarchingVoxelT {
  using Scalar = ScalarType;
  using Vector3i = Eigen::Vector3<Index>;

  Vector3i index;
  Scalar distance{};

  bool operator>(const FastMarchingVoxelT<ScalarType>& other) const {
    return distance > other.distance;
  }
};

// ================================================================================================
// STEP 3: SIGN DETERMINATION
// ================================================================================================

/**
 * Determine if a point is inside the mesh using ray casting.
 * Casts rays in multiple directions and uses parity rule.
 *
 * @param point Query point
 * @param bvh Pre-built BVH for efficient ray casting
 * @return True if point is inside
 */
template <typename ScalarType>
bool isPointInsideByRayCasting(
    const Eigen::Vector3<ScalarType>& point,
    const TriBvh<ScalarType>& bvh);

} // namespace

// ================================================================================================
// SHARED PIPELINE
// ================================================================================================

namespace detail {

template <typename ScalarType>
void meshToSdfImpl(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignedDistanceField<ScalarType>& sdf,
    const MeshToSdfConfig<ScalarType>& config) {
  // Convert narrow band width from voxel units to world units
  const auto voxelSize = sdf.voxelSize();
  const ScalarType bandWidthWorld = config.narrowBandWidth * voxelSize.maxCoeff();

  // STEP 1: Initialize narrow band with exact triangle distances
  if (config.verbose) {
    std::cout << "Step 1: Narrow band initialization..." << std::endl;
  }
  initializeNarrowBand(vertices, triangles, sdf, bandWidthWorld);
  if (config.verbose) {
    std::cout << "Narrow band initialized" << std::endl;
  }

  // STEP 2: Fast marching propagation to fill entire grid
  if (config.verbose) {
    std::cout << "Step 2: Fast marching propagation..." << std::endl;
  }
  fastMarchingPropagate(sdf);
  if (config.verbose) {
    std::cout << "Fast marching completed" << std::endl;
  }

  // STEP 3: Apply signs based on inside/outside determination
  if (config.verbose) {
    std::cout << "Step 3: Applying signs..." << std::endl;
  }
  detail::applySignsToDistanceField(sdf, vertices, triangles, config.signMethod);
  if (config.verbose) {
    std::cout << "Signs applied" << std::endl;
  }

  // Apply distance clamping if configured
  if (config.maxDistance > ScalarType{0}) {
    auto& data = sdf.data();
    for (auto& value : data) {
      value = std::clamp(value, -config.maxDistance, config.maxDistance);
    }
  }
}

} // namespace detail

// ================================================================================================
// MAIN API FUNCTIONS
// ================================================================================================

template <typename ScalarType>
SignedDistanceField<ScalarType> meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    const BoundingBox<ScalarType>& bounds,
    const Eigen::Vector3<Index>& resolution,
    const MeshToSdfConfig<ScalarType>& config) {
  // Create empty SDF
  SignedDistanceField<ScalarType> sdf(
      bounds, resolution, SignedDistanceField<ScalarType>::kVeryFarDistance);

  detail::meshToSdfImpl(vertices, triangles, sdf, config);

  return sdf;
}

template <typename ScalarType>
SignedDistanceField<ScalarType> meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    ScalarType voxelSize,
    const MeshToSdfPadding<ScalarType>& padding,
    const MeshToSdfConfig<ScalarType>& config) {
  // Step 1: Compute mesh bounding box
  const auto meshBounds = detail::computeMeshBounds(vertices);
  const auto extent = meshBounds.max() - meshBounds.min();

  // Step 2: Compute per-axis padding = max(extent[i] * fractional, absolute)
  Eigen::Vector3<ScalarType> paddingVec;
  for (int i = 0; i < 3; ++i) {
    paddingVec[i] = std::max(extent[i] * padding.fractional, padding.absolute);
  }

  // Step 3: Expand bounds by padding on each side
  const Eigen::Vector3<ScalarType> paddedMin = meshBounds.min() - paddingVec;
  const Eigen::Vector3<ScalarType> paddedMax = meshBounds.max() + paddingVec;
  const Eigen::Vector3<ScalarType> paddedExtent = paddedMax - paddedMin;

  // Step 4: Compute resolution per axis = max(1, ceil(paddedExtent[i] / voxelSize))
  Eigen::Vector3<Index> resolution;
  for (int i = 0; i < 3; ++i) {
    resolution[i] = std::max(Index{1}, static_cast<Index>(std::ceil(paddedExtent[i] / voxelSize)));
  }

  // Step 5: Adjust bounds so that extent / resolution == voxelSize exactly (center-preserving)
  const Eigen::Vector3<ScalarType> center = (paddedMin + paddedMax) * ScalarType{0.5};
  Eigen::Vector3<ScalarType> adjustedHalfExtent;
  for (int i = 0; i < 3; ++i) {
    adjustedHalfExtent[i] = (voxelSize * static_cast<ScalarType>(resolution[i])) * ScalarType{0.5};
  }

  const BoundingBox<ScalarType> adjustedBounds(
      center - adjustedHalfExtent, center + adjustedHalfExtent);

  // Step 6: Call overload 1 with computed uniform-voxel bounds and resolution
  return meshToSdf(vertices, triangles, adjustedBounds, resolution, config);
}

template <typename ScalarType>
void meshToSdf(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignedDistanceField<ScalarType>& sdf,
    const MeshToSdfConfig<ScalarType>& config) {
  // Reset SDF to initial far distance
  sdf.fill(SignedDistanceField<ScalarType>::kVeryFarDistance);

  detail::meshToSdfImpl(vertices, triangles, sdf, config);
}

namespace detail {

// ================================================================================================
// STEP 1: NARROW BAND INITIALIZATION
// ================================================================================================

template <typename ScalarType>
void rasterizeTriangleToNarrowBand(
    const Eigen::Vector3<ScalarType>& v0,
    const Eigen::Vector3<ScalarType>& v1,
    const Eigen::Vector3<ScalarType>& v2,
    SignedDistanceField<ScalarType>& sdf,
    ScalarType bandWidth) {
  using Vector3 = Eigen::Vector3<ScalarType>;
  using Vector3i = Eigen::Vector3<Index>;

  // Compute triangle bounding box in world space
  const Vector3 minBounds = v0.cwiseMin(v1).cwiseMin(v2);
  const Vector3 maxBounds = v0.cwiseMax(v1).cwiseMax(v2);

  // Expand by band width
  const Vector3 expandedMin = minBounds - Vector3::Constant(bandWidth);
  const Vector3 expandedMax = maxBounds + Vector3::Constant(bandWidth);

  // Convert to grid coordinates
  const Vector3 gridMin = sdf.worldToGrid(expandedMin);
  const Vector3 gridMax = sdf.worldToGrid(expandedMax);

  // Clamp to valid grid bounds
  const Vector3i startIdx = Vector3i(
      std::max(Index{0}, static_cast<Index>(std::floor(gridMin.x()))),
      std::max(Index{0}, static_cast<Index>(std::floor(gridMin.y()))),
      std::max(Index{0}, static_cast<Index>(std::floor(gridMin.z()))));

  const Vector3i endIdx = Vector3i(
      std::min(sdf.resolution().x() - 1, static_cast<Index>(std::ceil(gridMax.x()))),
      std::min(sdf.resolution().y() - 1, static_cast<Index>(std::ceil(gridMax.y()))),
      std::min(sdf.resolution().z() - 1, static_cast<Index>(std::ceil(gridMax.z()))));

  // Test each voxel in the bounding box
  for (Index i = startIdx.x(); i <= endIdx.x(); ++i) {
    for (Index j = startIdx.y(); j <= endIdx.y(); ++j) {
      for (Index k = startIdx.z(); k <= endIdx.z(); ++k) {
        const Vector3 worldPos = sdf.gridLocation(i, j, k);

        // Calculate distance from voxel center to triangle
        Vector3 closestPoint;
        axel::projectOnTriangle(worldPos, v0, v1, v2, closestPoint);
        const ScalarType distance = (worldPos - closestPoint).norm();

        // Only consider voxels within the grid cell diagonal distance to reduce grid alignment
        // dependency
        if (distance <= bandWidth) {
          sdf.set(i, j, k, std::min(distance, sdf.at(i, j, k)));
        }
      }
    }
  }
}

template <typename ScalarType>
void initializeNarrowBand(
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignedDistanceField<ScalarType>& sdf,
    ScalarType bandWidth) {
  // Initialize SDF with maximum values
  sdf.fill(std::numeric_limits<ScalarType>::max());

  // Process each triangle
  for (size_t triIdx = 0; triIdx < triangles.size(); ++triIdx) {
    const auto& triangle = triangles[triIdx];

    // Validate triangle indices
    if (triangle.x() >= static_cast<int>(vertices.size()) ||
        triangle.y() >= static_cast<int>(vertices.size()) ||
        triangle.z() >= static_cast<int>(vertices.size())) {
      std::cerr << "Warning: Invalid triangle indices at triangle " << triIdx << std::endl;
      continue;
    }

    const auto& v0 = vertices[triangle.x()];
    const auto& v1 = vertices[triangle.y()];
    const auto& v2 = vertices[triangle.z()];

    rasterizeTriangleToNarrowBand(v0, v1, v2, sdf, bandWidth);
  }
}

namespace {

// ================================================================================================
// STEP 2: FAST MARCHING PROPAGATION
// ================================================================================================

/**
 * Solve Eikonal equation at a voxel using upwind finite differences.
 * |∇u| = 1, where u is the distance function.
 *
 * @param sdf Distance field
 * @param voxelIndex Voxel to solve at
 * @param states Voxel states array
 * @return Computed distance value
 */
template <typename ScalarType>
ScalarType solveEikonal(
    const SignedDistanceField<ScalarType>& sdf,
    const Eigen::Vector3<Index>& voxelIndex,
    const std::vector<VoxelState>& states) {
  const auto& resolution = sdf.resolution();
  const auto voxelSize = sdf.voxelSize();

  // Find minimum known distances in each direction (upwind scheme)
  ScalarType minDistX = std::numeric_limits<ScalarType>::max();
  ScalarType minDistY = std::numeric_limits<ScalarType>::max();
  ScalarType minDistZ = std::numeric_limits<ScalarType>::max();

  // Check X direction neighbors
  if (voxelIndex.x() > 0) {
    const size_t idx = (voxelIndex.x() - 1) + voxelIndex.y() * resolution.x() +
        voxelIndex.z() * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistX = std::min(minDistX, sdf.at(voxelIndex.x() - 1, voxelIndex.y(), voxelIndex.z()));
    }
  }
  if (voxelIndex.x() < resolution.x() - 1) {
    const size_t idx = (voxelIndex.x() + 1) + voxelIndex.y() * resolution.x() +
        voxelIndex.z() * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistX = std::min(minDistX, sdf.at(voxelIndex.x() + 1, voxelIndex.y(), voxelIndex.z()));
    }
  }

  // Check Y direction neighbors
  if (voxelIndex.y() > 0) {
    const size_t idx = voxelIndex.x() + (voxelIndex.y() - 1) * resolution.x() +
        voxelIndex.z() * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistY = std::min(minDistY, sdf.at(voxelIndex.x(), voxelIndex.y() - 1, voxelIndex.z()));
    }
  }
  if (voxelIndex.y() < resolution.y() - 1) {
    const size_t idx = voxelIndex.x() + (voxelIndex.y() + 1) * resolution.x() +
        voxelIndex.z() * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistY = std::min(minDistY, sdf.at(voxelIndex.x(), voxelIndex.y() + 1, voxelIndex.z()));
    }
  }

  // Check Z direction neighbors
  if (voxelIndex.z() > 0) {
    const size_t idx = voxelIndex.x() + voxelIndex.y() * resolution.x() +
        (voxelIndex.z() - 1) * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistZ = std::min(minDistZ, sdf.at(voxelIndex.x(), voxelIndex.y(), voxelIndex.z() - 1));
    }
  }
  if (voxelIndex.z() < resolution.z() - 1) {
    const size_t idx = voxelIndex.x() + voxelIndex.y() * resolution.x() +
        (voxelIndex.z() + 1) * resolution.x() * resolution.y();
    if (states[idx] == VoxelState::KNOWN) {
      minDistZ = std::min(minDistZ, sdf.at(voxelIndex.x(), voxelIndex.y(), voxelIndex.z() + 1));
    }
  }

  // Solve Eikonal equation |∇T| = 1/F where F = 1 (speed)
  // Use Godunov's upwind finite difference scheme

  // Collect valid upwind differences with their corresponding grid spacing
  std::vector<std::pair<ScalarType, ScalarType>> validDists; // (distance, grid_spacing)

  if (minDistX < std::numeric_limits<ScalarType>::max()) {
    validDists.emplace_back(minDistX, voxelSize.x());
  }
  if (minDistY < std::numeric_limits<ScalarType>::max()) {
    validDists.emplace_back(minDistY, voxelSize.y());
  }
  if (minDistZ < std::numeric_limits<ScalarType>::max()) {
    validDists.emplace_back(minDistZ, voxelSize.z());
  }

  if (validDists.empty()) {
    return std::numeric_limits<ScalarType>::max();
  }

  // Sort by distance (smallest first)
  std::sort(validDists.begin(), validDists.end());

  // Try solving with increasing number of dimensions
  ScalarType result = std::numeric_limits<ScalarType>::max();

  // 1D update: T = T_min + h
  ScalarType T1 = validDists[0].first + validDists[0].second;
  result = T1;

  // 2D update: solve quadratic equation
  if (validDists.size() >= 2) {
    const ScalarType T_a = validDists[0].first;
    const ScalarType T_b = validDists[1].first;
    const ScalarType h_a = validDists[0].second;
    const ScalarType h_b = validDists[1].second;

    // Solve: (T-T_a)²/h_a² + (T-T_b)²/h_b² = 1
    const ScalarType a = ScalarType{1} / (h_a * h_a) + ScalarType{1} / (h_b * h_b);
    const ScalarType b = -ScalarType{2} * (T_a / (h_a * h_a) + T_b / (h_b * h_b));
    const ScalarType c = (T_a * T_a) / (h_a * h_a) + (T_b * T_b) / (h_b * h_b) - ScalarType{1};

    const ScalarType discriminant = b * b - ScalarType{4} * a * c;
    if (discriminant >= ScalarType{0}) {
      const ScalarType T2 = (-b + std::sqrt(discriminant)) / (ScalarType{2} * a);
      // Only use if it satisfies upwind condition and improves 1D solution
      if (T2 >= std::max(T_a, T_b) && T2 < result) {
        result = T2;
      }
    }
  }

  // 3D update: solve cubic equation (simplified)
  if (validDists.size() >= 3) {
    const ScalarType T_a = validDists[0].first;
    const ScalarType T_b = validDists[1].first;
    const ScalarType T_c = validDists[2].first;
    const ScalarType h_a = validDists[0].second;
    const ScalarType h_b = validDists[1].second;
    const ScalarType h_c = validDists[2].second;

    // Solve: (T-T_a)²/h_a² + (T-T_b)²/h_b² + (T-T_c)²/h_c² = 1
    const ScalarType a =
        ScalarType{1} / (h_a * h_a) + ScalarType{1} / (h_b * h_b) + ScalarType{1} / (h_c * h_c);
    const ScalarType b =
        -ScalarType{2} * (T_a / (h_a * h_a) + T_b / (h_b * h_b) + T_c / (h_c * h_c));
    const ScalarType c = (T_a * T_a) / (h_a * h_a) + (T_b * T_b) / (h_b * h_b) +
        (T_c * T_c) / (h_c * h_c) - ScalarType{1};

    const ScalarType discriminant = b * b - ScalarType{4} * a * c;
    if (discriminant >= ScalarType{0}) {
      const ScalarType T3 = (-b + std::sqrt(discriminant)) / (ScalarType{2} * a);
      // Only use if it satisfies upwind condition and improves 2D solution
      if (T3 >= std::max({T_a, T_b, T_c}) && T3 < result) {
        result = T3;
      }
    }
  }

  return result;
}

} // namespace

template <typename ScalarType>
void fastMarchingPropagate(SignedDistanceField<ScalarType>& sdf) {
  using Vector3i = Eigen::Vector3<Index>;
  using FastMarchingVoxel = FastMarchingVoxelT<ScalarType>;

  const auto& resolution = sdf.resolution();
  const size_t totalVoxels = sdf.totalVoxels();

  // Initialize voxel states
  std::vector<VoxelState> states(totalVoxels, VoxelState::UNKNOWN);

  // Priority queue for fast marching
  std::priority_queue<FastMarchingVoxel, std::vector<FastMarchingVoxel>, std::greater<>> queue;

  // Reconstruct known voxels by looking for values smaller than FLT_MAX
  std::vector<Vector3i> knownVoxels;
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const ScalarType value = sdf.at(i, j, k);
        if (value < SignedDistanceField<ScalarType>::kVeryFarDistance) {
          const Vector3i voxelIdx(i, j, k);
          knownVoxels.push_back(voxelIdx);

          const size_t linearIdx = i + j * resolution.x() + k * resolution.x() * resolution.y();
          states[linearIdx] = VoxelState::KNOWN;
        }
      }
    }
  }

  // Initialize neighbors of known voxels
  for (const auto& knownIdx : knownVoxels) {
    // Add 6-connected neighbors to trial set
    const std::array<Vector3i, 6> neighbors = {
        {Vector3i(knownIdx.x() - 1, knownIdx.y(), knownIdx.z()),
         Vector3i(knownIdx.x() + 1, knownIdx.y(), knownIdx.z()),
         Vector3i(knownIdx.x(), knownIdx.y() - 1, knownIdx.z()),
         Vector3i(knownIdx.x(), knownIdx.y() + 1, knownIdx.z()),
         Vector3i(knownIdx.x(), knownIdx.y(), knownIdx.z() - 1),
         Vector3i(knownIdx.x(), knownIdx.y(), knownIdx.z() + 1)}};

    for (const auto& neighbor : neighbors) {
      if (sdf.isValidIndex(neighbor.x(), neighbor.y(), neighbor.z())) {
        const size_t neighborIdx = neighbor.x() + neighbor.y() * resolution.x() +
            neighbor.z() * resolution.x() * resolution.y();

        if (states[neighborIdx] == VoxelState::UNKNOWN) {
          states[neighborIdx] = VoxelState::TRIAL;

          // Compute initial distance estimate
          const auto distance = solveEikonal<ScalarType>(sdf, neighbor, states);
          sdf.set(neighbor.x(), neighbor.y(), neighbor.z(), distance);

          // Add to priority queue
          FastMarchingVoxel voxel;
          voxel.index = neighbor;
          voxel.distance = sdf.at(neighbor.x(), neighbor.y(), neighbor.z());
          queue.push(voxel);
        }
      }
    }
  }

  // Fast marching main loop
  while (!queue.empty()) {
    const auto current = queue.top();
    queue.pop();

    const auto& index = current.index;
    const size_t linearIdx =
        index.x() + index.y() * resolution.x() + index.z() * resolution.x() * resolution.y();

    // Skip if already processed (may happen due to multiple updates)
    if (states[linearIdx] == VoxelState::KNOWN) {
      continue;
    }

    // Mark as known
    states[linearIdx] = VoxelState::KNOWN;

    // Update 6-connected neighbors
    const std::array<Vector3i, 6> neighbors = {
        {Vector3i(index.x() - 1, index.y(), index.z()),
         Vector3i(index.x() + 1, index.y(), index.z()),
         Vector3i(index.x(), index.y() - 1, index.z()),
         Vector3i(index.x(), index.y() + 1, index.z()),
         Vector3i(index.x(), index.y(), index.z() - 1),
         Vector3i(index.x(), index.y(), index.z() + 1)}};

    for (const auto& neighbor : neighbors) {
      if (sdf.isValidIndex(neighbor.x(), neighbor.y(), neighbor.z())) {
        const size_t neighborIdx = neighbor.x() + neighbor.y() * resolution.x() +
            neighbor.z() * resolution.x() * resolution.y();

        if (states[neighborIdx] == VoxelState::UNKNOWN) {
          states[neighborIdx] = VoxelState::TRIAL;
        }

        if (states[neighborIdx] == VoxelState::TRIAL) {
          // Update distance estimate
          const auto newDistance = solveEikonal<ScalarType>(sdf, neighbor, states);
          const ScalarType currentDistance = sdf.at(neighbor.x(), neighbor.y(), neighbor.z());

          // Only update if new distance is better
          if (newDistance < currentDistance) {
            sdf.set(neighbor.x(), neighbor.y(), neighbor.z(), newDistance);

            // Add to priority queue
            FastMarchingVoxel voxel;
            voxel.index = neighbor;
            voxel.distance = newDistance;
            queue.push(voxel);
          }
        }
      }
    }
  }

  // Fill any remaining unknown voxels using simple distance propagation
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const size_t linearIdx = i + j * resolution.x() + k * resolution.x() * resolution.y();
        if (states[linearIdx] == VoxelState::UNKNOWN ||
            sdf.at(i, j, k) >= std::numeric_limits<ScalarType>::max()) {
          // Find the closest known voxel and use distance to it
          ScalarType minDistance = std::numeric_limits<ScalarType>::max();

          for (const auto& knownIdx : knownVoxels) {
            const ScalarType knownDist = sdf.at(knownIdx.x(), knownIdx.y(), knownIdx.z());
            const Vector3i deltaIdx = Vector3i(i, j, k) - knownIdx;
            const auto deltaWorld = deltaIdx.cast<ScalarType>().cwiseProduct(sdf.voxelSize());
            const ScalarType spatialDistance = deltaWorld.norm();

            minDistance = std::min(minDistance, knownDist + spatialDistance);
          }

          sdf.set(i, j, k, minDistance);
        }
      }
    }
  }
}

// ================================================================================================
// STEP 3: SIGN DETERMINATION
// ================================================================================================

template <typename ScalarType>
bool isPointInsideByRayCasting(
    const Eigen::Vector3<ScalarType>& point,
    const TriBvh<ScalarType>& bvh) {
  // Use fixed ray directions to avoid randomness in parallel execution
  const std::array<Eigen::Vector3<ScalarType>, 6> directions = {
      {Eigen::Vector3<ScalarType>(1, 0, 0),
       Eigen::Vector3<ScalarType>(-1, 0, 0),
       Eigen::Vector3<ScalarType>(0, 1, 0),
       Eigen::Vector3<ScalarType>(0, -1, 0),
       Eigen::Vector3<ScalarType>(0, 0, 1),
       Eigen::Vector3<ScalarType>(0, 0, -1)}};

  int insideCount = 0;

  for (int i = 0; i < directions.size(); ++i) {
    const Ray3<ScalarType> ray(point, directions[i]);
    auto hits = bvh.allHits(ray);
    const ScalarType minDelta = 0.01;
    auto lastHit = minDelta;
    int nHits = 0;

    // Sort hits by distance since allHits returns them in BVH traversal order
    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
      return a.hitDistance < b.hitDistance;
    });

    // A relatively common case is to hit right on the
    // edge between two triangles, in which case we end
    // up with both triangles having the same hit distance.
    // To avoid this, we add a small delta to the hit distance
    // to ensure that we don't count the same hit twice
    for (const auto& hit : hits) {
      if (hit.hitDistance > lastHit) {
        lastHit = hit.hitDistance + minDelta;
        ++nHits;
      }
    }

    // Odd number of intersections means inside
    if ((nHits % 2) == 1) {
      insideCount++;
    }
  }

  // Use majority vote
  return insideCount > (directions.size() / 2);
}

/**
 * Compute the solid angle subtended by a triangle as seen from a point.
 * Uses the Van Oosterom-Strackee formula.
 */
template <typename ScalarType>
ScalarType triangleSolidAngle(
    const Eigen::Vector3<ScalarType>& point,
    const Eigen::Vector3<ScalarType>& v0,
    const Eigen::Vector3<ScalarType>& v1,
    const Eigen::Vector3<ScalarType>& v2) {
  const Eigen::Vector3<ScalarType> a = v0 - point;
  const Eigen::Vector3<ScalarType> b = v1 - point;
  const Eigen::Vector3<ScalarType> c = v2 - point;

  const ScalarType la = a.norm();
  const ScalarType lb = b.norm();
  const ScalarType lc = c.norm();

  // Degenerate case: point is on a vertex
  constexpr ScalarType kEps = std::numeric_limits<ScalarType>::epsilon() * ScalarType{100};
  if (la < kEps || lb < kEps || lc < kEps) {
    return ScalarType{0};
  }

  const ScalarType numerator = a.dot(b.cross(c));
  const ScalarType denominator = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;

  return ScalarType{2} * std::atan2(numerator, denominator);
}

/**
 * Compute the generalized winding number at a point.
 * Sums the solid angle of each triangle as seen from the query point,
 * normalized by 4pi. Returns +1 for inside (outward normals), -1 for
 * inside (inward normals), ~0 for outside.
 *
 * O(T) per query — brute-force over all triangles.
 */
template <typename ScalarType>
ScalarType computeWindingNumber(
    const Eigen::Vector3<ScalarType>& point,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles) {
  ScalarType totalSolidAngle{0};
  const ScalarType kFourPi = ScalarType{4} * std::numbers::pi_v<ScalarType>;

  for (const auto& tri : triangles) {
    totalSolidAngle +=
        triangleSolidAngle(point, vertices[tri.x()], vertices[tri.y()], vertices[tri.z()]);
  }

  return totalSolidAngle / kFourPi;
}

template <typename ScalarType>
void applySignsToDistanceField(
    SignedDistanceField<ScalarType>& sdf,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignMethod signMethod) {
  const auto& resolution = sdf.resolution();

  // Build BVH (only needed for RayCasting)
  std::unique_ptr<TriBvh<ScalarType>> bvh;
  if (signMethod == SignMethod::RayCasting) {
    Eigen::MatrixX3<ScalarType> vertexMatrix(vertices.size(), 3);
    for (size_t i = 0; i < vertices.size(); ++i) {
      vertexMatrix.row(i) = vertices[i];
    }

    Eigen::MatrixX3i triangleMatrix(triangles.size(), 3);
    for (size_t i = 0; i < triangles.size(); ++i) {
      triangleMatrix.row(i) = triangles[i];
    }

    bvh = std::make_unique<TriBvh<ScalarType>>(std::move(vertexMatrix), std::move(triangleMatrix));
  }

  // Process each voxel
  const auto processVoxel = [&](Index i, Index j, Index k) {
    const Eigen::Vector3<ScalarType> gridPos(
        static_cast<ScalarType>(i), static_cast<ScalarType>(j), static_cast<ScalarType>(k));
    const auto worldPos = sdf.gridToWorld(gridPos);

    bool isInside = false;
    switch (signMethod) {
      case SignMethod::RayCasting:
        isInside = isPointInsideByRayCasting(worldPos, *bvh);
        break;
      case SignMethod::WindingNumber:
        isInside = computeWindingNumber(worldPos, vertices, triangles) > ScalarType{0.5};
        break;
      case SignMethod::WindingNumberPermissive:
        isInside = std::abs(computeWindingNumber(worldPos, vertices, triangles)) > ScalarType{0.5};
        break;
    }

    // Apply sign: negative inside, positive outside
    if (isInside) {
      const ScalarType currentDistance = sdf.at(i, j, k);
      sdf.set(i, j, k, -std::abs(currentDistance));
    }
    // Outside voxels keep positive distances (already positive from previous steps)
  };

#ifdef AXEL_NO_DISPENSO
  // Serial version
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        processVoxel(i, j, k);
      }
    }
  }
#else
  // Parallel version
  dispenso::parallel_for(0, resolution.z(), [&](Index k) {
    for (Index i = 0; i < resolution.x(); ++i) {
      for (Index j = 0; j < resolution.y(); ++j) {
        processVoxel(i, j, k);
      }
    }
  });
#endif
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

template <typename ScalarType>
BoundingBox<ScalarType> computeMeshBounds(std::span<const Eigen::Vector3<ScalarType>> vertices) {
  if (vertices.empty()) {
    // Return unit cube centered at origin as fallback
    return BoundingBox<ScalarType>(
        Eigen::Vector3<ScalarType>(-1, -1, -1), Eigen::Vector3<ScalarType>(1, 1, 1));
  }

  Eigen::Vector3<ScalarType> minBounds = vertices[0];
  Eigen::Vector3<ScalarType> maxBounds = vertices[0];

  for (const auto& vertex : vertices) {
    minBounds = minBounds.cwiseMin(vertex);
    maxBounds = maxBounds.cwiseMax(vertex);
  }

  return BoundingBox<ScalarType>(minBounds, maxBounds);
}

} // namespace detail

// ================================================================================================
// PUBLIC API: applySignsToDistanceField
// ================================================================================================

template <typename ScalarType>
void applySignsToDistanceField(
    SignedDistanceField<ScalarType>& sdf,
    std::span<const Eigen::Vector3<ScalarType>> vertices,
    std::span<const Eigen::Vector3i> triangles,
    SignMethod signMethod) {
  detail::applySignsToDistanceField(sdf, vertices, triangles, signMethod);
}

namespace {

// Offsets of the 6 face-adjacent neighbors in a voxel grid, shared by the
// flood-fill and morphological passes below.
constexpr std::array<std::array<Index, 3>, 6> kFaceNeighborOffsets = {
    {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};

} // namespace

// ================================================================================================
// PUBLIC API: floodFillExterior
// ================================================================================================

template <typename ScalarType>
void floodFillExterior(SignedDistanceField<ScalarType>& sdf) {
  const auto& resolution = sdf.resolution();
  const Index nx = resolution.x();
  const Index ny = resolution.y();
  const Index nz = resolution.z();
  const size_t totalVoxels = sdf.totalVoxels();

  auto& data = sdf.data();
  std::vector<bool> visited(totalVoxels, false);
  std::queue<std::tuple<Index, Index, Index>> queue;

  auto linearIdx = [&](Index i, Index j, Index k) -> size_t {
    return static_cast<size_t>(k) * nx * ny + static_cast<size_t>(j) * nx + static_cast<size_t>(i);
  };

  // Seed from all 6 boundary faces
  auto trySeed = [&](Index i, Index j, Index k) {
    const size_t idx = linearIdx(i, j, k);
    if (!visited[idx] && data[idx] >= ScalarType{0}) {
      visited[idx] = true;
      queue.push({i, j, k});
    }
  };

  // Z faces (k=0 and k=nz-1)
  for (Index i = 0; i < nx; ++i) {
    for (Index j = 0; j < ny; ++j) {
      trySeed(i, j, 0);
      if (nz > 1) {
        trySeed(i, j, nz - 1);
      }
    }
  }
  // Y faces (j=0 and j=ny-1)
  for (Index i = 0; i < nx; ++i) {
    for (Index k = 0; k < nz; ++k) {
      trySeed(i, 0, k);
      if (ny > 1) {
        trySeed(i, ny - 1, k);
      }
    }
  }
  // X faces (i=0 and i=nx-1)
  for (Index j = 0; j < ny; ++j) {
    for (Index k = 0; k < nz; ++k) {
      trySeed(0, j, k);
      if (nx > 1) {
        trySeed(nx - 1, j, k);
      }
    }
  }

  // BFS through 6-connected neighbors with value >= 0
  while (!queue.empty()) {
    const auto [ci, cj, ck] = queue.front();
    queue.pop();

    for (const auto& offset : kFaceNeighborOffsets) {
      const Index ni = ci + offset[0];
      const Index nj = cj + offset[1];
      const Index nk = ck + offset[2];

      if (sdf.isValidIndex(ni, nj, nk)) {
        const size_t nIdx = linearIdx(ni, nj, nk);
        if (!visited[nIdx] && data[nIdx] >= ScalarType{0}) {
          visited[nIdx] = true;
          queue.push({ni, nj, nk});
        }
      }
    }
  }

  // Negate unreached positive voxels (interior voids)
  for (size_t idx = 0; idx < totalVoxels; ++idx) {
    if (!visited[idx] && data[idx] >= ScalarType{0}) {
      data[idx] = -data[idx];
    }
  }
}

// ================================================================================================
// PUBLIC API: closeInterior
// ================================================================================================

namespace {

/**
 * One pass of 6-connected binary dilation/erosion over a flat (k-major) mask.
 *
 * For dilation a voxel is set when itself or any face neighbor is set; for erosion
 * a voxel stays set only when itself and all face neighbors are set. Out-of-grid
 * neighbors are treated as unset (grid faces act as exterior), so erosion may
 * nibble the boundary — harmless here since the object interior is padded away
 * from the grid faces.
 */
void morphologicalPass(
    std::span<const char> in,
    std::span<char> out,
    Index nx,
    Index ny,
    Index nz,
    bool dilate) {
  const size_t totalVoxels =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  XR_CHECK(
      in.size() == totalVoxels && out.size() == totalVoxels,
      "morphologicalPass requires in/out spans sized to the voxel grid");
  auto idxOf = [&](Index i, Index j, Index k) -> size_t {
    return static_cast<size_t>(k) * nx * ny + static_cast<size_t>(j) * nx + static_cast<size_t>(i);
  };

  for (Index k = 0; k < nz; ++k) {
    for (Index j = 0; j < ny; ++j) {
      for (Index i = 0; i < nx; ++i) {
        const size_t idx = idxOf(i, j, k);
        // Dilation keeps any set voxel set; erosion keeps any unset voxel unset.
        if (in[idx] == (dilate ? 1 : 0)) {
          out[idx] = in[idx];
          continue;
        }
        bool result = !dilate; // erosion assumes set until a neighbor disproves
        for (const auto& off : kFaceNeighborOffsets) {
          const Index ni = i + off[0];
          const Index nj = j + off[1];
          const Index nk = k + off[2];
          const bool inBounds = ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz;
          const bool neighborSet = inBounds && in[idxOf(ni, nj, nk)] != 0;
          if (dilate && neighborSet) {
            result = true;
            break;
          }
          if (!dilate && !neighborSet) {
            result = false;
            break;
          }
        }
        out[idx] = result ? 1 : 0;
      }
    }
  }
}

/**
 * Shared body for closeInterior / openInterior.
 *
 * Builds an interior mask from the SDF sign, runs `iterations` passes of one
 * morphological op followed by `iterations` passes of its inverse, then flips
 * the sign of every voxel whose new interior classification disagrees with its
 * current sign. `firstDilate == true` performs a closing (dilate then erode);
 * `false` performs an opening (erode then dilate).
 */
template <typename ScalarType>
void morphInterior(SignedDistanceField<ScalarType>& sdf, int iterations, bool firstDilate) {
  if (iterations <= 0) {
    return;
  }

  const auto& resolution = sdf.resolution();
  const Index nx = resolution.x();
  const Index ny = resolution.y();
  const Index nz = resolution.z();
  const size_t totalVoxels = sdf.totalVoxels();

  auto& data = sdf.data();
  std::vector<char> mask(totalVoxels);
  for (size_t idx = 0; idx < totalVoxels; ++idx) {
    mask[idx] = data[idx] < ScalarType{0} ? 1 : 0;
  }

  std::vector<char> scratch(totalVoxels);
  for (int pass = 0; pass < iterations; ++pass) {
    morphologicalPass(mask, scratch, nx, ny, nz, /*dilate=*/firstDilate);
    mask.swap(scratch);
  }
  for (int pass = 0; pass < iterations; ++pass) {
    morphologicalPass(mask, scratch, nx, ny, nz, /*dilate=*/!firstDilate);
    mask.swap(scratch);
  }

  // Flip the sign wherever the new interior mask disagrees with the current
  // sign. Closing is extensive (only sets mask where data >= 0) and opening is
  // anti-extensive (only clears mask where data < 0), so both reduce to this
  // single disagreement predicate.
  for (size_t idx = 0; idx < totalVoxels; ++idx) {
    if ((mask[idx] != 0) != (data[idx] < ScalarType{0})) {
      data[idx] = -data[idx];
    }
  }
}

} // namespace

template <typename ScalarType>
void closeInterior(SignedDistanceField<ScalarType>& sdf, int iterations) {
  morphInterior(sdf, iterations, /*firstDilate=*/true);
}

template <typename ScalarType>
void openInterior(SignedDistanceField<ScalarType>& sdf, int iterations) {
  morphInterior(sdf, iterations, /*firstDilate=*/false);
}

// ================================================================================================
// EXPLICIT INSTANTIATIONS
// ================================================================================================

// Explicit instantiations for float and double
template SignedDistanceField<float> meshToSdf<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    const BoundingBox<float>&,
    const Eigen::Vector3<Index>&,
    const MeshToSdfConfig<float>&);

template SignedDistanceField<double> meshToSdf<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    const BoundingBox<double>&,
    const Eigen::Vector3<Index>&,
    const MeshToSdfConfig<double>&);

template SignedDistanceField<float> meshToSdf<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    float,
    const MeshToSdfPadding<float>&,
    const MeshToSdfConfig<float>&);

template SignedDistanceField<double> meshToSdf<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    double,
    const MeshToSdfPadding<double>&,
    const MeshToSdfConfig<double>&);

template void meshToSdf<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<float>&,
    const MeshToSdfConfig<float>&);

template void meshToSdf<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<double>&,
    const MeshToSdfConfig<double>&);

// Detail namespace explicit instantiations
namespace detail {

template void meshToSdfImpl<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<float>&,
    const MeshToSdfConfig<float>&);

template void meshToSdfImpl<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<double>&,
    const MeshToSdfConfig<double>&);

template void initializeNarrowBand<float>(
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<float>&,
    float);

template void initializeNarrowBand<double>(
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    SignedDistanceField<double>&,
    double);

template void fastMarchingPropagate<float>(SignedDistanceField<float>&);

template void fastMarchingPropagate<double>(SignedDistanceField<double>&);

template void applySignsToDistanceField<float>(
    SignedDistanceField<float>&,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    SignMethod);

template void applySignsToDistanceField<double>(
    SignedDistanceField<double>&,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    SignMethod);

template BoundingBox<float> computeMeshBounds<float>(std::span<const Eigen::Vector3<float>>);

template BoundingBox<double> computeMeshBounds<double>(std::span<const Eigen::Vector3<double>>);

} // namespace detail

// Public API explicit instantiations
template void applySignsToDistanceField<float>(
    SignedDistanceField<float>&,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Vector3i>,
    SignMethod);

template void applySignsToDistanceField<double>(
    SignedDistanceField<double>&,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Vector3i>,
    SignMethod);

template void floodFillExterior<float>(SignedDistanceField<float>&);

template void floodFillExterior<double>(SignedDistanceField<double>&);

template void closeInterior<float>(SignedDistanceField<float>&, int);

template void closeInterior<double>(SignedDistanceField<double>&, int);

template void openInterior<float>(SignedDistanceField<float>&, int);

template void openInterior<double>(SignedDistanceField<double>&, int);

} // namespace axel
