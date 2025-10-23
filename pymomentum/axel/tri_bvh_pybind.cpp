/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tri_bvh_pybind.h"

#include <axel/BoundingBox.h>
#include <axel/BvhCommon.h>
#include <axel/Ray.h>
#include <axel/TriBvh.h>
#include <momentum/common/exception.h>
#include <pymomentum/axel/axel_utility.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;

namespace pymomentum {

namespace {

/**
 * Validate that triangle indices reference valid vertex indices.
 * Checks that all triangle indices are in the range [0, num_vertices).
 */
void validateTriangleIndices(
    const py::array_t<int>& triangles,
    py::ssize_t num_vertices,
    const char* parameterName) {
  if (triangles.ndim() != 2 || triangles.shape(1) != 3) {
    throw std::runtime_error(fmt::format(
        "Invalid shape for {}: expected (N, 3), got {}", parameterName, getArrayDimStr(triangles)));
  }

  validateIndexArray(triangles, parameterName, num_vertices);
}

/**
 * Create an Eigen::MatrixX3<S> from a numpy array with validation.
 */
template <typename S>
Eigen::Matrix<S, Eigen::Dynamic, 3, Eigen::RowMajor> createPositionsMatrix(
    const py::array_t<S>& vertices,
    const char* parameterName) {
  validatePositionArray(vertices, parameterName);

  const auto data = vertices.template unchecked<2>();
  const py::ssize_t numRows = data.shape(0);

  Eigen::Matrix<S, Eigen::Dynamic, 3, Eigen::RowMajor> matrix(numRows, 3);
  for (py::ssize_t i = 0; i < numRows; ++i) {
    matrix(i, 0) = data(i, 0);
    matrix(i, 1) = data(i, 1);
    matrix(i, 2) = data(i, 2);
  }

  return matrix;
}

/**
 * Create an Eigen::MatrixX3i from a numpy array with validation.
 */
Eigen::MatrixX3i createTrianglesMatrix(
    const py::array_t<int>& triangles,
    py::ssize_t num_vertices,
    const char* parameterName) {
  validateTriangleIndices(triangles, num_vertices, parameterName);

  const auto data = triangles.unchecked<2>();
  const py::ssize_t numRows = data.shape(0);

  Eigen::MatrixX3i matrix(numRows, 3);
  for (py::ssize_t i = 0; i < numRows; ++i) {
    matrix(i, 0) = data(i, 0);
    matrix(i, 1) = data(i, 1);
    matrix(i, 2) = data(i, 2);
  }

  return matrix;
}

/**
 * Helper to perform batched ray queries for closest_hit
 * Returns tuple of (triangleIds, hitDistances, hitPoints, baryCoords)
 */
template <typename BvhType>
auto performBatchedClosestHitQuery(
    const BvhType& bvh,
    const py::array_t<float>& origins,
    const py::array_t<float>& directions,
    const py::array_t<float>& maxDistances) {
  // Validate input arrays
  validatePositionArray(origins, "origins");
  validatePositionArray(directions, "directions");

  const py::ssize_t numRays = origins.shape(0);
  if (directions.shape(0) != numRays) {
    throw std::runtime_error(fmt::format(
        "origins and directions must have same number of rays: {} vs {}",
        numRays,
        directions.shape(0)));
  }

  // Validate maxDistances if provided
  bool hasMaxDistances = maxDistances.size() > 0;
  if (hasMaxDistances) {
    if (maxDistances.ndim() != 1) {
      throw std::runtime_error(
          fmt::format("maxDistances must be 1-dimensional, got {}D", maxDistances.ndim()));
    }
    if (maxDistances.shape(0) != numRays) {
      throw std::runtime_error(fmt::format(
          "maxDistances must have same length as number of rays: {} vs {}",
          numRays,
          maxDistances.shape(0)));
    }
  }

  const auto originsData = origins.unchecked<2>();
  const auto directionsData = directions.unchecked<2>();

  // Create output arrays
  auto triangleIds = py::array_t<int32_t>(numRays);
  auto hitDistances = py::array_t<float>(numRays);
  std::vector<py::ssize_t> shape2d = {numRays, 3};
  auto hitPoints = py::array_t<float>(shape2d);
  auto baryCoords = py::array_t<float>(shape2d);

  auto triangleIdsData = triangleIds.mutable_unchecked<1>();
  auto hitDistancesData = hitDistances.mutable_unchecked<1>();
  auto hitPointsData = hitPoints.mutable_unchecked<2>();
  auto baryCoordsData = baryCoords.mutable_unchecked<2>();

  if (hasMaxDistances) {
    const auto maxDistancesData = maxDistances.unchecked<1>();
    for (py::ssize_t i = 0; i < numRays; ++i) {
      Eigen::Vector3f origin(originsData(i, 0), originsData(i, 1), originsData(i, 2));
      Eigen::Vector3f direction(directionsData(i, 0), directionsData(i, 1), directionsData(i, 2));

      axel::Ray3f ray(origin, direction, maxDistancesData(i));
      auto result = bvh.closestHit(ray);

      if (result.has_value()) {
        triangleIdsData(i) = result->triangleId;
        hitDistancesData(i) = result->hitDistance;
        hitPointsData(i, 0) = result->hitPoint.x();
        hitPointsData(i, 1) = result->hitPoint.y();
        hitPointsData(i, 2) = result->hitPoint.z();
        baryCoordsData(i, 0) = result->baryCoords.x();
        baryCoordsData(i, 1) = result->baryCoords.y();
        baryCoordsData(i, 2) = result->baryCoords.z();
      } else {
        triangleIdsData(i) = -1;
        hitDistancesData(i) = std::numeric_limits<float>::max();
        hitPointsData(i, 0) = 0.0f;
        hitPointsData(i, 1) = 0.0f;
        hitPointsData(i, 2) = 0.0f;
        baryCoordsData(i, 0) = 0.0f;
        baryCoordsData(i, 1) = 0.0f;
        baryCoordsData(i, 2) = 0.0f;
      }
    }
  } else {
    for (py::ssize_t i = 0; i < numRays; ++i) {
      Eigen::Vector3f origin(originsData(i, 0), originsData(i, 1), originsData(i, 2));
      Eigen::Vector3f direction(directionsData(i, 0), directionsData(i, 1), directionsData(i, 2));

      axel::Ray3f ray(origin, direction);
      auto result = bvh.closestHit(ray);

      if (result.has_value()) {
        triangleIdsData(i) = result->triangleId;
        hitDistancesData(i) = result->hitDistance;
        hitPointsData(i, 0) = result->hitPoint.x();
        hitPointsData(i, 1) = result->hitPoint.y();
        hitPointsData(i, 2) = result->hitPoint.z();
        baryCoordsData(i, 0) = result->baryCoords.x();
        baryCoordsData(i, 1) = result->baryCoords.y();
        baryCoordsData(i, 2) = result->baryCoords.z();
      } else {
        triangleIdsData(i) = -1;
        hitDistancesData(i) = std::numeric_limits<float>::max();
        hitPointsData(i, 0) = 0.0f;
        hitPointsData(i, 1) = 0.0f;
        hitPointsData(i, 2) = 0.0f;
        baryCoordsData(i, 0) = 0.0f;
        baryCoordsData(i, 1) = 0.0f;
        baryCoordsData(i, 2) = 0.0f;
      }
    }
  }

  return py::make_tuple(triangleIds, hitDistances, hitPoints, baryCoords);
}

/**
 * Helper to perform batched ray queries for any_hit
 * Returns array of bool indicating if each ray hit anything
 */
template <typename BvhType>
auto performBatchedAnyHitQuery(
    const BvhType& bvh,
    const py::array_t<float>& origins,
    const py::array_t<float>& directions,
    const py::array_t<float>& maxDistances) {
  // Validate input arrays
  validatePositionArray(origins, "origins");
  validatePositionArray(directions, "directions");

  const py::ssize_t numRays = origins.shape(0);
  if (directions.shape(0) != numRays) {
    throw std::runtime_error(fmt::format(
        "origins and directions must have same number of rays: {} vs {}",
        numRays,
        directions.shape(0)));
  }

  // Validate maxDistances if provided
  bool hasMaxDistances = maxDistances.size() > 0;
  if (hasMaxDistances) {
    if (maxDistances.ndim() != 1) {
      throw std::runtime_error(
          fmt::format("maxDistances must be 1-dimensional, got {}D", maxDistances.ndim()));
    }
    if (maxDistances.shape(0) != numRays) {
      throw std::runtime_error(fmt::format(
          "maxDistances must have same length as number of rays: {} vs {}",
          numRays,
          maxDistances.shape(0)));
    }
  }

  const auto originsData = origins.unchecked<2>();
  const auto directionsData = directions.unchecked<2>();

  auto hits = py::array_t<bool>(numRays);
  auto hitsData = hits.mutable_unchecked<1>();

  if (hasMaxDistances) {
    const auto maxDistancesData = maxDistances.unchecked<1>();
    for (py::ssize_t i = 0; i < numRays; ++i) {
      Eigen::Vector3f origin(originsData(i, 0), originsData(i, 1), originsData(i, 2));
      Eigen::Vector3f direction(directionsData(i, 0), directionsData(i, 1), directionsData(i, 2));

      axel::Ray3f ray(origin, direction, maxDistancesData(i));
      hitsData(i) = bvh.anyHit(ray);
    }
  } else {
    for (py::ssize_t i = 0; i < numRays; ++i) {
      Eigen::Vector3f origin(originsData(i, 0), originsData(i, 1), originsData(i, 2));
      Eigen::Vector3f direction(directionsData(i, 0), directionsData(i, 1), directionsData(i, 2));

      axel::Ray3f ray(origin, direction);
      hitsData(i) = bvh.anyHit(ray);
    }
  }

  return hits;
}

/**
 * Helper to perform single ray all_hits query
 * Returns tuple of (triangleIds, hitDistances, hitPoints, baryCoords)
 */
template <typename BvhType>
auto performAllHitsQuery(
    const BvhType& bvh,
    const py::array_t<float>& origin,
    const py::array_t<float>& direction,
    std::optional<float> max_distance) {
  // Validate single ray inputs
  if (origin.ndim() != 1 || origin.shape(0) != 3) {
    throw std::runtime_error(
        fmt::format("origin must be a 1D array of shape (3,), got {}", getArrayDimStr(origin)));
  }
  if (direction.ndim() != 1 || direction.shape(0) != 3) {
    throw std::runtime_error(fmt::format(
        "direction must be a 1D array of shape (3,), got {}", getArrayDimStr(direction)));
  }

  auto originData = origin.unchecked<1>();
  auto directionData = direction.unchecked<1>();

  Eigen::Vector3f originVec(originData(0), originData(1), originData(2));
  Eigen::Vector3f directionVec(directionData(0), directionData(1), directionData(2));

  axel::Ray3f ray(
      originVec, directionVec, max_distance.value_or(std::numeric_limits<float>::max()));
  auto results = bvh.allHits(ray);

  const size_t numHits = results.size();

  // Create output arrays
  auto triangleIds = py::array_t<int32_t>(numHits);
  auto hitDistances = py::array_t<float>(numHits);
  std::vector<py::ssize_t> shape2d = {static_cast<py::ssize_t>(numHits), 3};
  auto hitPoints = py::array_t<float>(shape2d);
  auto baryCoords = py::array_t<float>(shape2d);

  auto triangleIdsData = triangleIds.mutable_unchecked<1>();
  auto hitDistancesData = hitDistances.mutable_unchecked<1>();
  auto hitPointsData = hitPoints.mutable_unchecked<2>();
  auto baryCoordsData = baryCoords.mutable_unchecked<2>();

  for (size_t i = 0; i < numHits; ++i) {
    triangleIdsData(i) = results[i].triangleId;
    hitDistancesData(i) = results[i].hitDistance;
    hitPointsData(i, 0) = results[i].hitPoint.x();
    hitPointsData(i, 1) = results[i].hitPoint.y();
    hitPointsData(i, 2) = results[i].hitPoint.z();
    baryCoordsData(i, 0) = results[i].baryCoords.x();
    baryCoordsData(i, 1) = results[i].baryCoords.y();
    baryCoordsData(i, 2) = results[i].baryCoords.z();
  }

  return py::make_tuple(triangleIds, hitDistances, hitPoints, baryCoords);
}

/**
 * Helper to perform line hits query
 * Returns array of triangle indices hit by the infinite line
 */
template <typename BvhType>
auto performLineHitsQuery(
    const BvhType& bvh,
    const py::array_t<float>& origin,
    const py::array_t<float>& direction) {
  // Validate single ray inputs
  if (origin.ndim() != 1 || origin.shape(0) != 3) {
    throw std::runtime_error(
        fmt::format("origin must be a 1D array of shape (3,), got {}", getArrayDimStr(origin)));
  }
  if (direction.ndim() != 1 || direction.shape(0) != 3) {
    throw std::runtime_error(fmt::format(
        "direction must be a 1D array of shape (3,), got {}", getArrayDimStr(direction)));
  }

  auto originData = origin.unchecked<1>();
  auto directionData = direction.unchecked<1>();

  Eigen::Vector3f originVec(originData(0), originData(1), originData(2));
  Eigen::Vector3f directionVec(directionData(0), directionData(1), directionData(2));

  axel::Ray3f ray(originVec, directionVec);
  auto results = bvh.lineHits(ray);

  const size_t numHits = results.size();

  // Create output array
  auto triangleIds = py::array_t<int32_t>(numHits);
  auto triangleIdsData = triangleIds.mutable_unchecked<1>();

  for (size_t i = 0; i < numHits; ++i) {
    triangleIdsData(i) = results[i];
  }

  return triangleIds;
}

/**
 * Helper to perform batched closest surface point queries
 * Returns tuple of (valid, points, triangle_indices, bary_coords)
 */
template <typename BvhType>
auto performBatchedClosestSurfacePointQuery(const BvhType& bvh, const py::array_t<float>& queries) {
  validatePositionArray(queries, "queries");

  const py::ssize_t numQueries = queries.shape(0);
  auto queriesData = queries.unchecked<2>();

  // Create output arrays
  std::vector<py::ssize_t> shape2d = {numQueries, 3};
  auto valid = py::array_t<bool>(numQueries);
  auto points = py::array_t<float>(shape2d);
  auto triangle_indices = py::array_t<uint32_t>(numQueries);
  auto bary_coords = py::array_t<float>(shape2d);

  auto validData = valid.template mutable_unchecked<1>();
  auto pointsData = points.template mutable_unchecked<2>();
  auto triangleIndicesData = triangle_indices.template mutable_unchecked<1>();
  auto baryCoordsData = bary_coords.template mutable_unchecked<2>();

  for (py::ssize_t i = 0; i < numQueries; ++i) {
    Eigen::Vector3f pos(queriesData(i, 0), queriesData(i, 1), queriesData(i, 2));
    auto result = bvh.closestSurfacePoint(pos);

    // Check if result is valid
    validData(i) = (result.triangleIdx != axel::kInvalidTriangleIdx);
    pointsData(i, 0) = result.point.x();
    pointsData(i, 1) = result.point.y();
    pointsData(i, 2) = result.point.z();
    triangleIndicesData(i) = result.triangleIdx;

    if (result.baryCoords.has_value()) {
      baryCoordsData(i, 0) = result.baryCoords->x();
      baryCoordsData(i, 1) = result.baryCoords->y();
      baryCoordsData(i, 2) = result.baryCoords->z();
    } else {
      baryCoordsData(i, 0) = 0.0f;
      baryCoordsData(i, 1) = 0.0f;
      baryCoordsData(i, 2) = 0.0f;
    }
  }

  return py::make_tuple(valid, points, triangle_indices, bary_coords);
}

} // namespace

void registerTriBvhBindings(py::module& m) {
  // Bind TriBvh
  py::class_<axel::TriBvh<float>>(m, "TriBvh")
      .def(
          py::init([](const py::array_t<float>& vertices,
                      const py::array_t<int>& triangles,
                      std::optional<float> bounding_box_thickness) {
            // Create matrices from numpy arrays with validation
            auto positions = createPositionsMatrix<float>(vertices, "vertices");
            auto triangleIndices = createTrianglesMatrix(triangles, vertices.shape(0), "triangles");

            // Move the matrices into the TriBvh constructor
            return axel::TriBvh<float>(
                positions, std::move(triangleIndices), bounding_box_thickness);
          }),
          R"(Create a BVH (Bounding Volume Hierarchy) for triangle mesh ray and proximity queries.

The TriBvh accelerates spatial queries on triangle meshes including:
- Ray casting (closest hit, all hits, any hit)
- Closest point queries
- Bounding box overlap tests

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles.
                  Indices must be valid within the vertices array.
:param bounding_box_thickness: Optional thickness to add to triangle bounding boxes for more robust queries.
                               Useful when triangles are very thin. Default: 0.0

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Create a simple tetrahedron mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ], dtype=np.int32)

    # Build BVH
    bvh = axel.TriBvh(vertices, triangles)

    # Query closest point
    query_point = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    result = bvh.closest_surface_point(query_point)
    print(f"Closest point: {result.point}, on triangle: {result.triangle_idx}"))",
          py::arg("vertices"),
          py::arg("triangles"),
          py::arg("bounding_box_thickness") = std::nullopt)
      .def(
          "box_query",
          [](const axel::TriBvh<float>& self, const axel::BoundingBox<float>& box) {
            return self.boxQuery(box);
          },
          R"(Find all triangles whose bounding boxes intersect with the query box.

:param box: Query bounding box.
:return: List of triangle indices that intersect the box.)",
          py::arg("box"))
      .def(
          "line_hits",
          [](const axel::TriBvh<float>& self,
             const py::array_t<float>& origin,
             const py::array_t<float>& direction) {
            return performLineHitsQuery(self, origin, direction);
          },
          R"(Find all triangles hit by the infinite line defined by the ray direction.

Note: This queries the infinite line, not just the ray segment.

:param origin: Line origin as 1D array of shape (3,).
:param direction: Line direction as 1D array of shape (3,).
:return: int32 array of triangle indices hit by the line.)",
          py::arg("origin"),
          py::arg("direction"))
      .def(
          "closest_hit",
          [](const axel::TriBvh<float>& self,
             const py::array_t<float>& origins,
             const py::array_t<float>& directions,
             const py::array_t<float>& max_distances) {
            return performBatchedClosestHitQuery(self, origins, directions, max_distances);
          },
          R"(Find closest intersections for a batch of rays.

Supports batched queries for better performance when testing multiple rays.
Returns numpy arrays for efficient downstream processing.

:param origins: Ray origins as 2D array of shape (N, 3).
:param directions: Ray directions as 2D array of shape (N, 3).
:param max_distances: Optional maximum distances per ray as 1D array of shape (N,).
                      If not provided, uses default maximum distance.
:return: Tuple of (triangle_ids, hit_distances, hit_points, bary_coords) where:
         - triangle_ids: int32 array of shape (N,) with hit triangle indices (-1 if no hit)
         - hit_distances: float array of shape (N,) with hit distances (inf if no hit)
         - hit_points: float array of shape (N, 3) with hit point positions
         - bary_coords: float array of shape (N, 3) with barycentric coordinates)",
          py::arg("origins"),
          py::arg("directions"),
          py::arg("max_distances") = py::array_t<float>())
      .def(
          "all_hits",
          [](const axel::TriBvh<float>& self,
             const py::array_t<float>& origin,
             const py::array_t<float>& direction,
             std::optional<float> max_distance) {
            return performAllHitsQuery(self, origin, direction, max_distance);
          },
          R"(Find all intersections for a single ray.

:param origin: Ray origin as 1D array of shape (3,).
:param direction: Ray direction as 1D array of shape (3,).
:param max_distance: Optional maximum distance for the ray. If not provided, uses infinity.
:return: Tuple of (triangle_ids, hit_distances, hit_points, bary_coords) where:
         - triangle_ids: int32 array of shape (M,) with hit triangle indices
         - hit_distances: float array of shape (M,) with hit distances
         - hit_points: float array of shape (M, 3) with hit point positions
         - bary_coords: float array of shape (M, 3) with barycentric coordinates
         where M is the number of hits for this ray)",
          py::arg("origin"),
          py::arg("direction"),
          py::arg("max_distance") = std::nullopt)
      .def(
          "any_hit",
          [](const axel::TriBvh<float>& self,
             const py::array_t<float>& origins,
             const py::array_t<float>& directions,
             const py::array_t<float>& max_distances) {
            return performBatchedAnyHitQuery(self, origins, directions, max_distances);
          },
          R"(Test if rays intersect the mesh for a batch of rays.

:param origins: Ray origins as 2D array of shape (N, 3).
:param directions: Ray directions as 2D array of shape (N, 3).
:param max_distances: Optional maximum distances per ray as 1D array of shape (N,).
:return: Boolean numpy array of shape (N,) indicating if each ray hits the mesh.)",
          py::arg("origins"),
          py::arg("directions"),
          py::arg("max_distances") = py::array_t<float>())
      .def(
          "closest_surface_point",
          [](const axel::TriBvh<float>& self, const py::array_t<float>& queries) {
            return performBatchedClosestSurfacePointQuery(self, queries);
          },
          R"(Find the closest points on the mesh surface for a batch of query points.

Returns numpy arrays for efficient downstream processing.

:param queries: Query points as 2D array of shape (N, 3).
:return: Tuple of (valid, points, triangle_indices, bary_coords) where:
         - valid: bool array of shape (N,) indicating if query succeeded
         - points: float array of shape (N, 3) with closest points on the mesh surface
         - triangle_indices: uint32 array of shape (N,) with triangle indices (kInvalidTriangleIdx if invalid)
         - bary_coords: float array of shape (N, 3) with barycentric coordinates)",
          py::arg("queries"))
      .def_property_readonly(
          "node_count",
          &axel::TriBvh<float>::getNodeCount,
          "Total number of internal nodes in the BVH tree.")
      .def_property_readonly(
          "primitive_count",
          &axel::TriBvh<float>::getPrimitiveCount,
          "Total number of primitives (triangles) in the BVH.")
      .def("__repr__", [](const axel::TriBvh<float>& self) {
        return fmt::format(
            "TriBvh(primitives={}, nodes={})", self.getPrimitiveCount(), self.getNodeCount());
      });
}

} // namespace pymomentum
