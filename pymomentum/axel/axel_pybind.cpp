/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <axel/BoundingBox.h>
#include <axel/MeshToSdf.h>
#include <axel/SignedDistanceField.h>
#include <axel/common/Types.h>
#include <pymomentum/axel/axel_utility.h>

#include <fmt/format.h>
#include <pybind11/buffer_info.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <gsl/span>

namespace py = pybind11;

namespace pymomentum {

PYBIND11_MODULE(axel, m) {
  m.attr("__name__") = "pymomentum.axel";
  m.doc() = "Python bindings for Axel library classes including SignedDistanceField.";

  // Bind BoundingBox
  py::class_<axel::BoundingBox<float>>(m, "BoundingBox")
      .def(
          py::init<const Eigen::Vector3f&, const Eigen::Vector3f&, axel::Index>(),
          R"(Create a bounding box from minimum and maximum corners.

:param min_corner: Minimum corner of the bounding box (x, y, z).
:param max_corner: Maximum corner of the bounding box (x, y, z).
:param id: Optional ID for the bounding box (default: 0).)",
          py::arg("min_corner"),
          py::arg("max_corner"),
          py::arg("id") = 0)
      .def(
          py::init<const Eigen::Vector3f&, float>(),
          R"(Create a bounding box centered at a point with given thickness.

:param center: Center point of the bounding box (x, y, z).
:param thickness: Half-width in each dimension (default: 0.0).)",
          py::arg("center"),
          py::arg("thickness") = 0.0f)
      .def_property_readonly(
          "min", &axel::BoundingBox<float>::min, "Get the minimum corner of the bounding box.")
      .def_property_readonly(
          "max", &axel::BoundingBox<float>::max, "Get the maximum corner of the bounding box.")
      .def_property_readonly(
          "center", &axel::BoundingBox<float>::center, "Get the center of the bounding box.")
      .def(
          "contains",
          py::overload_cast<const Eigen::Vector3f&>(
              &axel::BoundingBox<float>::contains, py::const_),
          R"(Check if a point is contained within the bounding box.

:param point: Point to test (x, y, z).
:return: True if the point is inside the bounding box.)",
          py::arg("point"))
      .def(
          "extend",
          py::overload_cast<const Eigen::Vector3f&>(&axel::BoundingBox<float>::extend),
          R"(Extend the bounding box to include a point.

:param point: Point to include (x, y, z).)",
          py::arg("point"))
      .def("__repr__", [](const axel::BoundingBox<float>& self) {
        const auto& minPt = self.min();
        const auto& maxPt = self.max();
        return fmt::format(
            "BoundingBox(min=[{:.3f}, {:.3f}, {:.3f}], max=[{:.3f}, {:.3f}, {:.3f}])",
            minPt.x(),
            minPt.y(),
            minPt.z(),
            maxPt.x(),
            maxPt.y(),
            maxPt.z());
      });

  // Bind SignedDistanceField
  py::class_<axel::SignedDistanceField<float>>(m, "SignedDistanceField", py::buffer_protocol())
      .def(
          py::init<const axel::BoundingBox<float>&, const Eigen::Vector3<axel::Index>&, float>(),
          R"(Create a signed distance field with given bounds and resolution.

:param bounds: 3D bounding box defining the spatial extent of the SDF.
:param resolution: Grid resolution in each dimension (nx, ny, nz).
:param initial_value: Initial distance value for all voxels (default: very far distance).)",
          py::arg("bounds"),
          py::arg("resolution"),
          py::arg("initial_value") = axel::SignedDistanceField<float>::kVeryFarDistance)
      .def(
          py::init<
              const axel::BoundingBox<float>&,
              const Eigen::Vector3<axel::Index>&,
              std::vector<float>>(),
          R"(Create a signed distance field with given bounds, resolution, and initial data.

:param bounds: 3D bounding box defining the spatial extent of the SDF.
:param resolution: Grid resolution in each dimension (nx, ny, nz).
:param data: Initial distance values. Must have size nx * ny * nz.)",
          py::arg("bounds"),
          py::arg("resolution"),
          py::arg("data"))
      .def_property_readonly(
          "bounds", &axel::SignedDistanceField<float>::bounds, "Get the bounding box of the SDF.")
      .def_property_readonly(
          "resolution",
          &axel::SignedDistanceField<float>::resolution,
          "Get the grid resolution as (nx, ny, nz).")
      .def_property_readonly(
          "voxel_size",
          &axel::SignedDistanceField<float>::voxelSize,
          "Get the voxel size in each dimension as (dx, dy, dz).")
      .def_property_readonly(
          "total_voxels",
          &axel::SignedDistanceField<float>::totalVoxels,
          "Get the total number of voxels in the SDF.")
      .def(
          "sample",
          [](const axel::SignedDistanceField<float>& self, const py::array_t<float>& positions) {
            return applyBatchedScalarOperation(
                self,
                positions,
                [](const axel::SignedDistanceField<float>& sdf, const Eigen::Vector3f& pos) {
                  return sdf.sample(pos);
                });
          },
          R"(Sample the SDF at continuous 3D positions using trilinear interpolation.

Supports both single position and batch operations:
- Single position: Pass 1D array of shape (3,) to get a scalar result
- Batch positions: Pass 2D array of shape (N, 3) to get 1D array of N results

:param positions: Position(s) to query. Either (3,) for single position or (N, 3) for batch of positions.
:return: Interpolated signed distance value(s). Scalar for single position, 1D array for batch.)",
          py::arg("positions"))
      .def(
          "gradient",
          [](const axel::SignedDistanceField<float>& self, const py::array_t<float>& positions) {
            return applyBatchedVectorOperation(
                self,
                positions,
                [](const axel::SignedDistanceField<float>& sdf, const Eigen::Vector3f& pos) {
                  return sdf.gradient(pos);
                });
          },
          R"(Sample the SDF gradient at continuous 3D positions.

Supports both single position and batch operations:
- Single position: Pass 1D array of shape (3,) to get a 1D array of shape (3,)
- Batch positions: Pass 2D array of shape (N, 3) to get 2D array of shape (N, 3)

The gradient points in the direction of increasing distance.

:param positions: Position(s) to query. Either (3,) for single position or (N, 3) for batch of positions.
:return: Gradient vector(s) at the given position(s). Shape (3,) for single position, (N, 3) for batch.)",
          py::arg("positions"))
      .def(
          "sample_with_gradient",
          [](const axel::SignedDistanceField<float>& self, const py::array_t<float>& positions) {
            return applyBatchedSampleGradientOperation(
                self,
                positions,
                [](const axel::SignedDistanceField<float>& sdf, const Eigen::Vector3f& pos) {
                  return sdf.sampleWithGradient(pos);
                });
          },
          R"(Sample both the SDF value and gradient at continuous 3D positions.

Supports both single position and batch operations:
- Single position: Pass 1D array of shape (3,) to get tuple of (scalar, 1D array of shape (3,))
- Batch positions: Pass 2D array of shape (N, 3) to get tuple of (1D array of N values, 2D array of shape (N, 3))

More efficient than calling sample() and gradient() separately.

:param positions: Position(s) to query. Either (3,) for single position or (N, 3) for batch of positions.
:return: Tuple of (value(s), gradient(s)) at the given position(s).)",
          py::arg("positions"))
      .def(
          "world_to_grid",
          &axel::SignedDistanceField<float>::worldToGrid,
          R"(Convert a 3D world-space position to continuous grid coordinates.

:param position: 3D world-space position (x, y, z).
:return: Continuous grid coordinates (may be fractional).)",
          py::arg("position"))
      .def(
          "grid_to_world",
          &axel::SignedDistanceField<float>::gridToWorld,
          R"(Convert continuous grid coordinates to 3D world-space position.

:param grid_pos: Continuous grid coordinates.
:return: 3D world-space position (x, y, z).)",
          py::arg("grid_pos"))
      .def(
          "is_valid_index",
          &axel::SignedDistanceField<float>::isValidIndex,
          R"(Check if the given grid coordinates are within bounds.

:param i: Grid index in x dimension.
:param j: Grid index in y dimension.
:param k: Grid index in z dimension.
:return: True if indices are within valid range.)",
          py::arg("i"),
          py::arg("j"),
          py::arg("k"))
      .def(
          "fill",
          &axel::SignedDistanceField<float>::fill,
          R"(Fill the entire SDF with a constant value.

:param value: The value to fill with.)",
          py::arg("value"))
      .def_buffer([](axel::SignedDistanceField<float>& self) -> py::buffer_info {
        const auto& res = self.resolution();
        auto& data = self.data();

        return py::buffer_info(
            data.data(), // Pointer to buffer
            sizeof(float), // Size of one scalar
            py::format_descriptor<float>::format(), // Python format
                                                    // descriptor
            3, // Number of dimensions
            {static_cast<py::ssize_t>(res.x()), // Shape: nx
             static_cast<py::ssize_t>(res.y()), //        ny
             static_cast<py::ssize_t>(res.z())}, //        nz
            {sizeof(float) * static_cast<py::ssize_t>(res.y()) *
                 static_cast<py::ssize_t>(res.z()), // Strides: x dimension
             sizeof(float) * static_cast<py::ssize_t>(res.z()), //          y dimension
             sizeof(float)} //          z dimension
        );
      })
      .def("__repr__", [](const axel::SignedDistanceField<float>& self) {
        const auto& res = self.resolution();
        const auto& bounds = self.bounds();
        const auto& minPt = bounds.min();
        const auto& maxPt = bounds.max();
        return fmt::format(
            "SignedDistanceField(resolution=[{}, {}, {}], bounds=([{:.3f}, {:.3f}, {:.3f}], [{:.3f}, {:.3f}, {:.3f}]))",
            res.x(),
            res.y(),
            res.z(),
            minPt.x(),
            minPt.y(),
            minPt.z(),
            maxPt.x(),
            maxPt.y(),
            maxPt.z());
      });

  // Bind MeshToSdfConfig
  py::class_<axel::MeshToSdfConfig<float>>(m, "MeshToSdfConfig")
      .def(py::init<>(), "Create MeshToSdfConfig with default parameters.")
      .def_readwrite(
          "narrow_band_width",
          &axel::MeshToSdfConfig<float>::narrowBandWidth,
          R"(Narrow band width around triangles (in voxel units). Default: 1.5)")
      .def_readwrite(
          "max_distance",
          &axel::MeshToSdfConfig<float>::maxDistance,
          R"(Maximum distance to compute (distances beyond this are clamped). Set to 0 to disable clamping. Default: 0)")
      .def_readwrite(
          "tolerance",
          &axel::MeshToSdfConfig<float>::tolerance,
          R"(Numerical tolerance for computations. Default: machine epsilon * 1000)")
      .def("__repr__", [](const axel::MeshToSdfConfig<float>& self) {
        return fmt::format(
            "MeshToSdfConfig(narrow_band_width={:.3f}, max_distance={:.3f}, tolerance={:.6e})",
            self.narrowBandWidth,
            self.maxDistance,
            self.tolerance);
      });

  // Bind mesh_to_sdf function
  m.def(
      "mesh_to_sdf",
      [](const py::array_t<float>& vertices,
         const py::array_t<int>& triangles,
         const axel::BoundingBox<float>& bounds,
         const py::array_t<axel::Index>& resolution,
         const axel::MeshToSdfConfig<float>& config) {
        // Validate input arrays
        validatePositionArray(vertices, "vertices");
        validateIndexArray(triangles, "triangles");

        if (resolution.ndim() != 1 || resolution.shape(0) != 3) {
          throw std::runtime_error(fmt::format(
              "Invalid shape for resolution: expected (3,), got {}", getArrayDimStr(resolution)));
        }

        // Convert numpy arrays to spans
        const auto verticesData = vertices.unchecked<2>();
        const auto trianglesData = triangles.unchecked<2>();
        const auto resolutionData = resolution.unchecked<1>();

        // Convert vertex data to std::vector<Eigen::Vector3f>
        std::vector<Eigen::Vector3f> vertexVector;
        vertexVector.reserve(verticesData.shape(0));
        for (py::ssize_t i = 0; i < verticesData.shape(0); ++i) {
          vertexVector.emplace_back(verticesData(i, 0), verticesData(i, 1), verticesData(i, 2));
        }

        // Convert triangle data to std::vector<Eigen::Vector3i>
        std::vector<Eigen::Vector3i> triangleVector;
        triangleVector.reserve(trianglesData.shape(0));
        for (py::ssize_t i = 0; i < trianglesData.shape(0); ++i) {
          triangleVector.emplace_back(
              trianglesData(i, 0), trianglesData(i, 1), trianglesData(i, 2));
        }

        // Convert resolution to Eigen::Vector3<axel::Index>
        const Eigen::Vector3<axel::Index> resolutionVector(
            resolutionData(0), resolutionData(1), resolutionData(2));

        // Call the mesh_to_sdf function
        return axel::meshToSdf<float>(
            gsl::span<const Eigen::Vector3f>(vertexVector),
            gsl::span<const Eigen::Vector3i>(triangleVector),
            bounds,
            resolutionVector,
            config);
      },
      R"(Convert a triangle mesh to a signed distance field using modern 3-step approach.

This function creates a high-quality signed distance field from a triangle mesh using:
1. Narrow band initialization with exact triangle distances
2. Fast marching propagation using Eikonal equation  
3. Sign determination using ray casting

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles. 
                  Indices must be valid within the vertices array.
:param bounds: Spatial bounds for the SDF as a :class:`BoundingBox`.
:param resolution: Grid resolution as 1D array of shape (3,) containing (nx, ny, nz).
:param config: Configuration parameters as :class:`MeshToSdfConfig` (optional).
:return: Generated :class:`SignedDistanceField`.

Example usage::

    import numpy as np
    import pymomentum.axel as axel
    
    # Create a simple cube mesh
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]   # top face
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom face
        [4, 7, 6], [4, 6, 5],  # top face
        [0, 4, 5], [0, 5, 1],  # front face
        [2, 6, 7], [2, 7, 3],  # back face
        [0, 3, 7], [0, 7, 4],  # left face
        [1, 5, 6], [1, 6, 2]   # right face
    ], dtype=np.int32)
    
    bounds = axel.BoundingBox(
        min_corner=np.array([-1.5, -1.5, -1.5]), 
        max_corner=np.array([1.5, 1.5, 1.5])
    )
    resolution = np.array([32, 32, 32])
    
    config = axel.MeshToSdfConfig()
    config.narrow_band_width = 3.0
    
    sdf = axel.mesh_to_sdf(vertices, triangles, bounds, resolution, config))",
      py::arg("vertices"),
      py::arg("triangles"),
      py::arg("bounds"),
      py::arg("resolution"),
      py::arg("config") = axel::MeshToSdfConfig<float>{});

  // Bind convenience overload with automatic bounds computation
  m.def(
      "mesh_to_sdf",
      [](const py::array_t<float>& vertices,
         const py::array_t<int>& triangles,
         const py::array_t<axel::Index>& resolution,
         float padding,
         const axel::MeshToSdfConfig<float>& config) {
        // Validate input arrays
        validatePositionArray(vertices, "vertices");
        validateIndexArray(triangles, "triangles");

        if (resolution.ndim() != 1 || resolution.shape(0) != 3) {
          throw std::runtime_error(fmt::format(
              "Invalid shape for resolution: expected (3,), got {}", getArrayDimStr(resolution)));
        }

        // Convert numpy arrays to spans
        const auto verticesData = vertices.unchecked<2>();
        const auto trianglesData = triangles.unchecked<2>();
        const auto resolutionData = resolution.unchecked<1>();

        // Convert vertex data to std::vector<Eigen::Vector3f>
        std::vector<Eigen::Vector3f> vertexVector;
        vertexVector.reserve(verticesData.shape(0));
        for (py::ssize_t i = 0; i < verticesData.shape(0); ++i) {
          vertexVector.emplace_back(verticesData(i, 0), verticesData(i, 1), verticesData(i, 2));
        }

        // Convert triangle data to std::vector<Eigen::Vector3i>
        std::vector<Eigen::Vector3i> triangleVector;
        triangleVector.reserve(trianglesData.shape(0));
        for (py::ssize_t i = 0; i < trianglesData.shape(0); ++i) {
          triangleVector.emplace_back(
              trianglesData(i, 0), trianglesData(i, 1), trianglesData(i, 2));
        }

        // Convert resolution to Eigen::Vector3<axel::Index>
        const Eigen::Vector3<axel::Index> resolutionVector(
            resolutionData(0), resolutionData(1), resolutionData(2));

        // Call the mesh_to_sdf function with automatic bounds
        return axel::meshToSdf<float>(
            gsl::span<const Eigen::Vector3f>(vertexVector),
            gsl::span<const Eigen::Vector3i>(triangleVector),
            resolutionVector,
            padding,
            config);
      },
      R"(Convert a triangle mesh to a signed distance field with automatic bounds computation.

This convenience function automatically computes the bounding box from the mesh vertices,
adds padding, and creates a signed distance field.

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles.
:param resolution: Grid resolution as 1D array of shape (3,) containing (nx, ny, nz).
:param padding: Extra space around mesh bounds as fraction of bounding box size (default: 0.1).
:param config: Configuration parameters as :class:`MeshToSdfConfig` (optional).
:return: Generated :class:`SignedDistanceField`.

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
    
    resolution = np.array([32, 32, 32])
    
    # Automatically compute bounds with 20% padding
    sdf = axel.mesh_to_sdf(vertices, triangles, resolution, padding=0.2))",
      py::arg("vertices"),
      py::arg("triangles"),
      py::arg("resolution"),
      py::arg("padding") = 0.1f,
      py::arg("config") = axel::MeshToSdfConfig<float>{});
}

} // namespace pymomentum
