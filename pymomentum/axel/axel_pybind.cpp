/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <axel/BoundingBox.h>
#include <axel/DualContouring.h>
#include <axel/MeshToSdf.h>
#include <axel/SignedDistanceField.h>
#include <axel/common/Types.h>
#include <axel/math/MeshHoleFilling.h>
#include <momentum/common/exception.h>
#include <pymomentum/axel/axel_utility.h>
#include <pymomentum/axel/tri_bvh_pybind.h>

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

namespace {

py::array_t<float> smoothMeshLaplacianBinding(
    const py::array_t<float>& vertices,
    const py::array_t<int>& faces,
    const py::array_t<bool>& vertex_mask,
    int iterations,
    float step) {
  // Validate input arrays
  validatePositionArray(vertices, "vertices");

  // Validate faces array (but allow either 3 or 4 columns)
  if (faces.ndim() != 2) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for faces: expected 2-dimensional array, got {}D", faces.ndim()));
  }

  const py::ssize_t num_face_verts = faces.shape(1);
  if (num_face_verts != 3 && num_face_verts != 4) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for faces: expected (M, 3) for triangles or (M, 4) for quads, got ({}, {})",
            faces.shape(0),
            num_face_verts));
  }

  // Validate face indices are within bounds
  const int max_vertex_index = static_cast<int>(vertices.shape(0));
  validateIndexArray(faces, "faces", max_vertex_index);

  // Convert vertices using utility function
  const auto vertexVector = arrayToEigenVectors<float, 3>(vertices, "vertices");

  // Convert vertex mask if provided
  std::vector<bool> maskVector;
  if (vertex_mask.size() > 0) {
    // Validate mask array
    if (vertex_mask.ndim() != 1) {
      throw std::runtime_error("Vertex mask must be 1-dimensional");
    }
    if (vertex_mask.shape(0) != vertices.shape(0)) {
      throw std::runtime_error(
          fmt::format(
              "Vertex mask size ({}) must match number of vertices ({})",
              vertex_mask.shape(0),
              vertices.shape(0)));
    }

    const auto maskData = vertex_mask.unchecked<1>();
    maskVector.reserve(maskData.shape(0));
    for (py::ssize_t i = 0; i < maskData.shape(0); ++i) {
      maskVector.push_back(maskData(i));
    }
  }

  std::vector<Eigen::Vector3f> smoothedVertices;

  // Handle triangles vs quads based on face array shape
  if (num_face_verts == 3) {
    // Triangle mesh - convert faces using utility function
    const auto triangleVector = arrayToEigenVectors<int, 3>(faces, "faces");

    smoothedVertices = axel::smoothMeshLaplacian<float, Eigen::Vector3i>(
        std::span<const Eigen::Vector3f>(vertexVector),
        std::span<const Eigen::Vector3i>(triangleVector),
        maskVector,
        static_cast<axel::Index>(iterations),
        step);
  } else {
    // Quad mesh - convert faces using utility function
    const auto quadVector = arrayToEigenVectors<int, 4>(faces, "faces");

    smoothedVertices = axel::smoothMeshLaplacian<float, Eigen::Vector4i>(
        std::span<const Eigen::Vector3f>(vertexVector),
        std::span<const Eigen::Vector4i>(quadVector),
        maskVector,
        static_cast<axel::Index>(iterations),
        step);
  }

  // Convert results back to numpy array using utility function
  return eigenVectorsToArray<float, 3>(smoothedVertices);
}

} // namespace

PYBIND11_MODULE(axel, m) {
  m.attr("__name__") = "pymomentum.axel";
  m.doc() = "Python bindings for Axel library classes including SignedDistanceField.";

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  // Bind HoleFillingMethod enum
  py::enum_<axel::HoleFillingMethod>(m, "HoleFillingMethod", R"(Method for filling mesh holes.

Available methods:
- Centroid: Add a centroid vertex and create fan triangles (default, best for SDF generation)
- EarClipping: Use ear clipping without adding new vertices
- Auto: Automatically choose based on hole size (centroid for ≤8 vertices, ear clipping for larger))")
      .value(
          "Centroid",
          axel::HoleFillingMethod::Centroid,
          "Add centroid vertex, create fan triangles")
      .value(
          "EarClipping", axel::HoleFillingMethod::EarClipping, "Use ear clipping, no new vertices")
      .value("Auto", axel::HoleFillingMethod::Auto, "Auto-select based on hole size")
      .export_values();

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

  // Bind SignedDistanceField with shared_ptr holder for proper lifetime management
  // when shared with SDFCollider and other types that store shared_ptr<SDF>
  py::class_<axel::SignedDistanceField<float>, std::shared_ptr<axel::SignedDistanceField<float>>>(
      m, "SignedDistanceField", py::buffer_protocol())
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
          &axel::SignedDistanceField<float>::worldToGrid<float>,
          R"(Convert a 3D world-space position to continuous grid coordinates.

:param position: 3D world-space position (x, y, z).
:return: Continuous grid coordinates (may be fractional).)",
          py::arg("position"))
      .def(
          "grid_to_world",
          &axel::SignedDistanceField<float>::gridToWorld<float>,
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

  // Bind MeshToSdfPadding
  py::class_<axel::MeshToSdfPadding<float>>(
      m,
      "MeshToSdfPadding",
      R"(Padding configuration for mesh-to-SDF bounds computation.

The final per-axis padding is computed as ``max(extent[i] * fractional, absolute)``,
where ``extent`` is the mesh bounding box size in each dimension.

This allows both proportional and absolute minimum padding to be specified,
ensuring adequate padding even for very thin mesh regions.

:param fractional: Fractional padding as a fraction of bounding box extent per axis (default: 0.1).
:param absolute: Absolute minimum padding in world units (e.g. cm), applied per axis (default: 0).)")
      .def(py::init<>(), "Create :class:`MeshToSdfPadding` with default parameters.")
      .def(
          py::init([](float fractional, float absolute) {
            axel::MeshToSdfPadding<float> p;
            p.fractional = fractional;
            p.absolute = absolute;
            return p;
          }),
          R"(Create :class:`MeshToSdfPadding` with specified parameters.

:param fractional: Fractional padding as a fraction of bounding box extent per axis.
:param absolute: Absolute minimum padding in world units.)",
          py::arg("fractional"),
          py::arg("absolute") = 0.0f)
      .def_readwrite(
          "fractional",
          &axel::MeshToSdfPadding<float>::fractional,
          R"(Fractional padding as a fraction of bounding box extent per axis. Default: 0.1)")
      .def_readwrite(
          "absolute",
          &axel::MeshToSdfPadding<float>::absolute,
          R"(Absolute minimum padding in world units (e.g. cm), applied per axis. Default: 0)")
      .def("__repr__", [](const axel::MeshToSdfPadding<float>& self) {
        return fmt::format(
            "MeshToSdfPadding(fractional={:.3f}, absolute={:.3f})", self.fractional, self.absolute);
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
        validateTriangleIndexArray(triangles, "triangles", vertices.shape(0));

        if (resolution.ndim() != 1 || resolution.shape(0) != 3) {
          throw std::runtime_error(
              fmt::format(
                  "Invalid shape for resolution: expected (3,), got {}",
                  getArrayDimStr(resolution)));
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
            std::span<const Eigen::Vector3f>(vertexVector),
            std::span<const Eigen::Vector3i>(triangleVector),
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

  // Bind convenience overload with voxel size and automatic bounds computation
  m.def(
      "mesh_to_sdf",
      [](const py::array_t<float>& vertices,
         const py::array_t<int>& triangles,
         float voxelSize,
         const axel::MeshToSdfPadding<float>& padding,
         const axel::MeshToSdfConfig<float>& config) {
        // Validate input arrays
        validatePositionArray(vertices, "vertices");
        validateTriangleIndexArray(triangles, "triangles", vertices.shape(0));

        if (voxelSize <= 0.0f) {
          throw std::runtime_error(fmt::format("voxel_size must be positive, got {}", voxelSize));
        }

        // Convert numpy arrays to spans
        const auto verticesData = vertices.unchecked<2>();
        const auto trianglesData = triangles.unchecked<2>();

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

        // Call the mesh_to_sdf function with voxel size
        return axel::meshToSdf<float>(
            std::span<const Eigen::Vector3f>(vertexVector),
            std::span<const Eigen::Vector3i>(triangleVector),
            voxelSize,
            padding,
            config);
      },
      R"(Convert a triangle mesh to a signed distance field with uniform voxel size.

This convenience function automatically computes the bounding box from the mesh vertices,
adds padding, and creates a signed distance field with uniform (isotropic) voxels.
Grid dimensions adapt to mesh shape automatically.

The bounds are computed as follows:

1. Compute mesh bounding box from vertices
2. Per-axis padding = ``max(extent[i] * padding.fractional, padding.absolute)``
3. Expand bounds by padding on each side
4. Compute resolution per axis = ``max(1, ceil(padded_extent[i] / voxel_size))``
5. Adjust bounds so that ``extent / resolution`` is exactly ``voxel_size`` in each dimension

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles.
:param voxel_size: Uniform voxel size in world units (must be positive).
:param padding: Padding configuration as :class:`MeshToSdfPadding` (optional, default: 10% fractional).
:param config: Configuration parameters as :class:`MeshToSdfConfig` (optional).
:return: Generated :class:`SignedDistanceField` with isotropic voxels.

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

    # Create SDF with 0.05 cm voxels
    sdf = axel.mesh_to_sdf(vertices, triangles, voxel_size=0.05)

    # With custom padding
    padding = axel.MeshToSdfPadding(fractional=0.2, absolute=0.5)
    sdf = axel.mesh_to_sdf(vertices, triangles, voxel_size=0.05, padding=padding))",
      py::arg("vertices"),
      py::arg("triangles"),
      py::arg("voxel_size"),
      py::arg("padding") = axel::MeshToSdfPadding<float>{},
      py::arg("config") = axel::MeshToSdfConfig<float>{});

  // Bind in-place overload that writes into an existing SDF
  m.def(
      "mesh_to_sdf",
      [](const py::array_t<float>& vertices,
         const py::array_t<int>& triangles,
         axel::SignedDistanceField<float>& sdf,
         const axel::MeshToSdfConfig<float>& config) {
        // Validate input arrays
        validatePositionArray(vertices, "vertices");
        validateTriangleIndexArray(triangles, "triangles", vertices.shape(0));

        // Convert numpy arrays to spans
        const auto verticesData = vertices.unchecked<2>();
        const auto trianglesData = triangles.unchecked<2>();

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

        // Call the in-place mesh_to_sdf function
        axel::meshToSdf<float>(
            std::span<const Eigen::Vector3f>(vertexVector),
            std::span<const Eigen::Vector3i>(triangleVector),
            sdf,
            config);
      },
      R"(Write mesh-to-SDF conversion into a pre-existing signed distance field.

This in-place overload reuses the existing SDF's bounds and resolution,
writing the computed distance values directly into the grid.
This is useful when you want to recompute an SDF with a different mesh
but the same grid layout.

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles.
:param sdf: Pre-existing :class:`SignedDistanceField` to write into. Its contents will be overwritten.
:param config: Configuration parameters as :class:`MeshToSdfConfig` (optional).

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Create an SDF grid
    bounds = axel.BoundingBox(
        min_corner=np.array([-1.5, -1.5, -1.5]),
        max_corner=np.array([1.5, 1.5, 1.5])
    )
    resolution = np.array([32, 32, 32])
    sdf = axel.SignedDistanceField(bounds, resolution)

    # Compute SDF in-place for a mesh
    vertices = np.array([...], dtype=np.float32)
    triangles = np.array([...], dtype=np.int32)
    axel.mesh_to_sdf(vertices, triangles, sdf))",
      py::arg("vertices"),
      py::arg("triangles"),
      py::arg("sdf"),
      py::arg("config") = axel::MeshToSdfConfig<float>{});

  // Bind fill_holes function
  m.def(
      "fill_holes",
      [](const py::array_t<float>& vertices,
         const py::array_t<int>& triangles,
         axel::HoleFillingMethod method) {
        // Validate input arrays
        validatePositionArray(vertices, "vertices");
        validateTriangleIndexArray(triangles, "triangles", vertices.shape(0));

        // Convert numpy arrays to spans
        const auto verticesData = vertices.unchecked<2>();
        const auto trianglesData = triangles.unchecked<2>();

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

        // Call the fillMeshHolesComplete function with the specified method
        const auto [filledVertices, filledTriangles] = axel::fillMeshHolesComplete<float>(
            std::span<const Eigen::Vector3f>(vertexVector),
            std::span<const Eigen::Vector3i>(triangleVector),
            method);

        // Convert results back to numpy arrays
        // Convert vertices
        std::vector<py::ssize_t> vertexShape = {static_cast<py::ssize_t>(filledVertices.size()), 3};
        auto resultVertices = py::array_t<float>(vertexShape);
        auto verticesData_out = resultVertices.template mutable_unchecked<2>();
        for (size_t i = 0; i < filledVertices.size(); ++i) {
          verticesData_out(i, 0) = filledVertices[i].x();
          verticesData_out(i, 1) = filledVertices[i].y();
          verticesData_out(i, 2) = filledVertices[i].z();
        }

        // Convert triangles
        std::vector<py::ssize_t> triangleShape = {
            static_cast<py::ssize_t>(filledTriangles.size()), 3};
        auto resultTriangles = py::array_t<int>(triangleShape);
        auto trianglesData_out = resultTriangles.template mutable_unchecked<2>();
        for (size_t i = 0; i < filledTriangles.size(); ++i) {
          trianglesData_out(i, 0) = filledTriangles[i].x();
          trianglesData_out(i, 1) = filledTriangles[i].y();
          trianglesData_out(i, 2) = filledTriangles[i].z();
        }

        return py::make_tuple(resultVertices, resultTriangles);
      },
      R"(Fill holes in a triangle mesh to create a watertight surface.

This function identifies holes in the mesh and fills them with new triangles.
The result is a complete mesh suitable for operations that require watertight
surfaces, such as SDF generation.

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param triangles: Triangle indices as 2D array of shape (M, 3) where M is number of triangles.
                  Indices must be valid within the vertices array.
:param method: Hole filling method (default: Centroid). Options are:
               - HoleFillingMethod.Centroid: Add centroid vertex, create fan triangles (best quality)
               - HoleFillingMethod.EarClipping: Use ear clipping, no new vertices
               - HoleFillingMethod.Auto: Auto-select based on hole size
:return: Tuple of (filled_vertices, filled_triangles) where:
         - filled_vertices: 2D array of shape (N', 3) with original + new vertices
         - filled_triangles: 2D array of shape (M', 3) with original + new triangles

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Create a cube mesh with a missing face (hole)
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]   # top face
    ], dtype=np.float32)

    # Missing top face triangles to create a hole
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom face
        # [4, 7, 6], [4, 6, 5],  # top face (missing - creates hole)
        [0, 4, 5], [0, 5, 1],  # front face
        [2, 6, 7], [2, 7, 3],  # back face
        [0, 3, 7], [0, 7, 4],  # left face
        [1, 5, 6], [1, 6, 2]   # right face
    ], dtype=np.int32)

    # Use centroid method (default) for best triangle quality
    filled_vertices, filled_triangles = axel.fill_holes(vertices, triangles)

    print(f"Original mesh: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"Filled mesh: {len(filled_vertices)} vertices, {len(filled_triangles)} triangles"))",
      py::arg("vertices"),
      py::arg("triangles"),
      py::arg("method") = axel::HoleFillingMethod::Centroid);

  // Bind dual_contouring function (always returns quads)
  m.def(
      "dual_contouring",
      [](const axel::SignedDistanceField<float>& sdf, float isovalue, bool triangulate) {
        // Call the dual contouring function
        const auto result = axel::dualContouring<float>(sdf, isovalue);

        if (!result.success) {
          throw std::runtime_error("Dual contouring failed");
        }

        // Convert vertices to numpy array using helper
        auto vertices = pymomentum::eigenVectorsToArray<float, 3>(result.vertices);

        // Compute normals at vertices by sampling SDF gradients
        std::vector<Eigen::Vector3f> normals;
        normals.reserve(result.vertices.size());
        for (const auto& vertex : result.vertices) {
          normals.push_back(sdf.gradient(vertex));
        }
        auto normalsArray = pymomentum::eigenVectorsToArray<float, 3>(normals);

        // Convert quads to numpy array using helper
        auto faces = triangulate
            ? pymomentum::eigenVectorsToArray<int, 3>(axel::triangulateQuads(result.quads))
            : pymomentum::eigenVectorsToArray<int, 4>(result.quads);

        return py::make_tuple(vertices, normalsArray, faces);
      },
      R"(Extract an isosurface from a signed distance field using dual contouring.

Dual contouring places vertices inside grid cells and generates quad faces between
adjacent cells that both contain vertices. This naturally produces quads rather than
triangles, which better preserves surface topology and reduces mesh artifacts.

The algorithm works by:
1. Finding all cells that intersect the isosurface (sign changes across cell corners)
2. Placing one vertex at each intersecting cell, positioned on the surface using gradient descent
3. Generating quads for each edge crossing that connects 4 adjacent cells

:param sdf: The :class:`SignedDistanceField` to extract the isosurface from.
:param isovalue: The isovalue to extract (typically 0.0 for zero level set). Default: 0.0
:param triangulate: Whether to triangulate the quads (default: False).
:return: Tuple of (vertices, normals, quads) where:
         - vertices: 2D array of shape (N, 3) with vertex positions
         - normals: 2D array of shape (N, 3) with vertex normals (computed from SDF gradients)
         - quads: Quad indices of shape (M, 4) connecting the vertices

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Create a sphere SDF
    bounds = axel.BoundingBox(
        min_corner=np.array([-2.0, -2.0, -2.0]),
        max_corner=np.array([2.0, 2.0, 2.0])
    )
    resolution = np.array([32, 32, 32])
    sdf = axel.SignedDistanceField(bounds, resolution)

    # Fill with sphere distance values
    for k in range(resolution[2]):
        for j in range(resolution[1]):
            for i in range(resolution[0]):
                grid_pos = np.array([i, j, k], dtype=np.float32)
                world_pos = sdf.grid_to_world(grid_pos)
                distance = np.linalg.norm(world_pos) - 1.0  # Unit sphere
                sdf.set(i, j, k, distance)

    # Extract mesh (always returns quads)
    vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

    print(f"Extracted {len(vertices)} vertices, {len(quads)} quads")
    print(f"Vertex normals have shape: {normals.shape}"))",
      py::arg("sdf"),
      py::arg("isovalue") = 0.0f,
      py::arg("triangulate") = false);

  // Bind triangulate_quads function
  m.def(
      "triangulate_quads",
      [](const py::array_t<int>& quads) {
        // Validate input array - need custom validation for quads (N, 4)
        MT_THROW_IF_T(quads.ndim() != 2, py::value_error, "Quads array must be 2-dimensional");

        MT_THROW_IF_T(quads.shape(1) != 4, py::value_error, "Quads array must have shape (N, 4)");

        const auto quadsData = quads.unchecked<2>();

        // Convert to std::vector<Eigen::Vector4i>
        std::vector<Eigen::Vector4i> quadVector;
        quadVector.reserve(quadsData.shape(0));
        for (py::ssize_t i = 0; i < quadsData.shape(0); ++i) {
          quadVector.emplace_back(
              quadsData(i, 0), quadsData(i, 1), quadsData(i, 2), quadsData(i, 3));
        }

        // Triangulate
        const auto triangles = axel::triangulateQuads(quadVector);

        // Convert back to numpy array
        std::vector<py::ssize_t> triangleShape = {static_cast<py::ssize_t>(triangles.size()), 3};
        auto resultTriangles = py::array_t<int>(triangleShape);
        auto trianglesData = resultTriangles.template mutable_unchecked<2>();
        for (size_t i = 0; i < triangles.size(); ++i) {
          trianglesData(i, 0) = triangles[i].x();
          trianglesData(i, 1) = triangles[i].y();
          trianglesData(i, 2) = triangles[i].z();
        }

        return resultTriangles;
      },
      R"(Triangulate a quad mesh into triangles.

Each quad is split into two triangles using the diagonal (0,2).
This converts a quad mesh (as produced by dual contouring) into a triangle mesh
suitable for rendering or processing with triangle-based algorithms.

:param quads: Quad indices as 2D array of shape (M, 4) where M is number of quads.
              Each row contains 4 vertex indices defining a quad.
:return: Triangle indices as 2D array of shape (2M, 3) where each quad produces 2 triangles.

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Get quads from dual contouring
    vertices, normals, quads = axel.dual_contouring(sdf, config)

    # Convert to triangles if needed for rendering
    triangles = axel.triangulate_quads(quads)

    print(f"Converted {len(quads)} quads to {len(triangles)} triangles"))",
      py::arg("quads"));

  // Bind smooth_mesh_laplacian function
  m.def(
      "smooth_mesh_laplacian",
      &smoothMeshLaplacianBinding,
      R"(Smooth a triangle or quad mesh using Laplacian smoothing with optional vertex masking.

This function applies Laplacian smoothing to mesh vertices, where each vertex is moved toward
the average position of its neighboring vertices. Optionally, you can specify which vertices
to smooth using a boolean mask. Both triangle and quad meshes are supported automatically
based on the shape of the faces array.

The smoothing is applied iteratively using the formula:
new_position = (1 - step) * old_position + step * average_neighbor_position

:param vertices: Vertex positions as 2D array of shape (N, 3) where N is number of vertices.
:param faces: Face indices as 2D array of shape (M, 3) for triangles or (M, 4) for quads.
              Indices must be valid within the vertices array.
:param vertex_mask: Optional boolean mask of shape (N,) indicating which vertices to smooth.
                   If empty array or not provided, all vertices will be smoothed.
:param iterations: Number of smoothing iterations to apply (default: 1).
:param step: Smoothing step size between 0.0 and 1.0 (default: 0.5).
             Smaller values preserve original shape better, larger values smooth more aggressively.
:return: Smoothed vertex positions as 2D array of shape (N, 3).

Example usage::

    import numpy as np
    import pymomentum.axel as axel

    # Example 1: Triangle mesh
    tri_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.0, 1.0]
    ], dtype=np.float32)

    tri_faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ], dtype=np.int32)

    # Smooth all vertices of triangle mesh
    smoothed_tri = axel.smooth_mesh_laplacian(
        tri_vertices, tri_faces, np.array([]), iterations=5, step=0.3)

    # Example 2: Quad mesh
    quad_vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0]   # 7
    ], dtype=np.float32)

    quad_faces = np.array([
        [0, 1, 2, 3],  # Bottom face
        [4, 7, 6, 5],  # Top face
        [0, 4, 5, 1],  # Front face
        [2, 6, 7, 3],  # Back face
        [0, 3, 7, 4],  # Left face
        [1, 5, 6, 2]   # Right face
    ], dtype=np.int32)

    # Smooth only internal vertices (exclude corners)
    vertex_mask = np.array([False, True, True, False, False, True, True, False])
    smoothed_quad = axel.smooth_mesh_laplacian(
        quad_vertices, quad_faces, vertex_mask, iterations=3, step=0.5))",
      py::arg("vertices"),
      py::arg("faces"),
      py::arg("vertex_mask") = py::array_t<bool>(),
      py::arg("iterations") = 1,
      py::arg("step") = 0.5f);

  // Register TriBvh bindings
  registerTriBvhBindings(m);
}

} // namespace pymomentum
