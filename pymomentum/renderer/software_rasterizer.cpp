/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/software_rasterizer.h"

#include "pymomentum/array_utility/array_utility.h"
#include "pymomentum/array_utility/geometry_accessors.h"
#include "pymomentum/renderer/rasterizer_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>
#include <momentum/math/fwd.h>
#include <momentum/math/mesh.h>

#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>

#include <dispenso/parallel_for.h>
#include <drjit/packet.h>
#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pymomentum {

using rasterizer_detail::bufferToEigenMap;
using rasterizer_detail::convertLightsToEyeSpace;
using rasterizer_detail::make_mdspan;
using rasterizer_detail::validateRasterizerBuffers;

namespace {

constexpr size_t kMaxBufferBaseDepth = 64;

size_t availableBufferBytes(
    const py::buffer& buffer,
    const py::buffer_info& bufferInfo,
    const char* name) {
  auto owner = py::reinterpret_borrow<py::object>(buffer);
  std::unordered_set<PyObject*> visited;
  for (size_t depth = 0; py::hasattr(owner, "base"); ++depth) {
    MT_THROW_IF(
        depth >= kMaxBufferBaseDepth,
        "{} backing storage base chain exceeds {} objects",
        name,
        kMaxBufferBaseDepth);
    MT_THROW_IF(
        !visited.insert(owner.ptr()).second,
        "{} backing storage base chain contains a cycle",
        name);
    py::object base = owner.attr("base");
    if (base.is_none()) {
      break;
    }
    owner = std::move(base);
  }

  size_t ownerBytes = 0;
  uintptr_t ownerAddress = 0;
  if (py::hasattr(owner, "untyped_storage") && py::hasattr(owner, "storage_offset") &&
      py::hasattr(owner, "element_size") && py::hasattr(owner, "data_ptr")) {
    const auto storageBytes = owner.attr("untyped_storage")().attr("nbytes")().cast<size_t>();
    const auto storageOffset = owner.attr("storage_offset")().cast<size_t>();
    const auto elementSize = owner.attr("element_size")().cast<size_t>();
    MT_THROW_IF(
        elementSize == 0 || storageOffset > storageBytes / elementSize,
        "{} has an invalid backing storage offset",
        name);
    ownerBytes = storageBytes - storageOffset * elementSize;
    ownerAddress = owner.attr("data_ptr")().cast<uintptr_t>();
  } else if (PyObject_CheckBuffer(owner.ptr()) != 0) {
    const py::buffer_info ownerInfo = py::reinterpret_borrow<py::buffer>(owner).request();
    MT_THROW_IF(
        ownerInfo.size < 0 || ownerInfo.itemsize <= 0 ||
            static_cast<size_t>(ownerInfo.size) >
                std::numeric_limits<size_t>::max() / static_cast<size_t>(ownerInfo.itemsize),
        "{} backing storage size overflows",
        name);
    ownerBytes = static_cast<size_t>(ownerInfo.size) * static_cast<size_t>(ownerInfo.itemsize);
    ownerAddress = reinterpret_cast<uintptr_t>(ownerInfo.ptr);
  } else {
    MT_THROW("{} backing storage size cannot be determined", name);
  }

  const auto dataAddress = reinterpret_cast<uintptr_t>(bufferInfo.ptr);
  MT_THROW_IF(
      dataAddress < ownerAddress || dataAddress - ownerAddress > ownerBytes,
      "{} data pointer is outside its backing storage",
      name);
  return ownerBytes - (dataAddress - ownerAddress);
}

} // namespace

// rasterize with CameraData3d which considers distortion
void rasterizeMesh(
    const ContiguousArray<float>& positions,
    std::optional<ContiguousArray<float>> normals,
    const ContiguousArray<int32_t>& triangles,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<pybind11::buffer>& vertexIndexBuffer,
    const std::optional<pybind11::buffer>& triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<ContiguousArray<float>> textureCoordinates,
    std::optional<ContiguousArray<int32_t>> textureTriangles,
    std::optional<ContiguousArray<float>> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nVertsBindingID = -1;
  const int nTrisBindingID = -2;
  const int nTexCoordsBindingID = -3;

  ArrayChecker checker("rasterize_mesh");
  checker.validateBuffer(positions, "positions", {nVertsBindingID, 3}, {"nVertices", "xyz"}, false);

  if (normals.has_value()) {
    checker.validateBuffer(*normals, "normals", {nVertsBindingID, 3}, {"nVertices", "xyz"}, true);
  }

  checker.validateBuffer(triangles, "triangles", {nTrisBindingID, 3}, {"nTriangles", "xyz"}, false);

  if (textureCoordinates.has_value()) {
    checker.validateBuffer(
        *textureCoordinates,
        "texture_coordinates",
        {nTexCoordsBindingID, 2},
        {"nTexCoords", "uv"},
        false);
  }

  if (textureTriangles.has_value()) {
    checker.validateBuffer(
        *textureTriangles, "texture_triangles", {nTrisBindingID, 3}, {"nTriangles", "xyz"}, false);
  }

  if (perVertexDiffuseColor.has_value()) {
    checker.validateBuffer(
        *perVertexDiffuseColor,
        "per_vertex_diffuse_color",
        {nVertsBindingID, 3},
        {"nVerts", "rgb"},
        false);
  }

  validateRasterizerBuffers(
      camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  // Get buffer infos for creating Eigen maps (requires GIL)
  auto positionsInfo = positions.request();
  auto trianglesInfo = triangles.request();

  std::optional<py::buffer_info> normalsInfo;
  if (normals.has_value()) {
    normalsInfo = normals->request();
  }
  std::optional<py::buffer_info> texCoordsInfo;
  if (textureCoordinates.has_value()) {
    texCoordsInfo = textureCoordinates->request();
  }
  std::optional<py::buffer_info> texTrianglesInfo;
  if (textureTriangles.has_value()) {
    texTrianglesInfo = textureTriangles->request();
  }
  std::optional<py::buffer_info> perVertexColorInfo;
  if (perVertexDiffuseColor.has_value()) {
    perVertexColorInfo = perVertexDiffuseColor->request();
  }

  // Create Eigen maps from buffer data
  Eigen::VectorXf emptyFloatVec;
  Eigen::VectorXi emptyIntVec;
  auto positionsMap = bufferToEigenMap<float>(positionsInfo);
  auto normalsMap = bufferToEigenMap<float>(normalsInfo, emptyFloatVec);
  auto trianglesMap = bufferToEigenMap<int32_t>(trianglesInfo);
  auto texCoordsMap = bufferToEigenMap<float>(texCoordsInfo, emptyFloatVec);
  auto texTrianglesMap = bufferToEigenMap<int32_t>(texTrianglesInfo, emptyIntVec);
  auto perVertexColorMap = bufferToEigenMap<float>(perVertexColorInfo, emptyFloatVec);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);
  auto vertexIndexBufferMdspan = make_mdspan<int, 2>(vertexIndexBuffer);
  auto triangleIndexBufferMdspan = make_mdspan<int, 2>(triangleIndexBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeMesh(
      positionsMap,
      normalsMap,
      trianglesMap,
      texCoordsMap,
      texTrianglesMap,
      perVertexColorMap,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      material.value_or(momentum::rasterizer::PhongMaterial{}),
      zBufferMdspan,
      rgbBufferMdspan,
      surfaceNormalsBufferMdspan,
      vertexIndexBufferMdspan,
      triangleIndexBufferMdspan,
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeMeshes(
    const ContiguousArray<float>& positions,
    const ContiguousArray<int32_t>& triangles,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const pybind11::buffer& triangleIndexBuffer,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const py::buffer_info positionsInfo = positions.request();
  const py::buffer_info trianglesInfo = triangles.request();
  const py::buffer_info zBufferInfo = zBuffer.request();
  const py::buffer_info triangleIndexInfo = triangleIndexBuffer.request();
  MT_THROW_IF(
      positionsInfo.ndim != 3 || positionsInfo.shape[0] <= 0 || positionsInfo.shape[2] != 3,
      "positions must have shape [batch, numVertices, 3]");
  MT_THROW_IF(
      trianglesInfo.ndim != 2 || trianglesInfo.shape[1] != 3,
      "triangles must have shape [numTriangles, 3]");
  MT_THROW_IF(
      !rasterizer_detail::isCompatibleBufferFormat(
          zBufferInfo, py::format_descriptor<float>::format()) ||
          !rasterizer_detail::isCompatibleBufferFormat(
              triangleIndexInfo, py::format_descriptor<int32_t>::format()),
      "Batched rasterizer outputs have incorrect data types");

  const int64_t batchSize = positionsInfo.shape[0];
  const int64_t imageHeight = camera.imageHeight();
  const int64_t imageWidth = camera.imageWidth();
  MT_THROW_IF(imageHeight <= 0 || imageWidth <= 0, "Camera image must be non-empty");
  const std::vector<py::ssize_t> expectedOutputShape{batchSize, imageHeight, imageWidth};
  MT_THROW_IF(
      zBufferInfo.ndim != 3 || zBufferInfo.shape != expectedOutputShape,
      "z_buffer must have shape [{}, {}, {}]",
      batchSize,
      imageHeight,
      imageWidth);
  MT_THROW_IF(
      triangleIndexInfo.ndim != 3 || triangleIndexInfo.shape != expectedOutputShape,
      "triangle_index_buffer must have shape [{}, {}, {}]",
      batchSize,
      imageHeight,
      imageWidth);
  MT_THROW_IF(
      zBufferInfo.strides != triangleIndexInfo.strides,
      "Batched rasterizer outputs must have matching strides");

  const int64_t rowStride = zBufferInfo.strides[1] / static_cast<int64_t>(sizeof(float));
  const int64_t imageStride = zBufferInfo.strides[0] / static_cast<int64_t>(sizeof(float));
  MT_THROW_IF(
      zBufferInfo.strides[2] != sizeof(float) || rowStride < imageWidth,
      "Batched rasterizer outputs have invalid strides");
  MT_THROW_IF(
      imageHeight > std::numeric_limits<int64_t>::max() / rowStride,
      "Batched rasterizer output size overflows");
  MT_THROW_IF(
      imageStride < imageHeight * rowStride, "Batched rasterizer outputs have invalid strides");
  const int64_t imageRowsSize = imageHeight * rowStride;
  MT_THROW_IF(
      batchSize - 1 > (std::numeric_limits<int64_t>::max() - imageRowsSize) / imageStride,
      "Batched rasterizer output size overflows");
  const int64_t requiredElements = (batchSize - 1) * imageStride + imageRowsSize;
  MT_THROW_IF(
      static_cast<uint64_t>(requiredElements) > std::numeric_limits<size_t>::max() / sizeof(float),
      "Batched rasterizer output byte size overflows");
  const size_t requiredBytes = static_cast<size_t>(requiredElements) * sizeof(float);
  MT_THROW_IF(
      availableBufferBytes(zBuffer, zBufferInfo, "z_buffer") < requiredBytes,
      "z_buffer backing storage is too small");
  MT_THROW_IF(
      availableBufferBytes(triangleIndexBuffer, triangleIndexInfo, "triangle_index_buffer") <
          requiredBytes,
      "triangle_index_buffer backing storage is too small");
  MT_THROW_IF(
      reinterpret_cast<uintptr_t>(zBufferInfo.ptr) % momentum::kSimdAlignment != 0 ||
          reinterpret_cast<uintptr_t>(triangleIndexInfo.ptr) % momentum::kSimdAlignment != 0 ||
          zBufferInfo.strides[0] % momentum::kSimdAlignment != 0 ||
          triangleIndexInfo.strides[0] % momentum::kSimdAlignment != 0 ||
          zBufferInfo.strides[1] % momentum::kSimdAlignment != 0 ||
          triangleIndexInfo.strides[1] % momentum::kSimdAlignment != 0,
      "Batched rasterizer outputs must be SIMD-aligned");

  if (positionsInfo.ptr == nullptr || zBufferInfo.ptr == nullptr ||
      triangleIndexInfo.ptr == nullptr ||
      (trianglesInfo.shape[0] > 0 && trianglesInfo.ptr == nullptr)) {
    MT_THROW("Batched rasterizer tensors must have storage");
  }

  auto* positionsData = static_cast<float*>(positionsInfo.ptr);
  auto* zBufferData = static_cast<float*>(zBufferInfo.ptr);
  auto* triangleIndexData = static_cast<int32_t*>(triangleIndexInfo.ptr);
  const int64_t positionStride = positionsInfo.strides[0] / static_cast<int64_t>(sizeof(float));
  const int64_t numPositionValues = positionsInfo.shape[1] * positionsInfo.shape[2];
  auto trianglesMap = bufferToEigenMap<int32_t>(trianglesInfo);
  const Eigen::Matrix4f modelMatrixValue = modelMatrix.value_or(Eigen::Matrix4f::Identity());
  const std::array<momentum::rasterizer::index_t, 2> imageExtents{imageHeight, imageWidth};
  const std::array<momentum::rasterizer::index_t, 2> imageStrides{rowStride, 1};
  Eigen::VectorXf emptyFloat;
  Eigen::VectorXi emptyInt;

  pybind11::gil_scoped_release release;
  // The waiting parallel_for overload captures worker exceptions in its
  // internal TaskSet and rethrows them on this calling thread.
  dispenso::parallel_for(int64_t(0), batchSize, [&](int64_t frame) {
    drjit::scoped_flush_denormals flushDenorm(true);
    Eigen::Map<Eigen::VectorXf> positionsMap(
        positionsData + frame * positionStride, numPositionValues);
    auto depth =
        make_mdspan<float, 2>(zBufferData + frame * imageStride, imageExtents, imageStrides);
    auto triangleIndex = make_mdspan<int32_t, 2>(
        triangleIndexData + frame * imageStride, imageExtents, imageStrides);
    momentum::rasterizer::rasterizeMesh(
        positionsMap,
        emptyFloat,
        trianglesMap,
        emptyFloat,
        emptyInt,
        emptyFloat,
        camera,
        modelMatrixValue,
        nearClip,
        momentum::rasterizer::PhongMaterial{},
        depth,
        {},
        {},
        {},
        triangleIndex,
        {},
        backfaceCulling,
        depthOffset,
        Eigen::Vector2f::Zero());
  });
}

void rasterizeWireframe(
    const ContiguousArray<float>& positions,
    const ContiguousArray<int32_t>& triangles,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nVertsBindingID = -1;
  const int nTrisBindingID = -2;

  ArrayChecker checker("rasterize_wireframe");
  checker.validateBuffer(positions, "positions", {nVertsBindingID, 3}, {"nVertices", "xyz"}, false);
  checker.validateBuffer(triangles, "triangles", {nTrisBindingID, 3}, {"nTriangles", "xyz"}, false);

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, {}, {}, {});

  // Get buffer infos for creating Eigen maps (requires GIL)
  auto positionsInfo = positions.request();
  auto trianglesInfo = triangles.request();

  auto positionsMap = bufferToEigenMap<float>(positionsInfo);
  auto trianglesMap = bufferToEigenMap<int32_t>(trianglesInfo);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  momentum::rasterizer::rasterizeWireframe(
      positionsMap,
      trianglesMap,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f{1, 1, 1}),
      thickness,
      zBufferMdspan,
      rgbBufferMdspan,
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeSpheres(
    const pybind11::buffer& center,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    std::optional<pybind11::buffer> radius,
    std::optional<pybind11::buffer> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int subdivisionLevel) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nSpheresBindingID = -1;

  ArrayChecker checker("rasterize_spheres");
  checker.validateBuffer(center, "center", {nSpheresBindingID, 3}, {"nSpheres", "xyz"}, false);

  if (radius.has_value()) {
    checker.validateBuffer(*radius, "radius", {nSpheresBindingID}, {"nSpheres"}, false);
  }

  if (color.has_value()) {
    checker.validateBuffer(*color, "color", {nSpheresBindingID, 3}, {"nSpheres", "rgb"}, false);
  }

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nSpheres = checker.getBoundValue(nSpheresBindingID);

  const auto sphereMesh = momentum::rasterizer::makeSphere(subdivisionLevel);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  VectorArrayAccessor<float, 3> centerAccessor(center, LeadingDimensions{}, nSpheres);
  auto centerView = centerAccessor.view({});

  std::optional<ScalarArrayAccessor<float>> radiusAccessor;
  if (radius.has_value()) {
    radiusAccessor.emplace(*radius, nSpheres);
  }

  std::optional<VectorArrayAccessor<float, 3>> colorAccessor;
  std::optional<typename VectorArrayAccessor<float, 3>::ElementView> colorView;
  if (color.has_value()) {
    colorAccessor.emplace(*color, LeadingDimensions{}, nSpheres);
    colorView = colorAccessor->view({});
  }

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  for (int i = 0; i < nSpheres; ++i) {
    const float radiusVal = radiusAccessor ? radiusAccessor->get(i) : 1.0f;
    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
    if (colorView) {
      materialCur.diffuseColor = colorView->get(i);
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(centerView.get(i));
    transform.scale(radiusVal);

    momentum::rasterizer::rasterizeMesh(
        sphereMesh,
        camera,
        modelMatrix * transform.matrix(),
        nearClip,
        materialCur,
        zBufferMdspan,
        rgbBufferMdspan,
        surfaceNormalsBufferMdspan,
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCylinders(
    const pybind11::buffer& start_position,
    const pybind11::buffer& end_position,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    std::optional<pybind11::buffer> radius,
    std::optional<pybind11::buffer> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int lengthSubdivisions,
    int radiusSubdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nCylindersBindingID = -1;

  ArrayChecker checker("rasterize_cylinders");
  checker.validateBuffer(
      start_position, "start_position", {nCylindersBindingID, 3}, {"nCylinders", "xyz"}, false);

  checker.validateBuffer(
      end_position, "end_position", {nCylindersBindingID, 3}, {"nCylinders", "xyz"}, false);

  if (radius.has_value()) {
    checker.validateBuffer(*radius, "radius", {nCylindersBindingID}, {"nCylinders"}, false);
  }

  if (color.has_value()) {
    checker.validateBuffer(*color, "color", {nCylindersBindingID, 3}, {"nCylinders", "rgb"}, false);
  }

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer);

  const auto nCylinders = checker.getBoundValue(nCylindersBindingID);

  const auto cylinderMesh =
      momentum::rasterizer::makeCylinder(radiusSubdivisions, lengthSubdivisions);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  VectorArrayAccessor<float, 3> startPosAccessor(start_position, LeadingDimensions{}, nCylinders);
  auto startPosView = startPosAccessor.view({});
  VectorArrayAccessor<float, 3> endPosAccessor(end_position, LeadingDimensions{}, nCylinders);
  auto endPosView = endPosAccessor.view({});

  std::optional<ScalarArrayAccessor<float>> radiusAccessor;
  if (radius.has_value()) {
    radiusAccessor.emplace(*radius, nCylinders);
  }

  std::optional<VectorArrayAccessor<float, 3>> colorAccessor;
  std::optional<typename VectorArrayAccessor<float, 3>::ElementView> colorView;
  if (color.has_value()) {
    colorAccessor.emplace(*color, LeadingDimensions{}, nCylinders);
    colorView = colorAccessor->view({});
  }

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  for (int i = 0; i < nCylinders; ++i) {
    const Eigen::Vector3f startPos = startPosView.get(i);
    const Eigen::Vector3f endPos = endPosView.get(i);
    const float radiusVal = radiusAccessor ? radiusAccessor->get(i) : 1.0f;
    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
    if (colorView) {
      materialCur.diffuseColor = colorView->get(i);
    }

    const Eigen::Vector3f direction = endPos - startPos;
    const float length = direction.norm();
    if (length <= 1e-8f) {
      continue;
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(startPos);
    transform.rotate(
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), direction.stableNormalized()));
    transform.scale(Eigen::Vector3f(length, radiusVal, radiusVal));

    momentum::rasterizer::rasterizeMesh(
        cylinderMesh,
        camera,
        modelMatrix * transform.matrix(),
        nearClip,
        materialCur,
        zBufferMdspan,
        rgbBufferMdspan,
        surfaceNormalsBufferMdspan,
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCapsules(
    const pybind11::buffer& transformation,
    const pybind11::buffer& radius,
    const pybind11::buffer& length,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nCapsulesBindingId = -1;

  ArrayChecker checker("rasterize_capsules");
  checker.validateBuffer(
      transformation,
      "transformation",
      {nCapsulesBindingId, 4, 4},
      {"nCapsules", "rows", "cols"},
      false);

  checker.validateBuffer(
      radius, "radius", {nCapsulesBindingId, 2}, {"nCapsules", "startEnd"}, false);

  checker.validateBuffer(length, "length", {nCapsulesBindingId}, {"nCapsules"}, false);

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer);

  const auto nCapsules = checker.getBoundValue(nCapsulesBindingId);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // Capsules are closed surfaces so we might as well always cull:
  const bool backfaceCulling = true;

  TransformAccessor<float> transformAccessor(transformation, LeadingDimensions{}, nCapsules);
  const auto capsuleMatrices = transformAccessor.getMatrices({});

  VectorArrayAccessor<float, 2> radiusAccessor(radius, LeadingDimensions{}, nCapsules);
  auto radiusView = radiusAccessor.view({});
  ScalarArrayAccessor<float> lengthAccessor(length, nCapsules);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  for (int i = 0; i < nCapsules; ++i) {
    const Eigen::Matrix4f& transform = capsuleMatrices[i];

    const Eigen::Vector2f radiusVal = radiusView.get(i);
    const float lengthVal = lengthAccessor.get(i);

    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());

    momentum::rasterizer::rasterizeMesh(
        momentum::rasterizer::makeCapsule(
            cylinderRadiusSubdivisions,
            cylinderLengthSubdivisions,
            radiusVal[0],
            radiusVal[1],
            lengthVal),
        camera,
        modelMatrix * transform,
        nearClip,
        materialCur,
        zBufferMdspan,
        rgbBufferMdspan,
        surfaceNormalsBufferMdspan,
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

namespace {

struct SkeletonRenderingAssets {
  momentum::rasterizer::Mesh jointMesh;
  std::vector<Eigen::Vector3f> jointLines;
  std::vector<Eigen::Vector3f> jointCircles;
  momentum::rasterizer::Mesh sphereMesh;
  float lineThickness{1.0f};
  float circleRadius{3.0f};
  Eigen::Vector3f lineColor{Eigen::Vector3f(1.0f, 0.0f, 0.0f)};
  Eigen::Vector3f circleColor{Eigen::Vector3f(1.0f, 0.0f, 0.0f)};
};

SkeletonRenderingAssets initSkeletonAssets(
    SkeletonStyle style,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    float sphereRadius,
    float cylinderRadius,
    int sphereSubdivisionLevel,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions) {
  SkeletonRenderingAssets assets;

  if (style == SkeletonStyle::Octahedrons) {
    std::tie(assets.jointMesh, assets.jointLines) = momentum::rasterizer::makeOctahedron(0.15, 0.1);
    assets.sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);

  } else if (style == SkeletonStyle::Lines) {
    for (size_t i = 0; i < static_cast<size_t>(cylinderLengthSubdivisions); ++i) {
      assets.jointLines.emplace_back(i / float(cylinderLengthSubdivisions), 0, 0);
      assets.jointLines.emplace_back((i + 1) / float(cylinderLengthSubdivisions), 0, 0);
    }
    assets.jointCircles.emplace_back(0, 0, 0);
  } else {
    assets.sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);
    assets.jointMesh =
        momentum::rasterizer::makeCylinder(cylinderRadiusSubdivisions, cylinderLengthSubdivisions);
  }

  assets.lineThickness =
      (style == SkeletonStyle::Lines) ? std::max(1.0f, 2.0f * cylinderRadius) : 1.0f;
  assets.circleRadius = std::max(1.0f, sphereRadius);

  assets.lineColor = (style == SkeletonStyle::Lines)
      ? cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor
      : Eigen::Vector3f::Constant(0.3f);
  assets.circleColor = jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor;

  return assets;
}

void rasterizeSkeletonJoints(
    const SkeletonRenderingAssets& assets,
    const std::vector<bool>& activeJoints,
    const momentum::TransformListT<float>& jointTransforms,
    const momentum::rasterizer::Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    rasterizer_detail::Span<float, 2> zBufferSpan,
    rasterizer_detail::Span<float, 3> rgbBufferSpan,
    rasterizer_detail::Span<float, 3> surfaceNormalsSpan,
    const std::vector<momentum::rasterizer::Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float sphereRadius) {
  const size_t nJoints = activeJoints.size();

  for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
    if (!activeJoints[jJoint]) {
      continue;
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(jointTransforms[jJoint].translation);
    transform.scale(sphereRadius);

    if (!assets.jointCircles.empty()) {
      momentum::rasterizer::rasterizeCircles(
          assets.jointCircles,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          assets.lineColor,
          assets.circleColor,
          0.5f * assets.lineThickness,
          assets.circleRadius,
          zBufferSpan,
          rgbBufferSpan,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }

    if (!assets.sphereMesh.vertices.empty()) {
      momentum::rasterizer::rasterizeMesh(
          assets.sphereMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
          zBufferSpan,
          rgbBufferSpan,
          surfaceNormalsSpan,
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

void rasterizeSkeletonBones(
    const SkeletonRenderingAssets& assets,
    SkeletonStyle style,
    const std::vector<std::pair<size_t, size_t>>& jointConnections,
    const momentum::TransformListT<float>& jointTransforms,
    const momentum::rasterizer::Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    rasterizer_detail::Span<float, 2> zBufferSpan,
    rasterizer_detail::Span<float, 3> rgbBufferSpan,
    rasterizer_detail::Span<float, 3> surfaceNormalsSpan,
    const std::vector<momentum::rasterizer::Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float cylinderRadius) {
  for (const auto& [childJoint, parentJoint] : jointConnections) {
    const auto& parentXF = jointTransforms[parentJoint];
    const auto& childXF = jointTransforms[childJoint];

    const Eigen::Vector3f diff = childXF.translation - parentXF.translation;
    const float length = diff.norm();
    if (length < 1e-6) {
      continue;
    }

    // x always points to the next joint, y and z axes are aligned
    // to the local y and z axes of the parent joint where possible:
    const Eigen::Vector3f xVec = diff.normalized();
    Eigen::Vector3f yVec = (parentXF.toLinear() * Eigen::Vector3f::UnitY()).normalized();
    Eigen::Vector3f zVec = (parentXF.toLinear() * Eigen::Vector3f::UnitZ()).normalized();
    yVec = xVec.cross(zVec).normalized();
    zVec = xVec.cross(yVec).normalized();

    Eigen::Matrix3f rotMat = Eigen::Matrix3f::Identity();
    rotMat.col(0) = xVec;
    rotMat.col(1) = yVec;
    rotMat.col(2) = zVec;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(parentXF.translation);
    transform.linear() = rotMat;

    if (style == SkeletonStyle::Octahedrons) {
      transform.scale(Eigen::Vector3f::Constant(length));
    } else {
      transform.scale(Eigen::Vector3f(length, cylinderRadius, cylinderRadius));
    }

    if (!assets.jointMesh.vertices.empty()) {
      momentum::rasterizer::rasterizeMesh(
          assets.jointMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
          zBufferSpan,
          rgbBufferSpan,
          surfaceNormalsSpan,
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }

    if (!assets.jointLines.empty()) {
      momentum::rasterizer::rasterizeLines(
          assets.jointLines,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          assets.lineColor,
          assets.lineThickness,
          zBufferSpan,
          rgbBufferSpan,
          depthOffset - 0.001f,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

} // namespace

void rasterizeSkeleton(
    const momentum::Character& character,
    const pybind11::buffer& skeletonState,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    std::optional<pybind11::buffer> activeJoints_in,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float sphereRadius,
    float cylinderRadius,
    int sphereSubdivisionLevel,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions,
    SkeletonStyle style) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  ArrayChecker checker("rasterize_skeleton");
  checker.validateBuffer(
      skeletonState,
      "skeleton_state",
      {(int)character.skeleton.joints.size(), 8},
      {"nJoints", "pos+rot+scale"},
      false);

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer);

  const auto nJoints = character.skeleton.joints.size();

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  std::vector<bool> activeJoints(nJoints);
  if (activeJoints_in.has_value()) {
    auto activeInfo = activeJoints_in->request();
    if (activeInfo.ndim != 1 || static_cast<size_t>(activeInfo.shape[0]) != nJoints) {
      throw std::runtime_error(
          fmt::format(
              "active_joints must be a 1D array with {} elements but got shape {}",
              nJoints,
              formatArrayDims(activeInfo.shape)));
    }
    if (activeInfo.itemsize != 1) {
      throw std::runtime_error(
          fmt::format(
              "active_joints must be a bool or uint8 array but got itemsize={}",
              activeInfo.itemsize));
    }
    auto* ptr = static_cast<uint8_t*>(activeInfo.ptr);
    for (size_t k = 0; k < nJoints; ++k) {
      activeJoints[k] = (ptr[k] != 0);
    }
  } else {
    // Make all joints active except b_world, which we usually don't want to
    // render because it's confusing:
    for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
      activeJoints[i] = (character.skeleton.joints[i].name != "body_world");
    }
  }

  // Now construct the parent relationships:
  std::vector<std::pair<size_t, size_t>> jointConnections;
  for (size_t iJoint = 0; iJoint < character.skeleton.joints.size(); ++iJoint) {
    if (!activeJoints[iJoint]) {
      continue;
    }

    size_t grandparent = character.skeleton.joints[iJoint].parent;
    while (grandparent != momentum::kInvalidIndex) {
      MT_THROW_IF(
          grandparent >= character.skeleton.joints.size(),
          "Grandparent joint index {} exceeds skeleton size {}",
          grandparent,
          character.skeleton.joints.size());
      if (activeJoints[grandparent]) {
        break;
      }
      grandparent = character.skeleton.joints[grandparent].parent;
    }

    if (grandparent != momentum::kInvalidIndex) {
      jointConnections.emplace_back(iJoint, grandparent);
    }
  }

  const auto assets = initSkeletonAssets(
      style,
      jointMaterial,
      cylinderMaterial,
      sphereRadius,
      cylinderRadius,
      sphereSubdivisionLevel,
      cylinderLengthSubdivisions,
      cylinderRadiusSubdivisions);

  // Use SkeletonStateAccessor to parse joint transforms from the buffer (requires GIL)
  SkeletonStateAccessor<float> skelStateAccessor(skeletonState, LeadingDimensions{}, nJoints);
  const auto jointTransforms = skelStateAccessor.getTransforms({});

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  rasterizeSkeletonJoints(
      assets,
      activeJoints,
      jointTransforms,
      camera,
      modelMatrix,
      nearClip,
      jointMaterial,
      zBufferMdspan,
      rgbBufferMdspan,
      surfaceNormalsBufferMdspan,
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset,
      sphereRadius);

  rasterizeSkeletonBones(
      assets,
      style,
      jointConnections,
      jointTransforms,
      camera,
      modelMatrix,
      nearClip,
      cylinderMaterial,
      zBufferMdspan,
      rgbBufferMdspan,
      surfaceNormalsBufferMdspan,
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset,
      cylinderRadius);
}

void rasterizeCharacter(
    const momentum::Character& character,
    const pybind11::buffer& skeletonState,
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<pybind11::buffer>& vertexIndexBuffer,
    const std::optional<pybind11::buffer>& triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<pybind11::buffer> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    std::optional<Eigen::Vector3f> wireframeColor) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (!character.skinWeights || !character.mesh) {
    throw std::runtime_error("Mesh is missing from character.");
  }

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  ArrayChecker checker("rasterize_character");
  checker.validateBuffer(
      skeletonState,
      "skeleton_state",
      {(int)character.skeleton.joints.size(), 8},
      {"nJoints", "pos+rot+scale"},
      false);

  if (perVertexDiffuseColor.has_value()) {
    checker.validateBuffer(
        *perVertexDiffuseColor,
        "per_vertex_diffuse_color",
        {(int)character.mesh->vertices.size(), 3},
        {"nVertices", "rgb"},
        false);
  }

  validateRasterizerBuffers(
      camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  const auto nJoints = character.skeleton.joints.size();

  // Use SkeletonStateAccessor to parse joint transforms from the buffer (requires GIL)
  SkeletonStateAccessor<float> skelStateAccessor(skeletonState, LeadingDimensions{}, nJoints);
  const auto jointTransforms = skelStateAccessor.getTransforms({});

  std::optional<py::buffer_info> perVertexColorInfo;
  if (perVertexDiffuseColor.has_value()) {
    perVertexColorInfo = perVertexDiffuseColor->request();
  }
  Eigen::VectorXf emptyFloatVec;
  auto perVertexColorMap = bufferToEigenMap<float>(perVertexColorInfo, emptyFloatVec);

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);
  auto vertexIndexBufferMdspan = make_mdspan<int, 2>(vertexIndexBuffer);
  auto triangleIndexBufferMdspan = make_mdspan<int, 2>(triangleIndexBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  std::vector<momentum::JointState> jointStates(nJoints);
  for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
    jointStates[jJoint].transform = jointTransforms[jJoint];
  }

  momentum::Mesh posedMesh = *character.mesh;
  momentum::applySSD(
      character.inverseBindPose, *character.skinWeights, *character.mesh, jointStates, posedMesh);

  // Don't trust the skinned normals:
  posedMesh.updateNormals();

  momentum::rasterizer::rasterizeMesh(
      posedMesh.vertices,
      posedMesh.normals,
      posedMesh.faces,
      posedMesh.texcoords,
      posedMesh.texcoord_faces,
      perVertexColorMap,
      camera,
      modelMatrix,
      nearClip,
      material.value_or(momentum::rasterizer::PhongMaterial{}),
      zBufferMdspan,
      rgbBufferMdspan,
      surfaceNormalsBufferMdspan,
      vertexIndexBufferMdspan,
      triangleIndexBufferMdspan,
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));

  if (wireframeColor.has_value()) {
    momentum::rasterizer::rasterizeWireframe(
        posedMesh.vertices,
        posedMesh.faces,
        camera,
        modelMatrix,
        nearClip,
        *wireframeColor,
        1.0f,
        zBufferMdspan,
        rgbBufferMdspan,
        backfaceCulling,
        depthOffset - 0.001,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCheckerboard(
    const momentum::rasterizer::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material1_in,
    const std::optional<momentum::rasterizer::PhongMaterial>& material2_in,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numChecks,
    int subdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (numChecks <= 0) {
    throw std::runtime_error("num_checks should be positive.");
  }

  if (width <= 0) {
    throw std::runtime_error("width should be positive.");
  }

  if (subdivisions < 1) {
    throw std::runtime_error("subdivisions should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  validateRasterizerBuffers(camera, zBuffer, rgbBuffer, surfaceNormalsBuffer);

  const auto& meshes = momentum::rasterizer::makeCheckerboard(width, numChecks, subdivisions);

  // Simply white/gray scheme.  All-white or all-black tends to look bad in the
  // render:
  std::array<momentum::rasterizer::PhongMaterial, 2> materials = {
      material1_in.value_or(momentum::rasterizer::PhongMaterial{Eigen::Vector3f(0.8f, 0.8f, 0.8f)}),
      material2_in.value_or(
          momentum::rasterizer::PhongMaterial{Eigen::Vector3f(0.5f, 0.5f, 0.5f)})};

  // Create mdspans while holding GIL
  auto zBufferMdspan = make_mdspan<float, 2>(zBuffer);
  auto rgbBufferMdspan = make_mdspan<float, 3>(rgbBuffer);
  auto surfaceNormalsBufferMdspan = make_mdspan<float, 3>(surfaceNormalsBuffer);

  // Release GIL before rendering
  pybind11::gil_scoped_release release;

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (int k = 0; k < 2; ++k) {
    momentum::rasterizer::rasterizeMesh(
        meshes[k],
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        materials[k],
        zBufferMdspan,
        rgbBufferMdspan,
        surfaceNormalsBufferMdspan,
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

} // namespace pymomentum
