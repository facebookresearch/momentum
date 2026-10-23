/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/torch_bridge.h"

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/cuda_rasterizer.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace py = pybind11;

namespace pymomentum {
namespace {

void validateInputTensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalarType,
    int64_t secondDimension) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == scalarType, name, " must have dtype ", c10::toString(scalarType));
  TORCH_CHECK(tensor.dim() == 2, name, " must have rank 2");
  TORCH_CHECK(
      tensor.size(1) == secondDimension, name, " must have shape [n, ", secondDimension, "]");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

size_t availableStorageElements(const at::Tensor& tensor, const char* name) {
  const size_t elementSize = tensor.element_size();
  const size_t storageBytes = tensor.storage().nbytes();
  TORCH_CHECK(
      elementSize > 0 && storageBytes % elementSize == 0, name, " has invalid backing storage");
  const int64_t storageOffset = tensor.storage_offset();
  const size_t storageElements = storageBytes / elementSize;
  TORCH_CHECK(
      storageOffset >= 0 && static_cast<size_t>(storageOffset) <= storageElements,
      name,
      " has an invalid storage offset");
  return storageElements - static_cast<size_t>(storageOffset);
}

void validateBatchedVertexTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype Float");
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) > 0 && tensor.size(2) == 3,
      name,
      " must have shape [batch, n, 3] with a non-empty batch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

size_t validateOutputTensor(
    const at::Tensor& tensor,
    const char* name,
    const at::Device& device,
    at::ScalarType scalarType,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(tensor.device() == device, name, " must be on the same CUDA device as the vertices");
  TORCH_CHECK(
      tensor.scalar_type() == scalarType, name, " must have dtype ", c10::toString(scalarType));
  TORCH_CHECK(tensor.dim() == 2, name, " must have rank 2");
  TORCH_CHECK(tensor.size(0) == height && tensor.size(1) == width, name, " has the wrong shape");
  TORCH_CHECK(
      tensor.stride(1) == 1 && tensor.stride(0) >= width, name, " has an invalid row stride");
  TORCH_CHECK(height > 0 && width > 0, "camera image must be non-empty");
  const auto rowStride = static_cast<size_t>(tensor.stride(0));
  const auto imageHeight = static_cast<size_t>(height);
  TORCH_CHECK(
      rowStride <= std::numeric_limits<size_t>::max() / imageHeight,
      name,
      " required storage size overflows");
  const size_t requiredElements = imageHeight * rowStride;
  const size_t availableElements = availableStorageElements(tensor, name);
  TORCH_CHECK(
      availableElements >= requiredElements,
      name,
      " backing storage has ",
      availableElements,
      " elements from the data pointer; at least ",
      requiredElements,
      " are required by its shape and strides");
  return availableElements;
}

size_t validateBatchedOutputTensor(
    const at::Tensor& tensor,
    const char* name,
    const at::Device& device,
    at::ScalarType scalarType,
    int64_t batchSize,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(tensor.device() == device, name, " must be on the same CUDA device as the vertices");
  TORCH_CHECK(
      tensor.scalar_type() == scalarType, name, " must have dtype ", c10::toString(scalarType));
  TORCH_CHECK(tensor.dim() == 3, name, " must have rank 3");
  TORCH_CHECK(
      tensor.size(0) == batchSize && tensor.size(1) == height && tensor.size(2) == width,
      name,
      " has the wrong shape");
  TORCH_CHECK(batchSize > 0 && height > 0 && width > 0, "camera image and batch must be non-empty");
  TORCH_CHECK(
      tensor.stride(2) == 1 && tensor.stride(1) >= width, name, " has an invalid row stride");
  const auto rowStride = static_cast<size_t>(tensor.stride(1));
  const auto imageHeight = static_cast<size_t>(height);
  TORCH_CHECK(
      rowStride <= std::numeric_limits<size_t>::max() / imageHeight,
      name,
      " required storage size overflows");
  const size_t imageRowsSize = imageHeight * rowStride;
  TORCH_CHECK(
      tensor.stride(0) >= 0 && static_cast<size_t>(tensor.stride(0)) >= imageRowsSize,
      name,
      " has an invalid image stride");
  const auto imageStride = static_cast<size_t>(tensor.stride(0));
  const auto batchCount = static_cast<size_t>(batchSize);
  TORCH_CHECK(
      batchCount - 1 <= (std::numeric_limits<size_t>::max() - imageRowsSize) / imageStride,
      name,
      " required storage size overflows");
  const size_t requiredElements = (batchCount - 1) * imageStride + imageRowsSize;
  const size_t availableElements = availableStorageElements(tensor, name);
  TORCH_CHECK(
      availableElements >= requiredElements,
      name,
      " backing storage has ",
      availableElements,
      " elements from the data pointer; at least ",
      requiredElements,
      " are required by its shape and strides");
  return availableElements;
}

size_t validateVectorOutputTensor(
    const at::Tensor& tensor,
    const char* name,
    const at::Device& device,
    int64_t height,
    int64_t width,
    int64_t depthRowStride) {
  TORCH_CHECK(tensor.device() == device, name, " must be on the CUDA input device");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype Float");
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) == height && tensor.size(1) == width &&
          tensor.size(2) == 3,
      name,
      " must have shape [height, width, 3]");
  TORCH_CHECK(
      depthRowStride >= 0 && depthRowStride <= std::numeric_limits<int64_t>::max() / 3,
      name,
      " required storage size overflows");
  TORCH_CHECK(
      tensor.stride(2) == 1 && tensor.stride(1) == 3 && tensor.stride(0) == 3 * depthRowStride,
      name,
      " must use the same padded rows as z_buffer");
  const auto rowStride = static_cast<size_t>(depthRowStride);
  const auto imageHeight = static_cast<size_t>(height);
  TORCH_CHECK(
      rowStride <= std::numeric_limits<size_t>::max() / imageHeight &&
          imageHeight * rowStride <= std::numeric_limits<size_t>::max() / 3,
      name,
      " required storage size overflows");
  const size_t requiredElements = 3 * imageHeight * rowStride;
  const size_t availableElements = availableStorageElements(tensor, name);
  TORCH_CHECK(
      availableElements >= requiredElements,
      name,
      " backing storage has ",
      availableElements,
      " elements from the data pointer; at least ",
      requiredElements,
      " are required by its shape and strides");
  return availableElements;
}

size_t validateBatchedVectorOutputTensor(
    const at::Tensor& tensor,
    const char* name,
    const at::Device& device,
    int64_t batchSize,
    int64_t height,
    int64_t width,
    int64_t depthImageStride,
    int64_t depthRowStride) {
  TORCH_CHECK(tensor.device() == device, name, " must be on the CUDA input device");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype Float");
  TORCH_CHECK(
      tensor.dim() == 4 && tensor.size(0) == batchSize && tensor.size(1) == height &&
          tensor.size(2) == width && tensor.size(3) == 3,
      name,
      " must have shape [batch, height, width, 3]");
  TORCH_CHECK(
      depthRowStride >= 0 && depthImageStride >= 0 &&
          depthRowStride <= std::numeric_limits<int64_t>::max() / 3 &&
          depthImageStride <= std::numeric_limits<int64_t>::max() / 3,
      name,
      " required storage size overflows");
  TORCH_CHECK(
      tensor.stride(3) == 1 && tensor.stride(2) == 3 && tensor.stride(1) == 3 * depthRowStride &&
          tensor.stride(0) == 3 * depthImageStride,
      name,
      " must use the same padded images as z_buffer");
  const auto rowStride = static_cast<size_t>(depthRowStride);
  const auto imageStride = static_cast<size_t>(depthImageStride);
  const auto imageHeight = static_cast<size_t>(height);
  const auto batchCount = static_cast<size_t>(batchSize);
  TORCH_CHECK(
      rowStride <= std::numeric_limits<size_t>::max() / imageHeight,
      name,
      " required storage size overflows");
  const size_t imageRowsSize = imageHeight * rowStride;
  TORCH_CHECK(
      batchCount - 1 <= (std::numeric_limits<size_t>::max() - imageRowsSize) / imageStride,
      name,
      " required storage size overflows");
  const size_t requiredScalarElements = (batchCount - 1) * imageStride + imageRowsSize;
  TORCH_CHECK(
      requiredScalarElements <= std::numeric_limits<size_t>::max() / 3,
      name,
      " required storage size overflows");
  const size_t requiredElements = 3 * requiredScalarElements;
  const size_t availableElements = availableStorageElements(tensor, name);
  TORCH_CHECK(
      availableElements >= requiredElements,
      name,
      " backing storage has ",
      availableElements,
      " elements from the data pointer; at least ",
      requiredElements,
      " are required by its shape and strides");
  return availableElements;
}

std::tuple<at::Tensor, std::optional<at::Tensor>> rasterizeMeshCuda(
    const at::Tensor& vertexPositions,
    const at::Tensor& triangles,
    const momentum::Camera& camera,
    const std::optional<at::Tensor>& vertexNormals,
    const std::optional<at::Tensor>& zBuffer,
    const std::optional<at::Tensor>& triangleIndexBuffer,
    const std::optional<at::Tensor>& vertexIndexBuffer,
    const std::optional<at::Tensor>& surfaceNormalsBuffer,
    const std::optional<at::Tensor>& barycentricCoordinatesBuffer,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  validateInputTensor(vertexPositions, "vertex_positions", at::kFloat, 3);
  validateInputTensor(triangles, "triangles", at::kInt, 3);
  TORCH_CHECK(
      triangles.device() == vertexPositions.device(),
      "triangles must be on the same CUDA device as vertex_positions");
  if (vertexNormals.has_value()) {
    validateInputTensor(*vertexNormals, "vertex_normals", at::kFloat, 3);
    TORCH_CHECK(
        vertexNormals->device() == vertexPositions.device(),
        "vertex_normals must be on the same CUDA device as vertex_positions");
    TORCH_CHECK(
        vertexNormals->size(0) == vertexPositions.size(0),
        "vertex_normals must match vertex_positions shape");
  }
  TORCH_CHECK(nearClip > 0.0f, "near_clip must be positive");

  const int64_t height = camera.imageHeight();
  const int64_t width = camera.imageWidth();
  TORCH_CHECK(height > 0 && width > 0, "camera image must be non-empty");
  const at::Device device = vertexPositions.device();
  const c10::cuda::CUDAGuard deviceGuard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  const auto floatOptions = vertexPositions.options().dtype(at::kFloat);

  const int64_t paddedWidth = ((width + 7) / 8) * 8;
  at::Tensor depth = zBuffer.has_value()
      ? *zBuffer
      : at::full({height, paddedWidth}, std::numeric_limits<float>::max(), floatOptions)
            .narrow(1, 0, width);
  std::optional<at::Tensor> triangleIndex = triangleIndexBuffer;
  const size_t depthBufferSize =
      validateOutputTensor(depth, "z_buffer", device, at::kFloat, height, width);
  size_t triangleIndexBufferSize = 0;
  if (triangleIndex.has_value()) {
    triangleIndexBufferSize = validateOutputTensor(
        *triangleIndex, "triangle_index_buffer", device, at::kInt, height, width);
    TORCH_CHECK(
        triangleIndex->strides() == depth.strides(),
        "triangle_index_buffer must have the same strides as z_buffer");
  }
  size_t vertexIndexBufferSize = 0;
  if (vertexIndexBuffer.has_value()) {
    vertexIndexBufferSize = validateOutputTensor(
        *vertexIndexBuffer, "vertex_index_buffer", device, at::kInt, height, width);
    TORCH_CHECK(
        vertexIndexBuffer->strides() == depth.strides(),
        "vertex_index_buffer must have the same strides as z_buffer");
  }
  size_t surfaceNormalsBufferSize = 0;
  if (surfaceNormalsBuffer.has_value()) {
    surfaceNormalsBufferSize = validateVectorOutputTensor(
        *surfaceNormalsBuffer, "surface_normals_buffer", device, height, width, depth.stride(0));
  }
  size_t barycentricCoordinatesBufferSize = 0;
  if (barycentricCoordinatesBuffer.has_value()) {
    barycentricCoordinatesBufferSize = validateVectorOutputTensor(
        *barycentricCoordinatesBuffer,
        "barycentric_coordinates_buffer",
        device,
        height,
        width,
        depth.stride(0));
  }

  momentum::rasterizer::rasterizeMeshCuda(
      momentum::rasterizer::CudaRasterizerInputs{
          .positionsModel = vertexPositions.const_data_ptr<float>(),
          .positionsModelSize = static_cast<size_t>(vertexPositions.numel()),
          .normalsModel =
              vertexNormals.has_value() ? vertexNormals->const_data_ptr<float>() : nullptr,
          .normalsModelSize =
              vertexNormals.has_value() ? static_cast<size_t>(vertexNormals->numel()) : 0,
          .numVertices = static_cast<size_t>(vertexPositions.size(0)),
          .triangles = triangles.const_data_ptr<int32_t>(),
          .trianglesSize = static_cast<size_t>(triangles.numel()),
          .numTriangles = static_cast<size_t>(triangles.size(0)),
      },
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      momentum::rasterizer::CudaRasterizerBuffers{
          .zBuffer = depth.mutable_data_ptr<float>(),
          .zBufferSize = depthBufferSize,
          .triangleIndexBuffer =
              triangleIndex.has_value() ? triangleIndex->mutable_data_ptr<int32_t>() : nullptr,
          .triangleIndexBufferSize = triangleIndexBufferSize,
          .vertexIndexBuffer = vertexIndexBuffer.has_value()
              ? vertexIndexBuffer->mutable_data_ptr<int32_t>()
              : nullptr,
          .vertexIndexBufferSize = vertexIndexBufferSize,
          .surfaceNormalsBuffer = surfaceNormalsBuffer.has_value()
              ? surfaceNormalsBuffer->mutable_data_ptr<float>()
              : nullptr,
          .surfaceNormalsBufferSize = surfaceNormalsBufferSize,
          .barycentricCoordinatesBuffer = barycentricCoordinatesBuffer.has_value()
              ? barycentricCoordinatesBuffer->mutable_data_ptr<float>()
              : nullptr,
          .barycentricCoordinatesBufferSize = barycentricCoordinatesBufferSize,
          .rowStride = depth.stride(0),
      },
      backfaceCulling,
      depthOffset,
      stream);

  return {std::move(depth), std::move(triangleIndex)};
}

std::tuple<at::Tensor, std::optional<at::Tensor>> rasterizeMeshesCuda(
    const at::Tensor& vertexPositions,
    const at::Tensor& triangles,
    const momentum::Camera& camera,
    const std::optional<at::Tensor>& vertexNormals,
    const std::optional<at::Tensor>& zBuffer,
    const std::optional<at::Tensor>& triangleIndexBuffer,
    const std::optional<at::Tensor>& vertexIndexBuffer,
    const std::optional<at::Tensor>& surfaceNormalsBuffer,
    const std::optional<at::Tensor>& barycentricCoordinatesBuffer,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  validateBatchedVertexTensor(vertexPositions, "vertex_positions");
  validateInputTensor(triangles, "triangles", at::kInt, 3);
  TORCH_CHECK(
      triangles.device() == vertexPositions.device(),
      "triangles must be on the same CUDA device as vertex_positions");
  if (vertexNormals.has_value()) {
    validateBatchedVertexTensor(*vertexNormals, "vertex_normals");
    TORCH_CHECK(
        vertexNormals->device() == vertexPositions.device(),
        "vertex_normals must be on the same CUDA device as vertex_positions");
    TORCH_CHECK(
        vertexNormals->sizes() == vertexPositions.sizes(),
        "vertex_normals must match vertex_positions shape");
  }
  TORCH_CHECK(nearClip > 0.0f, "near_clip must be positive");

  const int64_t batchSize = vertexPositions.size(0);
  const int64_t height = camera.imageHeight();
  const int64_t width = camera.imageWidth();
  TORCH_CHECK(height > 0 && width > 0, "camera image must be non-empty");
  const int64_t paddedWidth = ((width + 7) / 8) * 8;
  const at::Device device = vertexPositions.device();
  const c10::cuda::CUDAGuard deviceGuard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  const auto floatOptions = vertexPositions.options().dtype(at::kFloat);

  at::Tensor depth = zBuffer.has_value()
      ? *zBuffer
      : at::full({batchSize, height, paddedWidth}, std::numeric_limits<float>::max(), floatOptions)
            .narrow(2, 0, width);
  std::optional<at::Tensor> triangleIndex = triangleIndexBuffer;
  const size_t depthBufferSize =
      validateBatchedOutputTensor(depth, "z_buffer", device, at::kFloat, batchSize, height, width);
  size_t triangleIndexBufferSize = 0;
  if (triangleIndex.has_value()) {
    triangleIndexBufferSize = validateBatchedOutputTensor(
        *triangleIndex, "triangle_index_buffer", device, at::kInt, batchSize, height, width);
    TORCH_CHECK(
        triangleIndex->strides() == depth.strides(),
        "triangle_index_buffer must have the same strides as z_buffer");
  }
  size_t vertexIndexBufferSize = 0;
  if (vertexIndexBuffer.has_value()) {
    vertexIndexBufferSize = validateBatchedOutputTensor(
        *vertexIndexBuffer, "vertex_index_buffer", device, at::kInt, batchSize, height, width);
    TORCH_CHECK(
        vertexIndexBuffer->strides() == depth.strides(),
        "vertex_index_buffer must have the same strides as z_buffer");
  }
  size_t surfaceNormalsBufferSize = 0;
  if (surfaceNormalsBuffer.has_value()) {
    surfaceNormalsBufferSize = validateBatchedVectorOutputTensor(
        *surfaceNormalsBuffer,
        "surface_normals_buffer",
        device,
        batchSize,
        height,
        width,
        depth.stride(0),
        depth.stride(1));
  }
  size_t barycentricCoordinatesBufferSize = 0;
  if (barycentricCoordinatesBuffer.has_value()) {
    barycentricCoordinatesBufferSize = validateBatchedVectorOutputTensor(
        *barycentricCoordinatesBuffer,
        "barycentric_coordinates_buffer",
        device,
        batchSize,
        height,
        width,
        depth.stride(0),
        depth.stride(1));
  }

  momentum::rasterizer::rasterizeMeshesCuda(
      momentum::rasterizer::CudaRasterizerInputs{
          .positionsModel = vertexPositions.const_data_ptr<float>(),
          .positionsModelSize = static_cast<size_t>(vertexPositions.numel()),
          .normalsModel =
              vertexNormals.has_value() ? vertexNormals->const_data_ptr<float>() : nullptr,
          .normalsModelSize =
              vertexNormals.has_value() ? static_cast<size_t>(vertexNormals->numel()) : 0,
          .numVertices = static_cast<size_t>(vertexPositions.size(1)),
          .triangles = triangles.const_data_ptr<int32_t>(),
          .trianglesSize = static_cast<size_t>(triangles.numel()),
          .numTriangles = static_cast<size_t>(triangles.size(0)),
      },
      static_cast<size_t>(batchSize),
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      momentum::rasterizer::CudaBatchedRasterizerBuffers{
          .zBuffer = depth.mutable_data_ptr<float>(),
          .zBufferSize = depthBufferSize,
          .triangleIndexBuffer =
              triangleIndex.has_value() ? triangleIndex->mutable_data_ptr<int32_t>() : nullptr,
          .triangleIndexBufferSize = triangleIndexBufferSize,
          .vertexIndexBuffer = vertexIndexBuffer.has_value()
              ? vertexIndexBuffer->mutable_data_ptr<int32_t>()
              : nullptr,
          .vertexIndexBufferSize = vertexIndexBufferSize,
          .surfaceNormalsBuffer = surfaceNormalsBuffer.has_value()
              ? surfaceNormalsBuffer->mutable_data_ptr<float>()
              : nullptr,
          .surfaceNormalsBufferSize = surfaceNormalsBufferSize,
          .barycentricCoordinatesBuffer = barycentricCoordinatesBuffer.has_value()
              ? barycentricCoordinatesBuffer->mutable_data_ptr<float>()
              : nullptr,
          .barycentricCoordinatesBufferSize = barycentricCoordinatesBufferSize,
          .rowStride = depth.stride(1),
          .imageStride = depth.stride(0),
      },
      backfaceCulling,
      depthOffset,
      stream);

  return {std::move(depth), std::move(triangleIndex)};
}

} // namespace
} // namespace pymomentum

PYBIND11_MODULE(_renderer_cuda, module) {
  module.attr("__name__") = "pymomentum._renderer_cuda";
  module.doc() = "CUDA rasterization functions for Momentum meshes.";

  py::module_::import("torch");
  py::module_::import("pymomentum.camera");

  module.def(
      "rasterize_mesh",
      &pymomentum::rasterizeMeshCuda,
      R"(Rasterize a triangle mesh into CUDA-resident output tensors.

This non-differentiable CUDA path atomically updates depth on the current PyTorch CUDA stream. It launches a second pass only when a triangle-index, vertex-index, surface-normal, or barycentric-coordinate output is provided. For non-tied depths, every requested attribute corresponds to finalized depth; exact-depth ties are unspecified and may mix attributes from tied triangles. Callers must establish stream ordering when inputs were produced on a different stream.

:param vertex_positions: Contiguous CUDA float32 tensor with shape ``(n_vertices, 3)``.
:param triangles: Contiguous CUDA int32 tensor with shape ``(n_triangles, 3)``.
:param camera: Camera used to transform and project vertices.
:param vertex_normals: Optional contiguous CUDA float32 tensor matching ``vertex_positions``. Face normals are used when omitted.
:param z_buffer: Optional CUDA float32 tensor with shape ``(height, width)`` and unit column stride. Existing depth values participate in the depth test.
:param triangle_index_buffer: Optional CUDA int32 tensor with the same shape and row stride as ``z_buffer``. No triangle-index tensor is allocated when omitted.
:param vertex_index_buffer: Optional CUDA int32 output tensor with the same shape and strides as ``z_buffer``.
:param surface_normals_buffer: Optional CUDA float32 output tensor with shape ``(height, width, 3)`` and matching padded rows.
:param barycentric_coordinates_buffer: Optional CUDA float32 output tensor with shape ``(height, width, 3)`` and matching padded rows. Stores perspective-correct triangle barycentric coordinates.
:param model_matrix: Optional model transformation matrix.
:param back_face_culling: Whether to discard back-facing triangles.
:param near_clip: Per-pixel near clipping distance.
:param depth_offset: Value added to rasterized depth before depth testing.
:return: Tuple containing depth and the provided triangle-index tensor, or ``None`` when omitted.
)",
      py::arg("vertex_positions"),
      py::arg("triangles"),
      py::arg("camera"),
      py::kw_only(),
      py::arg("vertex_normals") = std::nullopt,
      py::arg("z_buffer") = std::nullopt,
      py::arg("triangle_index_buffer") = std::nullopt,
      py::arg("vertex_index_buffer") = std::nullopt,
      py::arg("surface_normals_buffer") = std::nullopt,
      py::arg("barycentric_coordinates_buffer") = std::nullopt,
      py::arg("model_matrix") = std::nullopt,
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f);

  module.def(
      "rasterize_meshes",
      &pymomentum::rasterizeMeshesCuda,
      R"(Rasterize a batch of triangle meshes with shared topology and camera.

This non-differentiable CUDA path renders the entire batch with one depth kernel launch on the current PyTorch CUDA stream. It launches a second batched pass only when a triangle-index, vertex-index, surface-normal, or barycentric-coordinate output is provided. For non-tied depths, every requested attribute corresponds to finalized depth; exact-depth ties are unspecified. The triangle topology, camera, and model matrix are shared across the batch.

:param vertex_positions: Contiguous CUDA float32 tensor with shape ``(batch, n_vertices, 3)``.
:param triangles: Contiguous CUDA int32 tensor with shape ``(n_triangles, 3)`` shared by the batch.
:param camera: Camera shared by every batch entry.
:param vertex_normals: Optional contiguous CUDA float32 tensor matching ``vertex_positions``. Face normals are used when omitted.
:param z_buffer: Optional CUDA float32 tensor with shape ``(batch, height, width)`` and unit column stride. Existing depth values participate in the depth test.
:param triangle_index_buffer: Optional CUDA int32 tensor with the same shape and strides as ``z_buffer``. No tensor is allocated when omitted.
:param vertex_index_buffer: Optional CUDA int32 output tensor with the same shape and strides as ``z_buffer``.
:param surface_normals_buffer: Optional CUDA float32 output tensor with shape ``(batch, height, width, 3)`` and matching padded images.
:param barycentric_coordinates_buffer: Optional CUDA float32 output tensor with shape ``(batch, height, width, 3)`` and matching padded images. Stores perspective-correct triangle barycentric coordinates.
:param model_matrix: Optional model transformation matrix shared by every batch entry.
:param back_face_culling: Whether to discard back-facing triangles.
:param near_clip: Per-pixel near clipping distance.
:param depth_offset: Value added to rasterized depth before depth testing.
:return: Tuple containing batched depth and the provided triangle-index tensor, or ``None`` when omitted.
)",
      py::arg("vertex_positions"),
      py::arg("triangles"),
      py::arg("camera"),
      py::kw_only(),
      py::arg("vertex_normals") = std::nullopt,
      py::arg("z_buffer") = std::nullopt,
      py::arg("triangle_index_buffer") = std::nullopt,
      py::arg("vertex_index_buffer") = std::nullopt,
      py::arg("surface_normals_buffer") = std::nullopt,
      py::arg("barycentric_coordinates_buffer") = std::nullopt,
      py::arg("model_matrix") = std::nullopt,
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f);
}
