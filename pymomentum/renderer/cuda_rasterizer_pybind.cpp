/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/torch_bridge.h"

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/cuda_rasterizer.h>
#include <momentum/rasterizer/rasterizer.h>

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
#include <utility>
#include <vector>

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

void validateDistinctOutputStorage(
    const std::vector<std::pair<const char*, const at::Tensor*>>& outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j) {
      TORCH_CHECK(
          !outputs[i].second->storage().is_alias_of(outputs[j].second->storage()),
          outputs[i].first,
          " and ",
          outputs[j].first,
          " must not share backing storage");
    }
  }
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

void rasterizeMeshCuda(
    const at::Tensor& vertexPositions,
    const std::optional<at::Tensor>& vertexNormals,
    const at::Tensor& triangles,
    const momentum::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    std::optional<at::Tensor> vertexIndexBuffer,
    std::optional<at::Tensor> triangleIndexBuffer,
    const momentum::rasterizer::PhongMaterial& material,
    const std::optional<at::Tensor>& textureCoordinates,
    const std::optional<at::Tensor>& textureTriangles,
    const std::optional<at::Tensor>& perVertexDiffuseColor,
    const std::optional<at::Tensor>& diffuseTexture,
    const std::optional<at::Tensor>& emissiveTexture,
    const std::vector<momentum::rasterizer::Light>& lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  TORCH_CHECK(
      vertexPositions.dim() == 2 || vertexPositions.dim() == 3,
      "vertex_positions must have rank 2 or 3");
  const bool batched = vertexPositions.dim() == 3;
  if (batched) {
    validateBatchedVertexTensor(vertexPositions, "vertex_positions");
  } else {
    validateInputTensor(vertexPositions, "vertex_positions", at::kFloat, 3);
  }
  validateInputTensor(triangles, "triangles", at::kInt, 3);
  const at::Device device = vertexPositions.device();
  TORCH_CHECK(triangles.device() == device, "triangles must be on the CUDA input device");
  const int64_t numVertices = batched ? vertexPositions.size(1) : vertexPositions.size(0);
  const int64_t batchSize = batched ? vertexPositions.size(0) : 1;

  if (vertexNormals.has_value()) {
    if (batched) {
      validateBatchedVertexTensor(*vertexNormals, "vertex_normals");
    } else {
      validateInputTensor(*vertexNormals, "vertex_normals", at::kFloat, 3);
    }
    TORCH_CHECK(
        vertexNormals->device() == device && vertexNormals->sizes() == vertexPositions.sizes(),
        "vertex_normals must match vertex_positions on the CUDA input device");
  }
  if (textureCoordinates.has_value()) {
    validateInputTensor(*textureCoordinates, "texture_coordinates", at::kFloat, 2);
    TORCH_CHECK(
        textureCoordinates->device() == device,
        "texture_coordinates must be on the CUDA input device");
  }
  if (textureTriangles.has_value()) {
    validateInputTensor(*textureTriangles, "texture_triangles", at::kInt, 3);
    TORCH_CHECK(
        textureTriangles->device() == device && textureTriangles->sizes() == triangles.sizes(),
        "texture_triangles must match triangles on the CUDA input device");
  }
  if (perVertexDiffuseColor.has_value()) {
    validateInputTensor(*perVertexDiffuseColor, "per_vertex_diffuse_color", at::kFloat, 3);
    TORCH_CHECK(
        perVertexDiffuseColor->device() == device && perVertexDiffuseColor->size(0) == numVertices,
        "per_vertex_diffuse_color must match the shared vertices on the CUDA input device");
  }

  const auto validateTextureTensor = [&device](const at::Tensor& texture, const char* name) {
    TORCH_CHECK(texture.device() == device, name, " must be on the CUDA input device");
    TORCH_CHECK(texture.scalar_type() == at::kFloat, name, " must have dtype Float");
    TORCH_CHECK(
        texture.dim() == 3 && texture.size(0) > 0 && texture.size(1) > 0 && texture.size(2) == 3,
        name,
        " must have shape [height, width, 3]");
    TORCH_CHECK(texture.is_contiguous(), name, " must be contiguous");
  };
  if (diffuseTexture.has_value()) {
    validateTextureTensor(*diffuseTexture, "diffuse_texture");
  }
  if (emissiveTexture.has_value()) {
    validateTextureTensor(*emissiveTexture, "emissive_texture");
  }
  TORCH_CHECK(
      !diffuseTexture.has_value() || !perVertexDiffuseColor.has_value(),
      "diffuse_texture and per_vertex_diffuse_color are mutually exclusive");
  TORCH_CHECK(
      (!diffuseTexture.has_value() && !emissiveTexture.has_value() &&
       !textureTriangles.has_value()) ||
          textureCoordinates.has_value(),
      "textures and texture_triangles require texture_coordinates");
  TORCH_CHECK(nearClip > 0.0f, "near_clip must be positive");

  const int64_t height = camera.imageHeight();
  const int64_t width = camera.imageWidth();
  TORCH_CHECK(height > 0 && width > 0, "camera image must be non-empty");
  const c10::cuda::CUDAGuard deviceGuard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  const size_t depthBufferSize = batched
      ? validateBatchedOutputTensor(
            zBuffer, "z_buffer", device, at::kFloat, batchSize, height, width)
      : validateOutputTensor(zBuffer, "z_buffer", device, at::kFloat, height, width);

  size_t rgbBufferSize = 0;
  if (rgbBuffer.has_value()) {
    rgbBufferSize = batched
        ? validateBatchedVectorOutputTensor(
              *rgbBuffer,
              "rgb_buffer",
              device,
              batchSize,
              height,
              width,
              zBuffer.stride(0),
              zBuffer.stride(1))
        : validateVectorOutputTensor(
              *rgbBuffer, "rgb_buffer", device, height, width, zBuffer.stride(0));
  }
  size_t triangleIndexBufferSize = 0;
  if (triangleIndexBuffer.has_value()) {
    triangleIndexBufferSize = batched
        ? validateBatchedOutputTensor(
              *triangleIndexBuffer,
              "triangle_index_buffer",
              device,
              at::kInt,
              batchSize,
              height,
              width)
        : validateOutputTensor(
              *triangleIndexBuffer, "triangle_index_buffer", device, at::kInt, height, width);
    TORCH_CHECK(
        triangleIndexBuffer->strides() == zBuffer.strides(),
        "triangle_index_buffer must have the same strides as z_buffer");
  }
  size_t vertexIndexBufferSize = 0;
  if (vertexIndexBuffer.has_value()) {
    vertexIndexBufferSize = batched
        ? validateBatchedOutputTensor(
              *vertexIndexBuffer, "vertex_index_buffer", device, at::kInt, batchSize, height, width)
        : validateOutputTensor(
              *vertexIndexBuffer, "vertex_index_buffer", device, at::kInt, height, width);
    TORCH_CHECK(
        vertexIndexBuffer->strides() == zBuffer.strides(),
        "vertex_index_buffer must have the same strides as z_buffer");
  }
  size_t surfaceNormalsBufferSize = 0;
  if (surfaceNormalsBuffer.has_value()) {
    surfaceNormalsBufferSize = batched ? validateBatchedVectorOutputTensor(
                                             *surfaceNormalsBuffer,
                                             "surface_normals_buffer",
                                             device,
                                             batchSize,
                                             height,
                                             width,
                                             zBuffer.stride(0),
                                             zBuffer.stride(1))
                                       : validateVectorOutputTensor(
                                             *surfaceNormalsBuffer,
                                             "surface_normals_buffer",
                                             device,
                                             height,
                                             width,
                                             zBuffer.stride(0));
  }
  std::vector<std::pair<const char*, const at::Tensor*>> outputs{{"z_buffer", &zBuffer}};
  if (rgbBuffer.has_value()) {
    outputs.emplace_back("rgb_buffer", &*rgbBuffer);
  }
  if (surfaceNormalsBuffer.has_value()) {
    outputs.emplace_back("surface_normals_buffer", &*surfaceNormalsBuffer);
  }
  if (vertexIndexBuffer.has_value()) {
    outputs.emplace_back("vertex_index_buffer", &*vertexIndexBuffer);
  }
  if (triangleIndexBuffer.has_value()) {
    outputs.emplace_back("triangle_index_buffer", &*triangleIndexBuffer);
  }
  validateDistinctOutputStorage(outputs);

  const momentum::rasterizer::CudaRasterizerInputs rasterizerInputs{
      .positionsModel = vertexPositions.const_data_ptr<float>(),
      .positionsModelSize = static_cast<size_t>(vertexPositions.numel()),
      .normalsModel = vertexNormals.has_value() ? vertexNormals->const_data_ptr<float>() : nullptr,
      .normalsModelSize =
          vertexNormals.has_value() ? static_cast<size_t>(vertexNormals->numel()) : 0,
      .numVertices = static_cast<size_t>(numVertices),
      .triangles = triangles.const_data_ptr<int32_t>(),
      .trianglesSize = static_cast<size_t>(triangles.numel()),
      .numTriangles = static_cast<size_t>(triangles.size(0)),
  };
  const momentum::rasterizer::CudaRasterizerShadingInputs shadingInputs{
      .textureCoordinates =
          textureCoordinates.has_value() ? textureCoordinates->const_data_ptr<float>() : nullptr,
      .textureCoordinatesSize =
          textureCoordinates.has_value() ? static_cast<size_t>(textureCoordinates->numel()) : 0,
      .numTextureCoordinates =
          textureCoordinates.has_value() ? static_cast<size_t>(textureCoordinates->size(0)) : 0,
      .textureTriangles =
          textureTriangles.has_value() ? textureTriangles->const_data_ptr<int32_t>() : nullptr,
      .textureTrianglesSize =
          textureTriangles.has_value() ? static_cast<size_t>(textureTriangles->numel()) : 0,
      .perVertexDiffuseColor = perVertexDiffuseColor.has_value()
          ? perVertexDiffuseColor->const_data_ptr<float>()
          : nullptr,
      .perVertexDiffuseColorSize = perVertexDiffuseColor.has_value()
          ? static_cast<size_t>(perVertexDiffuseColor->numel())
          : 0,
      .diffuseTexture =
          diffuseTexture.has_value() ? diffuseTexture->const_data_ptr<float>() : nullptr,
      .diffuseTextureSize =
          diffuseTexture.has_value() ? static_cast<size_t>(diffuseTexture->numel()) : 0,
      .diffuseTextureHeight =
          diffuseTexture.has_value() ? static_cast<int32_t>(diffuseTexture->size(0)) : 0,
      .diffuseTextureWidth =
          diffuseTexture.has_value() ? static_cast<int32_t>(diffuseTexture->size(1)) : 0,
      .emissiveTexture =
          emissiveTexture.has_value() ? emissiveTexture->const_data_ptr<float>() : nullptr,
      .emissiveTextureSize =
          emissiveTexture.has_value() ? static_cast<size_t>(emissiveTexture->numel()) : 0,
      .emissiveTextureHeight =
          emissiveTexture.has_value() ? static_cast<int32_t>(emissiveTexture->size(0)) : 0,
      .emissiveTextureWidth =
          emissiveTexture.has_value() ? static_cast<int32_t>(emissiveTexture->size(1)) : 0,
      .material = &material,
      .lightsWorld = lights.empty() ? nullptr : lights.data(),
      .numLights = lights.size(),
  };
  const Eigen::Matrix4f model = modelMatrix.value_or(Eigen::Matrix4f::Identity());
  if (batched) {
    momentum::rasterizer::rasterizeMeshesCuda(
        rasterizerInputs,
        static_cast<size_t>(batchSize),
        camera,
        model,
        nearClip,
        momentum::rasterizer::CudaBatchedRasterizerBuffers{
            .zBuffer = zBuffer.mutable_data_ptr<float>(),
            .zBufferSize = depthBufferSize,
            .triangleIndexBuffer = triangleIndexBuffer.has_value()
                ? triangleIndexBuffer->mutable_data_ptr<int32_t>()
                : nullptr,
            .triangleIndexBufferSize = triangleIndexBufferSize,
            .vertexIndexBuffer = vertexIndexBuffer.has_value()
                ? vertexIndexBuffer->mutable_data_ptr<int32_t>()
                : nullptr,
            .vertexIndexBufferSize = vertexIndexBufferSize,
            .surfaceNormalsBuffer = surfaceNormalsBuffer.has_value()
                ? surfaceNormalsBuffer->mutable_data_ptr<float>()
                : nullptr,
            .surfaceNormalsBufferSize = surfaceNormalsBufferSize,
            .rgbBuffer = rgbBuffer.has_value() ? rgbBuffer->mutable_data_ptr<float>() : nullptr,
            .rgbBufferSize = rgbBufferSize,
            .rowStride = zBuffer.stride(1),
            .imageStride = zBuffer.stride(0),
        },
        rgbBuffer.has_value() ? &shadingInputs : nullptr,
        backfaceCulling,
        depthOffset,
        stream);
  } else {
    momentum::rasterizer::rasterizeMeshCuda(
        rasterizerInputs,
        camera,
        model,
        nearClip,
        momentum::rasterizer::CudaRasterizerBuffers{
            .zBuffer = zBuffer.mutable_data_ptr<float>(),
            .zBufferSize = depthBufferSize,
            .triangleIndexBuffer = triangleIndexBuffer.has_value()
                ? triangleIndexBuffer->mutable_data_ptr<int32_t>()
                : nullptr,
            .triangleIndexBufferSize = triangleIndexBufferSize,
            .vertexIndexBuffer = vertexIndexBuffer.has_value()
                ? vertexIndexBuffer->mutable_data_ptr<int32_t>()
                : nullptr,
            .vertexIndexBufferSize = vertexIndexBufferSize,
            .surfaceNormalsBuffer = surfaceNormalsBuffer.has_value()
                ? surfaceNormalsBuffer->mutable_data_ptr<float>()
                : nullptr,
            .surfaceNormalsBufferSize = surfaceNormalsBufferSize,
            .rgbBuffer = rgbBuffer.has_value() ? rgbBuffer->mutable_data_ptr<float>() : nullptr,
            .rgbBufferSize = rgbBufferSize,
            .rowStride = zBuffer.stride(0),
        },
        rgbBuffer.has_value() ? &shadingInputs : nullptr,
        backfaceCulling,
        depthOffset,
        stream);
  }
}

} // namespace
} // namespace pymomentum

PYBIND11_MODULE(_renderer_cuda, module) {
  module.attr("__name__") = "pymomentum._renderer_cuda";
  module.doc() = "CUDA rasterization functions for Momentum meshes.";

  py::module_::import("torch");
  py::module_::import("pymomentum.camera");
  py::module_::import("pymomentum.renderer");

  module.def(
      "rasterize_mesh",
      &pymomentum::rasterizeMeshCuda,
      R"(Rasterize a mesh into caller-owned CUDA buffers.

This non-differentiable path mutates every supplied output in place and never allocates output buffers. The required depth target participates in the depth test. Supplying an RGB target enables native Phong shading; triangle-index, vertex-index, and eye-space surface-normal targets are independently optional. Batched positions share topology, camera, material, textures, and lights. Perspective-correct barycentric coordinates and interpolated normals remain transient in the CUDA kernel. Exact-depth ties are unspecified and may mix independently written RGB channels or optional outputs.

:param vertex_positions: Contiguous CUDA float32 tensor with shape ``(n_vertices, 3)`` or ``(batch, n_vertices, 3)``.
:param vertex_normals: Optional contiguous CUDA float32 tensor matching ``vertex_positions``. Face normals are used when omitted.
:param triangles: Contiguous CUDA int32 tensor with shape ``(n_triangles, 3)`` shared by the batch.
:param camera: Camera shared by every batch entry.
:param z_buffer: Required caller-owned padded CUDA depth target.
:param rgb_buffer: Optional caller-owned padded CUDA RGB target. Its presence enables shading.
:param surface_normals_buffer: Optional caller-owned eye-space surface-normal target.
:param vertex_index_buffer: Optional caller-owned closest-vertex target.
:param triangle_index_buffer: Optional caller-owned triangle-index target.
:param material: Host-side Phong material parameters. Embedded NumPy textures are not supported.
:param texture_coordinates: Optional contiguous CUDA float32 UV coordinates with shape ``(n_coordinates, 2)``.
:param texture_triangles: Optional contiguous CUDA int32 UV indices matching ``triangles``. Mesh topology is reused when omitted.
:param per_vertex_diffuse_color: Optional contiguous CUDA float32 colors with shape ``(n_vertices, 3)``, shared by every batch entry.
:param diffuse_texture: Optional contiguous CUDA float32 RGB texture with shape ``(height, width, 3)``.
:param emissive_texture: Optional contiguous CUDA float32 RGB texture with shape ``(height, width, 3)``.
:param lights: Host-side lights in world space. An empty list selects the default camera light; at most 16 explicit lights are supported.
:param model_matrix: Optional model transformation matrix shared by the batch.
:param back_face_culling: Whether to discard back-facing triangles.
:param near_clip: Per-pixel near clipping distance.
:param depth_offset: Value added to rasterized depth before depth testing.
)",
      py::arg("vertex_positions"),
      py::arg("vertex_normals"),
      py::arg("triangles"),
      py::arg("camera"),
      py::arg("z_buffer"),
      py::arg("rgb_buffer") = std::nullopt,
      py::kw_only(),
      py::arg("surface_normals_buffer") = std::nullopt,
      py::arg("vertex_index_buffer") = std::nullopt,
      py::arg("triangle_index_buffer") = std::nullopt,
      py::arg("material") = momentum::rasterizer::PhongMaterial{},
      py::arg("texture_coordinates") = std::nullopt,
      py::arg("texture_triangles") = std::nullopt,
      py::arg("per_vertex_diffuse_color") = std::nullopt,
      py::arg("diffuse_texture") = std::nullopt,
      py::arg("emissive_texture") = std::nullopt,
      py::arg("lights") = std::vector<momentum::rasterizer::Light>{},
      py::arg("model_matrix") = std::nullopt,
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f);
}
