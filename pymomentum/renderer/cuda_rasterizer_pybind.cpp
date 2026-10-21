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

std::tuple<at::Tensor, at::Tensor> rasterizeMeshCuda(
    const at::Tensor& vertexPositions,
    const at::Tensor& triangles,
    const momentum::Camera& camera,
    const std::optional<at::Tensor>& zBuffer,
    const std::optional<at::Tensor>& triangleIndexBuffer,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset) {
  validateInputTensor(vertexPositions, "vertex_positions", at::kFloat, 3);
  validateInputTensor(triangles, "triangles", at::kInt, 3);
  TORCH_CHECK(
      triangles.device() == vertexPositions.device(),
      "triangles must be on the same CUDA device as vertex_positions");
  TORCH_CHECK(nearClip > 0.0f, "near_clip must be positive");

  const int64_t height = camera.imageHeight();
  const int64_t width = camera.imageWidth();
  TORCH_CHECK(height > 0 && width > 0, "camera image must be non-empty");
  const at::Device device = vertexPositions.device();
  const c10::cuda::CUDAGuard deviceGuard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  const auto floatOptions = vertexPositions.options().dtype(at::kFloat);
  const auto intOptions = vertexPositions.options().dtype(at::kInt);
  const auto visibilityOptions = vertexPositions.options().dtype(at::kLong);

  const int64_t paddedWidth = ((width + 7) / 8) * 8;
  at::Tensor depth = zBuffer.has_value()
      ? *zBuffer
      : at::full({height, paddedWidth}, std::numeric_limits<float>::max(), floatOptions)
            .narrow(1, 0, width);
  at::Tensor triangleIndex = triangleIndexBuffer.has_value()
      ? *triangleIndexBuffer
      : at::full({height, paddedWidth}, -1, intOptions).narrow(1, 0, width);
  const size_t depthBufferSize =
      validateOutputTensor(depth, "z_buffer", device, at::kFloat, height, width);
  const size_t triangleIndexBufferSize =
      validateOutputTensor(triangleIndex, "triangle_index_buffer", device, at::kInt, height, width);
  TORCH_CHECK(
      depth.sizes() == triangleIndex.sizes() && depth.strides() == triangleIndex.strides(),
      "z_buffer and triangle_index_buffer must have the same shape and strides");

  at::Tensor visibility = at::empty({height, depth.stride(0)}, visibilityOptions);

  momentum::rasterizer::rasterizeMeshCuda(
      momentum::rasterizer::CudaRasterizerInputs{
          .positionsModel = vertexPositions.const_data_ptr<float>(),
          .positionsModelSize = static_cast<size_t>(vertexPositions.numel()),
          .numVertices = static_cast<size_t>(vertexPositions.size(0)),
          .triangles = triangles.const_data_ptr<int32_t>(),
          .trianglesSize = static_cast<size_t>(triangles.numel()),
          .numTriangles = static_cast<size_t>(triangles.size(0)),
      },
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      momentum::rasterizer::CudaRasterizerBuffers{
          .visibilityBuffer = reinterpret_cast<uint64_t*>(visibility.mutable_data_ptr<int64_t>()),
          .visibilityBufferSize = static_cast<size_t>(visibility.numel()),
          .zBuffer = depth.mutable_data_ptr<float>(),
          .zBufferSize = depthBufferSize,
          .triangleIndexBuffer = triangleIndex.mutable_data_ptr<int32_t>(),
          .triangleIndexBufferSize = triangleIndexBufferSize,
          .rowStride = depth.stride(0),
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
      R"(Rasterize a triangle mesh into CUDA-resident depth and triangle-index tensors.

This initial CUDA path is non-differentiable and supports depth testing plus triangle IDs. It uses the current PyTorch CUDA stream and may update caller-provided output tensors in place. Callers must establish stream ordering when inputs were produced on a different stream.

:param vertex_positions: Contiguous CUDA float32 tensor with shape ``(n_vertices, 3)``.
:param triangles: Contiguous CUDA int32 tensor with shape ``(n_triangles, 3)``.
:param camera: Camera used to transform and project vertices.
:param z_buffer: Optional CUDA float32 tensor with shape ``(height, width)`` and unit column stride. Existing depth values participate in the depth test.
:param triangle_index_buffer: Optional CUDA int32 tensor with the same shape and row stride as ``z_buffer``. Existing values are preserved where no new triangle wins.
:param model_matrix: Optional model transformation matrix.
:param back_face_culling: Whether to discard back-facing triangles.
:param near_clip: Per-pixel near clipping distance.
:param depth_offset: Value added to rasterized depth before depth testing.
:return: Tuple containing the depth and triangle-index tensors.
)",
      py::arg("vertex_positions"),
      py::arg("triangles"),
      py::arg("camera"),
      py::kw_only(),
      py::arg("z_buffer") = std::nullopt,
      py::arg("triangle_index_buffer") = std::nullopt,
      py::arg("model_matrix") = std::nullopt,
      py::arg("back_face_culling") = true,
      py::arg("near_clip") = 0.1f,
      py::arg("depth_offset") = 0.0f);
}
