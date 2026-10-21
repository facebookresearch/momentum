/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>

#include <cuda_runtime_api.h>
#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace momentum::rasterizer {

/// Device inputs used by the CUDA rasterizer.
///
/// Buffer sizes are element counts, not byte counts. positionsModel must contain at least
/// 3 * numVertices floats, and triangles must contain at least 3 * numTriangles indices.
struct CudaRasterizerInputs {
  const float* positionsModel = nullptr;
  size_t positionsModelSize = 0;
  size_t numVertices = 0;
  const int32_t* triangles = nullptr;
  size_t trianglesSize = 0;
  size_t numTriangles = 0;
};

/// Device buffers used by the CUDA rasterizer.
///
/// All buffers use rowStride elements per row. The corresponding size fields are element counts,
/// not byte counts. Existing zBuffer and triangleIndexBuffer values are preserved where no new
/// triangle wins the depth test.
struct CudaRasterizerBuffers {
  uint64_t* visibilityBuffer = nullptr;
  size_t visibilityBufferSize = 0;
  float* zBuffer = nullptr;
  size_t zBufferSize = 0;
  int32_t* triangleIndexBuffer = nullptr;
  size_t triangleIndexBufferSize = 0;
  int64_t rowStride = 0;
};

/// Rasterize a triangle mesh into CUDA-resident depth and triangle-index buffers.
///
/// The input arrays must be contiguous CUDA buffers with shapes [numVertices, 3] and
/// [numTriangles, 3]. The function launches work asynchronously on stream. The camera object and
/// model matrix are consumed on the host before launch. Pinhole, OpenCV, and OpenCV fisheye
/// intrinsics are supported. Triangles containing an index outside [0, numVertices) are ignored.
///
/// @param inputs CUDA input buffers, capacities, and mesh dimensions.
/// @param camera Camera used for projection.
/// @param modelMatrix Additional model transform.
/// @param nearClip Per-pixel near clipping distance; must be positive.
/// @param buffers CUDA output and scratch buffers.
/// @param backfaceCulling Whether to discard back-facing triangles.
/// @param depthOffset Value added to depth before the depth test and write.
/// @param stream CUDA stream used for all launches.
void rasterizeMeshCuda(
    CudaRasterizerInputs inputs,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    CudaRasterizerBuffers buffers,
    bool backfaceCulling = true,
    float depthOffset = 0.0f,
    cudaStream_t stream = nullptr);

} // namespace momentum::rasterizer
