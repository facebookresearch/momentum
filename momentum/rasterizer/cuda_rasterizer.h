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
/// 3 * numVertices floats, normalsModel must contain the same number when provided, and triangles
/// must contain at least 3 * numTriangles indices.
struct CudaRasterizerInputs {
  const float* positionsModel = nullptr;
  size_t positionsModelSize = 0;
  const float* normalsModel = nullptr;
  size_t normalsModelSize = 0;
  size_t numVertices = 0;
  const int32_t* triangles = nullptr;
  size_t trianglesSize = 0;
  size_t numTriangles = 0;
};

/// Device buffers used by the CUDA rasterizer.
///
/// All scalar buffers use rowStride elements per row. surfaceNormalsBuffer, when provided, stores
/// three interleaved floats per pixel and uses 3 * rowStride elements per row. The corresponding
/// size fields are element counts, not byte counts. Existing output values are preserved where no
/// new triangle wins the depth test. zBuffer must be non-null; triangle, vertex, and normal outputs
/// are optional and written in a conditional second pass. For non-tied depth values, every
/// requested attribute corresponds to finalized depth. Exact-depth ties are unspecified and may mix
/// independently written attributes from tied triangles.
struct CudaRasterizerBuffers {
  float* zBuffer = nullptr;
  size_t zBufferSize = 0;
  int32_t* triangleIndexBuffer = nullptr;
  size_t triangleIndexBufferSize = 0;
  int32_t* vertexIndexBuffer = nullptr;
  size_t vertexIndexBufferSize = 0;
  float* surfaceNormalsBuffer = nullptr;
  size_t surfaceNormalsBufferSize = 0;
  int64_t rowStride = 0;
};

/// Device buffers used by batched CUDA rasterization.
///
/// Each image uses rowStride scalar elements per row and imageStride scalar elements between batch
/// entries. surfaceNormalsBuffer uses three times those strides. The corresponding size fields are
/// element counts, not byte counts, and must cover the final image's rows from each buffer pointer.
/// Existing output values are preserved where no new triangle wins the depth test. Optional outputs
/// have the same exact-depth tie semantics as CudaRasterizerBuffers.
struct CudaBatchedRasterizerBuffers {
  float* zBuffer = nullptr;
  size_t zBufferSize = 0;
  int32_t* triangleIndexBuffer = nullptr;
  size_t triangleIndexBufferSize = 0;
  int32_t* vertexIndexBuffer = nullptr;
  size_t vertexIndexBufferSize = 0;
  float* surfaceNormalsBuffer = nullptr;
  size_t surfaceNormalsBufferSize = 0;
  int64_t rowStride = 0;
  int64_t imageStride = 0;
};

/// Rasterize a triangle mesh into CUDA-resident output buffers.
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
/// @param buffers CUDA output buffers.
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

/// Rasterize a batch of triangle meshes with shared topology and camera.
///
/// inputs.positionsModel must be a contiguous CUDA buffer with shape
/// [batchSize, inputs.numVertices, 3]. inputs.normalsModel must match it when provided. The
/// triangle buffer, camera, and model matrix are shared by every batch entry. The depth pass for
/// the entire batch uses one kernel launch; a second batched launch occurs only when an optional
/// output is requested.
///
/// @param inputs CUDA input buffers, capacities, and mesh dimensions.
/// @param batchSize Number of meshes; must be positive and at most 65535.
/// @param camera Camera used for every batch entry.
/// @param modelMatrix Additional model transform shared by every batch entry.
/// @param nearClip Per-pixel near clipping distance; must be positive.
/// @param buffers Batched CUDA output buffers.
/// @param backfaceCulling Whether to discard back-facing triangles.
/// @param depthOffset Value added to depth before the depth test and write.
/// @param stream CUDA stream used for all launches.
void rasterizeMeshesCuda(
    CudaRasterizerInputs inputs,
    size_t batchSize,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    CudaBatchedRasterizerBuffers buffers,
    bool backfaceCulling = true,
    float depthOffset = 0.0f,
    cudaStream_t stream = nullptr);

} // namespace momentum::rasterizer
