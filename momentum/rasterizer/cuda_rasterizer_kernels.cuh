/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace momentum::rasterizer::detail {

inline constexpr int32_t kCudaRasterizerThreadsPerBlock = 256;

enum class CudaCameraModel : int32_t {
  Pinhole,
  OpenCV,
  OpenCVFisheye,
};

struct CudaCamera {
  CudaCameraModel model = CudaCameraModel::Pinhole;
  int32_t imageWidth = 0;
  int32_t imageHeight = 0;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
  float distortion[10] = {};
  float maxRSquared = 0.0f;
  float modelToWorld[16] = {};
  float worldToEye[12] = {};
};

cudaError_t launchCudaRasterizeMesh(
    const float* positionsModel,
    size_t numVertices,
    const int32_t* triangles,
    size_t numTriangles,
    const CudaCamera& camera,
    float nearClip,
    float depthOffset,
    bool backfaceCulling,
    int64_t rowStride,
    uint64_t* visibilityBuffer,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    cudaStream_t stream);

} // namespace momentum::rasterizer::detail
