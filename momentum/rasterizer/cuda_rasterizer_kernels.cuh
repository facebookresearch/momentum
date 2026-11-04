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

/// Camera intrinsics models implemented by the CUDA projection path.
enum class CudaCameraModel : int32_t {
  Pinhole,
  OpenCV,
  OpenCVFisheye,
};

/// Trivially copyable camera parameters passed to CUDA kernels.
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
  float normalToEye[9] = {};
};

constexpr int32_t kMaxCudaLights = 16;

enum class CudaLightType : int32_t {
  Point,
  Directional,
  Ambient,
};

struct CudaLight {
  float3 position = {};
  float3 color = {};
  CudaLightType type = CudaLightType::Point;
};

/// Trivially copyable native fragment-shading parameters passed to CUDA kernels.
struct CudaShadingParameters {
  const float* textureCoordinates = nullptr;
  size_t numTextureCoordinates = 0;
  const int32_t* textureTriangles = nullptr;
  const float* perVertexDiffuseColor = nullptr;
  const float* diffuseTexture = nullptr;
  int32_t diffuseTextureHeight = 0;
  int32_t diffuseTextureWidth = 0;
  const float* emissiveTexture = nullptr;
  int32_t emissiveTextureHeight = 0;
  int32_t emissiveTextureWidth = 0;
  float3 diffuseColor = {};
  float3 specularColor = {};
  float specularExponent = 0.0f;
  float3 emissiveColor = {};
  CudaLight lights[kMaxCudaLights] = {};
  int32_t numLights = 0;
  int32_t enabled = 0;
};

/// Launch the CUDA rasterization kernel for one mesh.
cudaError_t launchCudaRasterizeMesh(
    const float* positionsModel,
    const float* positionsEye,
    const float* positionsImage,
    const bool* projectionValidity,
    const float* normalsModel,
    size_t numVertices,
    const int32_t* triangles,
    size_t numTriangles,
    const CudaCamera& camera,
    float nearClip,
    float depthOffset,
    bool backfaceCulling,
    int64_t rowStride,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    int32_t* vertexIndexBuffer,
    float* surfaceNormalsBuffer,
    const CudaShadingParameters& shading,
    float* rgbBuffer,
    cudaStream_t stream);

/// Launch the CUDA rasterization kernel for a batch with shared topology and camera.
cudaError_t launchCudaRasterizeMeshes(
    const float* positionsModel,
    const float* positionsEye,
    const float* positionsImage,
    const bool* projectionValidity,
    const float* normalsModel,
    size_t batchSize,
    size_t numVertices,
    const int32_t* triangles,
    size_t numTriangles,
    const CudaCamera& camera,
    float nearClip,
    float depthOffset,
    bool backfaceCulling,
    int64_t rowStride,
    int64_t imageStride,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    int32_t* vertexIndexBuffer,
    float* surfaceNormalsBuffer,
    const CudaShadingParameters& shading,
    float* rgbBuffer,
    cudaStream_t stream);

} // namespace momentum::rasterizer::detail
