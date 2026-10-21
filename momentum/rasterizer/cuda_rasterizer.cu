/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/cuda_rasterizer_kernels.cuh"

#include <cuda_runtime.h>

namespace momentum::rasterizer::detail {
namespace {

struct TriangleSetup {
  float u[3];
  float v[3];
  float z[3];
  float denominator;
  int32_t minX;
  int32_t maxX;
  int32_t minY;
  int32_t maxY;
  int32_t valid;
};

__device__ uint32_t floatToOrderedUint(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) != 0U ? ~bits : bits ^ 0x80000000U;
}

__device__ float orderedUintToFloat(uint32_t value) {
  const uint32_t bits = (value & 0x80000000U) != 0U ? value ^ 0x80000000U : ~value;
  return __uint_as_float(bits);
}

__device__ bool
projectPoint(const CudaCamera& camera, const float3& pointModel, float& u, float& v, float& z) {
  const float worldXUnnormalized = camera.modelToWorld[0] * pointModel.x +
      camera.modelToWorld[1] * pointModel.y + camera.modelToWorld[2] * pointModel.z +
      camera.modelToWorld[3];
  const float worldYUnnormalized = camera.modelToWorld[4] * pointModel.x +
      camera.modelToWorld[5] * pointModel.y + camera.modelToWorld[6] * pointModel.z +
      camera.modelToWorld[7];
  const float worldZUnnormalized = camera.modelToWorld[8] * pointModel.x +
      camera.modelToWorld[9] * pointModel.y + camera.modelToWorld[10] * pointModel.z +
      camera.modelToWorld[11];
  const float worldW = camera.modelToWorld[12] * pointModel.x +
      camera.modelToWorld[13] * pointModel.y + camera.modelToWorld[14] * pointModel.z +
      camera.modelToWorld[15];

  const float worldX = worldXUnnormalized / worldW;
  const float worldY = worldYUnnormalized / worldW;
  const float worldZ = worldZUnnormalized / worldW;

  const float eyeX = camera.worldToEye[0] * worldX + camera.worldToEye[1] * worldY +
      camera.worldToEye[2] * worldZ + camera.worldToEye[3];
  const float eyeY = camera.worldToEye[4] * worldX + camera.worldToEye[5] * worldY +
      camera.worldToEye[6] * worldZ + camera.worldToEye[7];
  z = camera.worldToEye[8] * worldX + camera.worldToEye[9] * worldY +
      camera.worldToEye[10] * worldZ + camera.worldToEye[11];

  if (!(z > 0.0f) || !isfinite(z)) {
    return false;
  }

  const float x = eyeX / z;
  const float y = eyeY / z;
  if (camera.model == CudaCameraModel::Pinhole) {
    u = camera.fx * x + camera.cx;
    v = camera.fy * y + camera.cy;
  } else if (camera.model == CudaCameraModel::OpenCV) {
    const float radiusSquared = x * x + y * y;
    const float numerator = 1.0f +
        radiusSquared *
            (camera.distortion[0] +
             radiusSquared * (camera.distortion[1] + radiusSquared * camera.distortion[2]));
    const float denominator = 1.0f +
        radiusSquared *
            (camera.distortion[3] +
             radiusSquared * (camera.distortion[4] + radiusSquared * camera.distortion[5]));
    const float radial = numerator / denominator;
    const float distortedX = x * radial + 2.0f * camera.distortion[6] * x * y +
        camera.distortion[7] * (radiusSquared + 2.0f * x * x);
    const float distortedY = y * radial + camera.distortion[6] * (radiusSquared + 2.0f * y * y) +
        2.0f * camera.distortion[7] * x * y;
    u = camera.fx * distortedX + camera.cx;
    v = camera.fy * distortedY + camera.cy;
  } else {
    const float radiusSquared = x * x + y * y;
    if (radiusSquared > camera.maxRSquared) {
      return false;
    }
    const float radius = sqrtf(radiusSquared);
    const float theta = atanf(radius);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float distortedTheta = theta *
        (1.0f + camera.distortion[0] * theta2 + camera.distortion[1] * theta4 +
         camera.distortion[2] * theta6 + camera.distortion[3] * theta8);
    const float scale = radius > 1e-8f ? distortedTheta / radius : 1.0f;
    u = camera.fx * scale * x + camera.cx;
    v = camera.fy * scale * y + camera.cy;
  }

  return isfinite(u) && isfinite(v);
}

__global__ void initializeVisibilityKernel(
    int32_t imageWidth,
    int32_t imageHeight,
    int64_t rowStride,
    const float* zBuffer,
    uint64_t* visibilityBuffer) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t numPixels = static_cast<int64_t>(imageWidth) * imageHeight;
  if (pixel >= numPixels) {
    return;
  }

  const int32_t y = static_cast<int32_t>(pixel / imageWidth);
  const int32_t x = static_cast<int32_t>(pixel - static_cast<int64_t>(y) * imageWidth);
  const int64_t offset = static_cast<int64_t>(y) * rowStride + x;
  const float existingDepth = zBuffer[offset];
  visibilityBuffer[offset] =
      isnan(existingDepth) ? 0 : static_cast<uint64_t>(floatToOrderedUint(existingDepth)) << 32;
}

__global__ void rasterizeVisibilityKernel(
    const float* positionsModel,
    size_t numVertices,
    const int32_t* triangles,
    size_t numTriangles,
    CudaCamera camera,
    float nearClip,
    float depthOffset,
    bool backfaceCulling,
    int64_t rowStride,
    uint64_t* visibilityBuffer) {
  const size_t triangleIndex = blockIdx.x;
  if (triangleIndex >= numTriangles) {
    return;
  }

  __shared__ TriangleSetup setup;
  if (threadIdx.x == 0) {
    setup.valid = 1;
    for (int32_t vertex = 0; vertex < 3; ++vertex) {
      const int32_t vertexIndex = triangles[3 * triangleIndex + vertex];
      if (vertexIndex < 0 || static_cast<size_t>(vertexIndex) >= numVertices) {
        setup.valid = 0;
        break;
      }
      const float3 point = make_float3(
          positionsModel[3 * vertexIndex],
          positionsModel[3 * vertexIndex + 1],
          positionsModel[3 * vertexIndex + 2]);
      setup.valid &= projectPoint(camera, point, setup.u[vertex], setup.v[vertex], setup.z[vertex]);
    }

    if (setup.valid) {
      setup.denominator = (setup.v[1] - setup.v[2]) * (setup.u[0] - setup.u[2]) +
          (setup.u[2] - setup.u[1]) * (setup.v[0] - setup.v[2]);
      setup.valid &= isfinite(setup.denominator) && setup.denominator != 0.0f;
      if (backfaceCulling) {
        setup.valid &= setup.denominator < 0.0f;
      }

      const float minU = fminf(setup.u[0], fminf(setup.u[1], setup.u[2]));
      const float maxU = fmaxf(setup.u[0], fmaxf(setup.u[1], setup.u[2]));
      const float minV = fminf(setup.v[0], fminf(setup.v[1], setup.v[2]));
      const float maxV = fmaxf(setup.v[0], fmaxf(setup.v[1], setup.v[2]));
      setup.minX = max(0, min(camera.imageWidth - 1, static_cast<int32_t>(floorf(minU))));
      setup.maxX = max(0, min(camera.imageWidth - 1, static_cast<int32_t>(ceilf(maxU))));
      setup.minY = max(0, min(camera.imageHeight - 1, static_cast<int32_t>(floorf(minV))));
      setup.maxY = max(0, min(camera.imageHeight - 1, static_cast<int32_t>(ceilf(maxV))));
    }
  }
  __syncthreads();

  if (!setup.valid) {
    return;
  }

  const int32_t boxWidth = setup.maxX - setup.minX + 1;
  const int32_t boxHeight = setup.maxY - setup.minY + 1;
  const int64_t boxPixels = static_cast<int64_t>(boxWidth) * boxHeight;
  for (int64_t boxPixel = threadIdx.x; boxPixel < boxPixels; boxPixel += blockDim.x) {
    const int32_t y = setup.minY + static_cast<int32_t>(boxPixel / boxWidth);
    const int32_t x = setup.minX + static_cast<int32_t>(boxPixel % boxWidth);
    const float pixelX = static_cast<float>(x);
    const float pixelY = static_cast<float>(y);

    const float lambda0 = ((setup.v[1] - setup.v[2]) * (pixelX - setup.u[2]) +
                           (setup.u[2] - setup.u[1]) * (pixelY - setup.v[2])) /
        setup.denominator;
    const float lambda1 = ((setup.v[2] - setup.v[0]) * (pixelX - setup.u[2]) +
                           (setup.u[0] - setup.u[2]) * (pixelY - setup.v[2])) /
        setup.denominator;
    const float lambda2 = 1.0f - lambda0 - lambda1;
    if (lambda0 < 0.0f || lambda1 < 0.0f || lambda2 < 0.0f) {
      continue;
    }

    const float reciprocalDepth =
        lambda0 / setup.z[0] + lambda1 / setup.z[1] + lambda2 / setup.z[2];
    const float depth = 1.0f / reciprocalDepth;
    if (!(depth > nearClip) || !isfinite(depth)) {
      continue;
    }

    const float outputDepth = depth + depthOffset;
    if (!isfinite(outputDepth)) {
      continue;
    }

    const uint64_t key = (static_cast<uint64_t>(floatToOrderedUint(outputDepth)) << 32) |
        static_cast<uint32_t>(triangleIndex + 1);
    auto* address = reinterpret_cast<unsigned long long*>(
        visibilityBuffer + static_cast<int64_t>(y) * rowStride + x);
    atomicMin(address, static_cast<unsigned long long>(key));
  }
}

__global__ void resolveVisibilityKernel(
    int32_t imageWidth,
    int32_t imageHeight,
    int64_t rowStride,
    const uint64_t* visibilityBuffer,
    float* zBuffer,
    int32_t* triangleIndexBuffer) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t numPixels = static_cast<int64_t>(imageWidth) * imageHeight;
  if (pixel >= numPixels) {
    return;
  }

  const int32_t y = static_cast<int32_t>(pixel / imageWidth);
  const int32_t x = static_cast<int32_t>(pixel - static_cast<int64_t>(y) * imageWidth);
  const int64_t offset = static_cast<int64_t>(y) * rowStride + x;
  const uint64_t key = visibilityBuffer[offset];
  const uint32_t triangleKey = static_cast<uint32_t>(key);
  if (triangleKey == 0) {
    return;
  }

  zBuffer[offset] = orderedUintToFloat(static_cast<uint32_t>(key >> 32));
  triangleIndexBuffer[offset] = static_cast<int32_t>(triangleKey - 1);
}

} // namespace

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
    cudaStream_t stream) {
  const int64_t numPixels = static_cast<int64_t>(camera.imageWidth) * camera.imageHeight;
  const int32_t pixelBlocks = static_cast<int32_t>(
      (numPixels + kCudaRasterizerThreadsPerBlock - 1) / kCudaRasterizerThreadsPerBlock);

  initializeVisibilityKernel<<<pixelBlocks, kCudaRasterizerThreadsPerBlock, 0, stream>>>(
      camera.imageWidth, camera.imageHeight, rowStride, zBuffer, visibilityBuffer);
  if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
    return error;
  }

  if (numTriangles > 0) {
    rasterizeVisibilityKernel<<<
        static_cast<uint32_t>(numTriangles),
        kCudaRasterizerThreadsPerBlock,
        0,
        stream>>>(
        positionsModel,
        numVertices,
        triangles,
        numTriangles,
        camera,
        nearClip,
        depthOffset,
        backfaceCulling,
        rowStride,
        visibilityBuffer);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
      return error;
    }
  }

  resolveVisibilityKernel<<<pixelBlocks, kCudaRasterizerThreadsPerBlock, 0, stream>>>(
      camera.imageWidth,
      camera.imageHeight,
      rowStride,
      visibilityBuffer,
      zBuffer,
      triangleIndexBuffer);
  return cudaGetLastError();
}

} // namespace momentum::rasterizer::detail
