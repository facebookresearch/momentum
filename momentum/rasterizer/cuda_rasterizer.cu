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

constexpr int32_t kRasterThreadsPerBlock = 32;

struct TriangleSetup {
  float u[3];
  float v[3];
  float3 eye[3];
  float3 normal[3];
  int32_t vertexIndex[3];
  float denominator;
  int32_t minX;
  int32_t maxX;
  int32_t minY;
  int32_t maxY;
  int32_t valid;
};

__device__ void atomicMinDepth(float* address, float value) {
  auto* addressBits = reinterpret_cast<unsigned int*>(address);
  // Caller-owned buffers may contain any prior depth, including zero.
  unsigned int oldBits = atomicAdd(addressBits, 0U);
  while (isnan(__uint_as_float(oldBits)) || value < __uint_as_float(oldBits)) {
    const unsigned int assumed = oldBits;
    oldBits = atomicCAS(addressBits, assumed, __float_as_uint(value));
    if (oldBits == assumed) {
      return;
    }
  }
}

__device__ void atomicWriteFloat(float* address, float value) {
  atomicExch(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
}

__device__ float3 normalizeVector(float3 value) {
  const float squaredLength = value.x * value.x + value.y * value.y + value.z * value.z;
  if (!(squaredLength > 0.0f) || !isfinite(squaredLength)) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  const float inverseLength = rsqrtf(squaredLength);
  return make_float3(value.x * inverseLength, value.y * inverseLength, value.z * inverseLength);
}

__device__ float3 subtract(float3 first, float3 second) {
  return make_float3(first.x - second.x, first.y - second.y, first.z - second.z);
}

__device__ float3 cross(float3 first, float3 second) {
  return make_float3(
      first.y * second.z - first.z * second.y,
      first.z * second.x - first.x * second.z,
      first.x * second.y - first.y * second.x);
}

__device__ float3 transformNormal(const CudaCamera& camera, float3 normalModel) {
  return normalizeVector(make_float3(
      camera.normalToEye[0] * normalModel.x + camera.normalToEye[1] * normalModel.y +
          camera.normalToEye[2] * normalModel.z,
      camera.normalToEye[3] * normalModel.x + camera.normalToEye[4] * normalModel.y +
          camera.normalToEye[5] * normalModel.z,
      camera.normalToEye[6] * normalModel.x + camera.normalToEye[7] * normalModel.y +
          camera.normalToEye[8] * normalModel.z));
}

__device__ bool projectPoint(
    const CudaCamera& camera,
    const float3& pointModel,
    float& u,
    float& v,
    float3& pointEye) {
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
  const float eyeZ = camera.worldToEye[8] * worldX + camera.worldToEye[9] * worldY +
      camera.worldToEye[10] * worldZ + camera.worldToEye[11];
  pointEye = make_float3(eyeX, eyeY, eyeZ);

  if (!(eyeZ > 0.0f) || !isfinite(eyeZ)) {
    return false;
  }

  const float x = eyeX / eyeZ;
  const float y = eyeY / eyeZ;
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

__global__ void rasterizeKernel(
    const float* positionsModel,
    const float* normalsModel,
    size_t numVertices,
    const int32_t* triangles,
    size_t numTriangles,
    CudaCamera camera,
    float nearClip,
    float depthOffset,
    bool backfaceCulling,
    bool writeDepth,
    int64_t rowStride,
    int64_t imageStride,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    int32_t* vertexIndexBuffer,
    float* surfaceNormalsBuffer) {
  const size_t batchIndex = blockIdx.y;
  const size_t triangleIndex = blockIdx.x;
  positionsModel += 3 * batchIndex * numVertices;
  if (normalsModel != nullptr) {
    normalsModel += 3 * batchIndex * numVertices;
  }
  zBuffer += static_cast<int64_t>(batchIndex) * imageStride;
  if (triangleIndexBuffer != nullptr) {
    triangleIndexBuffer += static_cast<int64_t>(batchIndex) * imageStride;
  }
  if (vertexIndexBuffer != nullptr) {
    vertexIndexBuffer += static_cast<int64_t>(batchIndex) * imageStride;
  }
  if (surfaceNormalsBuffer != nullptr) {
    surfaceNormalsBuffer += 3 * static_cast<int64_t>(batchIndex) * imageStride;
  }
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
      setup.vertexIndex[vertex] = vertexIndex;
      const float3 point = make_float3(
          positionsModel[3 * vertexIndex],
          positionsModel[3 * vertexIndex + 1],
          positionsModel[3 * vertexIndex + 2]);
      setup.valid &=
          projectPoint(camera, point, setup.u[vertex], setup.v[vertex], setup.eye[vertex]);
      if (!writeDepth) {
        if (surfaceNormalsBuffer != nullptr && normalsModel != nullptr) {
          setup.normal[vertex] = transformNormal(
              camera,
              make_float3(
                  normalsModel[3 * vertexIndex],
                  normalsModel[3 * vertexIndex + 1],
                  normalsModel[3 * vertexIndex + 2]));
        }
      }
    }

    if (!writeDepth) {
      if (setup.valid && surfaceNormalsBuffer != nullptr && normalsModel == nullptr) {
        const float3 faceNormal = normalizeVector(
            cross(subtract(setup.eye[1], setup.eye[0]), subtract(setup.eye[2], setup.eye[0])));
        setup.normal[0] = faceNormal;
        setup.normal[1] = faceNormal;
        setup.normal[2] = faceNormal;
      }
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
        lambda0 / setup.eye[0].z + lambda1 / setup.eye[1].z + lambda2 / setup.eye[2].z;
    const float depth = 1.0f / reciprocalDepth;
    if (!(depth > nearClip) || !isfinite(depth)) {
      continue;
    }

    const float outputDepth = depth + depthOffset;
    if (!isfinite(outputDepth)) {
      continue;
    }

    const int64_t offset = static_cast<int64_t>(y) * rowStride + x;
    if (writeDepth) {
      atomicMinDepth(zBuffer + offset, outputDepth);
    } else {
      // Both passes execute this same runtime-parameterized kernel, so a given
      // fragment recomputes the same depth bits. Exact matching avoids races
      // between attributes from different triangles at nearly equal depths.
      if (__float_as_uint(zBuffer[offset]) != __float_as_uint(outputDepth)) {
        continue;
      }

      if (triangleIndexBuffer != nullptr) {
        atomicExch(triangleIndexBuffer + offset, static_cast<int32_t>(triangleIndex));
      }
      const float bary0 = lambda0 * depth / setup.eye[0].z;
      const float bary1 = lambda1 * depth / setup.eye[1].z;
      const float bary2 = lambda2 * depth / setup.eye[2].z;
      if (vertexIndexBuffer != nullptr) {
        const int32_t vertexIndex = bary0 > bary1 && bary0 > bary2
            ? setup.vertexIndex[0]
            : (bary1 > bary2 ? setup.vertexIndex[1] : setup.vertexIndex[2]);
        atomicExch(vertexIndexBuffer + offset, vertexIndex);
      }
      if (surfaceNormalsBuffer != nullptr) {
        const float3 normal = normalizeVector(make_float3(
            bary0 * setup.normal[0].x + bary1 * setup.normal[1].x + bary2 * setup.normal[2].x,
            bary0 * setup.normal[0].y + bary1 * setup.normal[1].y + bary2 * setup.normal[2].y,
            bary0 * setup.normal[0].z + bary1 * setup.normal[1].z + bary2 * setup.normal[2].z));
        float* output = surfaceNormalsBuffer + 3 * offset;
        atomicWriteFloat(output, normal.x);
        atomicWriteFloat(output + 1, normal.y);
        atomicWriteFloat(output + 2, normal.z);
      }
    }
  }
}

} // namespace

cudaError_t launchCudaRasterizeMeshes(
    const float* positionsModel,
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
    cudaStream_t stream) {
  if (numTriangles == 0) {
    return cudaSuccess;
  }

  const dim3 grid(static_cast<uint32_t>(numTriangles), static_cast<uint32_t>(batchSize));
  rasterizeKernel<<<grid, kRasterThreadsPerBlock, 0, stream>>>(
      positionsModel,
      normalsModel,
      numVertices,
      triangles,
      numTriangles,
      camera,
      nearClip,
      depthOffset,
      backfaceCulling,
      true,
      rowStride,
      imageStride,
      zBuffer,
      triangleIndexBuffer,
      vertexIndexBuffer,
      surfaceNormalsBuffer);
  if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
    return error;
  }

  if (triangleIndexBuffer != nullptr || vertexIndexBuffer != nullptr ||
      surfaceNormalsBuffer != nullptr) {
    rasterizeKernel<<<grid, kRasterThreadsPerBlock, 0, stream>>>(
        positionsModel,
        normalsModel,
        numVertices,
        triangles,
        numTriangles,
        camera,
        nearClip,
        depthOffset,
        backfaceCulling,
        false,
        rowStride,
        imageStride,
        zBuffer,
        triangleIndexBuffer,
        vertexIndexBuffer,
        surfaceNormalsBuffer);
    return cudaGetLastError();
  }
  return cudaSuccess;
}

cudaError_t launchCudaRasterizeMesh(
    const float* positionsModel,
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
    cudaStream_t stream) {
  return launchCudaRasterizeMeshes(
      positionsModel,
      normalsModel,
      1,
      numVertices,
      triangles,
      numTriangles,
      camera,
      nearClip,
      depthOffset,
      backfaceCulling,
      rowStride,
      static_cast<int64_t>(camera.imageHeight) * rowStride,
      zBuffer,
      triangleIndexBuffer,
      vertexIndexBuffer,
      surfaceNormalsBuffer,
      stream);
}

} // namespace momentum::rasterizer::detail
