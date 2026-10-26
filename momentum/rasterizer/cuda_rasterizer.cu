/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/cuda_rasterizer_kernels.cuh"

#include <cuda_runtime.h>

#include <cfloat>

namespace momentum::rasterizer::detail {
namespace {

constexpr int32_t kRasterThreadsPerBlock = 32;

struct TriangleSetup {
  float u[3];
  float v[3];
  float3 eye[3];
  float3 normal[3];
  int32_t vertexIndex[3];
  int32_t textureIndex[3];
  float edgeSlopeHi[3];
  float edgeSlopeLo[3];
  double edgeRowSlope[3];
  double edgeConstant[3];
  float edgeThreshold[3];
  float denominator;
  int32_t robustEdges;
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

__device__ float dot(float3 first, float3 second) {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

__device__ float3 add(float3 first, float3 second) {
  return make_float3(first.x + second.x, first.y + second.y, first.z + second.z);
}

__device__ float3 multiply(float3 first, float3 second) {
  return make_float3(first.x * second.x, first.y * second.y, first.z * second.z);
}

__device__ float3 multiply(float value, float3 vector) {
  return make_float3(value * vector.x, value * vector.y, value * vector.z);
}

__device__ float3
interpolate(float3 first, float3 second, float3 third, float bary0, float bary1, float bary2) {
  return make_float3(
      bary0 * first.x + bary1 * second.x + bary2 * third.x,
      bary0 * first.y + bary1 * second.y + bary2 * third.y,
      bary0 * first.z + bary1 * second.z + bary2 * third.z);
}

__device__ float3 loadTexturePixel(const float* texture, int32_t width, int32_t x, int32_t y) {
  const int64_t offset = 3 * (static_cast<int64_t>(y) * width + x);
  return make_float3(texture[offset], texture[offset + 1], texture[offset + 2]);
}

__device__ float3
sampleTexture(const float* texture, int32_t height, int32_t width, float u, float v) {
  const float pixelX = fminf(fmaxf(u, 0.0f), 1.0f) * float(width - 1);
  const float pixelY = fminf(fmaxf(v, 0.0f), 1.0f) * float(height - 1);
  const int32_t lowX = static_cast<int32_t>(floorf(pixelX));
  const int32_t lowY = static_cast<int32_t>(floorf(pixelY));
  const int32_t highX = min(lowX + 1, width - 1);
  const int32_t highY = min(lowY + 1, height - 1);
  const float xWeight = pixelX - float(lowX);
  const float yWeight = pixelY - float(lowY);
  return add(
      add(multiply(
              (1.0f - xWeight) * (1.0f - yWeight), loadTexturePixel(texture, width, lowX, lowY)),
          multiply((1.0f - xWeight) * yWeight, loadTexturePixel(texture, width, lowX, highY))),
      add(multiply(xWeight * (1.0f - yWeight), loadTexturePixel(texture, width, highX, lowY)),
          multiply(xWeight * yWeight, loadTexturePixel(texture, width, highX, highY))));
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

__device__ void shadeFragment(
    const TriangleSetup& setup,
    float bary0,
    float bary1,
    float bary2,
    const CudaShadingParameters& shading,
    float3 normal,
    float* rgbOutput) {
  float textureU = 0.0f;
  float textureV = 0.0f;
  if (shading.textureCoordinates != nullptr &&
      (shading.diffuseTexture != nullptr || shading.emissiveTexture != nullptr)) {
    const int32_t first = setup.textureIndex[0];
    const int32_t second = setup.textureIndex[1];
    const int32_t third = setup.textureIndex[2];
    textureU = bary0 * shading.textureCoordinates[2 * first] +
        bary1 * shading.textureCoordinates[2 * second] +
        bary2 * shading.textureCoordinates[2 * third];
    textureV = bary0 * shading.textureCoordinates[2 * first + 1] +
        bary1 * shading.textureCoordinates[2 * second + 1] +
        bary2 * shading.textureCoordinates[2 * third + 1];
  }

  float3 diffuseColor = shading.diffuseColor;
  if (shading.perVertexDiffuseColor != nullptr) {
    const float3 first = make_float3(
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[0]],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[0] + 1],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[0] + 2]);
    const float3 second = make_float3(
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[1]],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[1] + 1],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[1] + 2]);
    const float3 third = make_float3(
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[2]],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[2] + 1],
        shading.perVertexDiffuseColor[3 * setup.vertexIndex[2] + 2]);
    diffuseColor = interpolate(first, second, third, bary0, bary1, bary2);
  }
  if (shading.diffuseTexture != nullptr) {
    diffuseColor = sampleTexture(
        shading.diffuseTexture,
        shading.diffuseTextureHeight,
        shading.diffuseTextureWidth,
        textureU,
        textureV);
  }

  float3 shaded = shading.emissiveColor;
  if (shading.emissiveTexture != nullptr) {
    shaded =
        add(shaded,
            sampleTexture(
                shading.emissiveTexture,
                shading.emissiveTextureHeight,
                shading.emissiveTextureWidth,
                textureU,
                textureV));
  }

  const float3 position =
      interpolate(setup.eye[0], setup.eye[1], setup.eye[2], bary0, bary1, bary2);
  const float3 viewVector = normalizeVector(multiply(-1.0f, position));
  const bool hasSpecular = shading.specularExponent != 0.0f &&
      (shading.specularColor.x != 0.0f || shading.specularColor.y != 0.0f ||
       shading.specularColor.z != 0.0f);
  for (int32_t index = 0; index < shading.numLights; ++index) {
    const CudaLight& light = shading.lights[index];
    if (light.type == CudaLightType::Ambient) {
      shaded = add(shaded, multiply(diffuseColor, light.color));
      continue;
    }

    const float3 lightVector = light.type == CudaLightType::Directional
        ? multiply(-1.0f, light.position)
        : normalizeVector(subtract(light.position, position));
    const float intensity = fminf(fmaxf(dot(lightVector, normal), 0.0f), 1.0f);
    shaded = add(shaded, multiply(intensity, multiply(diffuseColor, light.color)));
    if (hasSpecular) {
      const float3 reflected =
          subtract(multiply(2.0f * dot(lightVector, normal), normal), lightVector);
      const float specularIntensity =
          powf(fminf(fmaxf(dot(reflected, viewVector), 0.0f), 1.0f), shading.specularExponent);
      shaded =
          add(shaded, multiply(specularIntensity, multiply(shading.specularColor, light.color)));
    }
  }

  atomicWriteFloat(rgbOutput, fminf(fmaxf(shaded.x, 0.0f), 1.0f));
  atomicWriteFloat(rgbOutput + 1, fminf(fmaxf(shaded.y, 0.0f), 1.0f));
  atomicWriteFloat(rgbOutput + 2, fminf(fmaxf(shaded.z, 0.0f), 1.0f));
}

__device__ __forceinline__ void rasterizeFragment(
    const TriangleSetup& setup,
    size_t triangleIndex,
    int32_t x,
    int32_t y,
    float lambda0,
    float lambda1,
    float lambda2,
    float nearClip,
    float depthOffset,
    bool writeDepth,
    int64_t rowStride,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    int32_t* vertexIndexBuffer,
    float* surfaceNormalsBuffer,
    const CudaShadingParameters& shading,
    float* rgbBuffer) {
  const float reciprocalDepth =
      lambda0 / setup.eye[0].z + lambda1 / setup.eye[1].z + lambda2 / setup.eye[2].z;
  const float depth = 1.0f / reciprocalDepth;
  if (!(depth > nearClip) || !isfinite(depth)) {
    return;
  }

  const float outputDepth = depth + depthOffset;
  if (!isfinite(outputDepth)) {
    return;
  }

  const int64_t offset = static_cast<int64_t>(y) * rowStride + x;
  if (writeDepth) {
    atomicMinDepth(zBuffer + offset, outputDepth);
    return;
  }

  // Both passes execute this same runtime-parameterized kernel, so a given
  // fragment recomputes the same depth bits. Exact matching avoids races
  // between attributes from different triangles at nearly equal depths.
  if (__float_as_uint(zBuffer[offset]) != __float_as_uint(outputDepth)) {
    return;
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
  float3 normal = {};
  if (surfaceNormalsBuffer != nullptr || shading.enabled) {
    normal = normalizeVector(
        interpolate(setup.normal[0], setup.normal[1], setup.normal[2], bary0, bary1, bary2));
  }
  if (surfaceNormalsBuffer != nullptr) {
    float* output = surfaceNormalsBuffer + 3 * offset;
    atomicWriteFloat(output, normal.x);
    atomicWriteFloat(output + 1, normal.y);
    atomicWriteFloat(output + 2, normal.z);
  }
  if (shading.enabled) {
    shadeFragment(setup, bary0, bary1, bary2, shading, normal, rgbBuffer + 3 * offset);
  }
}

__device__ __noinline__ void rasterizeRobustTriangle(
    const TriangleSetup& setup,
    size_t triangleIndex,
    float nearClip,
    float depthOffset,
    bool writeDepth,
    int64_t rowStride,
    float* zBuffer,
    int32_t* triangleIndexBuffer,
    int32_t* vertexIndexBuffer,
    float* surfaceNormalsBuffer,
    const CudaShadingParameters& shading,
    float* rgbBuffer) {
  for (int32_t y = setup.minY; y <= setup.maxY; ++y) {
    const double localY = double(y - setup.minY);
    float rowInterceptHi[3];
    float rowInterceptLo[3];
    for (int32_t edge = 0; edge < 3; ++edge) {
      const double intercept = fma(setup.edgeRowSlope[edge], localY, setup.edgeConstant[edge]);
      rowInterceptHi[edge] = float(intercept);
      rowInterceptLo[edge] = float(intercept - double(rowInterceptHi[edge]));
    }

    for (int32_t x = setup.minX + threadIdx.x; x <= setup.maxX; x += blockDim.x) {
      const float localX = float(x - setup.minX);
      float edgeValue[3];
      for (int32_t edge = 0; edge < 3; ++edge) {
        edgeValue[edge] = fmaf(setup.edgeSlopeHi[edge], localX, rowInterceptHi[edge]) +
            fmaf(setup.edgeSlopeLo[edge], localX, rowInterceptLo[edge]);
      }
      if (edgeValue[0] < setup.edgeThreshold[0] || edgeValue[1] < setup.edgeThreshold[1] ||
          edgeValue[2] < setup.edgeThreshold[2]) {
        continue;
      }

      const float edgeSum = edgeValue[0] + edgeValue[1] + edgeValue[2];
      if (!(edgeSum > 0.0f) || !isfinite(edgeSum)) {
        continue;
      }
      const float inverseEdgeSum = 1.0f / edgeSum;
      rasterizeFragment(
          setup,
          triangleIndex,
          x,
          y,
          edgeValue[0] * inverseEdgeSum,
          edgeValue[1] * inverseEdgeSum,
          edgeValue[2] * inverseEdgeSum,
          nearClip,
          depthOffset,
          writeDepth,
          rowStride,
          zBuffer,
          triangleIndexBuffer,
          vertexIndexBuffer,
          surfaceNormalsBuffer,
          shading,
          rgbBuffer);
    }
  }
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
    float* surfaceNormalsBuffer,
    CudaShadingParameters shading,
    float* rgbBuffer) {
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
  if (rgbBuffer != nullptr) {
    rgbBuffer += 3 * static_cast<int64_t>(batchIndex) * imageStride;
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
      if (shading.enabled && shading.textureCoordinates != nullptr &&
          (shading.diffuseTexture != nullptr || shading.emissiveTexture != nullptr)) {
        const int32_t textureIndex = shading.textureTriangles == nullptr
            ? vertexIndex
            : shading.textureTriangles[3 * triangleIndex + vertex];
        if (textureIndex < 0 ||
            static_cast<size_t>(textureIndex) >= shading.numTextureCoordinates) {
          setup.valid = 0;
          break;
        }
        setup.textureIndex[vertex] = textureIndex;
      }
      if (!writeDepth && (surfaceNormalsBuffer != nullptr || shading.enabled) &&
          normalsModel != nullptr) {
        setup.normal[vertex] = transformNormal(
            camera,
            make_float3(
                normalsModel[3 * vertexIndex],
                normalsModel[3 * vertexIndex + 1],
                normalsModel[3 * vertexIndex + 2]));
      }
    }

    if (!writeDepth) {
      if (setup.valid && (surfaceNormalsBuffer != nullptr || shading.enabled) &&
          normalsModel == nullptr) {
        const float3 faceNormal = normalizeVector(
            cross(subtract(setup.eye[1], setup.eye[0]), subtract(setup.eye[2], setup.eye[0])));
        setup.normal[0] = faceNormal;
        setup.normal[1] = faceNormal;
        setup.normal[2] = faceNormal;
      }
    }

    if (setup.valid) {
      // Widen projected-coordinate differences before forming the determinant.
      // The FP64 error bound rejects results that are indistinguishable from
      // zero after the two products and their sum are rounded.
      const double term0 =
          (double(setup.v[1]) - double(setup.v[2])) * (double(setup.u[0]) - double(setup.u[2]));
      const double term1 =
          (double(setup.u[2]) - double(setup.u[1])) * (double(setup.v[0]) - double(setup.v[2]));
      const double denominator = term0 + term1;
      const double denominatorError = 4.0 * DBL_EPSILON * (fabs(term0) + fabs(term1));
      setup.valid &= isfinite(denominator) && fabs(denominator) > denominatorError;
      if (backfaceCulling) {
        setup.valid &= denominator < 0.0;
      }

      setup.denominator = (setup.v[1] - setup.v[2]) * (setup.u[0] - setup.u[2]) +
          (setup.u[2] - setup.u[1]) * (setup.v[0] - setup.v[2]);

      const float minU = fminf(setup.u[0], fminf(setup.u[1], setup.u[2]));
      const float maxU = fmaxf(setup.u[0], fmaxf(setup.u[1], setup.u[2]));
      const float minV = fminf(setup.v[0], fminf(setup.v[1], setup.v[2]));
      const float maxV = fmaxf(setup.v[0], fmaxf(setup.v[1], setup.v[2]));
      setup.minX =
          static_cast<int32_t>(floorf(fminf(fmaxf(minU, 0.0f), float(camera.imageWidth - 1))));
      setup.maxX =
          static_cast<int32_t>(ceilf(fminf(fmaxf(maxU, 0.0f), float(camera.imageWidth - 1))));
      setup.minY =
          static_cast<int32_t>(floorf(fminf(fmaxf(minV, 0.0f), float(camera.imageHeight - 1))));
      setup.maxY =
          static_cast<int32_t>(ceilf(fminf(fmaxf(maxV, 0.0f), float(camera.imageHeight - 1))));

      double maxFastEdgeScale = 0.0;
      for (int32_t edge = 0; edge < 3; ++edge) {
        const int32_t first = (edge + 1) % 3;
        const int32_t second = (edge + 2) % 3;
        const double maxDistanceX = fmax(
            fmax(
                fabs(double(setup.minX) - double(setup.u[first])),
                fabs(double(setup.maxX) - double(setup.u[first]))),
            fmax(
                fabs(double(setup.minX) - double(setup.u[second])),
                fabs(double(setup.maxX) - double(setup.u[second]))));
        const double maxDistanceY = fmax(
            fmax(
                fabs(double(setup.minY) - double(setup.v[first])),
                fabs(double(setup.maxY) - double(setup.v[first]))),
            fmax(
                fabs(double(setup.minY) - double(setup.v[second])),
                fabs(double(setup.maxY) - double(setup.v[second]))));
        const double edgeScale =
            fabs(double(setup.v[first]) - double(setup.v[second])) * maxDistanceX +
            fabs(double(setup.u[second]) - double(setup.u[first])) * maxDistanceY;
        maxFastEdgeScale = fmax(maxFastEdgeScale, edgeScale);
      }
      constexpr double kRobustEdgeCondition = 16.0 * double(FLT_EPSILON);
      setup.robustEdges = fabs(denominator) <= kRobustEdgeCondition * maxFastEdgeScale;
      if (!setup.robustEdges) {
        setup.valid &= isfinite(setup.denominator) && setup.denominator != 0.0f;
      }

      if (setup.robustEdges) {
        const double orientation = denominator < 0.0 ? -1.0 : 1.0;
        const double maxLocalX = double(setup.maxX - setup.minX);
        const double maxLocalY = double(setup.maxY - setup.minY);
        // Evaluate robust edges in bounding-box-local coordinates. Splitting the
        // x slope and each FP64 row intercept into FP32 high and low parts keeps
        // the pixel loop on FP32 fused multiply-adds. The edge threshold bounds
        // setup and split-evaluation rounding on this path.
        constexpr double kSplitEvaluationError = 4.0 * double(FLT_EPSILON) * double(FLT_EPSILON);
        constexpr double kDoubleSetupError = 4.0 * DBL_EPSILON;
        for (int32_t edge = 0; edge < 3; ++edge) {
          const int32_t first = (edge + 1) % 3;
          const int32_t second = (edge + 2) % 3;
          const double slope = orientation * (double(setup.v[first]) - double(setup.v[second]));
          const double rowSlope = orientation * (double(setup.u[second]) - double(setup.u[first]));
          const double firstAnchorDistance = fabs(double(setup.minX) - double(setup.u[first])) +
              fabs(double(setup.minY) - double(setup.v[first]));
          const double secondAnchorDistance = fabs(double(setup.minX) - double(setup.u[second])) +
              fabs(double(setup.minY) - double(setup.v[second]));
          const bool useFirstAnchor = firstAnchorDistance < secondAnchorDistance ||
              (firstAnchorDistance == secondAnchorDistance &&
               (setup.u[first] < setup.u[second] ||
                (setup.u[first] == setup.u[second] && setup.v[first] < setup.v[second])));
          const int32_t anchor = useFirstAnchor ? first : second;
          const double localVertexX = double(setup.minX) - double(setup.u[anchor]);
          const double localVertexY = double(setup.minY) - double(setup.v[anchor]);
          const double constant = fma(slope, localVertexX, rowSlope * localVertexY);
          setup.edgeSlopeHi[edge] = float(slope);
          setup.edgeSlopeLo[edge] = float(slope - double(setup.edgeSlopeHi[edge]));
          setup.edgeRowSlope[edge] = rowSlope;
          setup.edgeConstant[edge] = constant;
          const double maxEdgeValue =
              fabs(slope) * maxLocalX + fabs(rowSlope) * maxLocalY + fabs(constant);
          const double setupMagnitude =
              fabs(slope) * fabs(localVertexX) + fabs(rowSlope) * fabs(localVertexY);
          const double edgeError = kSplitEvaluationError * maxEdgeValue +
              kDoubleSetupError * (setupMagnitude + maxEdgeValue);
          setup.edgeThreshold[edge] = float(-edgeError);
          setup.valid &= isfinite(float(maxEdgeValue)) && isfinite(setup.edgeThreshold[edge]);
        }
      }
    }
  }
  __syncthreads();

  if (!setup.valid) {
    return;
  }

  if (!setup.robustEdges) {
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
      rasterizeFragment(
          setup,
          triangleIndex,
          x,
          y,
          lambda0,
          lambda1,
          lambda2,
          nearClip,
          depthOffset,
          writeDepth,
          rowStride,
          zBuffer,
          triangleIndexBuffer,
          vertexIndexBuffer,
          surfaceNormalsBuffer,
          shading,
          rgbBuffer);
    }
    return;
  }

  rasterizeRobustTriangle(
      setup,
      triangleIndex,
      nearClip,
      depthOffset,
      writeDepth,
      rowStride,
      zBuffer,
      triangleIndexBuffer,
      vertexIndexBuffer,
      surfaceNormalsBuffer,
      shading,
      rgbBuffer);
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
    const CudaShadingParameters& shading,
    float* rgbBuffer,
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
      surfaceNormalsBuffer,
      shading,
      rgbBuffer);
  if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
    return error;
  }

  if (triangleIndexBuffer != nullptr || vertexIndexBuffer != nullptr ||
      surfaceNormalsBuffer != nullptr || rgbBuffer != nullptr) {
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
        surfaceNormalsBuffer,
        shading,
        rgbBuffer);
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
    const CudaShadingParameters& shading,
    float* rgbBuffer,
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
      shading,
      rgbBuffer,
      stream);
}

} // namespace momentum::rasterizer::detail
