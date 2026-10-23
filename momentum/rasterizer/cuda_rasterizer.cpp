/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/rasterizer/cuda_rasterizer.h"

#include "momentum/common/exception.h"
#include "momentum/rasterizer/cuda_rasterizer_kernels.cuh"

#include <Eigen/LU>

#include <limits>

namespace momentum::rasterizer {
namespace {

void throwIfCudaError(cudaError_t error, const char* operation) {
  MT_THROW_IF(error != cudaSuccess, "{} failed: {}", operation, cudaGetErrorString(error));
}

detail::CudaCamera makeCudaCamera(const Camera& camera, const Eigen::Matrix4f& modelMatrix) {
  detail::CudaCamera result;
  result.imageWidth = camera.imageWidth();
  result.imageHeight = camera.imageHeight();
  result.fx = camera.fx();
  result.fy = camera.fy();

  const auto* intrinsics = camera.intrinsicsModel().get();
  if (const auto* pinhole = dynamic_cast<const PinholeIntrinsicsModel*>(intrinsics)) {
    result.model = detail::CudaCameraModel::Pinhole;
    result.cx = pinhole->cx();
    result.cy = pinhole->cy();
  } else if (const auto* openCV = dynamic_cast<const OpenCVIntrinsicsModel*>(intrinsics)) {
    result.model = detail::CudaCameraModel::OpenCV;
    result.cx = openCV->cx();
    result.cy = openCV->cy();
    const auto& distortion = openCV->distortionParameters();
    result.distortion[0] = distortion.k1;
    result.distortion[1] = distortion.k2;
    result.distortion[2] = distortion.k3;
    result.distortion[3] = distortion.k4;
    result.distortion[4] = distortion.k5;
    result.distortion[5] = distortion.k6;
    result.distortion[6] = distortion.p1;
    result.distortion[7] = distortion.p2;
    result.distortion[8] = distortion.p3;
    result.distortion[9] = distortion.p4;
  } else if (const auto* fisheye = dynamic_cast<const OpenCVFisheyeIntrinsicsModel*>(intrinsics)) {
    result.model = detail::CudaCameraModel::OpenCVFisheye;
    result.cx = fisheye->cx();
    result.cy = fisheye->cy();
    const auto& distortion = fisheye->distortionParameters();
    result.distortion[0] = distortion.k1;
    result.distortion[1] = distortion.k2;
    result.distortion[2] = distortion.k3;
    result.distortion[3] = distortion.k4;
    result.maxRSquared = fisheye->maxRSquared();
  } else {
    MT_THROW("CUDA rasterization does not support this camera intrinsics model");
  }

  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t column = 0; column < 4; ++column) {
      result.modelToWorld[4 * row + column] = modelMatrix(row, column);
    }
  }

  const Eigen::Affine3f& eyeFromWorld = camera.eyeFromWorld();
  for (int32_t row = 0; row < 3; ++row) {
    for (int32_t column = 0; column < 3; ++column) {
      result.worldToEye[4 * row + column] = eyeFromWorld.linear()(row, column);
    }
    result.worldToEye[4 * row + 3] = eyeFromWorld.translation()(row);
  }

  const Eigen::Matrix3f linearToEye = eyeFromWorld.linear() * modelMatrix.topLeftCorner<3, 3>();
  Eigen::Matrix3f normalToEye;
  normalToEye.col(0) = linearToEye.col(1).cross(linearToEye.col(2));
  normalToEye.col(1) = linearToEye.col(2).cross(linearToEye.col(0));
  normalToEye.col(2) = linearToEye.col(0).cross(linearToEye.col(1));
  if (linearToEye.determinant() < 0.0f) {
    normalToEye = -normalToEye;
  }
  for (int32_t row = 0; row < 3; ++row) {
    for (int32_t column = 0; column < 3; ++column) {
      result.normalToEye[3 * row + column] = normalToEye(row, column);
    }
  }

  return result;
}

} // namespace

void rasterizeMeshesCuda(
    CudaRasterizerInputs inputs,
    size_t batchSize,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    CudaBatchedRasterizerBuffers buffers,
    bool backfaceCulling,
    float depthOffset,
    cudaStream_t stream) {
  const auto* positionsModel = inputs.positionsModel;
  const auto* normalsModel = inputs.normalsModel;
  const size_t numVertices = inputs.numVertices;
  const auto* triangles = inputs.triangles;
  const size_t numTriangles = inputs.numTriangles;
  MT_THROW_IF(batchSize == 0 || batchSize > 65535, "CUDA rasterizer batch size is out of range");
  MT_THROW_IF(nearClip <= 0.0f, "Near clip must be positive");
  MT_THROW_IF(
      camera.imageWidth() <= 0 || camera.imageHeight() <= 0, "Camera image must be non-empty");
  MT_THROW_IF(buffers.rowStride < camera.imageWidth(), "CUDA rasterizer row stride is too small");
  const auto imageHeight = static_cast<size_t>(camera.imageHeight());
  const auto rowStride = static_cast<size_t>(buffers.rowStride);
  MT_THROW_IF(
      rowStride > std::numeric_limits<size_t>::max() / imageHeight,
      "CUDA rasterizer output size overflows");
  const size_t imageRowsSize = imageHeight * rowStride;
  MT_THROW_IF(
      buffers.imageStride < 0 || static_cast<size_t>(buffers.imageStride) < imageRowsSize,
      "CUDA rasterizer image stride is too small");
  const auto imageStride = static_cast<size_t>(buffers.imageStride);
  MT_THROW_IF(
      batchSize - 1 > (std::numeric_limits<size_t>::max() - imageRowsSize) / imageStride,
      "CUDA rasterizer output size overflows");
  const size_t requiredBufferSize = (batchSize - 1) * imageStride + imageRowsSize;
  MT_THROW_IF(buffers.zBuffer == nullptr, "CUDA z buffer is null");
  MT_THROW_IF(
      buffers.zBufferSize < requiredBufferSize,
      "CUDA z buffer has {} elements; at least {} are required",
      buffers.zBufferSize,
      requiredBufferSize);
  MT_THROW_IF(
      buffers.triangleIndexBuffer != nullptr &&
          buffers.triangleIndexBufferSize < requiredBufferSize,
      "CUDA triangle-index buffer has {} elements; at least {} are required",
      buffers.triangleIndexBufferSize,
      requiredBufferSize);
  MT_THROW_IF(
      buffers.vertexIndexBuffer != nullptr && buffers.vertexIndexBufferSize < requiredBufferSize,
      "CUDA vertex-index buffer has {} elements; at least {} are required",
      buffers.vertexIndexBufferSize,
      requiredBufferSize);
  MT_THROW_IF(
      buffers.surfaceNormalsBuffer != nullptr &&
          requiredBufferSize > std::numeric_limits<size_t>::max() / 3,
      "CUDA surface-normal buffer size overflows");
  const size_t requiredNormalBufferSize =
      buffers.surfaceNormalsBuffer == nullptr ? 0 : 3 * requiredBufferSize;
  MT_THROW_IF(
      buffers.surfaceNormalsBuffer != nullptr &&
          buffers.surfaceNormalsBufferSize < requiredNormalBufferSize,
      "CUDA surface-normal buffer has {} elements; at least {} are required",
      buffers.surfaceNormalsBufferSize,
      requiredNormalBufferSize);
  MT_THROW_IF(
      buffers.barycentricCoordinatesBuffer != nullptr &&
          requiredBufferSize > std::numeric_limits<size_t>::max() / 3,
      "CUDA barycentric-coordinate buffer size overflows");
  const size_t requiredBarycentricBufferSize =
      buffers.barycentricCoordinatesBuffer == nullptr ? 0 : 3 * requiredBufferSize;
  MT_THROW_IF(
      buffers.barycentricCoordinatesBuffer != nullptr &&
          buffers.barycentricCoordinatesBufferSize < requiredBarycentricBufferSize,
      "CUDA barycentric-coordinate buffer has {} elements; at least {} are required",
      buffers.barycentricCoordinatesBufferSize,
      requiredBarycentricBufferSize);
  MT_THROW_IF(
      numVertices > std::numeric_limits<size_t>::max() / 3 ||
          numTriangles > std::numeric_limits<size_t>::max() / 3,
      "CUDA mesh input size overflows");
  const size_t positionElementsPerMesh = 3 * numVertices;
  MT_THROW_IF(
      positionElementsPerMesh > 0 &&
          batchSize > std::numeric_limits<size_t>::max() / positionElementsPerMesh,
      "CUDA mesh input size overflows");
  const size_t requiredPositionElements = batchSize * positionElementsPerMesh;
  const size_t requiredTriangleElements = 3 * numTriangles;
  MT_THROW_IF(numVertices > 0 && positionsModel == nullptr, "CUDA positions buffer is null");
  MT_THROW_IF(numTriangles > 0 && triangles == nullptr, "CUDA triangle buffer is null");
  MT_THROW_IF(
      inputs.positionsModelSize < requiredPositionElements,
      "CUDA positions buffer has {} elements; at least {} are required",
      inputs.positionsModelSize,
      requiredPositionElements);
  MT_THROW_IF(
      normalsModel != nullptr && inputs.normalsModelSize < requiredPositionElements,
      "CUDA normals buffer has {} elements; at least {} are required",
      inputs.normalsModelSize,
      requiredPositionElements);
  MT_THROW_IF(
      inputs.trianglesSize < requiredTriangleElements,
      "CUDA triangle buffer has {} elements; at least {} are required",
      inputs.trianglesSize,
      requiredTriangleElements);
  MT_THROW_IF(numTriangles > 0 && numVertices == 0, "CUDA mesh has triangles but no vertices");
  MT_THROW_IF(
      numTriangles > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "CUDA rasterizer supports at most {} triangles",
      std::numeric_limits<int32_t>::max());

  int device = 0;
  throwIfCudaError(cudaGetDevice(&device), "CUDA rasterizer device lookup");
  int maxGridDimensionX = 0;
  throwIfCudaError(
      cudaDeviceGetAttribute(&maxGridDimensionX, cudaDevAttrMaxGridDimX, device),
      "CUDA rasterizer grid-limit lookup");
  MT_THROW_IF(
      numTriangles > static_cast<size_t>(maxGridDimensionX),
      "CUDA rasterizer supports at most {} triangles on device {}",
      maxGridDimensionX,
      device);

  throwIfCudaError(
      detail::launchCudaRasterizeMeshes(
          positionsModel,
          normalsModel,
          batchSize,
          numVertices,
          triangles,
          numTriangles,
          makeCudaCamera(camera, modelMatrix),
          nearClip,
          depthOffset,
          backfaceCulling,
          buffers.rowStride,
          buffers.imageStride,
          buffers.zBuffer,
          buffers.triangleIndexBuffer,
          buffers.vertexIndexBuffer,
          buffers.surfaceNormalsBuffer,
          buffers.barycentricCoordinatesBuffer,
          stream),
      "Batched CUDA rasterizer launch");
}

void rasterizeMeshCuda(
    CudaRasterizerInputs inputs,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    CudaRasterizerBuffers buffers,
    bool backfaceCulling,
    float depthOffset,
    cudaStream_t stream) {
  const int64_t imageHeight = camera.imageHeight();
  MT_THROW_IF(imageHeight <= 0 || buffers.rowStride < 0, "CUDA rasterizer output size is invalid");
  MT_THROW_IF(
      buffers.rowStride > std::numeric_limits<int64_t>::max() / imageHeight,
      "CUDA rasterizer output size overflows");
  const int64_t imageStride = imageHeight * buffers.rowStride;
  rasterizeMeshesCuda(
      inputs,
      1,
      camera,
      modelMatrix,
      nearClip,
      CudaBatchedRasterizerBuffers{
          .zBuffer = buffers.zBuffer,
          .zBufferSize = buffers.zBufferSize,
          .triangleIndexBuffer = buffers.triangleIndexBuffer,
          .triangleIndexBufferSize = buffers.triangleIndexBufferSize,
          .vertexIndexBuffer = buffers.vertexIndexBuffer,
          .vertexIndexBufferSize = buffers.vertexIndexBufferSize,
          .surfaceNormalsBuffer = buffers.surfaceNormalsBuffer,
          .surfaceNormalsBufferSize = buffers.surfaceNormalsBufferSize,
          .barycentricCoordinatesBuffer = buffers.barycentricCoordinatesBuffer,
          .barycentricCoordinatesBufferSize = buffers.barycentricCoordinatesBufferSize,
          .rowStride = buffers.rowStride,
          .imageStride = imageStride,
      },
      backfaceCulling,
      depthOffset,
      stream);
}

} // namespace momentum::rasterizer
