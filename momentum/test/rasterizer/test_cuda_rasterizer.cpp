/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/cuda_rasterizer.h>
#include <momentum/rasterizer/rasterizer.h>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace momentum::rasterizer {
namespace {

void throwIfCudaError(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) : size_(size) {
    throwIfCudaError(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() {
    return data_;
  }

  void copyFromHost(const T* source) {
    throwIfCudaError(
        cudaMemcpy(data_, source, size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

  std::vector<T> copyToHost() const {
    std::vector<T> result(size_);
    throwIfCudaError(
        cudaMemcpy(result.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
    return result;
  }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

struct MeshInputs {
  Eigen::VectorXf positions;
  Eigen::VectorXi triangles;
};

MeshInputs makeOverlappingTriangles() {
  MeshInputs result;
  result.positions.resize(18);
  result.positions << -1.0f, -0.8f, 2.0f, -1.0f, 0.8f, 2.0f, 1.0f, -0.8f, 2.0f, -0.4f, -0.4f, 1.0f,
      -0.4f, 0.4f, 1.0f, 0.4f, -0.4f, 1.0f;
  result.triangles.resize(6);
  result.triangles << 0, 1, 2, 3, 4, 5;
  return result;
}

std::vector<std::pair<std::string, Camera>> makeTestCameras() {
  constexpr int32_t kWidth = 63;
  constexpr int32_t kHeight = 48;
  constexpr float kFx = 40.0f;
  constexpr float kFy = 38.0f;
  constexpr float kCx = 31.5f;
  constexpr float kCy = 24.0f;

  OpenCVDistortionParametersT<float> openCVDistortion;
  openCVDistortion.k1 = 0.04f;
  openCVDistortion.k2 = -0.01f;
  openCVDistortion.k4 = 0.005f;
  openCVDistortion.p1 = 0.001f;
  openCVDistortion.p2 = -0.0005f;

  OpenCVFisheyeDistortionParametersT<float> fisheyeDistortion;
  fisheyeDistortion.k1 = 0.02f;
  fisheyeDistortion.k2 = -0.005f;

  Eigen::Affine3f eyeFromWorld = Eigen::Affine3f::Identity();
  eyeFromWorld.linear() = Eigen::AngleAxisf(0.04f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  eyeFromWorld.translation() = Eigen::Vector3f(-0.05f, 0.03f, -0.2f);

  return {
      {"pinhole",
       Camera(
           std::make_shared<PinholeIntrinsicsModel>(kWidth, kHeight, kFx, kFy, kCx, kCy),
           eyeFromWorld)},
      {"opencv",
       Camera(
           std::make_shared<OpenCVIntrinsicsModel>(
               kWidth, kHeight, kFx, kFy, kCx, kCy, openCVDistortion),
           eyeFromWorld)},
      {"fisheye",
       Camera(
           std::make_shared<OpenCVFisheyeIntrinsicsModel>(
               kWidth, kHeight, kFx, kFy, kCx, kCy, fisheyeDistortion),
           eyeFromWorld)},
  };
}

Eigen::Matrix4f makeModelMatrix() {
  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result(0, 0) = 0.9f;
  result(1, 1) = 1.1f;
  result(0, 3) = 0.1f;
  result(2, 3) = 0.3f;
  result(3, 0) = 0.02f;
  return result;
}

struct CudaRenderResult {
  std::vector<float> depth;
  std::vector<int32_t> triangleIndex;
  int64_t rowStride = 0;
};

CudaRenderResult renderCuda(
    const MeshInputs& mesh,
    const Camera& camera,
    float depthOffset = 0.0f,
    Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity(),
    const std::vector<float>* initialDepthOverride = nullptr,
    const std::vector<int32_t>* initialTriangleIndexOverride = nullptr) {
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<int32_t> initialTriangleIndex(bufferSize, -1);
  if (initialDepthOverride != nullptr) {
    initialDepth = *initialDepthOverride;
  }
  if (initialTriangleIndexOverride != nullptr) {
    initialTriangleIndex = *initialTriangleIndexOverride;
  }
  if (initialDepth.size() != bufferSize || initialTriangleIndex.size() != bufferSize) {
    throw std::invalid_argument("Initial CUDA rasterizer buffers have the wrong size");
  }

  DeviceBuffer<float> positions(static_cast<size_t>(mesh.positions.size()));
  DeviceBuffer<int32_t> triangles(static_cast<size_t>(mesh.triangles.size()));
  DeviceBuffer<uint64_t> visibility(bufferSize);
  DeviceBuffer<float> depth(bufferSize);
  DeviceBuffer<int32_t> triangleIndex(bufferSize);
  positions.copyFromHost(mesh.positions.data());
  triangles.copyFromHost(mesh.triangles.data());
  depth.copyFromHost(initialDepth.data());
  triangleIndex.copyFromHost(initialTriangleIndex.data());

  rasterizeMeshCuda(
      CudaRasterizerInputs{
          .positionsModel = positions.data(),
          .positionsModelSize = static_cast<size_t>(mesh.positions.size()),
          .numVertices = static_cast<size_t>(mesh.positions.size() / 3),
          .triangles = triangles.data(),
          .trianglesSize = static_cast<size_t>(mesh.triangles.size()),
          .numTriangles = static_cast<size_t>(mesh.triangles.size() / 3),
      },
      camera,
      modelMatrix,
      0.1f,
      CudaRasterizerBuffers{
          .visibilityBuffer = visibility.data(),
          .visibilityBufferSize = bufferSize,
          .zBuffer = depth.data(),
          .zBufferSize = bufferSize,
          .triangleIndexBuffer = triangleIndex.data(),
          .triangleIndexBufferSize = bufferSize,
          .rowStride = rowStride,
      },
      true,
      depthOffset);
  throwIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  return {
      .depth = depth.copyToHost(),
      .triangleIndex = triangleIndex.copyToHost(),
      .rowStride = rowStride,
  };
}

TEST(CudaRasterizerTest, DepthAndTriangleIdsMatchCpuForSupportedCameras) {
  const MeshInputs mesh = makeOverlappingTriangles();
  const Eigen::Matrix4f modelMatrix = makeModelMatrix();
  const Eigen::VectorXf emptyFloat;
  const Eigen::VectorXi emptyInt;

  for (const auto& [name, camera] : makeTestCameras()) {
    SCOPED_TRACE(name);
    Tensor2f cpuDepth = makeRasterizerZBuffer(camera);
    Tensor2i cpuTriangleIndex = makeRasterizerIndexBuffer(camera);
    rasterizeMesh(
        mesh.positions,
        emptyFloat,
        mesh.triangles,
        emptyFloat,
        emptyInt,
        emptyFloat,
        camera,
        modelMatrix,
        0.1f,
        PhongMaterial{},
        cpuDepth.view(),
        {},
        {},
        {},
        cpuTriangleIndex.view());

    const CudaRenderResult gpu = renderCuda(mesh, camera, 0.0f, modelMatrix);
    EXPECT_GT(gpu.rowStride, camera.imageWidth());
    int32_t coveredPixels = 0;
    bool sawTriangle0 = false;
    bool sawTriangle1 = false;
    int32_t coverageDifferences = 0;
    int32_t triangleDifferences = 0;
    for (int32_t y = 0; y < camera.imageHeight(); ++y) {
      for (int32_t x = 0; x < camera.imageWidth(); ++x) {
        const size_t offset = static_cast<size_t>(y * gpu.rowStride + x);
        const bool cpuCovered = cpuDepth(y, x) < std::numeric_limits<float>::max();
        const bool gpuCovered = std::isfinite(gpu.depth[offset]);
        if (cpuCovered) {
          ++coveredPixels;
          sawTriangle0 |= cpuTriangleIndex(y, x) == 0;
          sawTriangle1 |= cpuTriangleIndex(y, x) == 1;
        }
        coverageDifferences += cpuCovered != gpuCovered;
        if (cpuCovered && gpuCovered) {
          const bool sameTriangle = cpuTriangleIndex(y, x) == gpu.triangleIndex[offset];
          triangleDifferences += !sameTriangle;
          if (sameTriangle) {
            EXPECT_NEAR(cpuDepth(y, x), gpu.depth[offset], 2e-4f);
          }
        }
      }
    }

    EXPECT_GT(coveredPixels, 0);
    EXPECT_TRUE(sawTriangle0);
    EXPECT_TRUE(sawTriangle1);
    EXPECT_LE(coverageDifferences, 4);
    EXPECT_LE(triangleDifferences, 4);
  }
}

TEST(CudaRasterizerTest, BatchedRenderingMatchesIndependentRenders) {
  const MeshInputs firstMesh = makeOverlappingTriangles();
  MeshInputs secondMesh = firstMesh;
  for (Eigen::Index index = 0; index < secondMesh.positions.size(); index += 3) {
    secondMesh.positions[index] += 0.15f;
  }

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const int64_t imageStride = static_cast<int64_t>(camera.imageHeight()) * rowStride;
  constexpr size_t kBatchSize = 2;
  const size_t bufferSize = kBatchSize * static_cast<size_t>(imageStride);
  std::vector<float> positions(
      static_cast<size_t>(firstMesh.positions.size() + secondMesh.positions.size()));
  std::copy(
      firstMesh.positions.data(),
      firstMesh.positions.data() + firstMesh.positions.size(),
      positions.begin());
  std::copy(
      secondMesh.positions.data(),
      secondMesh.positions.data() + secondMesh.positions.size(),
      positions.begin() + firstMesh.positions.size());
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<int32_t> initialTriangleIndex(bufferSize, -1);

  DeviceBuffer<float> devicePositions(positions.size());
  DeviceBuffer<int32_t> deviceTriangles(static_cast<size_t>(firstMesh.triangles.size()));
  DeviceBuffer<uint64_t> visibility(bufferSize);
  DeviceBuffer<float> depth(bufferSize);
  DeviceBuffer<int32_t> triangleIndex(bufferSize);
  devicePositions.copyFromHost(positions.data());
  deviceTriangles.copyFromHost(firstMesh.triangles.data());
  depth.copyFromHost(initialDepth.data());
  triangleIndex.copyFromHost(initialTriangleIndex.data());

  rasterizeMeshesCuda(
      CudaRasterizerInputs{
          .positionsModel = devicePositions.data(),
          .positionsModelSize = positions.size(),
          .numVertices = static_cast<size_t>(firstMesh.positions.size() / 3),
          .triangles = deviceTriangles.data(),
          .trianglesSize = static_cast<size_t>(firstMesh.triangles.size()),
          .numTriangles = static_cast<size_t>(firstMesh.triangles.size() / 3),
      },
      kBatchSize,
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      CudaBatchedRasterizerBuffers{
          .visibilityBuffer = visibility.data(),
          .visibilityBufferSize = bufferSize,
          .zBuffer = depth.data(),
          .zBufferSize = bufferSize,
          .triangleIndexBuffer = triangleIndex.data(),
          .triangleIndexBufferSize = bufferSize,
          .rowStride = rowStride,
          .imageStride = imageStride,
      });
  throwIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  const std::vector<float> batchDepth = depth.copyToHost();
  const std::vector<int32_t> batchTriangleIndex = triangleIndex.copyToHost();
  for (const auto& [batchIndex, mesh] :
       std::vector<std::pair<size_t, const MeshInputs*>>{{0, &firstMesh}, {1, &secondMesh}}) {
    const CudaRenderResult independent = renderCuda(*mesh, camera);
    const size_t begin = batchIndex * static_cast<size_t>(imageStride);
    EXPECT_EQ(
        std::vector<float>(batchDepth.begin() + begin, batchDepth.begin() + begin + imageStride),
        independent.depth);
    EXPECT_EQ(
        std::vector<int32_t>(
            batchTriangleIndex.begin() + begin, batchTriangleIndex.begin() + begin + imageStride),
        independent.triangleIndex);
  }
}

TEST(CudaRasterizerTest, TieBreakAndNegativeDepthOffsetAreDeterministic) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << -0.5f, -0.5f, 1.0f, -0.5f, 0.5f, 1.0f, 0.5f, -0.5f, 1.0f;
  mesh.triangles.resize(6);
  mesh.triangles << 0, 1, 2, 0, 1, 2;

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const CudaRenderResult first = renderCuda(mesh, camera, -1.5f);
  const CudaRenderResult second = renderCuda(mesh, camera, -1.5f);
  const size_t interior = static_cast<size_t>(20 * first.rowStride + 24);

  EXPECT_EQ(first.triangleIndex[interior], 0);
  EXPECT_NEAR(first.depth[interior], -0.5f, 1e-5f);
  EXPECT_EQ(second.triangleIndex, first.triangleIndex);
  EXPECT_EQ(second.depth, first.depth);
}

TEST(CudaRasterizerTest, ExistingDepthWinsAndPaddedRowsRemainUsable) {
  const MeshInputs mesh = makeOverlappingTriangles();
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<int32_t> initialTriangleIndex(bufferSize, -1);
  const size_t preservedPixel = static_cast<size_t>(16 * rowStride + 24);
  initialDepth[preservedPixel] = 0.5f;
  initialTriangleIndex[preservedPixel] = 17;

  const CudaRenderResult result = renderCuda(
      mesh, camera, 0.0f, Eigen::Matrix4f::Identity(), &initialDepth, &initialTriangleIndex);
  const size_t rasterizedPixel = static_cast<size_t>(20 * rowStride + 32);

  EXPECT_GT(result.rowStride, camera.imageWidth());
  EXPECT_EQ(result.triangleIndex[preservedPixel], 17);
  EXPECT_FLOAT_EQ(result.depth[preservedPixel], 0.5f);
  EXPECT_EQ(result.triangleIndex[rasterizedPixel], 1);
  EXPECT_NEAR(result.depth[rasterizedPixel], 1.0f, 1e-5f);
}

TEST(CudaRasterizerTest, InvalidBackFacingAndNearClippedTrianglesDoNotWrite) {
  MeshInputs mesh;
  mesh.positions.resize(27);
  mesh.positions << -0.5f, -0.5f, 1.0f, -0.5f, 0.5f, 1.0f, 0.5f, -0.5f, 1.0f, -0.5f, -0.5f, 0.05f,
      -0.5f, 0.5f, 0.05f, 0.5f, -0.5f, 0.05f, -0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, -0.5f,
      0.5f;
  mesh.triangles.resize(15);
  mesh.triangles << 0, 1, 2, 6, 8, 7, 3, 4, 5, 0, 1, 99, 0, -1, 2;

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const CudaRenderResult result = renderCuda(mesh, camera);
  int32_t coveredPixels = 0;
  for (int32_t y = 0; y < camera.imageHeight(); ++y) {
    for (int32_t x = 0; x < camera.imageWidth(); ++x) {
      const int32_t triangleIndex =
          result.triangleIndex[static_cast<size_t>(y * result.rowStride + x)];
      if (triangleIndex >= 0) {
        ++coveredPixels;
        EXPECT_EQ(triangleIndex, 0);
      }
    }
  }
  EXPECT_GT(coveredPixels, 0);
}

TEST(CudaRasterizerTest, RejectsUndersizedBuffersBeforeLaunch) {
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  uint64_t visibility = 0;
  float depth = 0.0f;
  int32_t triangleIndex = -1;
  const CudaRasterizerBuffers validBuffers{
      .visibilityBuffer = &visibility,
      .visibilityBufferSize = bufferSize,
      .zBuffer = &depth,
      .zBufferSize = bufferSize,
      .triangleIndexBuffer = &triangleIndex,
      .triangleIndexBufferSize = bufferSize,
      .rowStride = rowStride,
  };
  float positions[3]{};
  int32_t triangles[3]{};
  const CudaRasterizerInputs validInputs{
      .positionsModel = positions,
      .positionsModelSize = 3,
      .numVertices = 1,
      .triangles = triangles,
      .trianglesSize = 3,
      .numTriangles = 1,
  };
  const auto expectRejected = [&](CudaRasterizerInputs inputs, CudaRasterizerBuffers buffers) {
    EXPECT_THROW(
        rasterizeMeshCuda(inputs, camera, Eigen::Matrix4f::Identity(), 0.1f, buffers),
        std::runtime_error);
  };

  auto buffers = validBuffers;
  buffers.visibilityBufferSize = bufferSize - 1;
  expectRejected(validInputs, buffers);
  buffers = validBuffers;
  buffers.zBufferSize = bufferSize - 1;
  expectRejected(validInputs, buffers);
  buffers = validBuffers;
  buffers.triangleIndexBufferSize = bufferSize - 1;
  expectRejected(validInputs, buffers);

  auto inputs = validInputs;
  inputs.positionsModelSize = 2;
  expectRejected(inputs, validBuffers);
  inputs = validInputs;
  inputs.trianglesSize = 2;
  expectRejected(inputs, validBuffers);
  buffers = validBuffers;
  buffers.rowStride = std::numeric_limits<int64_t>::max();
  expectRejected(validInputs, buffers);

  constexpr size_t kBatchSize = 2;
  const size_t batchedBufferSize = kBatchSize * bufferSize;
  float batchedPositions[6]{};
  const CudaRasterizerInputs validBatchedInputs{
      .positionsModel = batchedPositions,
      .positionsModelSize = 6,
      .numVertices = 1,
      .triangles = triangles,
      .trianglesSize = 3,
      .numTriangles = 1,
  };
  const CudaBatchedRasterizerBuffers validBatchedBuffers{
      .visibilityBuffer = &visibility,
      .visibilityBufferSize = batchedBufferSize,
      .zBuffer = &depth,
      .zBufferSize = batchedBufferSize,
      .triangleIndexBuffer = &triangleIndex,
      .triangleIndexBufferSize = batchedBufferSize,
      .rowStride = rowStride,
      .imageStride = static_cast<int64_t>(bufferSize),
  };
  const auto expectBatchedRejected = [&](CudaRasterizerInputs inputs,
                                         CudaBatchedRasterizerBuffers batchedBuffers) {
    EXPECT_THROW(
        rasterizeMeshesCuda(
            inputs, kBatchSize, camera, Eigen::Matrix4f::Identity(), 0.1f, batchedBuffers),
        std::runtime_error);
  };

  auto batchedBuffers = validBatchedBuffers;
  batchedBuffers.visibilityBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);
  batchedBuffers = validBatchedBuffers;
  batchedBuffers.zBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);
  batchedBuffers = validBatchedBuffers;
  batchedBuffers.triangleIndexBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);

  inputs = validBatchedInputs;
  inputs.positionsModelSize = 5;
  expectBatchedRejected(inputs, validBatchedBuffers);
}

} // namespace
} // namespace momentum::rasterizer
