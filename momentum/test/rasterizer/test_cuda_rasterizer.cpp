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
#include <array>
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

MeshInputs makeWindowTriangle(
    const std::array<Eigen::Vector2f, 3>& window,
    const std::array<float, 3>& depth) {
  MeshInputs result;
  result.positions.resize(9);
  for (int32_t vertex = 0; vertex < 3; ++vertex) {
    result.positions.segment<3>(3 * vertex) = Eigen::Vector3f(
        window[vertex].x() * depth[vertex], window[vertex].y() * depth[vertex], depth[vertex]);
  }
  result.triangles.resize(3);
  result.triangles << 0, 1, 2;
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
          .zBuffer = depth.data(),
          .zBufferSize = bufferSize,
          .triangleIndexBuffer = triangleIndex.data(),
          .triangleIndexBufferSize = bufferSize,
          .rowStride = rowStride,
      },
      nullptr,
      true,
      depthOffset);
  throwIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  return {
      .depth = depth.copyToHost(),
      .triangleIndex = triangleIndex.copyToHost(),
      .rowStride = rowStride,
  };
}

struct CudaAttributeRenderResult {
  std::vector<float> depth;
  std::vector<int32_t> triangleIndex;
  std::vector<int32_t> vertexIndex;
  std::vector<float> surfaceNormals;
  int64_t rowStride = 0;
};

CudaAttributeRenderResult renderCudaAttributes(
    const MeshInputs& mesh,
    const Eigen::VectorXf* normals,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix = Eigen::Matrix4f::Identity(),
    bool backfaceCulling = true) {
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<int32_t> initialIndex(bufferSize, -1);
  std::vector<float> initialNormals(3 * bufferSize, 0.0f);

  DeviceBuffer<float> positions(static_cast<size_t>(mesh.positions.size()));
  DeviceBuffer<int32_t> triangles(static_cast<size_t>(mesh.triangles.size()));
  std::unique_ptr<DeviceBuffer<float>> deviceNormals;
  if (normals != nullptr) {
    deviceNormals = std::make_unique<DeviceBuffer<float>>(static_cast<size_t>(normals->size()));
    deviceNormals->copyFromHost(normals->data());
  }
  DeviceBuffer<float> depth(bufferSize);
  DeviceBuffer<int32_t> triangleIndex(bufferSize);
  DeviceBuffer<int32_t> vertexIndex(bufferSize);
  DeviceBuffer<float> surfaceNormals(3 * bufferSize);
  positions.copyFromHost(mesh.positions.data());
  triangles.copyFromHost(mesh.triangles.data());
  depth.copyFromHost(initialDepth.data());
  triangleIndex.copyFromHost(initialIndex.data());
  vertexIndex.copyFromHost(initialIndex.data());
  surfaceNormals.copyFromHost(initialNormals.data());

  rasterizeMeshCuda(
      CudaRasterizerInputs{
          .positionsModel = positions.data(),
          .positionsModelSize = static_cast<size_t>(mesh.positions.size()),
          .normalsModel = deviceNormals ? deviceNormals->data() : nullptr,
          .normalsModelSize = normals == nullptr ? 0 : static_cast<size_t>(normals->size()),
          .numVertices = static_cast<size_t>(mesh.positions.size() / 3),
          .triangles = triangles.data(),
          .trianglesSize = static_cast<size_t>(mesh.triangles.size()),
          .numTriangles = static_cast<size_t>(mesh.triangles.size() / 3),
      },
      camera,
      modelMatrix,
      0.1f,
      CudaRasterizerBuffers{
          .zBuffer = depth.data(),
          .zBufferSize = bufferSize,
          .triangleIndexBuffer = triangleIndex.data(),
          .triangleIndexBufferSize = bufferSize,
          .vertexIndexBuffer = vertexIndex.data(),
          .vertexIndexBufferSize = bufferSize,
          .surfaceNormalsBuffer = surfaceNormals.data(),
          .surfaceNormalsBufferSize = 3 * bufferSize,
          .rowStride = rowStride,
      },
      nullptr,
      backfaceCulling);
  throwIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  return {
      .depth = depth.copyToHost(),
      .triangleIndex = triangleIndex.copyToHost(),
      .vertexIndex = vertexIndex.copyToHost(),
      .surfaceNormals = surfaceNormals.copyToHost(),
      .rowStride = rowStride,
  };
}

TEST(CudaRasterizerTest, ShadesDirectlyIntoCallerOwnedRgbBuffer) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << -0.5f, -0.5f, 1.0f, -0.5f, 0.5f, 1.0f, 0.5f, -0.5f, 1.0f;
  mesh.triangles.resize(3);
  mesh.triangles << 0, 1, 2;
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 48, 40.0f, 38.0f, 32.0f, 24.0f));
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<float> initialRgb(3 * bufferSize, 0.0f);
  const size_t preservedPixel = static_cast<size_t>(16 * rowStride + 24);
  initialDepth[preservedPixel] = 0.5f;
  initialRgb[3 * preservedPixel] = 0.9f;
  initialRgb[3 * preservedPixel + 1] = 0.8f;
  initialRgb[3 * preservedPixel + 2] = 0.7f;
  DeviceBuffer<float> positions(static_cast<size_t>(mesh.positions.size()));
  DeviceBuffer<int32_t> triangles(static_cast<size_t>(mesh.triangles.size()));
  DeviceBuffer<float> depth(bufferSize);
  DeviceBuffer<float> rgb(3 * bufferSize);
  positions.copyFromHost(mesh.positions.data());
  triangles.copyFromHost(mesh.triangles.data());
  depth.copyFromHost(initialDepth.data());
  rgb.copyFromHost(initialRgb.data());

  const PhongMaterial material(
      Eigen::Vector3f(0.2f, 0.4f, 0.6f),
      Eigen::Vector3f::Zero(),
      0.0f,
      Eigen::Vector3f(0.05f, 0.02f, 0.01f));
  const Light light = createAmbientLight(Eigen::Vector3f(0.5f, 0.25f, 1.0f));
  const CudaRasterizerShadingInputs shadingInputs{
      .material = &material,
      .lightsWorld = &light,
      .numLights = 1,
  };
  rasterizeMeshCuda(
      CudaRasterizerInputs{
          .positionsModel = positions.data(),
          .positionsModelSize = static_cast<size_t>(mesh.positions.size()),
          .numVertices = 3,
          .triangles = triangles.data(),
          .trianglesSize = static_cast<size_t>(mesh.triangles.size()),
          .numTriangles = 1,
      },
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      CudaRasterizerBuffers{
          .zBuffer = depth.data(),
          .zBufferSize = bufferSize,
          .rgbBuffer = rgb.data(),
          .rgbBufferSize = 3 * bufferSize,
          .rowStride = rowStride,
      },
      &shadingInputs);
  throwIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  const std::vector<float> renderedDepth = depth.copyToHost();
  const std::vector<float> renderedRgb = rgb.copyToHost();
  const size_t shadedPixel = static_cast<size_t>(20 * rowStride + 32);
  EXPECT_NEAR(renderedDepth[shadedPixel], 1.0f, 1e-5f);
  EXPECT_NEAR(renderedRgb[3 * shadedPixel], 0.15f, 1e-5f);
  EXPECT_NEAR(renderedRgb[3 * shadedPixel + 1], 0.12f, 1e-5f);
  EXPECT_NEAR(renderedRgb[3 * shadedPixel + 2], 0.61f, 1e-5f);
  EXPECT_FLOAT_EQ(renderedDepth[preservedPixel], 0.5f);
  EXPECT_FLOAT_EQ(renderedRgb[3 * preservedPixel], 0.9f);
  EXPECT_FLOAT_EQ(renderedRgb[3 * preservedPixel + 1], 0.8f);
  EXPECT_FLOAT_EQ(renderedRgb[3 * preservedPixel + 2], 0.7f);
}

TEST(CudaRasterizerTest, WritesVertexIndicesAndInverseTransposeSurfaceNormals) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << -0.4f, -0.4f, 1.7f, -0.4f, 0.4f, 2.2f, 0.4f, -0.4f, 2.8f;
  mesh.triangles.resize(3);
  mesh.triangles << 0, 1, 2;
  Eigen::VectorXf normals(9);
  const Eigen::Vector3f normalModel = Eigen::Vector3f(1.0f, 2.0f, 3.0f).normalized();
  for (int32_t vertex = 0; vertex < 3; ++vertex) {
    normals.segment<3>(3 * vertex) = normalModel;
  }

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 48, 40.0f, 38.0f, 32.0f, 24.0f));
  Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
  modelMatrix(0, 0) = 1.5f;
  modelMatrix(1, 1) = 0.75f;
  modelMatrix(2, 2) = 1.2f;
  modelMatrix(0, 1) = 0.2f;
  const Eigen::Vector3f expectedNormal =
      (modelMatrix.topLeftCorner<3, 3>().inverse().transpose() * normalModel).normalized();

  const CudaAttributeRenderResult result =
      renderCudaAttributes(mesh, &normals, camera, modelMatrix);
  int32_t coveredPixels = 0;
  for (int32_t y = 0; y < camera.imageHeight(); ++y) {
    for (int32_t x = 0; x < camera.imageWidth(); ++x) {
      const size_t offset = static_cast<size_t>(y * result.rowStride + x);
      if (!std::isfinite(result.depth[offset])) {
        continue;
      }
      ++coveredPixels;
      EXPECT_EQ(result.triangleIndex[offset], 0);
      EXPECT_GE(result.vertexIndex[offset], 0);
      EXPECT_LT(result.vertexIndex[offset], 3);
      EXPECT_NEAR(result.surfaceNormals[3 * offset], expectedNormal.x(), 1e-5f);
      EXPECT_NEAR(result.surfaceNormals[3 * offset + 1], expectedNormal.y(), 1e-5f);
      EXPECT_NEAR(result.surfaceNormals[3 * offset + 2], expectedNormal.z(), 1e-5f);
      const Eigen::Vector3f eyeDepths = modelMatrix(2, 2) *
          Eigen::Vector3f(mesh.positions[2], mesh.positions[5], mesh.positions[8]);
      EXPECT_GE(result.depth[offset], eyeDepths.minCoeff());
      EXPECT_LE(result.depth[offset], eyeDepths.maxCoeff());
    }
  }
  EXPECT_GT(coveredPixels, 0);
}

TEST(CudaRasterizerTest, ConditionedEdgesHandleNearDegenerateTriangles) {
  struct TestCase {
    std::array<Eigen::Vector2f, 3> window;
    std::vector<Eigen::Vector2i> expectedPixels;
  };
  const std::array<TestCase, 4> cases = {{
      {
          .window =
              {Eigen::Vector2f(28.817873001098633f, 33.81787109375f),
               Eigen::Vector2f(37.18212890625f, 42.18212890625f),
               Eigen::Vector2f(36.69539260864258f, 41.695404052734375f)},
          .expectedPixels =
              {Eigen::Vector2i(30, 35),
               Eigen::Vector2i(31, 36),
               Eigen::Vector2i(32, 37),
               Eigen::Vector2i(33, 38),
               Eigen::Vector2i(34, 39),
               Eigen::Vector2i(35, 40),
               Eigen::Vector2i(36, 41),
               Eigen::Vector2i(37, 42)},
      },
      {
          .window =
              {Eigen::Vector2f(34.674339294433594f, 28.674341201782227f),
               Eigen::Vector2f(43.325660705566406f, 37.325660705566406f),
               Eigen::Vector2f(37.590118408203125f, 31.590131759643555f)},
          .expectedPixels = {},
      },
      {
          .window =
              {Eigen::Vector2f(25.729915618896484f, 38.729915618896484f),
               Eigen::Vector2f(28.270084381103516f, 41.270084381103516f),
               Eigen::Vector2f(27.69745635986328f, 40.69745635986328f)},
          .expectedPixels = {},
      },
      {
          .window =
              {Eigen::Vector2f(64.0f, 63.999996185302734f),
               Eigen::Vector2f(63.999996185302734f, 63.99999237060547f),
               Eigen::Vector2f(32.0f, 32.0f)},
          .expectedPixels = {Eigen::Vector2i(32, 32)},
      },
  }};
  const std::array<std::array<int32_t, 3>, 6> permutations = {{
      {0, 1, 2},
      {0, 2, 1},
      {1, 0, 2},
      {1, 2, 0},
      {2, 0, 1},
      {2, 1, 0},
  }};
  const std::array<float, 3> depths = {64.0f, 128.0f, 256.0f};
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 64, 1.0f, 1.0f, 0.0f, 0.0f));

  for (const auto& testCase : cases) {
    for (const auto& permutation : permutations) {
      MeshInputs mesh = makeWindowTriangle(testCase.window, depths);
      mesh.triangles << permutation[0], permutation[1], permutation[2];
      const CudaAttributeRenderResult result =
          renderCudaAttributes(mesh, nullptr, camera, Eigen::Matrix4f::Identity(), false);
      EXPECT_EQ(
          std::count(result.triangleIndex.begin(), result.triangleIndex.end(), 0),
          static_cast<std::ptrdiff_t>(testCase.expectedPixels.size()));

      for (const Eigen::Vector2i& pixel : testCase.expectedPixels) {
        const size_t offset = static_cast<size_t>(pixel.y() * result.rowStride + pixel.x());
        ASSERT_EQ(result.triangleIndex[offset], 0);
        EXPECT_GE(result.depth[offset], depths.front());
        EXPECT_LE(result.depth[offset], depths.back());
        const Eigen::Vector3f normal(
            result.surfaceNormals[3 * offset],
            result.surfaceNormals[3 * offset + 1],
            result.surfaceNormals[3 * offset + 2]);
        EXPECT_TRUE(normal.allFinite());
      }
    }
  }
}

TEST(CudaRasterizerTest, ConditionedAndFastTrianglesPreserveCoverage) {
  const std::array<Eigen::Vector2f, 4> window = {
      Eigen::Vector2f(28.817873001098633f, 33.81787109375f),
      Eigen::Vector2f(37.18212890625f, 42.18212890625f),
      Eigen::Vector2f(36.69539260864258f, 41.695404052734375f),
      Eigen::Vector2f(28.0f, 30.0f),
  };
  constexpr float kDepth = 64.0f;
  MeshInputs mesh;
  mesh.positions.resize(12);
  for (int32_t vertex = 0; vertex < 4; ++vertex) {
    mesh.positions.segment<3>(3 * vertex) =
        Eigen::Vector3f(window[vertex].x() * kDepth, window[vertex].y() * kDepth, kDepth);
  }
  mesh.triangles.resize(6);
  mesh.triangles << 0, 1, 2, 1, 0, 3;
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 64, 1.0f, 1.0f, 0.0f, 0.0f));

  const CudaAttributeRenderResult result =
      renderCudaAttributes(mesh, nullptr, camera, Eigen::Matrix4f::Identity(), false);
  const std::array<Eigen::Vector2i, 18> expectedPixels = {
      Eigen::Vector2i(28, 30),
      Eigen::Vector2i(29, 32),
      Eigen::Vector2i(29, 33),
      Eigen::Vector2i(30, 33),
      Eigen::Vector2i(30, 34),
      Eigen::Vector2i(30, 35),
      Eigen::Vector2i(31, 34),
      Eigen::Vector2i(31, 35),
      Eigen::Vector2i(31, 36),
      Eigen::Vector2i(32, 36),
      Eigen::Vector2i(32, 37),
      Eigen::Vector2i(33, 37),
      Eigen::Vector2i(33, 38),
      Eigen::Vector2i(34, 38),
      Eigen::Vector2i(34, 39),
      Eigen::Vector2i(35, 40),
      Eigen::Vector2i(36, 41),
      Eigen::Vector2i(37, 42),
  };
  EXPECT_EQ(
      std::count_if(
          result.triangleIndex.begin(),
          result.triangleIndex.end(),
          [](int32_t triangleIndex) { return triangleIndex >= 0; }),
      static_cast<std::ptrdiff_t>(expectedPixels.size()));
  for (const Eigen::Vector2i& pixel : expectedPixels) {
    const size_t offset = static_cast<size_t>(pixel.y() * result.rowStride + pixel.x());
    EXPECT_TRUE(std::isfinite(result.depth[offset]));
    EXPECT_GE(result.triangleIndex[offset], 0);
  }
}

TEST(CudaRasterizerTest, RejectsOverflowingFastPathDeterminant) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << 1e20f, 0.0f, 1.0f, 0.0f, 1e20f, 1.0f, 32.0f, 32.0f, 1.0f;
  mesh.triangles.resize(3);
  mesh.triangles << 0, 1, 2;
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 64, 1.0f, 1.0f, 0.0f, 0.0f));

  const CudaAttributeRenderResult result =
      renderCudaAttributes(mesh, nullptr, camera, Eigen::Matrix4f::Identity(), false);
  EXPECT_EQ(std::count(result.triangleIndex.begin(), result.triangleIndex.end(), 0), 0);
}

TEST(CudaRasterizerTest, SingularNormalTransformAndZeroNormalsStayFinite) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << -0.4f, -0.4f, 1.7f, -0.4f, 0.4f, 2.2f, 0.4f, -0.4f, 2.8f;
  mesh.triangles.resize(3);
  mesh.triangles << 0, 1, 2;
  const Eigen::VectorXf normals = Eigen::VectorXf::Zero(9);

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(64, 48, 40.0f, 38.0f, 32.0f, 24.0f));
  Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
  modelMatrix(2, 2) = 0.0f;
  modelMatrix(2, 3) = 2.0f;

  const CudaAttributeRenderResult result =
      renderCudaAttributes(mesh, &normals, camera, modelMatrix);
  int32_t coveredPixels = 0;
  for (int32_t y = 0; y < camera.imageHeight(); ++y) {
    for (int32_t x = 0; x < camera.imageWidth(); ++x) {
      const size_t offset = static_cast<size_t>(y * result.rowStride + x);
      if (!std::isfinite(result.depth[offset])) {
        continue;
      }
      ++coveredPixels;
      EXPECT_TRUE(std::isfinite(result.surfaceNormals[3 * offset]));
      EXPECT_TRUE(std::isfinite(result.surfaceNormals[3 * offset + 1]));
      EXPECT_TRUE(std::isfinite(result.surfaceNormals[3 * offset + 2]));
      EXPECT_FLOAT_EQ(result.surfaceNormals[3 * offset], 0.0f);
      EXPECT_FLOAT_EQ(result.surfaceNormals[3 * offset + 1], 0.0f);
      EXPECT_FLOAT_EQ(result.surfaceNormals[3 * offset + 2], 0.0f);
    }
  }
  EXPECT_GT(coveredPixels, 0);
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
          EXPECT_NEAR(cpuDepth(y, x), gpu.depth[offset], 2e-4f);
          EXPECT_EQ(gpu.triangleIndex[offset], cpuTriangleIndex(y, x));
        }
      }
    }

    EXPECT_GT(coveredPixels, 0);
    EXPECT_TRUE(sawTriangle0);
    EXPECT_TRUE(sawTriangle1);
    EXPECT_LE(coverageDifferences, 4);
  }
}

TEST(CudaRasterizerTest, BatchedDepthMatchesIndependentRendersAndIndicesAreValid) {
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
    for (size_t pixel = 0; pixel < static_cast<size_t>(imageStride); ++pixel) {
      const int32_t batchId = batchTriangleIndex[begin + pixel];
      if (std::isfinite(batchDepth[begin + pixel])) {
        EXPECT_GE(batchId, 0);
        EXPECT_LT(batchId, 2);
      } else {
        EXPECT_EQ(batchId, -1);
      }
    }
  }
}

TEST(CudaRasterizerTest, NegativeDepthOffsetAndExactDepthTiesAreSupported) {
  MeshInputs mesh;
  mesh.positions.resize(9);
  mesh.positions << -0.5f, -0.5f, 1.0f, -0.5f, 0.5f, 1.0f, 0.5f, -0.5f, 1.0f;
  mesh.triangles.resize(6);
  mesh.triangles << 0, 1, 2, 0, 1, 2;

  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const CudaRenderResult first = renderCuda(mesh, camera, -1.5f);
  const CudaRenderResult second = renderCuda(mesh, camera, -1.5f);
  const size_t interior = static_cast<size_t>(20 * first.rowStride + 24);

  EXPECT_TRUE(first.triangleIndex[interior] == 0 || first.triangleIndex[interior] == 1);
  EXPECT_NEAR(first.depth[interior], -0.5f, 1e-5f);
  EXPECT_TRUE(second.triangleIndex[interior] == 0 || second.triangleIndex[interior] == 1);
  EXPECT_NEAR(second.depth[interior], -0.5f, 1e-5f);

  const CudaAttributeRenderResult attributes = renderCudaAttributes(mesh, nullptr, camera);
  EXPECT_TRUE(attributes.triangleIndex[interior] == 0 || attributes.triangleIndex[interior] == 1);
  EXPECT_GE(attributes.vertexIndex[interior], 0);
  EXPECT_LT(attributes.vertexIndex[interior], 3);
  const Eigen::Vector3f normal(
      attributes.surfaceNormals[3 * interior],
      attributes.surfaceNormals[3 * interior + 1],
      attributes.surfaceNormals[3 * interior + 2]);
  EXPECT_TRUE(normal.allFinite());
  EXPECT_NEAR(normal.norm(), 1.0f, 1e-5f);
}

TEST(CudaRasterizerTest, ExistingDepthWinsAndPaddedRowsRemainUsable) {
  const MeshInputs mesh = makeOverlappingTriangles();
  const Camera camera(std::make_shared<PinholeIntrinsicsModel>(63, 48, 40.0f, 38.0f, 31.5f, 24.0f));
  const int64_t rowStride = padImageWidthForRasterizer(camera.imageWidth());
  const size_t bufferSize = static_cast<size_t>(camera.imageHeight()) * rowStride;
  std::vector<float> initialDepth(bufferSize, std::numeric_limits<float>::infinity());
  std::vector<int32_t> initialTriangleIndex(bufferSize, -1);
  const size_t preservedPixel = static_cast<size_t>(16 * rowStride + 24);
  const size_t nanPixel = static_cast<size_t>(20 * rowStride + 31);
  initialDepth[preservedPixel] = 0.0f;
  initialTriangleIndex[preservedPixel] = 17;
  initialDepth[nanPixel] = std::numeric_limits<float>::quiet_NaN();
  initialTriangleIndex[nanPixel] = 18;

  const CudaRenderResult result = renderCuda(
      mesh, camera, 0.0f, Eigen::Matrix4f::Identity(), &initialDepth, &initialTriangleIndex);
  const size_t rasterizedPixel = static_cast<size_t>(20 * rowStride + 32);

  EXPECT_GT(result.rowStride, camera.imageWidth());
  EXPECT_EQ(result.triangleIndex[preservedPixel], 17);
  EXPECT_FLOAT_EQ(result.depth[preservedPixel], 0.0f);
  EXPECT_TRUE(result.triangleIndex[nanPixel] == 0 || result.triangleIndex[nanPixel] == 1);
  EXPECT_NEAR(result.depth[nanPixel], 1.0f, 1e-5f);
  EXPECT_TRUE(
      result.triangleIndex[rasterizedPixel] == 0 || result.triangleIndex[rasterizedPixel] == 1);
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
  float depth = 0.0f;
  int32_t triangleIndex = -1;
  int32_t vertexIndex = -1;
  float surfaceNormal = 0.0f;
  float rgb = 0.0f;
  const CudaRasterizerBuffers validBuffers{
      .zBuffer = &depth,
      .zBufferSize = bufferSize,
      .triangleIndexBuffer = &triangleIndex,
      .triangleIndexBufferSize = bufferSize,
      .vertexIndexBuffer = &vertexIndex,
      .vertexIndexBufferSize = bufferSize,
      .surfaceNormalsBuffer = &surfaceNormal,
      .surfaceNormalsBufferSize = 3 * bufferSize,
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
  inputs = validInputs;
  inputs.normalsModel = positions;
  inputs.normalsModelSize = 2;
  expectRejected(inputs, validBuffers);
  buffers = validBuffers;
  buffers.vertexIndexBufferSize = bufferSize - 1;
  expectRejected(validInputs, buffers);
  buffers = validBuffers;
  buffers.surfaceNormalsBufferSize = 3 * bufferSize - 1;
  expectRejected(validInputs, buffers);
  buffers = validBuffers;
  buffers.rgbBuffer = &rgb;
  buffers.rgbBufferSize = 3 * bufferSize;
  expectRejected(validInputs, buffers);
  const PhongMaterial material;
  const CudaRasterizerShadingInputs shadingInputs{.material = &material};
  buffers.rgbBufferSize = 3 * bufferSize - 1;
  EXPECT_THROW(
      rasterizeMeshCuda(
          validInputs, camera, Eigen::Matrix4f::Identity(), 0.1f, buffers, &shadingInputs),
      std::runtime_error);
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
      .zBuffer = &depth,
      .zBufferSize = batchedBufferSize,
      .triangleIndexBuffer = &triangleIndex,
      .triangleIndexBufferSize = batchedBufferSize,
      .vertexIndexBuffer = &vertexIndex,
      .vertexIndexBufferSize = batchedBufferSize,
      .surfaceNormalsBuffer = &surfaceNormal,
      .surfaceNormalsBufferSize = 3 * batchedBufferSize,
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
  batchedBuffers.zBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);
  batchedBuffers = validBatchedBuffers;
  batchedBuffers.triangleIndexBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);

  inputs = validBatchedInputs;
  inputs.positionsModelSize = 5;
  expectBatchedRejected(inputs, validBatchedBuffers);
  inputs = validBatchedInputs;
  inputs.normalsModel = batchedPositions;
  inputs.normalsModelSize = 5;
  expectBatchedRejected(inputs, validBatchedBuffers);
  batchedBuffers = validBatchedBuffers;
  batchedBuffers.vertexIndexBufferSize = batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);
  batchedBuffers = validBatchedBuffers;
  batchedBuffers.surfaceNormalsBufferSize = 3 * batchedBufferSize - 1;
  expectBatchedRejected(validBatchedInputs, batchedBuffers);
}

} // namespace
} // namespace momentum::rasterizer
