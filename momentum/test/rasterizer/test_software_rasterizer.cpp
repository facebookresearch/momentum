/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <axel/math/RayTriangleIntersection.h>
#include <gtest/gtest.h>
#include <momentum/rasterizer/camera.h>
#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>
#include <cmath>
#include <cstdint>
#include <iomanip>

using namespace momentum::rasterizer;

Eigen::Matrix4d
frustumMatrix(const double right, const double top, const double zNear, const double zFar) {
  Eigen::Matrix4d projectionMatrix;
  const double C = -(zFar + zNear) / (zFar - zNear);
  const double D = -2.0 * zFar * zNear / (zFar - zNear);
  projectionMatrix << zNear / right, 0, 0, 0, 0, zNear / top, 0, 0, 0, 0, C, D, 0, 0, -1, 0;
  return projectionMatrix;
}

void rasterizeMeshWithRayTracer(
    const Camera& camera,
    const Mesh& mesh,
    Span2f zBuffer,
    Span2i triangleIndexBuffer) {
  const auto height = camera.imageHeight();
  const auto width = camera.imageWidth();

  ASSERT_GE(zBuffer.extent(0), height);
  ASSERT_GE(zBuffer.extent(1), width);
  ASSERT_GE(triangleIndexBuffer.extent(0), height);
  ASSERT_GE(triangleIndexBuffer.extent(1), width);

  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      float min_dist = FLT_MAX;
      int triangle_index = -1;

      // Create ray through pixel using new camera system
      Eigen::Vector3f imagePoint(x, y, 1.0f);
      auto [rayDirection, valid] = camera.intrinsicsModel()->unproject(imagePoint);
      if (!valid) {
        continue;
      }

      // Transform ray to world space
      Eigen::Vector3f rayOrigin = camera.worldFromEye().translation();
      Eigen::Vector3f worldRayDirection = camera.worldFromEye().linear() * rayDirection;
      worldRayDirection.normalize();

      for (int iTri = 0; iTri < static_cast<int>(mesh.faces.size()); ++iTri) {
        const auto& tri = mesh.faces[iTri];
        Eigen::Vector3f intersectionPoint = Eigen::Vector3f::Zero();
        float t = 0;
        Eigen::Vector3f v0 = mesh.vertices[tri.x()];
        Eigen::Vector3f v1 = mesh.vertices[tri.y()];
        Eigen::Vector3f v2 = mesh.vertices[tri.z()];
        bool intersected = axel::rayTriangleIntersect(
            rayOrigin, worldRayDirection, v0, v1, v2, intersectionPoint, t);
        if (intersected && t >= 0.0f && t < min_dist) {
          min_dist = t;
          triangle_index = iTri;
        }
      }

      if (triangle_index >= 0) {
        // Transform intersection point to eye space to get Z depth
        Eigen::Vector3f worldIntersection = rayOrigin + min_dist * worldRayDirection;
        Eigen::Vector3f eyeIntersection = camera.eyeFromWorld() * worldIntersection;
        auto zView = zBuffer;
        auto triView = triangleIndexBuffer;
        zView(y, x) = eyeIntersection.z();
        triView(y, x) = triangle_index;
      } else {
        auto zView = zBuffer;
        auto triView = triangleIndexBuffer;
        zView(y, x) = FLT_MAX;
        triView(y, x) = -1;
      }
    }
  }
}

void visualizeZBuffer(Span2f zBuffer) {
  for (int i = 0; i < zBuffer.extent(0); ++i) {
    std::cout << std::setw(5) << i << ": ";
    for (int j = 0; j < zBuffer.extent(1); ++j) {
      const auto zVal = zBuffer(i, j);
      if (zVal < FLT_MAX) {
        std::cout << "x";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }
}

void visualizeIndexBuffer(Span2i idxBuffer) {
  for (int i = 0; i < idxBuffer.extent(0); ++i) {
    for (int j = 0; j < idxBuffer.extent(1); ++j) {
      const auto idx = idxBuffer(i, j);
      if (idx < 0) {
        std::cout << " ";
      } else {
        std::cout << idx;
      }
    }
    std::cout << "\n";
  }
}

void testRasterizerWithGeometry(const Camera& camera, const Mesh& mesh) {
  auto zBuffer_rasterizer = makeRasterizerZBuffer(camera);
  auto triangleIndexBuffer_rasterizer = makeRasterizerIndexBuffer(camera);
  auto rgbBuffer_rasterizer = makeRasterizerRGBBuffer(camera);

  const auto height = camera.imageHeight();
  const auto width = camera.imageWidth();

  rasterizeMesh(
      mesh,
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      PhongMaterial{Eigen::Vector3f(1, 1, 0)},
      zBuffer_rasterizer.view(),
      rgbBuffer_rasterizer.view(),
      {},
      {},
      triangleIndexBuffer_rasterizer.view(),
      std::vector<Light>{createAmbientLight()},
      true);
  visualizeZBuffer(zBuffer_rasterizer.view());
  // visualizeIndexBuffer(triangleIndexBuffer_rasterizer);

  auto zBuffer_rayTrace = makeRasterizerZBuffer(camera);
  auto triangleIndexBuffer_rayTrace = makeRasterizerIndexBuffer(camera);
  rasterizeMeshWithRayTracer(
      camera, mesh, zBuffer_rayTrace.view(), triangleIndexBuffer_rayTrace.view());
  visualizeZBuffer(zBuffer_rayTrace.view());
  // visualizeIndexBuffer(triangleIndexBuffer_rayTrace);

  int numTriDifferences = 0;
  int numDepthDifferences = 0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      if (std::abs(zBuffer_rayTrace(y, x) - zBuffer_rasterizer(y, x)) > 1e-3f) {
        numDepthDifferences += 1;
      }
      if (zBuffer_rayTrace(y, x) < FLT_MAX && zBuffer_rasterizer(y, x) < FLT_MAX) {
        // We sometimes see small numerical differences in where the triangle edges are.
        // If the triangle is hit by both the rasterizer and the ray tracer, though, the
        // depth should be very similar:
        ASSERT_NEAR(zBuffer_rayTrace(y, x), zBuffer_rasterizer(y, x), 1e-3f);
      }

      if (triangleIndexBuffer_rayTrace(y, x) != triangleIndexBuffer_rasterizer(y, x)) {
        numTriDifferences += 1;
      }
      // triangle index buffer should agree with the z buffer:
      if (zBuffer_rasterizer(y, x) == FLT_MAX) {
        ASSERT_EQ(-1, triangleIndexBuffer_rasterizer(y, x));
      } else {
        ASSERT_NE(-1, triangleIndexBuffer_rasterizer(y, x));
      }

      // RGB buffer should be black if the triangle index buffer is -1,
      // else should be the color of the mesh (1, 1, 0) with ambient light:
      if (zBuffer_rasterizer(y, x) == FLT_MAX) {
        ASSERT_TRUE(rgbBuffer_rasterizer(y, x, 0) == 0);
        ASSERT_TRUE(rgbBuffer_rasterizer(y, x, 1) == 0);
        ASSERT_TRUE(rgbBuffer_rasterizer(y, x, 2) == 0);
      } else {
        ASSERT_NEAR(1.0f, rgbBuffer_rasterizer(y, x, 0), 1e-3f);
        ASSERT_NEAR(1.0f, rgbBuffer_rasterizer(y, x, 1), 1e-3f);
        ASSERT_NEAR(0.0f, rgbBuffer_rasterizer(y, x, 2), 1e-3f);
      }
    }
  }

  // Allow some variation along the boundary between the two triangles:
  ASSERT_LE(numTriDifferences, 3);
  ASSERT_LE(numDepthDifferences, 3);
}

TEST(SoftwareRasterizer, OneQuad) {
  const int width = 20;
  const int height = 10;

  // Create OpenCV intrinsics with no distortion
  OpenCVDistortionParametersT<float> distortionParams; // All zeros by default
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  Mesh mesh;
  mesh.vertices = {
      Eigen::Vector3f(-0.51f, -0.51f, 1.5f),
      Eigen::Vector3f(-0.51f, 0.51f, 1.5f),
      Eigen::Vector3f(0.51f, -0.51f, 1.5f),
      Eigen::Vector3f(0.51f, 0.51f, 1.5f)};

  mesh.normals = {
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ()};

  // triangles need to be clockwise:
  //  1---3
  //  | \ |
  //  0---2

  mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 3, 2)};

  testRasterizerWithGeometry(camera, mesh);

  // Move the positions around a bit:
  mesh.vertices[0].z() = 1.4f;
  mesh.vertices[1].z() = 1.5f;
  mesh.vertices[2].z() = 1.6f;
  mesh.vertices[3].z() = 1.7f;

  testRasterizerWithGeometry(camera, mesh);
}

TEST(SoftwareRasterizer, EyeZIsZero) {
  const int width = 20;
  const int height = 10;

  // Create OpenCV intrinsics with no distortion
  OpenCVDistortionParametersT<float> distortionParams; // All zeros by default
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  Mesh mesh;
  mesh.vertices = {
      Eigen::Vector3f(-0.51f, -0.51f, 0.0f),
      Eigen::Vector3f(-0.51f, 0.51f, 1.5f),
      Eigen::Vector3f(0.51f, -0.51f, 1.5f),
      Eigen::Vector3f(0.51f, 0.51f, 0.0f)};

  mesh.normals = {
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ(),
      Eigen::Vector3f::UnitZ()};

  // triangles need to be clockwise:
  //  1---3
  //  | \ |
  //  0---2

  mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 3, 2)};

  auto zBuffer_rasterizer = makeRasterizerZBuffer(camera);
  auto triangleIndexBuffer_rasterizer = makeRasterizerIndexBuffer(camera);

  rasterizeMesh(
      mesh,
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      PhongMaterial{},
      zBuffer_rasterizer.view(),
      {},
      {},
      {},
      triangleIndexBuffer_rasterizer.view(),
      std::vector<Light>{},
      true);

  // Both triangles should not get rasterized due to having z==0
  // (and it shouldn't throw an error)
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      ASSERT_EQ(zBuffer_rasterizer(y, x), FLT_MAX);
      ASSERT_EQ(triangleIndexBuffer_rasterizer(y, x), -1);
    }
  }

  mesh.vertices[3] = Eigen::Vector3f(0.51f, 0.51f, 1.5f);

  rasterizeMesh(
      mesh,
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      PhongMaterial{},
      zBuffer_rasterizer.view(),
      {},
      {},
      {},
      triangleIndexBuffer_rasterizer.view(),
      std::vector<Light>{},
      true);

  // One triangle should get rasterized:
  size_t count = 0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      if (zBuffer_rasterizer(y, x) < FLT_MAX) {
        count++;
      }
    }
  }

  ASSERT_GT(count, 1);
}

TEST(SoftwareRasterizer, Splats) {
  // Rasterize a splat and a circle mesh and make sure they match up:
  const int width = 50;
  const int height = 40;

  // Create OpenCV intrinsics with no distortion
  OpenCVDistortionParametersT<float> distortionParams; // All zeros by default
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  const Eigen::Vector3f splatOrigin = Eigen::Vector3f(0, 0, 1.5);
  const Eigen::Vector3f splatNormal = Eigen::Vector3f::UnitZ();
  const float splatRadius = 0.5f;

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3i> triangles;

  size_t nCircleSamples = 72;
  vertices.push_back(splatOrigin);
  normals.push_back(splatNormal);

  for (size_t i = 0; i < nCircleSamples; ++i) {
    float theta_i = 2.0f * M_PI * float(i) / float(nCircleSamples);
    vertices.emplace_back(
        splatOrigin +
        Eigen::Vector3f(splatRadius * std::cos(theta_i), splatRadius * std::sin(theta_i), 0));
    normals.push_back(splatNormal);
    triangles.emplace_back(i + 1, 0, (i + 1) % nCircleSamples + 1);
  }

  auto zBuffer_triangles = makeRasterizerZBuffer(camera);
  rasterizeMesh(
      vertices,
      normals,
      triangles,
      {},
      {},
      Eigen::VectorXf{},
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      PhongMaterial{},
      zBuffer_triangles.view(),
      {},
      {},
      {},
      {},
      std::vector<Light>{},
      true);
  visualizeZBuffer(zBuffer_triangles.view());

  auto zBuffer_splats = makeRasterizerZBuffer(camera);
  rasterizeSplats(
      std::vector<Eigen::Vector3f>{splatOrigin},
      std::vector<Eigen::Vector3f>{splatNormal},
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      PhongMaterial{},
      PhongMaterial{},
      splatRadius,
      zBuffer_splats.view(),
      {},
      std::vector<Light>{},
      0);
  visualizeZBuffer(zBuffer_triangles.view());

  int numDepthDifferences = 0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      if (std::abs(zBuffer_triangles(y, x) - zBuffer_splats(y, x)) > 1e-3f) {
        numDepthDifferences += 1;
      }
    }
  }

  ASSERT_LE(numDepthDifferences, 2);
}

TEST(SoftwareRasterizer, AlphaMatte) {
  const std::vector<int> testWidths{1, 11, 32, 60};
  for (const auto width : testWidths) {
    const int height = 40;

    // Create OpenCV intrinsics with no distortion
    OpenCVDistortionParametersT<float> distortionParams; // All zeros by default
    auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
        width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

    Camera camera(intrinsics);

    auto zBuffer = makeRasterizerZBuffer(camera);
    auto rgbBuffer = makeRasterizerRGBBuffer(camera);

    for (size_t iRow = height / 4; iRow < 3 * height / 4; ++iRow) {
      for (size_t jCol = width / 4; jCol < 3 * width / 4; ++jCol) {
        zBuffer(iRow, jCol) = 1.0f;
        rgbBuffer(iRow, jCol, 0) = 1.0f;
      }
    }

    Tensor<uint8_t, 3> targetImage({height, width, 3});

    alphaMatte(zBuffer.view(), rgbBuffer.view(), targetImage.view());
    for (size_t iRow = 0; iRow < height; ++iRow) {
      for (size_t jCol = 0; jCol < width; ++jCol) {
        bool set = (zBuffer(iRow, jCol) < FLT_MAX);
        if (set) {
          ASSERT_EQ(targetImage(iRow, jCol, 0), 255);
        } else {
          ASSERT_EQ(targetImage(iRow, jCol, 0), 0);
        }
      }
    }

    Tensor<uint8_t, 3> targetImage2({height, width, 3});
    alphaMatte(zBuffer.view(), rgbBuffer.view(), targetImage2.view());
    for (size_t iRow = 0; iRow < height; ++iRow) {
      for (size_t jCol = 0; jCol < width; ++jCol) {
        ASSERT_EQ(targetImage2(iRow, jCol, 0), targetImage(iRow, jCol, 0));
      }
    }
  }
}
