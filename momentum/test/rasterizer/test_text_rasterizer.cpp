/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/text_rasterizer.h>

#include <gtest/gtest.h>

using namespace momentum;
using namespace momentum::rasterizer;

TEST(TextRasterizer, BasicText3D) {
  const int width = 200;
  const int height = 100;

  OpenCVDistortionParametersT<float> distortionParams;
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  auto zBuffer = makeRasterizerZBuffer(camera);
  auto rgbBuffer = makeRasterizerRGBBuffer(camera);

  std::vector<Eigen::Vector3f> positions = {Eigen::Vector3f(0.0f, 0.0f, 1.5f)};
  std::vector<std::string> texts = {"Hello"};

  rasterizeText(
      positions,
      texts,
      camera,
      Eigen::Matrix4f::Identity(),
      0.1f,
      Eigen::Vector3f(1.0f, 0.0f, 0.0f),
      1,
      zBuffer.view(),
      rgbBuffer.view());

  int pixelsSet = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (zBuffer(y, x) < FLT_MAX) {
        pixelsSet++;
        EXPECT_NEAR(rgbBuffer(y, x, 0), 1.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 1), 0.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 2), 0.0f, 1e-5f);
        EXPECT_NEAR(zBuffer(y, x), 1.5f, 1e-5f);
      }
    }
  }

  EXPECT_GT(pixelsSet, 0);
}

TEST(TextRasterizer, BasicText2D) {
  const int width = 200;
  const int height = 100;

  OpenCVDistortionParametersT<float> distortionParams;
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  auto zBuffer = makeRasterizerZBuffer(camera);
  auto rgbBuffer = makeRasterizerRGBBuffer(camera);

  std::vector<Eigen::Vector2f> positions = {Eigen::Vector2f(10.0f, 10.0f)};
  std::vector<std::string> texts = {"Test"};

  rasterizeText2D(
      positions, texts, Eigen::Vector3f(0.0f, 1.0f, 0.0f), 1, rgbBuffer.view(), zBuffer.view());

  int pixelsSet = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (zBuffer(y, x) < FLT_MAX) {
        pixelsSet++;
        EXPECT_NEAR(rgbBuffer(y, x, 0), 0.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 1), 1.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 2), 0.0f, 1e-5f);
        EXPECT_NEAR(zBuffer(y, x), 0.0f, 1e-5f);
      }
    }
  }

  EXPECT_GT(pixelsSet, 0);
}

TEST(TextRasterizer, TextScaling) {
  const int width = 400;
  const int height = 200;

  OpenCVDistortionParametersT<float> distortionParams;
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  auto rgbBuffer1 = makeRasterizerRGBBuffer(camera);
  auto rgbBuffer2 = makeRasterizerRGBBuffer(camera);

  std::vector<Eigen::Vector2f> positions = {Eigen::Vector2f(10.0f, 10.0f)};
  std::vector<std::string> texts = {"A"};

  rasterizeText2D(positions, texts, Eigen::Vector3f(1.0f, 1.0f, 1.0f), 1, rgbBuffer1.view());

  int pixelsScale1 = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (rgbBuffer1(y, x, 0) > 0.5f) {
        pixelsScale1++;
      }
    }
  }

  rasterizeText2D(positions, texts, Eigen::Vector3f(1.0f, 1.0f, 1.0f), 2, rgbBuffer2.view());

  int pixelsScale2 = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (rgbBuffer2(y, x, 0) > 0.5f) {
        pixelsScale2++;
      }
    }
  }

  EXPECT_GT(pixelsScale1, 0);
  EXPECT_GT(pixelsScale2, pixelsScale1);
  EXPECT_NEAR(static_cast<float>(pixelsScale2) / pixelsScale1, 4.0f, 1.0f);
}

TEST(TextRasterizer, MultipleTexts) {
  const int width = 400;
  const int height = 200;

  OpenCVDistortionParametersT<float> distortionParams;
  auto intrinsics = std::make_shared<OpenCVIntrinsicsModel>(
      width, height, width / 2.0f, height / 2.0f, width / 2.0f, height / 2.0f, distortionParams);

  Camera camera(intrinsics);

  auto zBuffer = makeRasterizerZBuffer(camera);
  auto rgbBuffer = makeRasterizerRGBBuffer(camera);

  std::vector<Eigen::Vector2f> positions = {
      Eigen::Vector2f(10.0f, 10.0f), Eigen::Vector2f(10.0f, 30.0f)};
  std::vector<std::string> texts = {"Line1", "Line2"};

  rasterizeText2D(
      positions, texts, Eigen::Vector3f(1.0f, 0.0f, 1.0f), 1, rgbBuffer.view(), zBuffer.view());

  int pixelsSet = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (zBuffer(y, x) < FLT_MAX) {
        pixelsSet++;
        EXPECT_NEAR(rgbBuffer(y, x, 0), 1.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 1), 0.0f, 1e-5f);
        EXPECT_NEAR(rgbBuffer(y, x, 2), 1.0f, 1e-5f);
      }
    }
  }

  EXPECT_GT(pixelsSet, 0);
}
