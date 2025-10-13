/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/PointTriangleProjection.h"

#include <gmock/gmock.h>
#include <array>

#include <gsl/span>

#include <perception/test_helpers/EigenChecks.h>
#include <test_helpers/EigenMatchers.h>

namespace axel::test {
namespace {

using ::testing::Combine;
using ::testing::DoubleEq;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

Eigen::Isometry3d buildTransform3D(
    const double roll,
    const double pitch,
    const double yaw,
    const Eigen::Vector3d& translation) {
  const Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity() * yawAngle * pitchAngle * rollAngle;
  t.translation() = translation;
  return t;
}

const auto testTransformations3D = std::to_array<Eigen::Isometry3d>({
    // Identity
    buildTransform3D(0.0, 0.0, 0.0, {0.0, 0.0, 0.0}),
    // Only rotation
    buildTransform3D(1.0, 0.5, 1.0, {0.0, 0.0, 0.0}),
    // Only translation
    buildTransform3D(0.0, 0.0, 0.0, {1.0, 2.0, 3.0}),
    // Rotation and translation
    buildTransform3D(1.0, 0.5, 1.0, {1.0, 2.0, 3.0}),
});

struct ProjectOnTriangleTestParams {
  Eigen::Vector3d particle;
  std::array<Eigen::Vector3d, 3> triangle;
  Eigen::Vector3d expectedProjection;
  bool expectedInside;
};

struct ProjectOnTriangleVectorizedTestParams {
  WideVec3d particle;
  WideVec3d a;
  WideVec3d b;
  WideVec3d c;
  WideVec3d expectedProjection;
  WideMask<WideScalard> expectedInside;
  std::array<bool, kNativeLaneWidth<double>> activeLanes = {false};
};

void setFromEigenVector(WideVec3d& wideVec, const size_t laneIdx, const Eigen::Vector3d& eigenVec) {
  wideVec[0][laneIdx] = eigenVec[0];
  wideVec[1][laneIdx] = eigenVec[1];
  wideVec[2][laneIdx] = eigenVec[2];
}

Eigen::Vector3d toEigenVector(const WideVec3d& wideVec, const size_t laneIdx) {
  return {
      wideVec[0][laneIdx],
      wideVec[1][laneIdx],
      wideVec[2][laneIdx],
  };
}

std::vector<ProjectOnTriangleVectorizedTestParams> createVectorizedTestParams(
    const gsl::span<const ProjectOnTriangleTestParams> params) {
  constexpr size_t N = kNativeLaneWidth<double>;
  const auto elementCount = static_cast<int32_t>(params.size());
  std::vector<ProjectOnTriangleVectorizedTestParams> vectParams((elementCount - 1) / N + 1);
  for (int32_t i = 0; i < elementCount; ++i) {
    const int32_t idx = i / N;
    const int32_t offset = i % N;
    setFromEigenVector(vectParams[idx].particle, offset, params[i].particle);
    setFromEigenVector(vectParams[idx].a, offset, params[i].triangle[0]);
    setFromEigenVector(vectParams[idx].b, offset, params[i].triangle[1]);
    setFromEigenVector(vectParams[idx].c, offset, params[i].triangle[2]);
    setFromEigenVector(vectParams[idx].expectedProjection, offset, params[i].expectedProjection);

    vectParams[idx].expectedInside[offset] = params[i].expectedInside;
    vectParams[idx].activeLanes[offset] = true;
  }
  return vectParams;
}

ProjectOnTriangleTestParams applyTransform(
    const ProjectOnTriangleTestParams& params,
    const Eigen::Isometry3d& transform) {
  ProjectOnTriangleTestParams result;
  result.particle = transform * params.particle;
  for (int i = 0; i < 3; ++i) {
    result.triangle[i] = transform * params.triangle[i];
  }
  result.expectedProjection = transform * params.expectedProjection;
  result.expectedInside = params.expectedInside;
  return result;
}

class ProjectOnTriangleTest
    : public TestWithParam<std::tuple<ProjectOnTriangleTestParams, Eigen::Isometry3d>> {};

TEST_P(ProjectOnTriangleTest, checkProjection) {
  const auto& transform = std::get<1>(GetParam());
  const auto params = applyTransform(std::get<0>(GetParam()), transform);

  Eigen::Vector3d q{};
  bool inside = projectOnTriangle<double>(
      params.particle, params.triangle[0], params.triangle[1], params.triangle[2], q);

  EXPECT_EQ(inside, params.expectedInside);
  EXPECT_THAT(q, Elementwise(DoubleEq(), params.expectedProjection));
}

class ProjectOnTriangleVectorizedTest
    : public TestWithParam<std::tuple<ProjectOnTriangleVectorizedTestParams, Eigen::Isometry3d>> {};

TEST_P(ProjectOnTriangleVectorizedTest, checkProjection) {
  const auto params = std::get<0>(GetParam());

  WideVec3d q;
  const auto inside = projectOnTriangle(params.particle, params.a, params.b, params.c, q);

  for (uint32_t i = 0; i < params.activeLanes.size() && params.activeLanes[i]; ++i) {
    EXPECT_EQ(inside[i], params.expectedInside[i]);
    EXPECT_THAT(
        toEigenVector(q, i), Elementwise(DoubleEq(), toEigenVector(params.expectedProjection, i)));
  }
}

const auto projectOnTriangleTestParams = std::to_array<ProjectOnTriangleTestParams>({
    // Particle inside triangle
    {{1.0, 1.0, 0.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}},
     {1.0, 1.0, 0.0},
     true},
    // Particle above triangle
    {{1.0, 1.0, 1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}},
     {1.0, 1.0, 0.0},
     true},
    // Particle below triangle
    {{1.0, 1.0, -1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}},
     {1.0, 1.0, 0.0},
     true},
    // Particle on triangle edge
    {{1.0, 1.0, 0.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {1.0, 1.0, 0.0},
     false},
    // Particle on triangle vertex
    {{2.0, 0.0, 0.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {2.0, 0.0, 0.0},
     false},
    // Particle in region outside of vertex A
    {{-1.0, -1.0, 0.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {0.0, 0.0, 0.0},
     false},
    // Particle in region outside of vertex B
    {{4.0, -1.0, -1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {2.0, 0.0, 0.0},
     false},
    // Particle in region outside of vertex C
    {{-1.0, 4.0, 1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {0.0, 2.0, 0.0},
     false},
    // Particle in edge region AB
    {{1.0, -1.0, 0.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {1.0, 0.0, 0.0},
     false},
    // Particle in edge region AC
    {{-1.0, 1.0, -1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {0.0, 1.0, 0.0},
     false},
    // Particle in edge region BC
    {{2.0, 2.0, 1.0},
     {Eigen::Vector3d{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
     {1.0, 1.0, 0.0},
     false},
});

INSTANTIATE_TEST_SUITE_P(
    ProjectOnTriangleParameterizedTest,
    ProjectOnTriangleTest,
    Combine(ValuesIn(projectOnTriangleTestParams), ValuesIn(testTransformations3D)));

INSTANTIATE_TEST_SUITE_P(
    ProjectOnTriangleVectorizedParameterizedTest,
    ProjectOnTriangleVectorizedTest,
    // Dummy transformation.
    Combine(
        ValuesIn(createVectorizedTestParams(projectOnTriangleTestParams)),
        ValuesIn({testTransformations3D[0]})));

} // namespace
} // namespace axel::test
