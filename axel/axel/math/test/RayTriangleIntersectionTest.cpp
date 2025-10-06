/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <perception/test_helpers/EigenChecks.h>

#include "axel/common/Constants.h"
#include "axel/math/RayTriangleIntersection.h"

namespace axel::test {
namespace {

using ::axel::detail::eps;

TEST(RayTriangleIntersectionTest, Basic) {
  Eigen::Vector3d collisionPoint;
  double t;
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(0., 5., 0.),
      Eigen::Vector3d(5., 5., 0.),
      Eigen::Vector3d(2.5, 5., 2.5),
      collisionPoint,
      t));
  EXPECT_EIGEN_MATRIX_NEAR(collisionPoint, Eigen::Vector3d(1., 5., 1.), eps<double>());
  EXPECT_NEAR(t, 4., eps<double>());

  EXPECT_FALSE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(10., 5., 0.),
      Eigen::Vector3d(10., 15., 0.),
      Eigen::Vector3d(12.5, 5., 2.5),
      collisionPoint,
      t));

  // same edge-triangle tests with different scale
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-2.3116044998168945, -3.557568311691284, 4.071789264678955),
      Eigen::Vector3d(-0.02631211280822754, -0.005295753479003906, -0.04990386962890625),
      Eigen::Vector3d(-2.322500705718994, -3.5661442279815674, 3.9918453693389893),
      Eigen::Vector3d(-2.381772518157959, -3.554487943649292, 4.0887017250061035),
      Eigen::Vector3d(-2.2690765857696533, -3.567392110824585, 3.994669198989868),
      collisionPoint,
      t));
  EXPECT_NEAR(t, 0.73062444794057768, eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-2.381772518157959, -3.554487943649292, 4.0887017250061035),
      Eigen::Vector3d(0.11269593238830566, -0.012904167175292969, -0.09403252601623535),
      Eigen::Vector3d(-2.3116044998168945, -3.557568311691284, 4.071789264678955),
      Eigen::Vector3d(-2.337916612625122, -3.562864065170288, 4.021885395050049),
      Eigen::Vector3d(-2.248196601867676, -3.580512285232544, 3.9052698612213135),
      collisionPoint,
      t));
  EXPECT_NEAR(t, 0.53861406334285833, eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.5650877952575684, -0.9797914624214172, 0.8701220750808716),
      Eigen::Vector3d(-0.006578028202056885, -0.0013239383697509766, -0.012475967407226562),
      Eigen::Vector3d(-0.5678118467330933, -0.981935441493988, 0.8501361608505249),
      Eigen::Vector3d(-0.5826297998428345, -0.9790213704109192, 0.8743501901626587),
      Eigen::Vector3d(-0.5544558167457581, -0.9822474122047424, 0.8508421182632446),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.5826297998428345, -0.9790213704109192, 0.8743501901626587),
      Eigen::Vector3d(0.028173983097076416, -0.003226041793823242, -0.023508071899414062),
      Eigen::Vector3d(-0.5650877952575684, -0.9797914624214172, 0.8701220750808716),
      Eigen::Vector3d(-0.5716658234596252, -0.9811154007911682, 0.857646107673645),
      Eigen::Vector3d(-0.5492358207702637, -0.9855274558067322, 0.8284921646118164),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.27400198578834534, -0.5501620173454285, 0.3365109860897064),
      Eigen::Vector3d(-0.0032890141010284424, -0.0006619691848754883, -0.006237983703613281),
      Eigen::Vector3d(-0.2753640115261078, -0.5512340068817139, 0.3265179991722107),
      Eigen::Vector3d(-0.2827729880809784, -0.5497769713401794, 0.3386250138282776),
      Eigen::Vector3d(-0.2686859965324402, -0.5513899922370911, 0.32687100768089294),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.2827729880809784, -0.5497769713401794, 0.3386250138282776),
      Eigen::Vector3d(0.014086991548538208, -0.001613020896911621, -0.011754006147384644),
      Eigen::Vector3d(-0.27400198578834534, -0.5501620173454285, 0.3365109860897064),
      Eigen::Vector3d(-0.2772909998893738, -0.550823986530304, 0.33027300238609314),
      Eigen::Vector3d(-0.266075998544693, -0.5530300140380859, 0.31569600105285645),
      collisionPoint,
      t));
}

TEST(RaytracingUtilsTest, rayTriangleIntersect_hit) {
  const Eigen::Vector3d p0(0., 5., 0.);
  const Eigen::Vector3d p1(5., 5., 0.);
  const Eigen::Vector3d p2(2.5, 5., 2.5);

  Eigen::Vector3d hitPoint;
  double t;
  double u;
  double v;
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.), Eigen::Vector3d(0., 1., 0.), p0, p1, p2, hitPoint, t, u, v));
  EXPECT_NEAR(t, 4., eps<double>());
  EXPECT_EIGEN_MATRIX_NEAR(hitPoint, Eigen::Vector3d(1., 5., 1.), eps<double>());
  EXPECT_EIGEN_MATRIX_NEAR(
      Eigen::Vector3d(1 - u - v, u, v), Eigen::Vector3d(0.6, 0.0, 0.4), eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 10., 1.), Eigen::Vector3d(0., 1., 0.), p0, p1, p2, hitPoint, t));
  EXPECT_NEAR(t, -5., eps<double>());
}

TEST(RaytracingUtilsTest, rayTriangleIntersect_noHit) {
  Eigen::Vector3d hitPoint;
  double t;
  EXPECT_FALSE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(10., 5., 0.),
      Eigen::Vector3d(10., 15., 0.),
      Eigen::Vector3d(12.5, 5., 2.5),
      hitPoint,
      t));
}

} // namespace
} // namespace axel::test
