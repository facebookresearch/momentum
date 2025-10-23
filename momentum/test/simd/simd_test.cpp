/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "momentum/simd/simd.h"

using namespace momentum;

TEST(SimdTest, Setters) {
  FloatP f1 = 0.0f;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(f1[i], 0);
  }

  f1 = 4.56f;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(f1[i], 4.56f);
  }

  f1 = -1.23f;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(f1[i], -1.23f);
  }

  IntP i1 = 1;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_EQ(i1[i], 1);
  }

  i1 = 2;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_EQ(i1[i], 2);
  }

  i1 = -3;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_EQ(i1[i], -3);
  }
}

TEST(SimdTest, Utilities) {
  // pX1 = {1, 2, ..., kSimdPacketSize}
  // pY1 = {2, 3, ..., kSimdPacketSize + 1}
  // pZ1 = {3, 4, ..., kSimdPacketSize + 2}
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pX1;
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pY1;
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pZ1;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    pX1[i] = i;
    pY1[i] = i + 1;
    pZ1[i] = i + 2;
  }

  // pX2 = {1, 2, ..., kSimdPacketSize + 3}
  // pY2 = {2, 3, ..., kSimdPacketSize + 4}
  // pZ2 = {3, 4, ..., kSimdPacketSize + 5}
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pX2;
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pY2;
  alignas(kSimdAlignment) std::array<float, kSimdPacketSize> pZ2;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    pX2[i] = i + 3;
    pY2[i] = i + 4;
    pZ2[i] = i + 5;
  }

  // Packets of {pX1, pY1, pZ1} and {pX2, pY2, pZ2}
  const Vector3fP vec1 = Vector3fP{
      drjit::load<FloatP>(pX1.data()),
      drjit::load<FloatP>(pY1.data()),
      drjit::load<FloatP>(pZ1.data())};
  const Vector3fP vec2 = Vector3fP{
      drjit::load<FloatP>(pX2.data()),
      drjit::load<FloatP>(pY2.data()),
      drjit::load<FloatP>(pZ2.data())};

  // A random Eigen::Vector3f
  const Eigen::Vector3f eigenVec1 = Eigen::Vector3f::Random();

  // Dot product of Eigen::Vector3f and Vector3fP
  const FloatP dot1 = momentum::dot(eigenVec1, vec1);
  const FloatP dot2 = momentum::dot(vec1, eigenVec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_NEAR(eigenVec1.dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])), dot1[i], 1e-6);
    EXPECT_NEAR(eigenVec1.dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])), dot2[i], 1e-6);
  }

  // Dot product of two Vector3fP-s
  const FloatP dot3 = dot(vec1, vec2);
  const FloatP dot4 = dot(vec2, vec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(
        Eigen::Vector3f(pX1[i], pY1[i], pZ1[i]).dot(Eigen::Vector3f(pX2[i], pY2[i], pZ2[i])),
        dot3[i]);
    EXPECT_FLOAT_EQ(
        Eigen::Vector3f(pX2[i], pY2[i], pZ2[i]).dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])),
        dot4[i]);
  }

  // Summation of Eigen::Vector3f and Vector3fP
  const Vector3fP plus1 = momentum::operator+(eigenVec1, vec1);
  const Vector3fP plus2 = momentum::operator+(vec1, eigenVec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(plus1.x()[i], eigenVec1.x() + vec1.x()[i]);
    EXPECT_FLOAT_EQ(plus1.y()[i], eigenVec1.y() + vec1.y()[i]);
    EXPECT_FLOAT_EQ(plus1.z()[i], eigenVec1.z() + vec1.z()[i]);

    EXPECT_FLOAT_EQ(plus2.x()[i], vec1.x()[i] + eigenVec1.x());
    EXPECT_FLOAT_EQ(plus2.y()[i], vec1.y()[i] + eigenVec1.y());
    EXPECT_FLOAT_EQ(plus2.z()[i], vec1.z()[i] + eigenVec1.z());
  }

  // Summation of two Vector3fP-s
  const Vector3fP plus3 = vec1 + vec2;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(plus3.x()[i], vec1.x()[i] + vec2.x()[i]);
    EXPECT_FLOAT_EQ(plus3.y()[i], vec1.y()[i] + vec2.y()[i]);
    EXPECT_FLOAT_EQ(plus3.z()[i], vec1.z()[i] + vec2.z()[i]);
  }

  // Subtraction of Eigen::Vector3f and Vector3fP
  const Vector3fP minus1 = momentum::operator-(eigenVec1, vec1);
  const Vector3fP minus2 = momentum::operator-(vec1, eigenVec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(minus1.x()[i], eigenVec1.x() - vec1.x()[i]);
    EXPECT_FLOAT_EQ(minus1.y()[i], eigenVec1.y() - vec1.y()[i]);
    EXPECT_FLOAT_EQ(minus1.z()[i], eigenVec1.z() - vec1.z()[i]);

    EXPECT_FLOAT_EQ(minus2.x()[i], vec1.x()[i] - eigenVec1.x());
    EXPECT_FLOAT_EQ(minus2.y()[i], vec1.y()[i] - eigenVec1.y());
    EXPECT_FLOAT_EQ(minus2.z()[i], vec1.z()[i] - eigenVec1.z());
  }

  // Subtraction of two Vector3fP-s
  const Vector3fP minus3 = vec1 - vec2;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_FLOAT_EQ(minus3.x()[i], vec1.x()[i] - vec2.x()[i]);
    EXPECT_FLOAT_EQ(minus3.y()[i], vec1.y()[i] - vec2.y()[i]);
    EXPECT_FLOAT_EQ(minus3.z()[i], vec1.z()[i] - vec2.z()[i]);
  }

  // Multiplication of 3x3 matrix and 3x1 vector in packet
  const Eigen::Matrix3f eigenMat = Eigen::Matrix3f::Random();
  const Vector3fP prod = momentum::operator*(eigenMat, vec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_NEAR(prod.x()[i], eigenMat.row(0).dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])), 5e-6f);
    EXPECT_NEAR(prod.y()[i], eigenMat.row(1).dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])), 5e-6f);
    EXPECT_NEAR(prod.z()[i], eigenMat.row(2).dot(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])), 5e-6f);
  }

  // Affine transformation
  Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
  eigenAffine.linear() = Eigen::Quaternionf::UnitRandom().toRotationMatrix();
  eigenAffine.translation() = Eigen::Vector3f::Random();
  const Vector3fP affineTransform = eigenAffine * vec1;
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_TRUE(
        Eigen::Vector3f(affineTransform.x()[i], affineTransform.y()[i], affineTransform.z()[i])
            .isApprox(eigenAffine * Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])));
  }

  // Cross product of Eigen::Vector3f and Vector3fP
  const Vector3fP crossProd1 = momentum::cross(eigenVec1, vec1);
  const Vector3fP crossProd2 = momentum::cross(vec1, eigenVec1);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_TRUE(
        Eigen::Vector3f(crossProd1.x()[i], crossProd1.y()[i], crossProd1.z()[i])
            .isApprox(eigenVec1.cross(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i]))));
    EXPECT_TRUE(
        Eigen::Vector3f(crossProd2.x()[i], crossProd2.y()[i], crossProd2.z()[i])
            .isApprox(Eigen::Vector3f(pX1[i], pY1[i], pZ1[i]).cross(eigenVec1)));
  }

  // Cross product of two Vector3fP-s
  const Vector3fP crossProd3 = cross(vec1, vec2);
  for (auto i = 0u; i < kSimdPacketSize; ++i) {
    EXPECT_TRUE(
        Eigen::Vector3f(crossProd3.x()[i], crossProd3.y()[i], crossProd3.z()[i])
            .isApprox(
                Eigen::Vector3f(pX1[i], pY1[i], pZ1[i])
                    .cross(Eigen::Vector3f(pX2[i], pY2[i], pZ2[i]))));
  }
}
