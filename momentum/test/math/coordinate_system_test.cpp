/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/math/constants.h>
#include <momentum/math/coordinate_system.h>

#include <gtest/gtest.h>

using namespace momentum;

// --- scaleFactor tests ---

TEST(CoordinateSystemTest, ScaleFactor_SameUnits) {
  constexpr CoordinateSystem a{UpAxis::Y, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem b{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  EXPECT_FLOAT_EQ(scaleFactor<float>(a, b), 1.0f);
}

TEST(CoordinateSystemTest, ScaleFactor_MetersToCentimeters) {
  constexpr CoordinateSystem m{UpAxis::Y, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem cm{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  EXPECT_FLOAT_EQ(scaleFactor<float>(m, cm), 100.0f);
  EXPECT_FLOAT_EQ(scaleFactor<float>(cm, m), 0.01f);
}

TEST(CoordinateSystemTest, ScaleFactor_MillimetersToMeters) {
  constexpr CoordinateSystem mm{UpAxis::Y, Handedness::Right, LengthUnit::Millimeter};
  constexpr CoordinateSystem m{UpAxis::Y, Handedness::Right, LengthUnit::Meter};
  EXPECT_FLOAT_EQ(scaleFactor<float>(mm, m), 0.001f);
  EXPECT_FLOAT_EQ(scaleFactor<float>(m, mm), 1000.0f);
}

TEST(CoordinateSystemTest, ScaleFactor_DecimetersToCentimeters) {
  constexpr CoordinateSystem dm{UpAxis::Y, Handedness::Right, LengthUnit::Decimeter};
  constexpr CoordinateSystem cm{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  EXPECT_FLOAT_EQ(scaleFactor<float>(dm, cm), 10.0f);
}

// --- changeVector identity tests ---

TEST(CoordinateSystemTest, ChangeVector_Identity) {
  constexpr CoordinateSystem cs{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto result = changeVector(v, cs, cs);
  EXPECT_TRUE(result.isApprox(v));
}

// --- changeVector known-value tests ---

TEST(CoordinateSystemTest, ChangeVector_ZUpMeters_To_YUpCentimeters) {
  // This is the XSens -> Momentum conversion
  constexpr CoordinateSystem zUpM{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto result = changeVector(v, zUpM, kMomentumCoordinateSystem);

  // Z-up right-hand: X=right, Y=forward, Z=up
  // Y-up right-hand: X=right, Y=up, Z=back
  // Mapping: X->X, Z->Y, -Y->Z, then scale *100
  // (1, 2, 3) -> (1, 3, -2) * 100 = (100, 300, -200)
  EXPECT_FLOAT_EQ(result.x(), 100.0f);
  EXPECT_FLOAT_EQ(result.y(), 300.0f);
  EXPECT_FLOAT_EQ(result.z(), -200.0f);
}

TEST(CoordinateSystemTest, ChangeVector_YUpMeters_To_YUpCentimeters) {
  // This is the glTF -> Momentum conversion (pure scaling, no axis change)
  constexpr CoordinateSystem yUpM{UpAxis::Y, Handedness::Right, LengthUnit::Meter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto result = changeVector(v, yUpM, kMomentumCoordinateSystem);
  EXPECT_TRUE(result.isApprox(Vector3f(100.0f, 200.0f, 300.0f)));
}

// --- changeVector round-trip tests ---

TEST(CoordinateSystemTest, ChangeVector_RoundTrip) {
  constexpr CoordinateSystem a{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem b{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Vector3f v(1.5f, -2.3f, 4.7f);
  const auto roundTrip = changeVector(changeVector(v, a, b), b, a);
  EXPECT_TRUE(roundTrip.isApprox(v, 1e-5f));
}

TEST(CoordinateSystemTest, ChangeVector_RoundTrip_XUp) {
  constexpr CoordinateSystem a{UpAxis::X, Handedness::Right, LengthUnit::Millimeter};
  constexpr CoordinateSystem b{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Vector3f v(3.0f, -1.0f, 7.0f);
  const auto roundTrip = changeVector(changeVector(v, a, b), b, a);
  EXPECT_TRUE(roundTrip.isApprox(v, 1e-5f));
}

// --- changeVector double precision ---

TEST(CoordinateSystemTest, ChangeVector_DoublePrecision) {
  constexpr CoordinateSystem zUpM{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  const Vector3d v(1.0, 2.0, 3.0);
  const auto result = changeVector(v, zUpM, kMomentumCoordinateSystem);
  EXPECT_DOUBLE_EQ(result.x(), 100.0);
  EXPECT_DOUBLE_EQ(result.y(), 300.0);
  EXPECT_DOUBLE_EQ(result.z(), -200.0);
}

// --- changeQuaternion tests ---

TEST(CoordinateSystemTest, ChangeQuaternion_Identity) {
  constexpr CoordinateSystem cs{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Quaternionf q = Quaternionf::Identity();
  const auto result = changeQuaternion(q, cs, cs);
  EXPECT_TRUE(result.isApprox(q));
}

TEST(CoordinateSystemTest, ChangeQuaternion_RoundTrip) {
  constexpr CoordinateSystem a{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem b{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Quaternionf q(AngleAxisf(0.5f, Vector3f::UnitX()));
  const auto roundTrip = changeQuaternion(changeQuaternion(q, a, b), b, a);
  // Quaternions may differ by sign (q == -q for rotations)
  EXPECT_TRUE(
      roundTrip.isApprox(q) || roundTrip.isApprox(Quaternionf(-q.w(), -q.x(), -q.y(), -q.z())));
}

TEST(CoordinateSystemTest, ChangeQuaternion_PreservesRotationSemantics) {
  // A rotation around Z (the up axis in Z-up) should become a rotation around Y (the up axis in
  // Y-up)
  constexpr CoordinateSystem zUp{UpAxis::Z, Handedness::Right, LengthUnit::Meter};

  const Quaternionf qAroundUp(AngleAxisf(pi<float>() / 2.0f, Vector3f::UnitZ()));
  const auto result = changeQuaternion(qAroundUp, zUp, kMomentumCoordinateSystem);

  const Quaternionf expected(AngleAxisf(pi<float>() / 2.0f, Vector3f::UnitY()));
  EXPECT_TRUE(
      result.isApprox(expected, 1e-5f) ||
      result.isApprox(
          Quaternionf(-expected.w(), -expected.x(), -expected.y(), -expected.z()), 1e-5f));
}

// --- changeRotationMatrix tests ---

TEST(CoordinateSystemTest, ChangeRotationMatrix_Identity) {
  constexpr CoordinateSystem cs{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Matrix3f m = Matrix3f::Identity();
  const auto result = changeRotationMatrix(m, cs, cs);
  EXPECT_TRUE(result.isApprox(m));
}

TEST(CoordinateSystemTest, ChangeRotationMatrix_RoundTrip) {
  constexpr CoordinateSystem a{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem b{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  const Matrix3f m = AngleAxisf(0.7f, Vector3f(1, 2, 3).normalized()).toRotationMatrix();
  const auto roundTrip = changeRotationMatrix(changeRotationMatrix(m, a, b), b, a);
  EXPECT_TRUE(roundTrip.isApprox(m, 1e-5f));
}

// --- changeQuaternion consistency with changeRotationMatrix ---

TEST(CoordinateSystemTest, QuaternionAndMatrixConsistency) {
  constexpr CoordinateSystem a{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  constexpr CoordinateSystem b{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};

  const Quaternionf q(AngleAxisf(0.7f, Vector3f(1, 2, 3).normalized()));
  const Matrix3f m = q.toRotationMatrix();

  const auto qResult = changeQuaternion(q, a, b);
  const auto mResult = changeRotationMatrix(m, a, b);

  EXPECT_TRUE(qResult.toRotationMatrix().isApprox(mResult, 1e-5f));
}

// --- toMomentum / fromMomentum convenience tests ---

TEST(CoordinateSystemTest, ToFromMomentumVector) {
  constexpr CoordinateSystem zUpM{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto toM = toMomentumVector(v, zUpM);
  const auto backToZ = fromMomentumVector(toM, zUpM);
  EXPECT_TRUE(backToZ.isApprox(v, 1e-5f));
}

TEST(CoordinateSystemTest, ToFromMomentumQuaternion) {
  constexpr CoordinateSystem zUpM{UpAxis::Z, Handedness::Right, LengthUnit::Meter};
  const Quaternionf q(AngleAxisf(1.0f, Vector3f::UnitZ()));
  const auto toM = toMomentumQuaternion(q, zUpM);
  const auto back = fromMomentumQuaternion(toM, zUpM);
  EXPECT_TRUE(back.isApprox(q) || back.isApprox(Quaternionf(-q.w(), -q.x(), -q.y(), -q.z())));
}

// --- Handedness flip tests ---

TEST(CoordinateSystemTest, ChangeVector_HandednessFlip) {
  constexpr CoordinateSystem rightHand{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  constexpr CoordinateSystem leftHand{UpAxis::Y, Handedness::Left, LengthUnit::Centimeter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto result = changeVector(v, rightHand, leftHand);
  // Handedness flip negates the forward axis (Z for Y-up)
  EXPECT_FLOAT_EQ(result.x(), 1.0f);
  EXPECT_FLOAT_EQ(result.y(), 2.0f);
  EXPECT_FLOAT_EQ(result.z(), -3.0f);
}

TEST(CoordinateSystemTest, ChangeVector_HandednessFlip_RoundTrip) {
  constexpr CoordinateSystem rightHand{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter};
  constexpr CoordinateSystem leftHand{UpAxis::Y, Handedness::Left, LengthUnit::Centimeter};
  const Vector3f v(1.0f, 2.0f, 3.0f);
  const auto roundTrip = changeVector(changeVector(v, rightHand, leftHand), leftHand, rightHand);
  EXPECT_TRUE(roundTrip.isApprox(v, 1e-5f));
}
