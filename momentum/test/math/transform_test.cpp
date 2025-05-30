/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/math/transform.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct TransformTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(TransformTest, Types);

// Test static factory methods
TYPED_TEST(TransformTest, FactoryMethods) {
  using T = typename TestFixture::Type;

  // Test makeRotation
  const Quaternion<T> rotation = Quaternion<T>::UnitRandom();
  const TransformT<T> rotTransform = TransformT<T>::makeRotation(rotation);
  EXPECT_TRUE(rotTransform.rotation.isApprox(rotation));
  EXPECT_TRUE(rotTransform.translation.isZero());
  EXPECT_EQ(rotTransform.scale, T(1));

  // Test makeTranslation
  const Vector3<T> translation = Vector3<T>::Random();
  const TransformT<T> transTransform = TransformT<T>::makeTranslation(translation);
  EXPECT_TRUE(transTransform.translation.isApprox(translation));
  EXPECT_TRUE(transTransform.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_EQ(transTransform.scale, T(1));

  // Test makeScale
  const T scale = uniform<T>(0.1, 10);
  const TransformT<T> scaleTransform = TransformT<T>::makeScale(scale);
  EXPECT_TRUE(scaleTransform.translation.isZero());
  EXPECT_TRUE(scaleTransform.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_EQ(scaleTransform.scale, scale);

  // Test makeRandom with different combinations
  const TransformT<T> randomTransform1 = TransformT<T>::makeRandom(true, true, true);
  EXPECT_FALSE(randomTransform1.translation.isZero());
  EXPECT_FALSE(randomTransform1.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_NE(randomTransform1.scale, T(1));

  const TransformT<T> randomTransform2 = TransformT<T>::makeRandom(true, false, false);
  EXPECT_FALSE(randomTransform2.translation.isZero());
  EXPECT_TRUE(randomTransform2.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_EQ(randomTransform2.scale, T(1));

  const TransformT<T> randomTransform3 = TransformT<T>::makeRandom(false, true, false);
  EXPECT_TRUE(randomTransform3.translation.isZero());
  EXPECT_FALSE(randomTransform3.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_EQ(randomTransform3.scale, T(1));

  const TransformT<T> randomTransform4 = TransformT<T>::makeRandom(false, false, true);
  EXPECT_TRUE(randomTransform4.translation.isZero());
  EXPECT_TRUE(randomTransform4.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_NE(randomTransform4.scale, T(1));
}

// Test constructors
TYPED_TEST(TransformTest, Constructors) {
  using T = typename TestFixture::Type;

  // Test default constructor
  const TransformT<T> defaultTransform;
  EXPECT_TRUE(defaultTransform.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_TRUE(defaultTransform.translation.isZero());
  EXPECT_EQ(defaultTransform.scale, T(1));

  // Test constructor with parameters
  const Vector3<T> translation = Vector3<T>::Random();
  const Quaternion<T> rotation = Quaternion<T>::UnitRandom();
  const T scale = uniform<T>(0.1, 10);

  const TransformT<T> paramTransform(translation, rotation, scale);
  EXPECT_TRUE(paramTransform.translation.isApprox(translation));
  EXPECT_TRUE(paramTransform.rotation.isApprox(rotation));
  EXPECT_EQ(paramTransform.scale, scale);

  // Test constructor with default parameters
  const TransformT<T> defaultParamTransform(translation);
  EXPECT_TRUE(defaultParamTransform.translation.isApprox(translation));
  EXPECT_TRUE(defaultParamTransform.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_EQ(defaultParamTransform.scale, T(1));

  // Test constructor from Affine3
  Affine3<T> affine = Affine3<T>::Identity();
  affine = Translation3<T>(translation) * rotation * Eigen::Scaling(scale);

  const TransformT<T> affineTransform(affine);
  EXPECT_TRUE(affineTransform.translation.isApprox(translation));
  // Using matrices to avoid quaternion sign ambiguity
  EXPECT_TRUE(affineTransform.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(affineTransform.scale, scale, Eps<T>(1e-5f, 1e-13));

  // Test constructor from Matrix4
  Matrix4<T> matrix = Matrix4<T>::Identity();
  matrix.template block<3, 1>(0, 3) = translation;
  matrix.template block<3, 3>(0, 0) = rotation.toRotationMatrix() * scale;

  const TransformT<T> matrixTransform(matrix);
  EXPECT_TRUE(matrixTransform.translation.isApprox(translation));
  // Using matrices to avoid quaternion sign ambiguity
  EXPECT_TRUE(matrixTransform.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(matrixTransform.scale, scale, Eps<T>(1e-5f, 1e-13));

  // Test copy constructor with type conversion
  const TransformT<float> floatTransform(Vector3f::Random(), Quaternionf::UnitRandom(), 2.5f);
  const TransformT<double> doubleTransform(floatTransform);

  EXPECT_TRUE(doubleTransform.translation.isApprox(floatTransform.translation.cast<double>()));
  EXPECT_TRUE(doubleTransform.rotation.isApprox(floatTransform.rotation.cast<double>()));
  EXPECT_DOUBLE_EQ(doubleTransform.scale, floatTransform.scale);
}

// Test assignment operators
TYPED_TEST(TransformTest, AssignmentOperators) {
  using T = typename TestFixture::Type;

  const Vector3<T> translation = Vector3<T>::Random();
  const Quaternion<T> rotation = Quaternion<T>::UnitRandom();
  const T scale = uniform<T>(0.1, 10);

  // Test assignment from Affine3
  Affine3<T> affine = Affine3<T>::Identity();
  affine.translate(translation);
  affine.rotate(rotation);
  affine.scale(scale);

  TransformT<T> transform1;
  transform1 = affine;

  EXPECT_TRUE(transform1.translation.isApprox(translation));
  // Compare rotation matrices instead of quaternions to handle sign ambiguity
  EXPECT_TRUE(transform1.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(transform1.scale, scale, Eps<T>(1e-5f, 1e-13));

  // Test assignment from Matrix4
  Matrix4<T> matrix = Matrix4<T>::Identity();
  matrix.template block<3, 1>(0, 3) = translation;
  matrix.template block<3, 3>(0, 0) = rotation.toRotationMatrix() * scale;

  TransformT<T> transform2;
  transform2 = matrix;

  EXPECT_TRUE(transform2.translation.isApprox(translation));
  // Compare rotation matrices instead of quaternions to handle sign ambiguity
  EXPECT_TRUE(transform2.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(transform2.scale, scale, Eps<T>(1e-5f, 1e-13));
}

// Test conversion methods
TYPED_TEST(TransformTest, ConversionMethods) {
  using T = typename TestFixture::Type;

  const Vector3<T> translation = Vector3<T>::Random();
  const Quaternion<T> rotation = Quaternion<T>::UnitRandom();
  const T scale = uniform<T>(0.1, 10);

  const TransformT<T> transform(translation, rotation, scale);

  // Test toMatrix
  const Matrix4<T> matrix = transform.toMatrix();
  auto topRight = matrix.block(0, 3, 3, 1);
  EXPECT_TRUE(topRight.isApprox(translation));
  auto topLeft = matrix.block(0, 0, 3, 3);
  auto scaledRotation = topLeft / scale;
  EXPECT_TRUE(scaledRotation.isApprox(rotation.toRotationMatrix()));
  EXPECT_EQ(matrix(3, 3), T(1));

  // Test toRotationMatrix
  const Matrix3<T> rotMatrix = transform.toRotationMatrix();
  EXPECT_TRUE(rotMatrix.isApprox(rotation.toRotationMatrix()));

  // Test toLinear
  const Matrix3<T> linear = transform.toLinear();
  EXPECT_TRUE(linear.isApprox(rotation.toRotationMatrix() * scale));

  // Test explicit conversion to Affine3
  const auto affine = static_cast<Affine3<T>>(transform);
  EXPECT_TRUE(affine.translation().isApprox(translation));
  EXPECT_TRUE((affine.linear() / scale).isApprox(rotation.toRotationMatrix()));

  // Test cast to different type
  const TransformT<float> floatTransform = transform.template cast<float>();
  EXPECT_TRUE(floatTransform.translation.isApprox(translation.template cast<float>()));
  EXPECT_TRUE(floatTransform.rotation.isApprox(rotation.template cast<float>()));
  EXPECT_FLOAT_EQ(floatTransform.scale, static_cast<float>(scale));
}

// Test operator* with Affine3 and Vector3
TYPED_TEST(TransformTest, AdditionalOperators) {
  using T = typename TestFixture::Type;

  const TransformT<T> transform = TransformT<T>::makeRandom();

  // Test operator*(const Affine3<T>&)
  const Vector3<T> translation = Vector3<T>::Random();
  const Affine3<T> affine = Affine3<T>(Translation3<T>(translation));

  const Affine3<T> result1 = transform * affine;
  const Affine3<T> expected1 = transform.toAffine3() * affine;

  EXPECT_LE(
      (result1.matrix() - expected1.matrix()).template lpNorm<Eigen::Infinity>(),
      Eps<T>(1e-5f, 1e-13));

  // Test operator*(const Vector3<T>&)
  const Vector3<T> point = Vector3<T>::Random();
  const Vector3<T> result2 = transform * point;
  const Vector3<T> expected2 = transform.transformPoint(point);

  EXPECT_TRUE(result2.isApprox(expected2));
}

// Test fromAffine3 and fromMatrix static methods
TYPED_TEST(TransformTest, FromMethods) {
  using T = typename TestFixture::Type;

  const Vector3<T> translation = Vector3<T>::Random();
  const Quaternion<T> rotation = Quaternion<T>::UnitRandom();
  const T scale = uniform<T>(0.1, 10);

  // Test fromAffine3
  Affine3<T> affine = Translation3<T>(translation) * rotation * Eigen::Scaling(scale);

  const TransformT<T> transform1 = TransformT<T>::fromAffine3(affine);

  EXPECT_TRUE(transform1.translation.isApprox(translation));
  // Compare rotation matrices instead of quaternions to handle sign ambiguity
  EXPECT_TRUE(transform1.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(transform1.scale, scale, Eps<T>(1e-5f, 1e-13));

  // Test fromMatrix
  Matrix4<T> matrix = Matrix4<T>::Identity();
  matrix.template block<3, 1>(0, 3) = translation;
  matrix.template block<3, 3>(0, 0) = rotation.toRotationMatrix() * scale;

  const TransformT<T> transform2 = TransformT<T>::fromMatrix(matrix);

  EXPECT_TRUE(transform2.translation.isApprox(translation));
  // Compare rotation matrices instead of quaternions to handle sign ambiguity
  EXPECT_TRUE(transform2.rotation.toRotationMatrix().isApprox(rotation.toRotationMatrix()));
  EXPECT_NEAR(transform2.scale, scale, Eps<T>(1e-5f, 1e-13));
}

// Test edge cases
TYPED_TEST(TransformTest, EdgeCases) {
  using T = typename TestFixture::Type;

  // Test identity transform
  const TransformT<T> identity;
  EXPECT_TRUE(identity.rotation.isApprox(Quaternion<T>::Identity()));
  EXPECT_TRUE(identity.translation.isZero());
  EXPECT_EQ(identity.scale, T(1));

  // Test that identity * transform = transform
  const TransformT<T> transform = TransformT<T>::makeRandom();
  const TransformT<T> result = identity * transform;

  EXPECT_TRUE(result.translation.isApprox(transform.translation));
  EXPECT_TRUE(result.rotation.isApprox(transform.rotation));
  EXPECT_NEAR(result.scale, transform.scale, Eps<T>(1e-5f, 1e-13));

  // Test that transform * identity = transform
  const TransformT<T> result2 = transform * identity;

  EXPECT_TRUE(result2.translation.isApprox(transform.translation));
  EXPECT_TRUE(result2.rotation.isApprox(transform.rotation));
  EXPECT_NEAR(result2.scale, transform.scale, Eps<T>(1e-5f, 1e-13));

  // Test that transform * transform.inverse() = identity
  const TransformT<T> inverse = transform.inverse();
  const TransformT<T> result3 = transform * inverse;

  EXPECT_TRUE(result3.translation.isZero(Eps<T>(1e-4f, 1e-12)));
  EXPECT_TRUE(result3.rotation.isApprox(Quaternion<T>::Identity(), Eps<T>(1e-4f, 1e-12)));
  EXPECT_NEAR(result3.scale, T(1), Eps<T>(1e-4f, 1e-12));

  // Test that transform.inverse() * transform = identity
  const TransformT<T> result4 = inverse * transform;

  EXPECT_TRUE(result4.translation.isZero(Eps<T>(1e-4f, 1e-12)));
  EXPECT_TRUE(result4.rotation.isApprox(Quaternion<T>::Identity(), Eps<T>(1e-4f, 1e-12)));
  EXPECT_NEAR(result4.scale, T(1), Eps<T>(1e-4f, 1e-12));
}

TYPED_TEST(TransformTest, Multiplication) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const TransformT<T> trans2 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);

    const TransformT<T> tmp = trans1 * trans2;
    const Affine3<T> res1 = tmp.toAffine3();
    const Affine3<T> res2 = trans1.toAffine3() * trans2.toAffine3();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, Inverse) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);

    const Affine3<T> res1 = trans1.inverse().toAffine3();
    const Affine3<T> res2 = trans1.toAffine3().inverse();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformPoint) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const auto randomPoint = uniform<Vector3<T>>(-10, 10);

    const Eigen::Vector3<T> res1 = trans1.transformPoint(randomPoint);
    const Eigen::Vector3<T> res2 = trans1.toAffine3() * randomPoint;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformVec) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const Eigen::Vector3<T> randomVec = uniform<Vector3<T>>(-1, 1).normalized();

    const Eigen::Vector3<T> res1 = trans1.rotate(randomVec);
    const Eigen::Vector3<T> res2 = trans1.toAffine3().rotation() * randomVec;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-6f, 5e-15));
  }
}

// This test is to make sure that the Eigen::Affine3f and momentum::Transformf are
// intercompatible.
TYPED_TEST(TransformTest, CompatibleWithEigenAffine) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    // const TransformT<T> tf1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const TransformT<T> tf1 = TransformT<T>::makeRandom();
    const Affine3<T> tf2 = tf1.toAffine3();
    TransformT<T> tf3;
    tf3 = tf2;
    const Affine3<T> tf4 = tf3.toAffine3();

    EXPECT_LE(
        (tf1.toMatrix() - tf2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf2.matrix() - tf3.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf3.toMatrix() - tf4.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf4.matrix() - tf1.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
  }
}
