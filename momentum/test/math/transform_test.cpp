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

TYPED_TEST(TransformTest, InverseScalePrecision) {
  using T = typename TestFixture::Type;

  for (const T scale : {T(0.1), T(1.0), T(10.0)}) {
    const TransformT<T> trans(Vector3<T>::Random(), Quaternion<T>::UnitRandom(), scale);
    const TransformT<T> identity = trans * trans.inverse();

    EXPECT_TRUE(identity.translation.isZero(Eps<T>(1e-4f, 1e-12)));
    EXPECT_TRUE(identity.rotation.isApprox(Quaternion<T>::Identity(), Eps<T>(1e-4f, 1e-12)));
    EXPECT_NEAR(identity.scale, T(1), Eps<T>(1e-4f, 1e-12));
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

TYPED_TEST(TransformTest, Blend1) {
  using T = typename TestFixture::Type;

  const auto nTest = 10;
  for (size_t i = 0; i < nTest; ++i) {
    const auto js = TransformT<T>::makeRandom();
    const std::vector<TransformT<T>> transforms{js};
    const std::vector<T> weights{T(1.0)};
    const auto blended =
        blendTransforms(gsl::span<const TransformT<T>>(transforms), gsl::span<const T>(weights));

    ASSERT_LT(
        (js.toMatrix() - blended.toMatrix()).template lpNorm<Eigen::Infinity>(),
        Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, Blend2) {
  using T = typename TestFixture::Type;

  const auto nTest = 10;
  for (size_t i = 0; i < nTest; ++i) {
    const auto js1 = TransformT<T>::makeRandom();
    const auto js2 = TransformT<T>::makeRandom();

    {
      const std::vector<TransformT<T>> transforms{js1, js2};
      const std::vector<T> weights{T(1.0), T(0.0)};
      const auto blended =
          blendTransforms(gsl::span<const TransformT<T>>(transforms), gsl::span<const T>(weights));
      ASSERT_LT(
          (js1.toMatrix() - blended.toMatrix()).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));
    }

    {
      const std::vector<TransformT<T>> transforms{js1, js2};
      const std::vector<T> weights{T(0.0), T(1.0)};
      const auto blended =
          blendTransforms(gsl::span<const TransformT<T>>(transforms), gsl::span<const T>(weights));
      ASSERT_LT(
          (js2.toMatrix() - blended.toMatrix()).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));
    }

    {
      const std::vector<TransformT<T>> transforms{js1, js2};
      const std::vector<T> weights{T(2.0), T(2.0)};
      const auto halfway =
          blendTransforms(gsl::span<const TransformT<T>>(transforms), gsl::span<const T>(weights));
      const auto lerped = slerp(js1, js2, T(0.5));
      ASSERT_LT(
          (halfway.toMatrix() - lerped.toMatrix()).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));
    }
  }
}

// Antipodal quaternions are the same quaternion, so they should blend to the
// same rotation no matter what the coeffs.
TYPED_TEST(TransformTest, BlendOpposites) {
  using T = typename TestFixture::Type;

  const auto nTest = 10;
  for (size_t iTest = 0; iTest < nTest; ++iTest) {
    const auto js1 = TransformT<T>::makeRandom();
    const TransformT<T> js2(js1.translation, Quaternion<T>(-js1.rotation.coeffs()), js1.scale);

    for (int j = 0; j < 5; ++j) {
      const std::vector<TransformT<T>> transforms{js1, js2};
      const std::vector<T> weights{T(j), T(4 - j)};
      const auto blended =
          blendTransforms(gsl::span<const TransformT<T>>(transforms), gsl::span<const T>(weights));
      ASSERT_LT(
          (js1.toMatrix() - blended.toMatrix()).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));
      ASSERT_LT(
          (js2.toMatrix() - blended.toMatrix()).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));
    }
  }
}

TYPED_TEST(TransformTest, SlerpTest) {
  using T = typename TestFixture::Type;

  const TransformT<T> t1; // identity
  const Vector3<T> trans(1, 0, 0);
  const Vector3<T> rotDir(0, 0, 1);
  const T rotAmt = pi<T>() / static_cast<T>(4.0);
  const TransformT<T> t2(trans, Quaternion<T>(Eigen::AngleAxis<T>(rotAmt, rotDir)));

  // Check the endpoints:
  {
    const TransformT<T> l1 = slerp(t1, t2, T(0.0));
    ASSERT_LT(
        (l1.toMatrix() - t1.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  {
    const TransformT<T> l2 = slerp(t1, t2, T(1.0));
    ASSERT_LT(
        (l2.toMatrix() - t2.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  const size_t numSamples = 10;
  for (size_t i = 0; i <= numSamples; ++i) {
    const T amt = T(i) / T(numSamples);
    const TransformT<T> l3 = slerp(t1, t2, amt);
    ASSERT_LT(
        (l3.translation - amt * trans).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
    const TransformT<T> expected(
        amt * trans, Quaternion<T>(Eigen::AngleAxis<T>(amt * rotAmt, rotDir)));
    ASSERT_LT(
        (l3.toMatrix() - expected.toMatrix()).template lpNorm<Eigen::Infinity>(),
        Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, SlerpRandomTransforms) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 20; ++iTest) {
    const TransformT<T> t1 = TransformT<T>::makeRandom();
    const TransformT<T> t2 = TransformT<T>::makeRandom();

    // Check the endpoints:
    {
      const TransformT<T> l1 = slerp(t1, t2, T(0.0));
      ASSERT_LT(
          (l1.toMatrix() - t1.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
    }

    {
      const TransformT<T> l2 = slerp(t1, t2, T(1.0));
      ASSERT_LT(
          (l2.toMatrix() - t2.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
    }

    // Check intermediate values
    for (size_t i = 1; i < 10; ++i) {
      const T amt = T(i) / T(10);
      const TransformT<T> lerped = slerp(t1, t2, amt);

      // Check that translation is linearly interpolated
      const Vector3<T> expectedTranslation =
          t1.translation + amt * (t2.translation - t1.translation);
      ASSERT_LT(
          (lerped.translation - expectedTranslation).template lpNorm<Eigen::Infinity>(),
          Eps<T>(1e-5f, 1e-13));

      // Check that scale is linearly interpolated
      const T expectedScale = t1.scale + amt * (t2.scale - t1.scale);
      ASSERT_NEAR(lerped.scale, expectedScale, Eps<T>(1e-5f, 1e-13));

      // Check that the rotation interpolation produces a valid rotation
      ASSERT_NEAR(lerped.rotation.norm(), T(1.0), Eps<T>(1e-5f, 1e-13));
    }
  }
}

TYPED_TEST(TransformTest, SlerpIdentityTransforms) {
  using T = typename TestFixture::Type;

  const TransformT<T> identity;

  // Slerp between identity and itself should be identity
  for (size_t i = 0; i <= 10; ++i) {
    const T amt = T(i) / T(10);
    const TransformT<T> result = slerp(identity, identity, amt);

    ASSERT_LT(
        (result.toMatrix() - identity.toMatrix()).template lpNorm<Eigen::Infinity>(),
        Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, SlerpScaleOnly) {
  using T = typename TestFixture::Type;

  const TransformT<T> t1 = TransformT<T>::makeScale(T(0.5));
  const TransformT<T> t2 = TransformT<T>::makeScale(T(2.0));

  // Check endpoints
  {
    const TransformT<T> l1 = slerp(t1, t2, T(0.0));
    ASSERT_NEAR(l1.scale, T(0.5), Eps<T>(1e-5f, 1e-13));
  }

  {
    const TransformT<T> l2 = slerp(t1, t2, T(1.0));
    ASSERT_NEAR(l2.scale, T(2.0), Eps<T>(1e-5f, 1e-13));
  }

  // Check midpoint
  {
    const TransformT<T> l3 = slerp(t1, t2, T(0.5));
    ASSERT_NEAR(l3.scale, T(1.25), Eps<T>(1e-5f, 1e-13)); // 0.5 + 0.5 * (2.0 - 0.5) = 1.25
  }
}

TYPED_TEST(TransformTest, SlerpTranslationOnly) {
  using T = typename TestFixture::Type;

  const Vector3<T> trans1(1, 0, 0);
  const Vector3<T> trans2(0, 1, 0);
  const TransformT<T> t1 = TransformT<T>::makeTranslation(trans1);
  const TransformT<T> t2 = TransformT<T>::makeTranslation(trans2);

  // Check endpoints
  {
    const TransformT<T> l1 = slerp(t1, t2, T(0.0));
    ASSERT_LT((l1.translation - trans1).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  {
    const TransformT<T> l2 = slerp(t1, t2, T(1.0));
    ASSERT_LT((l2.translation - trans2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  // Check midpoint
  {
    const TransformT<T> l3 = slerp(t1, t2, T(0.5));
    const Vector3<T> expectedMidpoint = (trans1 + trans2) / T(2);
    ASSERT_LT(
        (l3.translation - expectedMidpoint).template lpNorm<Eigen::Infinity>(),
        Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, SlerpRotationOnly) {
  using T = typename TestFixture::Type;

  const Vector3<T> axis(0, 0, 1);
  const T angle1 = T(0);
  const T angle2 = pi<T>() / static_cast<T>(2);

  const TransformT<T> t1 =
      TransformT<T>::makeRotation(Quaternion<T>(Eigen::AngleAxis<T>(angle1, axis)));
  const TransformT<T> t2 =
      TransformT<T>::makeRotation(Quaternion<T>(Eigen::AngleAxis<T>(angle2, axis)));

  // Check endpoints
  {
    const TransformT<T> l1 = slerp(t1, t2, T(0.0));
    ASSERT_LT(
        (l1.toMatrix() - t1.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  {
    const TransformT<T> l2 = slerp(t1, t2, T(1.0));
    ASSERT_LT(
        (l2.toMatrix() - t2.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }

  // Check that midpoint rotation is approximately at PI/4
  {
    const TransformT<T> l3 = slerp(t1, t2, T(0.5));
    const TransformT<T> expectedMidpoint = TransformT<T>::makeRotation(
        Quaternion<T>(Eigen::AngleAxis<T>(pi<T>() / static_cast<T>(4), axis)));
    ASSERT_LT(
        (l3.toMatrix() - expectedMidpoint.toMatrix()).template lpNorm<Eigen::Infinity>(),
        Eps<T>(1e-4f, 1e-12)); // Slightly looser tolerance for rotation interpolation
  }
}
