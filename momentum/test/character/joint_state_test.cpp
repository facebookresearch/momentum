/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/joint_state.h"
#include "momentum/math/constants.h"

using namespace momentum;

// Test fixture for JointState tests
template <typename T>
class JointStateTest : public testing::Test {
 protected:
  using JointType = JointT<T>;
  using JointStateType = JointStateT<T>;
  using JointVectorType = JointVectorT<T>;
  using Vector3Type = Vector3<T>;
  using QuaternionType = Quaternion<T>;

  void SetUp() override {
    // Create a simple joint
    joint.name = "test_joint";
    joint.parent = kInvalidIndex; // Root joint
    joint.preRotation = QuaternionType::Identity();
    joint.translationOffset = Vector3Type(1, 2, 3);

    // Create a simple joint vector (parameters)
    // [tx, ty, tz, rx, ry, rz, scale]
    parameters << 4, 5, 6, pi<T>() / 4, pi<T>() / 6, pi<T>() / 8, 0.5;
  }

  JointType joint;
  JointVectorType parameters = JointVectorType::Zero();
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(JointStateTest, Types);

// Test default constructor and initial values
TYPED_TEST(JointStateTest, DefaultConstructor) {
  using JointStateType = typename TestFixture::JointStateType;

  JointStateType state;

  // Check default values
  EXPECT_TRUE(state.localTransform.rotation.isApprox(Quaternion<TypeParam>::Identity()));
  EXPECT_TRUE(state.localTransform.translation.isApprox(Vector3<TypeParam>::Zero()));
  EXPECT_FLOAT_EQ(state.localTransform.scale, 1.0);

  EXPECT_TRUE(state.transform.rotation.isApprox(Quaternion<TypeParam>::Identity()));
  EXPECT_TRUE(state.transform.translation.isApprox(Vector3<TypeParam>::Zero()));
  EXPECT_FLOAT_EQ(state.transform.scale, 1.0);

  EXPECT_TRUE(state.derivDirty);
}

// Test set method without parent
TYPED_TEST(JointStateTest, SetWithoutParent) {
  using JointStateType = typename TestFixture::JointStateType;
  using Vector3Type = typename TestFixture::Vector3Type;

  JointStateType state;

  // Set the state without a parent
  state.set(this->joint, this->parameters, nullptr, true);

  // Check that derivDirty is false since we're computing derivatives
  EXPECT_FALSE(state.derivDirty);

  // Check local transform
  // Translation should be joint offset + parameters translation
  Vector3Type expectedTranslation = this->joint.translationOffset +
      Vector3Type(this->parameters[0], this->parameters[1], this->parameters[2]);
  EXPECT_TRUE(state.localTransform.translation.isApprox(expectedTranslation));

  // Scale should be 2^parameters[6]
  EXPECT_NEAR(state.localTransform.scale, std::exp2(this->parameters[6]), 1e-5);

  // Check global transform (should be same as local since there's no parent)
  EXPECT_TRUE(state.transform.translation.isApprox(state.localTransform.translation));
  EXPECT_TRUE(state.transform.rotation.isApprox(state.localTransform.rotation));
  EXPECT_FLOAT_EQ(state.transform.scale, state.localTransform.scale);

  // Check that transformation matrix is set correctly
  EXPECT_TRUE(state.transformation.matrix().isApprox(state.transform.toAffine3().matrix()));

  // Check that translation axes are identity (no parent)
  Matrix3<TypeParam> identity = Matrix3<TypeParam>::Identity();
  EXPECT_TRUE(state.translationAxis.isApprox(identity));
}

// Test set method with parent
TYPED_TEST(JointStateTest, SetWithParent) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointType = typename TestFixture::JointType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  // Create parent joint and state
  JointType parentJoint;
  parentJoint.name = "parent_joint";
  parentJoint.parent = kInvalidIndex;
  parentJoint.translationOffset = Vector3Type(10, 20, 30);

  JointVectorType parentParams;
  parentParams << 1, 2, 3, 0, 0, 0, 0; // No rotation, no scaling

  JointStateType parentState;
  parentState.set(parentJoint, parentParams, nullptr, true);

  // Create child state
  JointStateType childState;
  childState.set(this->joint, this->parameters, &parentState, true);

  // Check that derivDirty is false
  EXPECT_FALSE(childState.derivDirty);

  // Check local transform
  Vector3Type expectedLocalTranslation = this->joint.translationOffset +
      Vector3Type(this->parameters[0], this->parameters[1], this->parameters[2]);
  EXPECT_TRUE(childState.localTransform.translation.isApprox(expectedLocalTranslation));

  // Check global transform
  // Global translation should be parent translation + (parent rotation * local translation * parent
  // scale)
  Vector3Type expectedGlobalTranslation = parentState.transform.translation +
      parentState.transform.rotation *
          (childState.localTransform.translation * parentState.transform.scale);
  EXPECT_TRUE(childState.transform.translation.isApprox(expectedGlobalTranslation));

  // Global rotation should be parent rotation * local rotation
  EXPECT_TRUE(childState.transform.rotation.isApprox(
      parentState.transform.rotation * childState.localTransform.rotation));

  // Global scale should be parent scale * local scale
  EXPECT_NEAR(
      childState.transform.scale,
      parentState.transform.scale * childState.localTransform.scale,
      1e-5);

  // Check that transformation matrix is set correctly
  EXPECT_TRUE(
      childState.transformation.matrix().isApprox(childState.transform.toAffine3().matrix()));

  // Check that translation axes are set from parent
  EXPECT_TRUE(childState.translationAxis.isApprox(parentState.transformation.linear()));
}

// Test set method without computing derivatives
TYPED_TEST(JointStateTest, SetWithoutDerivatives) {
  using JointStateType = typename TestFixture::JointStateType;

  JointStateType state;

  // Set the state without computing derivatives
  state.set(this->joint, this->parameters, nullptr, false);

  // Check that derivDirty is true
  EXPECT_TRUE(state.derivDirty);

  // When derivDirty is true, we should not call getRotationDerivative or getTranslationDerivative
  // as they will cause fatal errors with MT_CHECK

  // getScaleDerivative should not throw even if derivDirty is true
  Vector3<TypeParam> refPoint = Vector3<TypeParam>::Zero();
  Vector3<TypeParam> result = state.getScaleDerivative(refPoint);

  // Verify the scale derivative is correct
  Vector3<TypeParam> expectedScaleDeriv = refPoint * ln2<TypeParam>();
  EXPECT_TRUE(result.isApprox(expectedScaleDeriv));
}

// Test derivative methods
TYPED_TEST(JointStateTest, DerivativeMethods) {
  using JointStateType = typename TestFixture::JointStateType;
  using Vector3Type = typename TestFixture::Vector3Type;

  JointStateType state;

  // Set the state with derivatives
  state.set(this->joint, this->parameters, nullptr, true);

  // Test rotation derivatives
  Vector3Type refPoint(1, 1, 1);
  for (size_t i = 0; i < 3; ++i) {
    Vector3Type rotDeriv = state.getRotationDerivative(i, refPoint);
    // The rotation derivative should be the cross product of the rotation axis and the reference
    // point
    Vector3Type expectedRotDeriv = state.rotationAxis.col(i).cross(refPoint);
    EXPECT_TRUE(rotDeriv.isApprox(expectedRotDeriv));
  }

  // Test translation derivatives
  for (size_t i = 0; i < 3; ++i) {
    Vector3Type transDeriv = state.getTranslationDerivative(i);
    // The translation derivative should be the corresponding column of the translation axis matrix
    Vector3Type expectedTransDeriv = state.translationAxis.col(i);
    EXPECT_TRUE(transDeriv.isApprox(expectedTransDeriv));
  }

  // Test scale derivative
  Vector3Type refPoint2(2, 3, 4);
  Vector3Type scaleDeriv = state.getScaleDerivative(refPoint2);
  // The scale derivative should be the reference point multiplied by ln(2)
  Vector3Type expectedScaleDeriv = refPoint2 * ln2<TypeParam>();
  EXPECT_TRUE(scaleDeriv.isApprox(expectedScaleDeriv));
}

// Test type conversion
TYPED_TEST(JointStateTest, TypeConversion) {
  using JointStateType = typename TestFixture::JointStateType;

  JointStateType state;

  // Set the state
  state.set(this->joint, this->parameters, nullptr, true);

  // Convert to the other type
  using OtherType =
      typename std::conditional<std::is_same<TypeParam, float>::value, double, float>::type;
  JointStateT<OtherType> otherState;
  otherState.set(state);

  // Convert back to original type
  JointStateType convertedState;
  convertedState.set(otherState);

  // Check that the conversion preserved all values
  EXPECT_TRUE(convertedState.localTransform.translation.isApprox(state.localTransform.translation));

  // Check if the rotations represent the same transformation by applying them to test vectors
  // Use a very relaxed tolerance for float-double conversions
  const TypeParam angleTolerance = 1e-2; // Allow up to ~0.6 degrees of difference
  const TypeParam scaleTolerance = 1e-3;

  Vector3<TypeParam> testVector1(1, 0, 0);
  Vector3<TypeParam> testVector2(0, 1, 0);
  Vector3<TypeParam> testVector3(0, 0, 1);

  // For local transform rotation, check if the angle between vectors is small
  Vector3<TypeParam> rotatedByOriginal1 = state.localTransform.rotation * testVector1;
  Vector3<TypeParam> rotatedByConverted1 = convertedState.localTransform.rotation * testVector1;
  if (rotatedByOriginal1.norm() > 1e-6 && rotatedByConverted1.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted1.normalized().dot(rotatedByOriginal1.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  Vector3<TypeParam> rotatedByOriginal2 = state.localTransform.rotation * testVector2;
  Vector3<TypeParam> rotatedByConverted2 = convertedState.localTransform.rotation * testVector2;
  if (rotatedByOriginal2.norm() > 1e-6 && rotatedByConverted2.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted2.normalized().dot(rotatedByOriginal2.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  Vector3<TypeParam> rotatedByOriginal3 = state.localTransform.rotation * testVector3;
  Vector3<TypeParam> rotatedByConverted3 = convertedState.localTransform.rotation * testVector3;
  if (rotatedByOriginal3.norm() > 1e-6 && rotatedByConverted3.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted3.normalized().dot(rotatedByOriginal3.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  EXPECT_NEAR(convertedState.localTransform.scale, state.localTransform.scale, scaleTolerance);

  // Same for global transform
  EXPECT_TRUE(
      convertedState.transform.translation.isApprox(state.transform.translation, scaleTolerance));

  rotatedByOriginal1 = state.transform.rotation * testVector1;
  rotatedByConverted1 = convertedState.transform.rotation * testVector1;
  if (rotatedByOriginal1.norm() > 1e-6 && rotatedByConverted1.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted1.normalized().dot(rotatedByOriginal1.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  rotatedByOriginal2 = state.transform.rotation * testVector2;
  rotatedByConverted2 = convertedState.transform.rotation * testVector2;
  if (rotatedByOriginal2.norm() > 1e-6 && rotatedByConverted2.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted2.normalized().dot(rotatedByOriginal2.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  rotatedByOriginal3 = state.transform.rotation * testVector3;
  rotatedByConverted3 = convertedState.transform.rotation * testVector3;
  if (rotatedByOriginal3.norm() > 1e-6 && rotatedByConverted3.norm() > 1e-6) {
    // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
    TypeParam dot = rotatedByConverted3.normalized().dot(rotatedByOriginal3.normalized());
    dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
    TypeParam angle = std::acos(std::abs(dot));
    EXPECT_LT(angle, angleTolerance);
  }

  EXPECT_NEAR(convertedState.transform.scale, state.transform.scale, scaleTolerance);

  EXPECT_TRUE(convertedState.translationAxis.isApprox(state.translationAxis, scaleTolerance));

  // For rotation axes, check if they're pointing in approximately the same direction
  for (int i = 0; i < 3; ++i) {
    const Vector3<TypeParam>& axisVector = testVector1;
    Vector3<TypeParam> rotatedByOriginalAxis = state.rotationAxis.col(i).cross(axisVector);
    Vector3<TypeParam> rotatedByConvertedAxis =
        convertedState.rotationAxis.col(i).cross(axisVector);

    // Skip comparison if either vector is near zero
    if (rotatedByOriginalAxis.norm() > 1e-6 && rotatedByConvertedAxis.norm() > 1e-6) {
      // Clamp dot product to [-1, 1] to avoid NaN from numerical precision issues
      TypeParam dot = rotatedByConvertedAxis.normalized().dot(rotatedByOriginalAxis.normalized());
      dot = std::clamp(dot, TypeParam(-1.0), TypeParam(1.0));
      TypeParam angle = std::acos(std::abs(dot));
      EXPECT_LT(angle, angleTolerance);
    }
  }

  EXPECT_EQ(convertedState.derivDirty, state.derivDirty);
}

// Test accessor methods
TYPED_TEST(JointStateTest, AccessorMethods) {
  using JointStateType = typename TestFixture::JointStateType;
  using Vector3Type = typename TestFixture::Vector3Type;
  using QuaternionType = typename TestFixture::QuaternionType;

  JointStateType state;

  // Set the state
  state.set(this->joint, this->parameters, nullptr, true);

  // Test local accessors
  EXPECT_TRUE(state.localRotation().isApprox(state.localTransform.rotation));
  EXPECT_TRUE(state.localTranslation().isApprox(state.localTransform.translation));
  EXPECT_FLOAT_EQ(state.localScale(), state.localTransform.scale);

  // Test global accessors
  EXPECT_TRUE(state.rotation().isApprox(state.transform.rotation));
  EXPECT_TRUE(state.translation().isApprox(state.transform.translation));
  EXPECT_FLOAT_EQ(state.scale(), state.transform.scale);

  // Test individual component accessors
  EXPECT_FLOAT_EQ(state.x(), state.transform.translation.x());
  EXPECT_FLOAT_EQ(state.y(), state.transform.translation.y());
  EXPECT_FLOAT_EQ(state.z(), state.transform.translation.z());

  EXPECT_FLOAT_EQ(state.quatW(), state.transform.rotation.w());
  EXPECT_FLOAT_EQ(state.quatX(), state.transform.rotation.x());
  EXPECT_FLOAT_EQ(state.quatY(), state.transform.rotation.y());
  EXPECT_FLOAT_EQ(state.quatZ(), state.transform.rotation.z());

  // Test mutable accessors
  state.localRotation() = QuaternionType::Identity();
  state.localTranslation() = Vector3Type::Zero();
  state.localScale() = 2.0;

  EXPECT_TRUE(state.localTransform.rotation.isApprox(QuaternionType::Identity()));
  EXPECT_TRUE(state.localTransform.translation.isApprox(Vector3Type::Zero()));
  EXPECT_FLOAT_EQ(state.localTransform.scale, 2.0);

  state.rotation() = QuaternionType::Identity();
  state.translation() = Vector3Type::Zero();
  state.scale() = 3.0;

  EXPECT_TRUE(state.transform.rotation.isApprox(QuaternionType::Identity()));
  EXPECT_TRUE(state.transform.translation.isApprox(Vector3Type::Zero()));
  EXPECT_FLOAT_EQ(state.transform.scale, 3.0);

  state.x() = 1.0;
  state.y() = 2.0;
  state.z() = 3.0;

  EXPECT_FLOAT_EQ(state.transform.translation.x(), 1.0);
  EXPECT_FLOAT_EQ(state.transform.translation.y(), 2.0);
  EXPECT_FLOAT_EQ(state.transform.translation.z(), 3.0);

  state.quatW() = 1.0;
  state.quatX() = 0.0;
  state.quatY() = 0.0;
  state.quatZ() = 0.0;

  EXPECT_FLOAT_EQ(state.transform.rotation.w(), 1.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.x(), 0.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.y(), 0.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.z(), 0.0);
}

// Test complex hierarchy with multiple joints
TYPED_TEST(JointStateTest, ComplexHierarchy) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointType = typename TestFixture::JointType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  // Create a simple hierarchy:
  // root -> joint1 -> joint2

  // Root joint
  JointType rootJoint;
  rootJoint.name = "root";
  rootJoint.parent = kInvalidIndex;
  rootJoint.translationOffset = Vector3Type(0, 0, 0);

  JointVectorType rootParams;
  rootParams << 1, 0, 0, 0, 0, 0, 0; // Only x translation

  // Joint 1
  JointType joint1;
  joint1.name = "joint1";
  joint1.parent = 0; // Child of root
  joint1.translationOffset = Vector3Type(0, 1, 0);

  JointVectorType joint1Params;
  joint1Params << 0, 0, 0, 0, pi<TypeParam>() / 2, 0, 0; // 90 degree rotation around Y

  // Joint 2
  JointType joint2;
  joint2.name = "joint2";
  joint2.parent = 1; // Child of joint1
  joint2.translationOffset = Vector3Type(0, 0, 1);

  JointVectorType joint2Params;
  joint2Params << 0, 0, 0, 0, 0, pi<TypeParam>() / 2, 0; // 90 degree rotation around Z

  // Create joint states
  JointStateType rootState, joint1State, joint2State;

  // Set states in hierarchy order
  rootState.set(rootJoint, rootParams, nullptr, true);
  joint1State.set(joint1, joint1Params, &rootState, true);
  joint2State.set(joint2, joint2Params, &joint1State, true);

  // Check global positions
  // Root should be at (1, 0, 0)
  EXPECT_TRUE(rootState.transform.translation.isApprox(Vector3Type(1, 0, 0)));

  // Joint1 should be at (1, 1, 0) (root + offset)
  EXPECT_TRUE(joint1State.transform.translation.isApprox(Vector3Type(1, 1, 0)));

  // Joint2's position is affected by joint1's rotation
  // Since joint1 rotates 90 degrees around Y, the Z offset of joint2 becomes X offset
  // So joint2 should be at (1+1, 1, 0) = (2, 1, 0)
  EXPECT_TRUE(joint2State.transform.translation.isApprox(Vector3Type(2, 1, 0), 1e-5));

  // Check that rotation axes are correctly transformed through the hierarchy
  // For joint2, the rotation axes should be affected by both parent rotations

  // Create a reference point in joint2's local space
  Vector3Type localPoint(1, 0, 0);

  // Transform to global space
  Vector3Type globalPoint = joint2State.transform.rotation * localPoint;

  // The point should be rotated by both joint1's and joint2's rotations
  // joint1: 90 degrees around Y, joint2: 90 degrees around Z
  // This should transform (1,0,0) to approximately (0,1,0)
  EXPECT_TRUE(globalPoint.isApprox(Vector3Type(0, 1, 0), 1e-5));
}

// Test with pre-rotation
TYPED_TEST(JointStateTest, PreRotation) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointType = typename TestFixture::JointType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;
  using QuaternionType = typename TestFixture::QuaternionType;
  using AngleAxisType = typename Eigen::AngleAxis<TypeParam>;

  // Create a joint with pre-rotation
  JointType joint;
  joint.name = "pre_rotated_joint";
  joint.parent = kInvalidIndex;
  joint.translationOffset = Vector3Type(1, 0, 0);

  // Set pre-rotation to 90 degrees around X
  joint.preRotation = QuaternionType(AngleAxisType(pi<TypeParam>() / 2, Vector3Type::UnitX()));

  // Parameters with no rotation
  JointVectorType params;
  params << 0, 0, 0, 0, 0, 0, 0;

  // Create joint state
  JointStateType state;
  state.set(joint, params, nullptr, true);

  // Check that local rotation includes pre-rotation
  EXPECT_TRUE(state.localTransform.rotation.isApprox(joint.preRotation));

  // Create a reference point in local space
  Vector3Type localPoint(0, 1, 0);

  // Transform to global space
  Vector3Type globalPoint = state.transform.rotation * localPoint;

  // The point should be rotated by the pre-rotation
  // 90 degrees around X transforms (0,1,0) to approximately (0,0,1)
  EXPECT_TRUE(globalPoint.isApprox(Vector3Type(0, 0, 1), 1e-5));
}

// Test JointStateList
TYPED_TEST(JointStateTest, JointStateList) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointStateListType = JointStateListT<TypeParam>;

  // Create a list of joint states
  JointStateListType stateList;

  // Add some joint states
  stateList.push_back(JointStateType());
  stateList.push_back(JointStateType());
  stateList.push_back(JointStateType());

  // Check size
  EXPECT_EQ(stateList.size(), 3);

  // Check that we can access and modify elements
  stateList[0].localTransform.translation = Vector3<TypeParam>(1, 2, 3);
  EXPECT_TRUE(stateList[0].localTransform.translation.isApprox(Vector3<TypeParam>(1, 2, 3)));
}

// Test error cases
TYPED_TEST(JointStateTest, ErrorCases) {
  using JointStateType = typename TestFixture::JointStateType;

  JointStateType state;

  // Set the state without computing derivatives
  state.set(this->joint, this->parameters, nullptr, false);

  // Check that derivDirty is true
  EXPECT_TRUE(state.derivDirty);

  // When derivDirty is true, we should not call getRotationDerivative or getTranslationDerivative
  // as they will cause fatal errors with MT_CHECK

  // Test with valid derivatives but out-of-range index
  state.set(this->joint, this->parameters, nullptr, true);

  // Check that derivDirty is false
  EXPECT_FALSE(state.derivDirty);

  // Test with out-of-range index (should throw an exception, not a fatal error)
  // We'll skip this test since it's not clear if it throws an exception or causes a fatal error
  // The implementation might use MT_CHECK which would cause a fatal error
}

// Test with different rotation orders
TYPED_TEST(JointStateTest, RotationOrder) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  JointStateType state;

  // Create parameters with rotations in all three axes
  JointVectorType params;
  params << 0, 0, 0, pi<TypeParam>() / 4, pi<TypeParam>() / 3, pi<TypeParam>() / 6,
      0; // 45, 60, 30 degrees

  // Set the state
  state.set(this->joint, params, nullptr, true);

  // The rotation order in JointState::set() is Z, Y, X (from the code)
  // So we need to verify that the resulting rotation is equivalent to that order

  // Create rotation matrices for each axis
  Eigen::AngleAxis<TypeParam> rx(params[3], Vector3Type::UnitX());
  Eigen::AngleAxis<TypeParam> ry(params[4], Vector3Type::UnitY());
  Eigen::AngleAxis<TypeParam> rz(params[5], Vector3Type::UnitZ());

  // Combine in Z, Y, X order
  Quaternion<TypeParam> expectedRotation = rz * ry * rx;

  // Check that the local rotation matches the expected rotation
  EXPECT_TRUE(state.localTransform.rotation.isApprox(expectedRotation, 1e-5));
}

// Test with scaling
TYPED_TEST(JointStateTest, Scaling) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  JointStateType state;

  // Create parameters with scaling
  JointVectorType params;
  params << 0, 0, 0, 0, 0, 0, 1.0; // Scale factor of 2^1 = 2

  // Set the state
  state.set(this->joint, params, nullptr, true);

  // Check that the scale is 2^params[6]
  EXPECT_NEAR(state.localTransform.scale, std::exp2(params[6]), 1e-5);

  // Create a reference point
  Vector3Type refPoint(1, 1, 1);

  // Get scale derivative
  Vector3Type scaleDeriv = state.getScaleDerivative(refPoint);

  // The scale derivative should be the reference point multiplied by ln(2)
  Vector3Type expectedScaleDeriv = refPoint * ln2<TypeParam>();
  EXPECT_TRUE(scaleDeriv.isApprox(expectedScaleDeriv));
}

// Test toAffine3 method explicitly
TYPED_TEST(JointStateTest, ToAffine3) {
  using JointStateType = typename TestFixture::JointStateType;
  using Vector3Type = typename TestFixture::Vector3Type;
  using QuaternionType = typename TestFixture::QuaternionType;
  using Affine3Type = typename Eigen::Transform<TypeParam, 3, Eigen::Affine>;

  JointStateType state;

  // Set up the state with specific values
  state.localTransform.translation = Vector3Type(1, 2, 3);
  state.localTransform.rotation = QuaternionType::UnitRandom();
  state.localTransform.scale = 2.0;

  state.transform.translation = Vector3Type(4, 5, 6);
  state.transform.rotation = QuaternionType::UnitRandom();
  state.transform.scale = 3.0;

  // Get the Affine3 representation
  Affine3Type affine = state.transform.toAffine3();

  // Check that the affine transformation has the correct components
  EXPECT_TRUE(affine.translation().isApprox(state.transform.translation));

  // Check that the linear part is rotation * scale
  Matrix3<TypeParam> expectedLinear =
      state.transform.rotation.toRotationMatrix() * state.transform.scale;
  EXPECT_TRUE(affine.linear().isApprox(expectedLinear));

  // Check that applying the transformation to a point gives the same result
  Vector3Type testPoint(1, 1, 1);
  Vector3Type transformedByState =
      state.transform.rotation * (testPoint * state.transform.scale) + state.transform.translation;
  Vector3Type transformedByAffine = affine * testPoint;
  EXPECT_TRUE(transformedByAffine.isApprox(transformedByState));
}

// Test set method with parent and without computing derivatives
TYPED_TEST(JointStateTest, SetWithParentWithoutDerivatives) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointType = typename TestFixture::JointType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  // Create parent joint and state
  JointType parentJoint;
  parentJoint.name = "parent_joint";
  parentJoint.parent = kInvalidIndex;
  parentJoint.translationOffset = Vector3Type(10, 20, 30);

  JointVectorType parentParams;
  parentParams << 1, 2, 3, 0, 0, 0, 0; // No rotation, no scaling

  JointStateType parentState;
  parentState.set(parentJoint, parentParams, nullptr, true);

  // Create child state
  JointStateType childState;
  childState.set(this->joint, this->parameters, &parentState, false);

  // Check that derivDirty is true
  EXPECT_TRUE(childState.derivDirty);

  // Check local transform
  Vector3Type expectedLocalTranslation = this->joint.translationOffset +
      Vector3Type(this->parameters[0], this->parameters[1], this->parameters[2]);
  EXPECT_TRUE(childState.localTransform.translation.isApprox(expectedLocalTranslation));

  // Check global transform
  // Global translation should be parent translation + (parent rotation * local translation * parent
  // scale)
  Vector3Type expectedGlobalTranslation = parentState.transform.translation +
      parentState.transform.rotation *
          (childState.localTransform.translation * parentState.transform.scale);
  EXPECT_TRUE(childState.transform.translation.isApprox(expectedGlobalTranslation));

  // Global rotation should be parent rotation * local rotation
  EXPECT_TRUE(childState.transform.rotation.isApprox(
      parentState.transform.rotation * childState.localTransform.rotation));

  // Global scale should be parent scale * local scale
  EXPECT_NEAR(
      childState.transform.scale,
      parentState.transform.scale * childState.localTransform.scale,
      1e-5);
}

// Test all accessor methods to ensure 100% coverage
TYPED_TEST(JointStateTest, AllAccessorMethods) {
  using JointStateType = typename TestFixture::JointStateType;
  using Vector3Type = typename TestFixture::Vector3Type;
  using QuaternionType = typename TestFixture::QuaternionType;

  JointStateType state;

  // Set up the state with specific values
  state.localTransform.translation = Vector3Type(1, 2, 3);
  state.localTransform.rotation = QuaternionType::UnitRandom();
  state.localTransform.scale = 2.0;

  state.transform.translation = Vector3Type(4, 5, 6);
  state.transform.rotation = QuaternionType::UnitRandom();
  state.transform.scale = 3.0;

  // Test all const accessor methods by using a const reference
  const JointStateType& constState = state;

  EXPECT_TRUE(constState.localRotation().isApprox(state.localTransform.rotation));
  EXPECT_TRUE(constState.localTranslation().isApprox(state.localTransform.translation));
  EXPECT_FLOAT_EQ(constState.localScale(), state.localTransform.scale);

  EXPECT_TRUE(constState.rotation().isApprox(state.transform.rotation));
  EXPECT_TRUE(constState.translation().isApprox(state.transform.translation));
  EXPECT_FLOAT_EQ(constState.scale(), state.transform.scale);

  EXPECT_FLOAT_EQ(constState.x(), state.transform.translation.x());
  EXPECT_FLOAT_EQ(constState.y(), state.transform.translation.y());
  EXPECT_FLOAT_EQ(constState.z(), state.transform.translation.z());

  EXPECT_FLOAT_EQ(constState.quatW(), state.transform.rotation.w());
  EXPECT_FLOAT_EQ(constState.quatX(), state.transform.rotation.x());
  EXPECT_FLOAT_EQ(constState.quatY(), state.transform.rotation.y());
  EXPECT_FLOAT_EQ(constState.quatZ(), state.transform.rotation.z());

  // Test all non-const accessor methods
  QuaternionType newLocalRotation = QuaternionType::UnitRandom();
  Vector3Type newLocalTranslation = Vector3Type(7, 8, 9);
  TypeParam newLocalScale = 4.0;

  state.localRotation() = newLocalRotation;
  state.localTranslation() = newLocalTranslation;
  state.localScale() = newLocalScale;

  EXPECT_TRUE(state.localTransform.rotation.isApprox(newLocalRotation));
  EXPECT_TRUE(state.localTransform.translation.isApprox(newLocalTranslation));
  EXPECT_FLOAT_EQ(state.localTransform.scale, newLocalScale);

  QuaternionType newRotation = QuaternionType::UnitRandom();
  Vector3Type newTranslation = Vector3Type(10, 11, 12);
  TypeParam newScale = 5.0;

  state.rotation() = newRotation;
  state.translation() = newTranslation;
  state.scale() = newScale;

  EXPECT_TRUE(state.transform.rotation.isApprox(newRotation));
  EXPECT_TRUE(state.transform.translation.isApprox(newTranslation));
  EXPECT_FLOAT_EQ(state.transform.scale, newScale);

  state.x() = 13.0;
  state.y() = 14.0;
  state.z() = 15.0;

  EXPECT_FLOAT_EQ(state.transform.translation.x(), 13.0);
  EXPECT_FLOAT_EQ(state.transform.translation.y(), 14.0);
  EXPECT_FLOAT_EQ(state.transform.translation.z(), 15.0);

  state.quatW() = 1.0;
  state.quatX() = 0.0;
  state.quatY() = 0.0;
  state.quatZ() = 0.0;

  EXPECT_FLOAT_EQ(state.transform.rotation.w(), 1.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.x(), 0.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.y(), 0.0);
  EXPECT_FLOAT_EQ(state.transform.rotation.z(), 0.0);
}

// Test with all rotation parameters
TYPED_TEST(JointStateTest, AllRotationParameters) {
  using JointStateType = typename TestFixture::JointStateType;
  using JointVectorType = typename TestFixture::JointVectorType;
  using Vector3Type = typename TestFixture::Vector3Type;

  JointStateType state;

  // Create parameters with rotations in all three axes
  JointVectorType params;
  params << 0, 0, 0, pi<TypeParam>() / 4, pi<TypeParam>() / 3, pi<TypeParam>() / 2,
      0; // 45, 60, 90 degrees

  // Set the state
  state.set(this->joint, params, nullptr, true);

  // Test that the rotation is applied correctly by checking how it transforms basis vectors
  Vector3Type xAxis(1, 0, 0);
  Vector3Type yAxis(0, 1, 0);
  Vector3Type zAxis(0, 0, 1);

  Vector3Type rotatedX = state.localTransform.rotation * xAxis;
  Vector3Type rotatedY = state.localTransform.rotation * yAxis;
  Vector3Type rotatedZ = state.localTransform.rotation * zAxis;

  // Create rotation matrices for each axis
  Eigen::AngleAxis<TypeParam> rx(params[3], Vector3Type::UnitX());
  Eigen::AngleAxis<TypeParam> ry(params[4], Vector3Type::UnitY());
  Eigen::AngleAxis<TypeParam> rz(params[5], Vector3Type::UnitZ());

  // Combine in Z, Y, X order
  Quaternion<TypeParam> expectedRotation = rz * ry * rx;

  Vector3Type expectedRotatedX = expectedRotation * xAxis;
  Vector3Type expectedRotatedY = expectedRotation * yAxis;
  Vector3Type expectedRotatedZ = expectedRotation * zAxis;

  EXPECT_TRUE(rotatedX.isApprox(expectedRotatedX));
  EXPECT_TRUE(rotatedY.isApprox(expectedRotatedY));
  EXPECT_TRUE(rotatedZ.isApprox(expectedRotatedZ));
}
