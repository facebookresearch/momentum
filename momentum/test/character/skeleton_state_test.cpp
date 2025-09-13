/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

// Test fixture for SkeletonState tests
template <typename T>
class SkeletonStateTest : public testing::Test {
 protected:
  using SkeletonStateType = SkeletonStateT<T>;
  using JointParametersType = JointParametersT<T>;
  using JointType = JointT<T>;
  using Vector3Type = Vector3<T>;
  using QuaternionType = Quaternion<T>;

  void SetUp() override {
    // Create a simple skeleton with 3 joints
    // Joint 0: Root
    // Joint 1: Child of Root
    // Joint 2: Child of Joint 1
    JointT<float> joint0;
    joint0.name = "root";
    joint0.parent = kInvalidIndex;
    joint0.translationOffset = Vector3<float>(0, 0, 0);
    joint0.preRotation = Quaternion<float>::Identity();

    JointT<float> joint1;
    joint1.name = "joint1";
    joint1.parent = 0; // Child of root
    joint1.translationOffset = Vector3<float>(1, 0, 0);
    joint1.preRotation = Quaternion<float>::Identity();

    JointT<float> joint2;
    joint2.name = "joint2";
    joint2.parent = 1; // Child of joint1
    joint2.translationOffset = Vector3<float>(0, 1, 0);
    joint2.preRotation = Quaternion<float>::Identity();

    skeleton.joints = {joint0, joint1, joint2};

    // Create joint parameters for the skeleton
    // Each joint has 7 parameters: 3 translation, 3 rotation, 1 scale
    // Parameters are arranged as [tx, ty, tz, rx, ry, rz, scale]
    jointParameters.v.resize(skeleton.joints.size() * kParametersPerJoint);

    // Root joint parameters
    jointParameters.v(0) = 0; // tx
    jointParameters.v(1) = 0; // ty
    jointParameters.v(2) = 0; // tz
    jointParameters.v(3) = 0; // rx
    jointParameters.v(4) = 0; // ry
    jointParameters.v(5) = 0; // rz
    jointParameters.v(6) = 0; // scale (2^0 = 1)

    // Joint 1 parameters
    jointParameters.v(7) = 0; // tx
    jointParameters.v(8) = 0; // ty
    jointParameters.v(9) = 0; // tz
    jointParameters.v(10) = 0; // rx
    jointParameters.v(11) = pi<T>() / 4; // ry (45 degrees)
    jointParameters.v(12) = 0; // rz
    jointParameters.v(13) = 0; // scale (2^0 = 1)

    // Joint 2 parameters
    jointParameters.v(14) = 0; // tx
    jointParameters.v(15) = 0; // ty
    jointParameters.v(16) = 0; // tz
    jointParameters.v(17) = 0; // rx
    jointParameters.v(18) = 0; // ry
    jointParameters.v(19) = pi<T>() / 4; // rz (45 degrees)
    jointParameters.v(20) = 0; // scale (2^0 = 1)
  }

  Skeleton skeleton;
  JointParametersType jointParameters;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(SkeletonStateTest, Types);

// Test default constructor
TYPED_TEST(SkeletonStateTest, DefaultConstructor) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;

  SkeletonStateType state;

  // Check that the state is empty
  EXPECT_EQ(state.jointParameters.size(), 0);
  EXPECT_EQ(state.jointState.size(), 0);
}

// Test constructor with parameters and reference skeleton
TYPED_TEST(SkeletonStateTest, ConstructorWithParameters) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;

  SkeletonStateType state(this->jointParameters, this->skeleton, true);

  // Check that the state has the correct size
  EXPECT_EQ(state.jointParameters.size(), this->jointParameters.size());
  EXPECT_EQ(state.jointState.size(), this->skeleton.joints.size());

  // Check that the joint parameters are correctly set
  for (Eigen::Index i = 0; i < state.jointParameters.size(); ++i) {
    EXPECT_FLOAT_EQ(state.jointParameters.v(i), this->jointParameters.v(i));
  }

  // Check that the joint states are correctly computed
  // Root joint should be at the origin
  EXPECT_TRUE(state.jointState[0].transform.translation.isApprox(Vector3<TypeParam>(0, 0, 0)));

  // Joint 1 should be at (1, 0, 0) relative to root
  EXPECT_TRUE(state.jointState[1].transform.translation.isApprox(Vector3<TypeParam>(1, 0, 0)));

  // Joint 2 should be rotated by 45 degrees around Y-axis from Joint 1
  // and then translated by (0, 1, 0) in the local frame
  // The rotation around Y by 45 degrees transforms (0, 1, 0) to approximately (0, 1, 0)
  // since Y-axis rotation doesn't affect Y coordinate
  // So Joint 2 should be at approximately (1, 1, 0)
  EXPECT_TRUE(
      state.jointState[2].transform.translation.isApprox(Vector3<TypeParam>(1, 1, 0), 1e-5));
}

// Test constructor with rvalue parameters and reference skeleton
TYPED_TEST(SkeletonStateTest, ConstructorWithRValueParameters) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using JointParametersType = typename TestFixture::JointParametersType;

  // Create a copy of the joint parameters to move from
  JointParametersType params = this->jointParameters;

  SkeletonStateType state(std::move(params), this->skeleton, true);

  // Check that the state has the correct size
  EXPECT_EQ(state.jointParameters.size(), this->jointParameters.size());
  EXPECT_EQ(state.jointState.size(), this->skeleton.joints.size());

  // Check that the joint parameters are correctly set
  for (Eigen::Index i = 0; i < state.jointParameters.size(); ++i) {
    EXPECT_FLOAT_EQ(state.jointParameters.v(i), this->jointParameters.v(i));
  }

  // Check that the joint states are correctly computed
  // Root joint should be at the origin
  EXPECT_TRUE(state.jointState[0].transform.translation.isApprox(Vector3<TypeParam>(0, 0, 0)));

  // Joint 1 should be at (1, 0, 0) relative to root
  EXPECT_TRUE(state.jointState[1].transform.translation.isApprox(Vector3<TypeParam>(1, 0, 0)));

  // Joint 2 should be at approximately (1, 1, 0)
  EXPECT_TRUE(
      state.jointState[2].transform.translation.isApprox(Vector3<TypeParam>(1, 1, 0), 1e-5));
}

// Test copy constructor with type conversion
TYPED_TEST(SkeletonStateTest, CopyConstructorWithTypeConversion) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using OtherType =
      typename std::conditional<std::is_same<TypeParam, float>::value, double, float>::type;

  // Create a state with the original type
  SkeletonStateType originalState(this->jointParameters, this->skeleton, true);

  // Create a state with the other type using the copy constructor
  SkeletonStateT<OtherType> convertedState(originalState);

  // Check that the converted state has the correct size
  EXPECT_EQ(convertedState.jointParameters.size(), originalState.jointParameters.size());
  EXPECT_EQ(convertedState.jointState.size(), originalState.jointState.size());

  // Check that the joint parameters are correctly converted
  for (Eigen::Index i = 0; i < convertedState.jointParameters.size(); ++i) {
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointParameters.v(i)),
        originalState.jointParameters.v(i),
        1e-5);
  }

  // Check that the joint states are correctly converted
  for (size_t i = 0; i < convertedState.jointState.size(); ++i) {
    // Check translation
    Vector3<TypeParam> originalTranslation = originalState.jointState[i].transform.translation;
    Vector3<OtherType> convertedTranslation = convertedState.jointState[i].transform.translation;
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(static_cast<TypeParam>(convertedTranslation[j]), originalTranslation[j], 1e-5);
    }

    // Check rotation (using dot product to handle quaternion sign ambiguity)
    Quaternion<TypeParam> originalRotation = originalState.jointState[i].transform.rotation;
    Quaternion<OtherType> convertedRotation = convertedState.jointState[i].transform.rotation;
    auto dot = static_cast<TypeParam>(std::abs(
        originalRotation.x() * convertedRotation.x() +
        originalRotation.y() * convertedRotation.y() +
        originalRotation.z() * convertedRotation.z() +
        originalRotation.w() * convertedRotation.w()));
    EXPECT_NEAR(dot, 1.0, 1e-5);

    // Check scale
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointState[i].transform.scale),
        originalState.jointState[i].transform.scale,
        1e-5);
  }
}

// Test set method with parameters and reference skeleton
TYPED_TEST(SkeletonStateTest, SetWithParameters) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;

  SkeletonStateType state;
  state.set(this->jointParameters, this->skeleton, true);

  // Check that the state has the correct size
  EXPECT_EQ(state.jointParameters.size(), this->jointParameters.size());
  EXPECT_EQ(state.jointState.size(), this->skeleton.joints.size());

  // Check that the joint parameters are correctly set
  for (Eigen::Index i = 0; i < state.jointParameters.size(); ++i) {
    EXPECT_FLOAT_EQ(state.jointParameters.v(i), this->jointParameters.v(i));
  }

  // Check that the joint states are correctly computed
  // Root joint should be at the origin
  EXPECT_TRUE(state.jointState[0].transform.translation.isApprox(Vector3<TypeParam>(0, 0, 0)));

  // Joint 1 should be at (1, 0, 0) relative to root
  EXPECT_TRUE(state.jointState[1].transform.translation.isApprox(Vector3<TypeParam>(1, 0, 0)));

  // Joint 2 should be at approximately (1, 1, 0)
  EXPECT_TRUE(
      state.jointState[2].transform.translation.isApprox(Vector3<TypeParam>(1, 1, 0), 1e-5));
}

// Test set method with rvalue parameters and reference skeleton
TYPED_TEST(SkeletonStateTest, SetWithRValueParameters) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using JointParametersType = typename TestFixture::JointParametersType;

  SkeletonStateType state;

  // Create a copy of the joint parameters to move from
  JointParametersType params = this->jointParameters;

  state.set(std::move(params), this->skeleton, true);

  // Check that the state has the correct size
  EXPECT_EQ(state.jointParameters.size(), this->jointParameters.size());
  EXPECT_EQ(state.jointState.size(), this->skeleton.joints.size());

  // Check that the joint parameters are correctly set
  for (Eigen::Index i = 0; i < state.jointParameters.size(); ++i) {
    EXPECT_FLOAT_EQ(state.jointParameters.v(i), this->jointParameters.v(i));
  }

  // Check that the joint states are correctly computed
  // Root joint should be at the origin
  EXPECT_TRUE(state.jointState[0].transform.translation.isApprox(Vector3<TypeParam>(0, 0, 0)));

  // Joint 1 should be at (1, 0, 0) relative to root
  EXPECT_TRUE(state.jointState[1].transform.translation.isApprox(Vector3<TypeParam>(1, 0, 0)));

  // Joint 2 should be at approximately (1, 1, 0)
  EXPECT_TRUE(
      state.jointState[2].transform.translation.isApprox(Vector3<TypeParam>(1, 1, 0), 1e-5));
}

// Test set method with type conversion
TYPED_TEST(SkeletonStateTest, SetWithTypeConversion) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using OtherType =
      typename std::conditional<std::is_same<TypeParam, float>::value, double, float>::type;

  // Create a state with the original type
  SkeletonStateType originalState(this->jointParameters, this->skeleton, true);

  // Create a state with the other type
  SkeletonStateT<OtherType> convertedState;

  // Set the converted state from the original state
  convertedState.set(originalState);

  // Check that the converted state has the correct size
  EXPECT_EQ(convertedState.jointParameters.size(), originalState.jointParameters.size());
  EXPECT_EQ(convertedState.jointState.size(), originalState.jointState.size());

  // Check that the joint parameters are correctly converted
  for (Eigen::Index i = 0; i < convertedState.jointParameters.size(); ++i) {
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointParameters.v(i)),
        originalState.jointParameters.v(i),
        1e-5);
  }

  // Check that the joint states are correctly converted
  for (size_t i = 0; i < convertedState.jointState.size(); ++i) {
    // Check translation
    Vector3<TypeParam> originalTranslation = originalState.jointState[i].transform.translation;
    Vector3<OtherType> convertedTranslation = convertedState.jointState[i].transform.translation;
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(static_cast<TypeParam>(convertedTranslation[j]), originalTranslation[j], 1e-5);
    }

    // Check rotation (using dot product to handle quaternion sign ambiguity)
    Quaternion<TypeParam> originalRotation = originalState.jointState[i].transform.rotation;
    Quaternion<OtherType> convertedRotation = convertedState.jointState[i].transform.rotation;
    auto dot = static_cast<TypeParam>(std::abs(
        originalRotation.x() * convertedRotation.x() +
        originalRotation.y() * convertedRotation.y() +
        originalRotation.z() * convertedRotation.z() +
        originalRotation.w() * convertedRotation.w()));
    EXPECT_NEAR(dot, 1.0, 1e-5);

    // Check scale
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointState[i].transform.scale),
        originalState.jointState[i].transform.scale,
        1e-5);
  }
}

// Test compare method
TYPED_TEST(SkeletonStateTest, Compare) {
  using T = TypeParam;
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using JointParametersType = typename TestFixture::JointParametersType;

  // Create two identical states
  SkeletonStateType state1(this->jointParameters, this->skeleton, true);
  SkeletonStateType state2(this->jointParameters, this->skeleton, true);

  // Compare the states
  StateSimilarity similarity = SkeletonStateType::compare(state1, state2);

  // Check that the similarity measures are correct for identical states
  // Position error should be zero for all joints
  for (Eigen::Index i = 0; i < similarity.positionError.size(); ++i) {
    EXPECT_NEAR(similarity.positionError[i], 0.0f, 1e-5);
  }

  // Orientation error should be zero for all joints
  for (Eigen::Index i = 0; i < similarity.orientationError.size(); ++i) {
    EXPECT_NEAR(similarity.orientationError[i], 0.0f, 1e-5);
  }

  // RMSE and max values should be zero
  EXPECT_NEAR(similarity.positionRMSE, 0.0f, 1e-5);
  EXPECT_NEAR(similarity.orientationRMSE, 0.0f, 1e-5);
  EXPECT_NEAR(similarity.positionMax, 0.0f, 1e-5);
  EXPECT_NEAR(similarity.orientationMax, 0.0f, 1e-5);

  // Now create a state with different parameters
  JointParametersType differentParams = this->jointParameters;
  // Modify the translation of the root joint
  differentParams.v(0) = 1.0; // tx = 1.0 (was 0.0)

  SkeletonStateType state3(differentParams, this->skeleton, true);

  // Compare the states
  StateSimilarity similarity2 = SkeletonStateType::compare(state1, state3);

  // Check that the position error for the root joint is 1.0
  EXPECT_NEAR(similarity2.positionError[0], 1.0f, 1e-5);

  // Check that the RMSE and max values are correct
  EXPECT_GT(similarity2.positionRMSE, 0.0f);
  EXPECT_NEAR(similarity2.positionMax, 1.0f, 1e-5);

  // Now create a state with different orientation
  JointParametersType differentOrientParams = this->jointParameters;
  // Modify the rotation of the root joint
  differentOrientParams.v(5) = pi<T>() / 2; // rz = 90 degrees (was 0.0)

  SkeletonStateType state4(differentOrientParams, this->skeleton, true);

  // Compare the states
  StateSimilarity similarity3 = SkeletonStateType::compare(state1, state4);

  // Check that the orientation error for the root joint is approximately PI/2
  EXPECT_NEAR(similarity3.orientationError[0], static_cast<float>(pi<T>() / 2), 1e-5);

  // Check that the RMSE and max values are correct
  EXPECT_GT(similarity3.orientationRMSE, 0.0f);
  EXPECT_NEAR(similarity3.orientationMax, static_cast<float>(pi<T>() / 2), 1e-5);
}

// Test toTransforms method
TYPED_TEST(SkeletonStateTest, ToTransforms) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;

  // Create a state
  SkeletonStateType state(this->jointParameters, this->skeleton, true);

  // Convert to transforms
  TransformListT<TypeParam> transforms = state.toTransforms();

  // Check that the transforms have the correct size
  EXPECT_EQ(transforms.size(), state.jointState.size());

  // Check that the transforms match the joint states
  for (size_t i = 0; i < transforms.size(); ++i) {
    EXPECT_TRUE(transforms[i].translation.isApprox(state.jointState[i].transform.translation));
    EXPECT_TRUE(transforms[i].rotation.isApprox(state.jointState[i].transform.rotation));
    EXPECT_FLOAT_EQ(transforms[i].scale, state.jointState[i].transform.scale);
  }
}

// Test cast method
TYPED_TEST(SkeletonStateTest, Cast) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using OtherType =
      typename std::conditional<std::is_same<TypeParam, float>::value, double, float>::type;

  // Create a state
  SkeletonStateType originalState(this->jointParameters, this->skeleton, true);

  // Cast to the other type
  SkeletonStateT<OtherType> convertedState = originalState.template cast<OtherType>();

  // Check that the converted state has the correct size
  EXPECT_EQ(convertedState.jointParameters.size(), originalState.jointParameters.size());
  EXPECT_EQ(convertedState.jointState.size(), originalState.jointState.size());

  // Check that the joint parameters are correctly converted
  for (Eigen::Index i = 0; i < convertedState.jointParameters.size(); ++i) {
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointParameters.v(i)),
        originalState.jointParameters.v(i),
        1e-5);
  }

  // Check that the joint states are correctly converted
  for (size_t i = 0; i < convertedState.jointState.size(); ++i) {
    // Check translation
    Vector3<TypeParam> originalTranslation = originalState.jointState[i].transform.translation;
    Vector3<OtherType> convertedTranslation = convertedState.jointState[i].transform.translation;
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(static_cast<TypeParam>(convertedTranslation[j]), originalTranslation[j], 1e-5);
    }

    // Check rotation (using dot product to handle quaternion sign ambiguity)
    Quaternion<TypeParam> originalRotation = originalState.jointState[i].transform.rotation;
    Quaternion<OtherType> convertedRotation = convertedState.jointState[i].transform.rotation;
    auto dot = static_cast<TypeParam>(std::abs(
        originalRotation.x() * convertedRotation.x() +
        originalRotation.y() * convertedRotation.y() +
        originalRotation.z() * convertedRotation.z() +
        originalRotation.w() * convertedRotation.w()));
    EXPECT_NEAR(dot, 1.0, 1e-5);

    // Check scale
    EXPECT_NEAR(
        static_cast<TypeParam>(convertedState.jointState[i].transform.scale),
        originalState.jointState[i].transform.scale,
        1e-5);
  }
}

// Test transformAtoB function
TYPED_TEST(SkeletonStateTest, TransformAtoB) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;
  using TransformType = TransformT<TypeParam>;
  using Vector3Type = Vector3<TypeParam>;

  // Create a state
  SkeletonStateType state(this->jointParameters, this->skeleton, true);

  // Test transform from joint 1 to joint 1 (identity)
  TransformType joint1ToJoint1 = transformAtoB(1, 1, this->skeleton, state);

  // The transform should be identity
  EXPECT_TRUE(joint1ToJoint1.translation.isApprox(Vector3Type(0, 0, 0)));
  EXPECT_TRUE(joint1ToJoint1.rotation.isApprox(Quaternion<TypeParam>::Identity()));
  EXPECT_FLOAT_EQ(joint1ToJoint1.scale, 1.0);

  // Test transform from joint 1 to joint 0 and back
  TransformType joint1ToJoint0 = transformAtoB(1, 0, this->skeleton, state);
  TransformType joint0ToJoint1 = transformAtoB(0, 1, this->skeleton, state);

  // Apply both transforms in sequence to a test point
  Vector3Type testPoint(1, 2, 3);
  Vector3Type transformedPoint = joint1ToJoint0 * (joint0ToJoint1 * testPoint);

  // The result should be the original point (within floating point precision)
  EXPECT_TRUE(transformedPoint.isApprox(testPoint, 1e-5));

  // Test that transformAtoB(A, C) is equivalent to transformAtoB(B, C) * transformAtoB(A, B)
  TransformType joint0ToJoint2 = transformAtoB(0, 2, this->skeleton, state);
  TransformType joint0ToJoint1ToJoint2 =
      transformAtoB(1, 2, this->skeleton, state) * transformAtoB(0, 1, this->skeleton, state);

  // Apply both transforms to a test point
  Vector3Type point1 = joint0ToJoint2 * testPoint;
  Vector3Type point2 = joint0ToJoint1ToJoint2 * testPoint;

  // The results should be the same
  EXPECT_TRUE(point1.isApprox(point2, 1e-5));
}

TYPED_TEST(SkeletonStateTest, SkeletonStateToJointParameters) {
  using SkeletonStateType = typename TestFixture::SkeletonStateType;

  const auto testCharacter = createTestCharacter();

  // Create a state
  Random<> rng(42);
  const momentum::JointParametersT<TypeParam> jointParameters =
      rng.uniform<Eigen::VectorX<TypeParam>>(
          kParametersPerJoint * testCharacter.skeleton.joints.size(), -1.0f, 1.0f);
  const SkeletonStateType state(jointParameters, this->skeleton, false);
  const auto jointParameters2 = skeletonStateToJointParameters(state, this->skeleton);
  const SkeletonStateType state2(jointParameters2, this->skeleton, false);

  ASSERT_EQ(state.jointState.size(), state2.jointState.size());

  for (size_t i = 0; i < state.jointState.size(); ++i) {
    EXPECT_LE(
        (state.jointState[i].transform.toAffine3().matrix() -
         state2.jointState[i].transform.toAffine3().matrix())
            .norm(),
        1e-5);
  }

  EXPECT_LT((jointParameters.v - jointParameters2.v).norm(), 1e-5);

  std::vector<TransformT<TypeParam>> transforms;
  for (const auto& s : state2.jointState) {
    transforms.push_back(s.transform);
  }
  const auto jointParameters3 = skeletonStateToJointParameters(transforms, this->skeleton);
  EXPECT_LT((jointParameters.v - jointParameters3.v).norm(), 1e-5);
}

// Note: We're not testing error cases because they use MT_CHECK which causes fatal errors
// rather than throwing exceptions that can be caught with EXPECT_THROW
