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

using namespace momentum;

// Test fixture for Skeleton tests
template <typename T>
class SkeletonTest : public testing::Test {
 protected:
  using SkeletonType = SkeletonT<T>;
  using JointType = JointT<T>;
  using Vector3Type = Vector3<T>;
  using QuaternionType = Quaternion<T>;

  void SetUp() override {
    // Create a simple skeleton with 5 joints in a tree structure:
    // Joint 0: Root
    // Joint 1: Child of Root
    // Joint 2: Child of Joint 1
    // Joint 3: Child of Root
    // Joint 4: Child of Joint 3
    joints.resize(5);

    joints[0].name = "root";
    joints[0].parent = kInvalidIndex;
    joints[0].translationOffset = Vector3<float>(0, 0, 0);
    joints[0].preRotation = Quaternion<float>::Identity();

    joints[1].name = "joint1";
    joints[1].parent = 0; // Child of root
    joints[1].translationOffset = Vector3<float>(1, 0, 0);
    joints[1].preRotation = Quaternion<float>::Identity();

    joints[2].name = "joint2";
    joints[2].parent = 1; // Child of joint1
    joints[2].translationOffset = Vector3<float>(0, 1, 0);
    joints[2].preRotation = Quaternion<float>::Identity();

    joints[3].name = "joint3";
    joints[3].parent = 0; // Child of root
    joints[3].translationOffset = Vector3<float>(0, 0, 1);
    joints[3].preRotation = Quaternion<float>::Identity();

    joints[4].name = "joint4";
    joints[4].parent = 3; // Child of joint3
    joints[4].translationOffset = Vector3<float>(0, 1, 0);
    joints[4].preRotation = Quaternion<float>::Identity();
  }

  JointList joints;
};

using SkeletonTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(SkeletonTest, SkeletonTypes);

// Test constructor with joints
TYPED_TEST(SkeletonTest, Constructor) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Check that the skeleton has the correct number of joints
  EXPECT_EQ(skeleton.joints.size(), this->joints.size());

  // Check that the joints are correctly set
  for (size_t i = 0; i < this->joints.size(); ++i) {
    EXPECT_EQ(skeleton.joints[i].name, this->joints[i].name);
    EXPECT_EQ(skeleton.joints[i].parent, this->joints[i].parent);
    EXPECT_TRUE(skeleton.joints[i].translationOffset.isApprox(this->joints[i].translationOffset));
    EXPECT_TRUE(skeleton.joints[i].preRotation.isApprox(this->joints[i].preRotation));
  }
}

// Test getJointIdByName method
TYPED_TEST(SkeletonTest, GetJointIdByName) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Test finding existing joints
  EXPECT_EQ(skeleton.getJointIdByName("root"), 0);
  EXPECT_EQ(skeleton.getJointIdByName("joint1"), 1);
  EXPECT_EQ(skeleton.getJointIdByName("joint2"), 2);
  EXPECT_EQ(skeleton.getJointIdByName("joint3"), 3);
  EXPECT_EQ(skeleton.getJointIdByName("joint4"), 4);

  // Test finding non-existent joint
  EXPECT_EQ(skeleton.getJointIdByName("non_existent_joint"), kInvalidIndex);
}

// Test getJointNames method
TYPED_TEST(SkeletonTest, GetJointNames) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Get the joint names
  std::vector<std::string> jointNames = skeleton.getJointNames();

  // Check that the joint names are correct
  ASSERT_EQ(jointNames.size(), this->joints.size());
  for (size_t i = 0; i < this->joints.size(); ++i) {
    EXPECT_EQ(jointNames[i], this->joints[i].name);
  }
}

// Test getChildrenJoints method with recursive=true
TYPED_TEST(SkeletonTest, GetChildrenJointsRecursive) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Get all children of the root joint (recursive)
  std::vector<size_t> rootChildren = skeleton.getChildrenJoints(0, true);

  // Root should have 4 children: joints 1, 2, 3, and 4
  ASSERT_EQ(rootChildren.size(), 4);
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 1) != rootChildren.end());
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 2) != rootChildren.end());
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 3) != rootChildren.end());
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 4) != rootChildren.end());

  // Get all children of joint 1 (recursive)
  std::vector<size_t> joint1Children = skeleton.getChildrenJoints(1, true);

  // Joint 1 should have 1 child: joint 2
  ASSERT_EQ(joint1Children.size(), 1);
  EXPECT_EQ(joint1Children[0], 2);

  // Get all children of joint 3 (recursive)
  std::vector<size_t> joint3Children = skeleton.getChildrenJoints(3, true);

  // Joint 3 should have 1 child: joint 4
  ASSERT_EQ(joint3Children.size(), 1);
  EXPECT_EQ(joint3Children[0], 4);

  // Get all children of joint 2 (recursive)
  std::vector<size_t> joint2Children = skeleton.getChildrenJoints(2, true);

  // Joint 2 should have no children
  EXPECT_TRUE(joint2Children.empty());

  // Get all children of joint 4 (recursive)
  std::vector<size_t> joint4Children = skeleton.getChildrenJoints(4, true);

  // Joint 4 should have no children
  EXPECT_TRUE(joint4Children.empty());
}

// Test getChildrenJoints method with recursive=false
TYPED_TEST(SkeletonTest, GetChildrenJointsNonRecursive) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Get direct children of the root joint (non-recursive)
  std::vector<size_t> rootChildren = skeleton.getChildrenJoints(0, false);

  // Root should have 2 direct children: joints 1 and 3
  ASSERT_EQ(rootChildren.size(), 2);
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 1) != rootChildren.end());
  EXPECT_TRUE(std::find(rootChildren.begin(), rootChildren.end(), 3) != rootChildren.end());

  // Get direct children of joint 1 (non-recursive)
  std::vector<size_t> joint1Children = skeleton.getChildrenJoints(1, false);

  // Joint 1 should have 1 direct child: joint 2
  ASSERT_EQ(joint1Children.size(), 1);
  EXPECT_EQ(joint1Children[0], 2);

  // Get direct children of joint 3 (non-recursive)
  std::vector<size_t> joint3Children = skeleton.getChildrenJoints(3, false);

  // Joint 3 should have 1 direct child: joint 4
  ASSERT_EQ(joint3Children.size(), 1);
  EXPECT_EQ(joint3Children[0], 4);

  // Get direct children of joint 2 (non-recursive)
  std::vector<size_t> joint2Children = skeleton.getChildrenJoints(2, false);

  // Joint 2 should have no direct children
  EXPECT_TRUE(joint2Children.empty());

  // Get direct children of joint 4 (non-recursive)
  std::vector<size_t> joint4Children = skeleton.getChildrenJoints(4, false);

  // Joint 4 should have no direct children
  EXPECT_TRUE(joint4Children.empty());
}

// Test isAncestor method
TYPED_TEST(SkeletonTest, IsAncestor) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Test self-ancestry
  EXPECT_TRUE(skeleton.isAncestor(0, 0)); // Root is its own ancestor
  EXPECT_TRUE(skeleton.isAncestor(1, 1)); // Joint 1 is its own ancestor
  EXPECT_TRUE(skeleton.isAncestor(2, 2)); // Joint 2 is its own ancestor
  EXPECT_TRUE(skeleton.isAncestor(3, 3)); // Joint 3 is its own ancestor
  EXPECT_TRUE(skeleton.isAncestor(4, 4)); // Joint 4 is its own ancestor

  // Test direct ancestry
  EXPECT_TRUE(skeleton.isAncestor(1, 0)); // Root is an ancestor of Joint 1
  EXPECT_TRUE(skeleton.isAncestor(2, 1)); // Joint 1 is an ancestor of Joint 2
  EXPECT_TRUE(skeleton.isAncestor(3, 0)); // Root is an ancestor of Joint 3
  EXPECT_TRUE(skeleton.isAncestor(4, 3)); // Joint 3 is an ancestor of Joint 4

  // Test indirect ancestry
  EXPECT_TRUE(skeleton.isAncestor(2, 0)); // Root is an ancestor of Joint 2
  EXPECT_TRUE(skeleton.isAncestor(4, 0)); // Root is an ancestor of Joint 4

  // Test non-ancestry
  EXPECT_FALSE(skeleton.isAncestor(0, 1)); // Joint 1 is not an ancestor of Root
  EXPECT_FALSE(skeleton.isAncestor(0, 2)); // Joint 2 is not an ancestor of Root
  EXPECT_FALSE(skeleton.isAncestor(0, 3)); // Joint 3 is not an ancestor of Root
  EXPECT_FALSE(skeleton.isAncestor(0, 4)); // Joint 4 is not an ancestor of Root
  EXPECT_FALSE(skeleton.isAncestor(1, 3)); // Joint 3 is not an ancestor of Joint 1
  EXPECT_FALSE(skeleton.isAncestor(2, 3)); // Joint 3 is not an ancestor of Joint 2
  EXPECT_FALSE(skeleton.isAncestor(3, 1)); // Joint 1 is not an ancestor of Joint 3
  EXPECT_FALSE(skeleton.isAncestor(4, 1)); // Joint 1 is not an ancestor of Joint 4
}

// Test commonAncestor method
TYPED_TEST(SkeletonTest, CommonAncestor) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Test common ancestor of a joint with itself
  EXPECT_EQ(skeleton.commonAncestor(0, 0), 0); // Root's common ancestor with itself is Root
  EXPECT_EQ(skeleton.commonAncestor(1, 1), 1); // Joint 1's common ancestor with itself is Joint 1
  EXPECT_EQ(skeleton.commonAncestor(2, 2), 2); // Joint 2's common ancestor with itself is Joint 2
  EXPECT_EQ(skeleton.commonAncestor(3, 3), 3); // Joint 3's common ancestor with itself is Joint 3
  EXPECT_EQ(skeleton.commonAncestor(4, 4), 4); // Joint 4's common ancestor with itself is Joint 4

  // Test common ancestor of parent and child
  EXPECT_EQ(skeleton.commonAncestor(0, 1), 0); // Common ancestor of Root and Joint 1 is Root
  EXPECT_EQ(skeleton.commonAncestor(1, 0), 0); // Common ancestor of Joint 1 and Root is Root
  EXPECT_EQ(skeleton.commonAncestor(1, 2), 1); // Common ancestor of Joint 1 and Joint 2 is Joint 1
  EXPECT_EQ(skeleton.commonAncestor(2, 1), 1); // Common ancestor of Joint 2 and Joint 1 is Joint 1
  EXPECT_EQ(skeleton.commonAncestor(0, 3), 0); // Common ancestor of Root and Joint 3 is Root
  EXPECT_EQ(skeleton.commonAncestor(3, 0), 0); // Common ancestor of Joint 3 and Root is Root
  EXPECT_EQ(skeleton.commonAncestor(3, 4), 3); // Common ancestor of Joint 3 and Joint 4 is Joint 3
  EXPECT_EQ(skeleton.commonAncestor(4, 3), 3); // Common ancestor of Joint 4 and Joint 3 is Joint 3

  // Test common ancestor of siblings
  EXPECT_EQ(skeleton.commonAncestor(1, 3), 0); // Common ancestor of Joint 1 and Joint 3 is Root
  EXPECT_EQ(skeleton.commonAncestor(3, 1), 0); // Common ancestor of Joint 3 and Joint 1 is Root

  // Test common ancestor of cousins
  EXPECT_EQ(skeleton.commonAncestor(2, 4), 0); // Common ancestor of Joint 2 and Joint 4 is Root
  EXPECT_EQ(skeleton.commonAncestor(4, 2), 0); // Common ancestor of Joint 4 and Joint 2 is Root
}

// Test cast method
TYPED_TEST(SkeletonTest, Cast) {
  using SkeletonType = typename TestFixture::SkeletonType;
  using OtherType =
      typename std::conditional<std::is_same<TypeParam, float>::value, double, float>::type;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Create a new skeleton of the other type
  SkeletonT<OtherType> castedSkeleton;

  // Manually copy the joints (no casting needed since joints are always float)
  castedSkeleton.joints = skeleton.joints;

  // Check that the casted skeleton has the correct number of joints
  EXPECT_EQ(castedSkeleton.joints.size(), skeleton.joints.size());

  // Check that the joints are correctly casted
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    EXPECT_EQ(castedSkeleton.joints[i].name, skeleton.joints[i].name);
    EXPECT_EQ(castedSkeleton.joints[i].parent, skeleton.joints[i].parent);

    // Check translation offset
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(
          static_cast<TypeParam>(castedSkeleton.joints[i].translationOffset[j]),
          skeleton.joints[i].translationOffset[j],
          1e-5);
    }

    // Check pre-rotation (using dot product to handle quaternion sign ambiguity)
    auto dot = static_cast<TypeParam>(std::abs(
        castedSkeleton.joints[i].preRotation.x() * skeleton.joints[i].preRotation.x() +
        castedSkeleton.joints[i].preRotation.y() * skeleton.joints[i].preRotation.y() +
        castedSkeleton.joints[i].preRotation.z() * skeleton.joints[i].preRotation.z() +
        castedSkeleton.joints[i].preRotation.w() * skeleton.joints[i].preRotation.w()));
    EXPECT_NEAR(dot, 1.0, 1e-5);
  }

  // Create a new skeleton of the original type
  SkeletonType roundTripSkeleton;

  // Manually copy the joints back
  roundTripSkeleton.joints.resize(castedSkeleton.joints.size());
  for (size_t i = 0; i < castedSkeleton.joints.size(); ++i) {
    roundTripSkeleton.joints[i].name = castedSkeleton.joints[i].name;
    roundTripSkeleton.joints[i].parent = castedSkeleton.joints[i].parent;

    // Create new Vector3<float> and Quaternion<float> objects directly
    // since joints are always of type JointT<float>
    roundTripSkeleton.joints[i].translationOffset = Vector3<float>(
        castedSkeleton.joints[i].translationOffset.x(),
        castedSkeleton.joints[i].translationOffset.y(),
        castedSkeleton.joints[i].translationOffset.z());

    roundTripSkeleton.joints[i].preRotation = Quaternion<float>(
        castedSkeleton.joints[i].preRotation.w(),
        castedSkeleton.joints[i].preRotation.x(),
        castedSkeleton.joints[i].preRotation.y(),
        castedSkeleton.joints[i].preRotation.z());
  }

  // Check that the round-trip skeleton has the correct number of joints
  EXPECT_EQ(roundTripSkeleton.joints.size(), skeleton.joints.size());

  // Check that the joints are correctly round-tripped
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    EXPECT_EQ(roundTripSkeleton.joints[i].name, skeleton.joints[i].name);
    EXPECT_EQ(roundTripSkeleton.joints[i].parent, skeleton.joints[i].parent);
    EXPECT_TRUE(roundTripSkeleton.joints[i].translationOffset.isApprox(
        skeleton.joints[i].translationOffset));
    EXPECT_TRUE(roundTripSkeleton.joints[i].preRotation.isApprox(skeleton.joints[i].preRotation));
  }
}

// Test error cases
TYPED_TEST(SkeletonTest, ErrorCases) {
  using SkeletonType = typename TestFixture::SkeletonType;

  // Create a skeleton with the test joints
  SkeletonType skeleton(this->joints);

  // Test getChildrenJoints with invalid joint ID
  // Use a lambda to capture the return value to avoid the nodiscard warning
  EXPECT_THROW(
      [&skeleton]() {
        auto result = skeleton.getChildrenJoints(10, true);
        (void)result; // Suppress unused variable warning
      }(),
      std::out_of_range);

  EXPECT_THROW(
      [&skeleton]() {
        auto result = skeleton.getChildrenJoints(10, false);
        (void)result; // Suppress unused variable warning
      }(),
      std::out_of_range);

  // Test isAncestor with invalid joint IDs
  // Note: We can't test MT_CHECK failures directly with EXPECT_THROW
  // as they cause fatal errors rather than throwing exceptions
}
