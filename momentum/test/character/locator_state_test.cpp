/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/locator.h"
#include "momentum/character/locator_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/test/character/character_helpers.h"

namespace momentum {

class LocatorStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a test character with a skeleton
    character = createTestCharacter();
    skeletonState.jointState.resize(character.skeleton.joints.size());

    // Initialize joint states with identity transforms
    for (auto& jointState : skeletonState.jointState) {
      jointState.localRotation() = Quaternionf::Identity();
      jointState.localTranslation().setZero();
      jointState.localScale() = 1.0f;
      jointState.transform.rotation = Quaternionf::Identity();
      jointState.transform.translation.setZero();
      jointState.transform.scale = 1.0f;
      jointState.transformation = Eigen::Affine3f::Identity();
    }
  }

  Character character;
  SkeletonState skeletonState;
};

// Test LocatorState update with a single locator
TEST_F(LocatorStateTest, UpdateSingle) {
  // Create a locator attached to the root joint
  Locator locator("root_locator", 0, Vector3f(1.0f, 2.0f, 3.0f));
  LocatorList locators = {locator};

  // Create a locator state and update it
  LocatorState locatorState;
  locatorState.update(skeletonState, locators);

  // Check that the position was computed correctly
  EXPECT_EQ(locatorState.position.size(), 1);
  EXPECT_TRUE(locatorState.position[0].isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
}

// Test LocatorState update with multiple locators
TEST_F(LocatorStateTest, UpdateMultiple) {
  // Create locators attached to different joints
  Locator locator1("root_locator", 0, Vector3f(1.0f, 0.0f, 0.0f));
  Locator locator2("joint1_locator", 1, Vector3f(0.0f, 1.0f, 0.0f));
  Locator locator3("joint2_locator", 2, Vector3f(0.0f, 0.0f, 1.0f));
  LocatorList locators = {locator1, locator2, locator3};

  // Create a locator state and update it
  LocatorState locatorState;
  locatorState.update(skeletonState, locators);

  // Check that the positions were computed correctly
  EXPECT_EQ(locatorState.position.size(), 3);
  EXPECT_TRUE(locatorState.position[0].isApprox(Vector3f(1.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(locatorState.position[1].isApprox(Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(locatorState.position[2].isApprox(Vector3f(0.0f, 0.0f, 1.0f)));
}

// Test LocatorState update with transformed joints
TEST_F(LocatorStateTest, UpdateTransformed) {
  // Create a locator attached to the root joint
  Locator locator("root_locator", 0, Vector3f(1.0f, 0.0f, 0.0f));
  LocatorList locators = {locator};

  // Transform the root joint
  skeletonState.jointState[0].transform.translation = Vector3f(2.0f, 3.0f, 4.0f);
  skeletonState.jointState[0].transform.rotation =
      Quaternionf(0.7071f, 0.0f, 0.7071f, 0.0f); // 90 degrees around Y

  // Set the transformation matrix directly
  Eigen::Affine3f transformation = Eigen::Affine3f::Identity();
  transformation.translate(Vector3f(2.0f, 3.0f, 4.0f));
  transformation.rotate(Quaternionf(0.7071f, 0.0f, 0.7071f, 0.0f));
  skeletonState.jointState[0].transformation = transformation;

  // Create a locator state and update it
  LocatorState locatorState;
  locatorState.update(skeletonState, locators);

  // Check that the position was computed correctly
  // The locator offset (1,0,0) rotated 90 degrees around Y becomes (0,0,-1)
  // Then translated by (2,3,4)
  EXPECT_EQ(locatorState.position.size(), 1);
  EXPECT_TRUE(locatorState.position[0].isApprox(Vector3f(2.0f, 3.0f, 3.0f)));
}

// Test LocatorState constructor with skeleton state and locators
TEST_F(LocatorStateTest, Constructor) {
  // Create a locator attached to the root joint
  Locator locator("root_locator", 0, Vector3f(1.0f, 2.0f, 3.0f));
  LocatorList locators = {locator};

  // Create a locator state using the constructor
  LocatorState locatorState(skeletonState, locators);

  // Check that the position was computed correctly
  EXPECT_EQ(locatorState.position.size(), 1);
  EXPECT_TRUE(locatorState.position[0].isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
}

// Test LocatorState with empty locator list
TEST_F(LocatorStateTest, EmptyList) {
  // Create an empty locator list
  LocatorList locators;

  // Create a locator state and update it
  LocatorState locatorState;
  locatorState.update(skeletonState, locators);

  // Check that the position list is empty
  EXPECT_TRUE(locatorState.position.empty());
}

// Test LocatorState default constructor
TEST_F(LocatorStateTest, DefaultConstructor) {
  // Create a locator state with the default constructor
  LocatorState locatorState;

  // Check that the position list is empty
  EXPECT_TRUE(locatorState.position.empty());
}

} // namespace momentum
