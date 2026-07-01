/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/plane_collision_query.h>
#include <momentum/character_solver/support_contacts.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/character_solver/error_function_helpers.h>
#include <momentum/test/helpers/expect_throw.h>

#include <memory>
#include <span>
#include <stdexcept>

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

namespace {

CollisionPrimitive createCapsulePrimitive() {
  TaperedCapsule capsule;
  capsule.parent = 0;
  capsule.length = 1.0f;
  capsule.radius = Eigen::Vector2f(0.4f, 0.4f);
  capsule.transformation.translation = Eigen::Vector3f(0.0f, 0.2f, 0.0f);
  return capsule;
}

Character createSinglePrimitiveCharacter(const CollisionPrimitive& primitive) {
  Character character = createTestCharacter(3);
  character.collision = std::make_unique<CollisionGeometry>();
  character.collision->push_back(primitive);
  return character;
}

template <typename T>
bool vectorContainsApprox(
    const std::vector<Vector3<T>>& values,
    const Vector3<T>& expected,
    const T tolerance = Eps<T>(1e-5f, 1e-8)) {
  for (const Vector3<T>& value : values) {
    if ((value - expected).norm() <= tolerance) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool contactContainsApprox(
    const std::vector<SupportContactT<T>>& contacts,
    const size_t parentJoint,
    const Vector3<T>& expectedPosition,
    const Vector3<T>& expectedParentOffset,
    const T tolerance = Eps<T>(1e-5f, 1e-8)) {
  for (const SupportContactT<T>& contact : contacts) {
    if (contact.parentJoint == parentJoint &&
        (contact.position - expectedPosition).norm() <= tolerance &&
        (contact.parentOffset - expectedParentOffset).norm() <= tolerance) {
      return true;
    }
  }
  return false;
}

} // namespace

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_UsesSignedFloorLocatorHeight) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter(3);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  const std::vector<Locator> locators = {
      Locator("FloorOrigin", 0, Vector3f(1.0f, 0.0f, 2.0f)),
      Locator("FloorBelow", 0, Vector3f(3.0f, -1.5f, 4.0f)),
      Locator("FloorFarBelow", 0, Vector3f(5.0f, -3.0f, 6.0f)),
      Locator("FloorTooHigh", 0, Vector3f(7.0f, 3.0f, 8.0f)),
      Locator("HandFloor", 0, Vector3f(9.0f, 0.0f, 10.0f)),
  };

  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const std::span<const Locator> locatorSpan{locators.data(), locators.size()};
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitY(), T(0));
  const SupportContactListT<T> supportContactList =
      computeFloorLocatorSupportContacts(stateSpan, locatorSpan, T(2), supportPlane);

  ASSERT_EQ(supportContactList.floorLocatorPositions.size(), 4u);
  EXPECT_TRUE(vectorContainsApprox(supportContactList.floorLocatorPositions, Vector3<T>(1, 0, 2)));
  EXPECT_TRUE(
      vectorContainsApprox(supportContactList.floorLocatorPositions, Vector3<T>(3, -1.5, 4)));
  EXPECT_TRUE(vectorContainsApprox(supportContactList.floorLocatorPositions, Vector3<T>(5, -3, 6)));
  EXPECT_TRUE(vectorContainsApprox(supportContactList.floorLocatorPositions, Vector3<T>(7, 3, 8)));

  ASSERT_EQ(supportContactList.contacts.size(), 3u);
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(1, 0, 2), Vector3<T>(1, 0, 2)));
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(3, -1.5, 4), Vector3<T>(3, -1.5, 4)));
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(5, -3, 6), Vector3<T>(5, -3, 6)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_ParentOffsetInvertsParentScale) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter(3);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  state.jointState.at(0).transform.scale = T(2);
  const std::vector<Locator> locators = {
      Locator("FloorScaled", 0, Vector3f(1.0f, 0.0f, 2.0f)),
  };

  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const std::span<const Locator> locatorSpan{locators.data(), locators.size()};
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitY(), T(0));
  const SupportContactListT<T> supportContactList =
      computeFloorLocatorSupportContacts(stateSpan, locatorSpan, T(1), supportPlane);

  ASSERT_EQ(supportContactList.contacts.size(), 1u);
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(2, 0, 4), Vector3<T>(1, 0, 2)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_UsesCustomSupportPlane) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter(3);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  const std::vector<Locator> locators = {
      Locator("FloorOnPlane", 0, Vector3f(2.0f, 1.0f, 0.0f)),
      Locator("FloorBelowPlane", 0, Vector3f(0.0f, 2.0f, 0.0f)),
      Locator("FloorAboveMargin", 0, Vector3f(4.0f, 3.0f, 0.0f)),
  };
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitX(), T(2), Vector3<T>::UnitY());

  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const std::span<const Locator> locatorSpan{locators.data(), locators.size()};
  const SupportContactListT<T> supportContactList =
      computeFloorLocatorSupportContacts(stateSpan, locatorSpan, T(1), supportPlane);

  ASSERT_EQ(supportContactList.floorLocatorPositions.size(), 3u);
  ASSERT_EQ(supportContactList.contacts.size(), 2u);
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(2, 1, 0), Vector3<T>(2, 1, 0)));
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(0, 2, 0), Vector3<T>(0, 2, 0)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_CombinesFloorAndCollisionContacts) {
  using T = typename TestFixture::Type;

  CollisionPrimitive primitive = createCapsulePrimitive();
  primitive.transformation.translation = Vector3f(2.0f, 0.2f, 0.0f);
  Character character = createSinglePrimitiveCharacter(primitive);
  character.locators = {
      Locator("FloorOrigin", 0, Vector3f(0.0f, 0.0f, 3.0f)),
  };
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);

  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitY(), T(0));
  const SupportContactListT<T> supportContactList = computeSupportContacts(
      character, stateSpan, T(0.5), static_cast<PlaneCollisionQueryT<T>*>(nullptr), supportPlane);

  ASSERT_EQ(supportContactList.floorLocatorPositions.size(), 1u);
  ASSERT_EQ(supportContactList.contacts.size(), 2u);
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(0, 0, 3), Vector3<T>(0, 0, 3)));
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(2, -0.2, 0), Vector3<T>(2, -0.2, 0)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactPositions_CombinesFloorAndCollisionContacts) {
  using T = typename TestFixture::Type;

  CollisionPrimitive primitive = createCapsulePrimitive();
  primitive.transformation.translation = Vector3f(2.0f, 0.2f, 0.0f);
  Character character = createSinglePrimitiveCharacter(primitive);
  character.locators = {
      Locator("FloorOrigin", 0, Vector3f(0.0f, 0.0f, 3.0f)),
      Locator("FloorAboveMargin", 0, Vector3f(0.0f, 2.0f, 5.0f)),
  };
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);

  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitY(), T(0));
  const std::vector<Vector3<T>> positions = computeSupportContactPositions(
      character, stateSpan, T(0.5), static_cast<PlaneCollisionQueryT<T>*>(nullptr), supportPlane);

  ASSERT_EQ(positions.size(), 2u);
  EXPECT_TRUE(vectorContainsApprox(positions, Vector3<T>(0, 0, 3)));
  EXPECT_TRUE(vectorContainsApprox(positions, Vector3<T>(2, -0.2, 0)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_CombinesCustomPlaneContacts) {
  using T = typename TestFixture::Type;

  Character character = createSinglePrimitiveCharacter(createCapsulePrimitive());
  character.locators = {
      Locator("FloorBehindPlane", 0, Vector3f(-2.0f, 0.0f, 3.0f)),
      Locator("FloorAboveMargin", 0, Vector3f(2.0f, 0.0f, 3.0f)),
  };
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const SupportPlaneT<T> supportPlane(Vector3<T>::UnitX(), T(0), Vector3<T>::UnitY());
  PlaneCollisionQueryT<T> collisionQuery(character, supportPlane.normal, supportPlane.offset);

  const SupportContactListT<T> supportContactList =
      computeSupportContacts(character, stateSpan, T(0.5), &collisionQuery, supportPlane);

  ASSERT_EQ(supportContactList.floorLocatorPositions.size(), 2u);
  ASSERT_EQ(supportContactList.contacts.size(), 2u);
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(-2, 0, 3), Vector3<T>(-2, 0, 3)));
  EXPECT_TRUE(contactContainsApprox(
      supportContactList.contacts, 0, Vector3<T>(-0.4, 0.2, 0), Vector3<T>(-0.4, 0.2, 0)));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_RejectsMismatchedCollisionPlane) {
  using T = typename TestFixture::Type;

  Character character = createSinglePrimitiveCharacter(createCapsulePrimitive());
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  PlaneCollisionQueryT<T> collisionQuery(character, Vector3<T>::UnitX(), /*planeOffset=*/T(0));

  EXPECT_THROW_WITH_MESSAGE(
      [&]() {
        static_cast<void>(computeSupportContacts(
            character,
            stateSpan,
            T(0.5),
            &collisionQuery,
            SupportPlaneT<T>(Vector3<T>::UnitY(), T(0))));
      },
      std::runtime_error,
      testing::HasSubstr("same oriented support plane"));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneCollisionQuery_NormalizesPlaneOffset) {
  using T = typename TestFixture::Type;

  const Character character = createSinglePrimitiveCharacter(createCapsulePrimitive());
  PlaneCollisionQueryT<T> collisionQuery(
      *character.collision, T(2) * Vector3<T>::UnitY(), /*planeOffset=*/T(6));

  EXPECT_NEAR(collisionQuery.getPlaneNormal().y(), T(1), Eps<T>(1e-5f, 1e-12));
  EXPECT_NEAR(collisionQuery.getPlaneOffset(), T(3), Eps<T>(1e-5f, 1e-12));
}

TYPED_TEST(Momentum_ErrorFunctionsTest, SupportContactList_AcceptsLargeMatchingPlaneOffset) {
  using T = typename TestFixture::Type;

  Character character = createSinglePrimitiveCharacter(createCapsulePrimitive());
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const ModelParametersT<T> params = ModelParametersT<T>::Zero(transform.numAllModelParameters());
  const SkeletonStateT<T> state(transform.apply(params), character.skeleton);
  const std::span<const JointStateT<T>> stateSpan{state.jointState.data(), state.jointState.size()};
  const T offset = T(10000);
  const T offsetDelta = Eps<T>(1e-5f, 1e-12) * T(10);
  PlaneCollisionQueryT<T> collisionQuery(
      character, Vector3<T>::UnitY(), /*planeOffset=*/offset + offsetDelta);

  EXPECT_NO_THROW(
      static_cast<void>(computeSupportContacts(
          character,
          stateSpan,
          T(0.5),
          &collisionQuery,
          SupportPlaneT<T>(Vector3<T>::UnitY(), offset))));
}
