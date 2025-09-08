/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/constants.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct Momentum_ForwardKinematicsTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_ForwardKinematicsTest, Types);

TEST(Momentum_ForwardKinematics, Definitions) {
  EXPECT_TRUE(kParametersPerJoint == 7);
}

TEST(Momentum_ForwardKinematics, IsAncestor) {
  const Character character = createTestCharacter(3);
  EXPECT_TRUE(character.skeleton.isAncestor(1, 0));
  EXPECT_TRUE(character.skeleton.isAncestor(1, 1));
  EXPECT_TRUE(character.skeleton.isAncestor(2, 0));
  EXPECT_TRUE(character.skeleton.isAncestor(2, 1));
  EXPECT_TRUE(character.skeleton.isAncestor(2, 2));

  EXPECT_FALSE(character.skeleton.isAncestor(0, 1));
  EXPECT_FALSE(character.skeleton.isAncestor(1, 2));
}

TYPED_TEST(Momentum_ForwardKinematicsTest, Kinematics) {
  using T = typename TestFixture::Type;

  // Test with increasing number of joints from 3 to 512
  for (int numJoints = 3; numJoints <= 512; numJoints++) {
    SCOPED_TRACE("Checking for numJoints = " + std::to_string(numJoints));

    // create skeleton and reference values
    const Character character = createTestCharacter(numJoints);
    const Skeleton& skeleton = character.skeleton;
    const ParameterTransform& transform = character.parameterTransform;
    const ParameterTransformT<T> castedCharactorParameterTransform = transform.cast<T>();
    VectorX<T> parameters = VectorX<T>::Zero(transform.numAllModelParameters());

    SkeletonStateT<T> state(castedCharactorParameterTransform.apply(parameters), skeleton);

    {
      SCOPED_TRACE("Checking Rest State");
      for (size_t i = 0; i < state.jointState.size(); i++) {
        auto& js = state.jointState[i];
        const Vector3<T> pos = skeleton.joints[i].translationOffset.cast<T>();
        EXPECT_TRUE(js.localRotation().coeffs() == Quaternion<T>::Identity().coeffs());
        EXPECT_TRUE(js.localTranslation().isApprox(pos, Eps<T>(1e-6f, 5e-7)));
        EXPECT_TRUE(js.localScale() == 1.0);

        EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
        EXPECT_TRUE(js.rotationAxis == Matrix3<T>::Identity());

        EXPECT_TRUE(js.rotation().coeffs() == Quaternion<T>::Identity().coeffs());
        // TODO: Add test for js.translation
        EXPECT_TRUE(js.scale() == 1.0);
      }
    }

    {
      SCOPED_TRACE("Checking Posed");
      parameters.template head<10>() << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(),
          pi<T>(), -pi<T>();
      state.set(castedCharactorParameterTransform.apply(parameters), skeleton);

      const Vector3<T> pos = state.jointState[2].transformation * Vector3<T>(1, 1, 1);
      EXPECT_LE(
          (pos - Vector3<T>(-1.14354682, 3.14354706, -0.0717732906)).norm(), Eps<T>(1e-6f, 5e-7));
    }
  }
}

Skeleton createComplexSkeleton() {
  // Create a skeleton with a more complicated tree (not just a single chain) for testing relative
  // transforms:
  //
  // (root)---(joint1)-+-(joint2)---(joint3)
  //                   |
  //                   +-(joint4)---(joint5)
  Skeleton result;
  Joint joint;
  joint.name = "root";
  joint.parent = kInvalidIndex;
  joint.preRotation = Eigen::Quaternionf::Identity();
  joint.translationOffset = Eigen::Vector3f::Zero();
  const size_t joint0_index = result.joints.size();
  result.joints.push_back(joint);

  joint.name = "joint1";
  joint.parent = joint0_index;
  joint.translationOffset = Eigen::Vector3f::UnitY();
  joint.preRotation = Eigen::Quaternionf(Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitX()));
  const size_t joint1_index = result.joints.size();
  result.joints.push_back(joint);

  joint.name = "joint2";
  joint.parent = joint1_index;
  joint.preRotation = Eigen::Quaternionf::Identity();
  const size_t joint2_index = result.joints.size();
  result.joints.push_back(joint);

  joint.name = "joint3";
  joint.parent = joint2_index;
  result.joints.push_back(joint);

  joint.name = "joint4";
  joint.parent = joint1_index;
  const size_t joint4_index = result.joints.size();
  result.joints.push_back(joint);

  joint.name = "joint5";
  joint.parent = joint4_index;
  result.joints.push_back(joint);

  return result;
}

TYPED_TEST(Momentum_ForwardKinematicsTest, RelativeTransform) {
  using T = typename TestFixture::Type;

  const Skeleton skeleton = createComplexSkeleton();
  VectorX<T> jointParameters = VectorX<T>::Zero(skeleton.joints.size() * kParametersPerJoint);
  SkeletonStateT<T> state(jointParameters, skeleton);

  for (size_t iJoint = 0; iJoint < skeleton.joints.size(); ++iJoint) {
    // A-to-world:
    EXPECT_TRUE(transformAtoB<T>(iJoint, kInvalidIndex, skeleton, state)
                    .toAffine3()
                    .isApprox(state.jointState[iJoint].transformation, Eps<T>(1e-8f, 1e-8)));

    // world-to-B:
    EXPECT_TRUE(
        transformAtoB<T>(kInvalidIndex, iJoint, skeleton, state)
            .toAffine3()
            .isApprox(state.jointState[iJoint].transformation.inverse(), Eps<T>(1e-8f, 1e-8)));
  }

  for (size_t jointA = 0; jointA < skeleton.joints.size(); ++jointA) {
    for (size_t jointB = 0; jointB < skeleton.joints.size(); ++jointB) {
      const Affine3<T> AtoB = state.jointState[jointB].transformation.inverse() *
          state.jointState[jointA].transformation;
      const Affine3<T> AtoB2 = transformAtoB<T>(jointA, jointB, skeleton, state).toAffine3();
      EXPECT_TRUE(AtoB.isApprox(AtoB2, Eps<T>(1e-7f, 1e-8)));
    }
  }
}
