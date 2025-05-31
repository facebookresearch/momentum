/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/character/parameter_limits.h>
#include <momentum/io/skeleton/parameter_limits_io.h>
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

namespace {

Character createCharacterWithLimits() {
  auto testCharacter = createTestCharacter();

  ParameterLimits limits;

  {
    ParameterLimit limit;
    limit.type = MinMax;
    limit.weight = 1.5f;
    limit.data.minMax.limits = Vector2f(-0.2, 0.1);
    limit.data.minMax.parameterIndex = 1;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::MinMaxJoint;
    limit.weight = 2.0f;
    limit.data.minMaxJoint.limits = Vector2f(-0.3, 0.0);
    limit.data.minMaxJoint.jointIndex = 1;
    limit.data.minMaxJoint.jointParameter = 2;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::MinMaxJointPassive;
    limit.weight = 2.0f;
    limit.data.minMaxJoint.limits = Vector2f(-0.0, 0.5);
    limit.data.minMaxJoint.jointIndex = 2;
    limit.data.minMaxJoint.jointParameter = 4;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::Ellipsoid;
    limit.weight = 4.0;
    limit.data.ellipsoid.ellipsoid = limit.data.ellipsoid.ellipsoid = Affine3f::Identity();
    limit.data.ellipsoid.ellipsoid.translation() = Eigen::Vector3f(2, 3, 4);
    const Vector3f eulerXYZ = Vector3f(pi() / 2.0f, pi() / 4.0f, -pi() / 3.0f);
    limit.data.ellipsoid.ellipsoid.linear() =
        eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
        Eigen::Scaling(0.8f, 0.9f, 1.3f);
    limit.data.ellipsoid.ellipsoidInv = limit.data.ellipsoid.ellipsoid.inverse();
    limit.data.ellipsoid.offset = Eigen::Vector3f(1, 2, 3);
    limit.data.ellipsoid.ellipsoidParent = 2;
    limit.data.ellipsoid.parent = 1;
    limits.push_back(limit);
  }

  {
    // ellipsoid constraints are the hardest to get right so let's include two of them:
    ParameterLimit limit;
    limit.type = LimitType::Ellipsoid;
    limit.weight = 5.0;
    limit.data.ellipsoid.ellipsoid = limit.data.ellipsoid.ellipsoid = Affine3f::Identity();
    limit.data.ellipsoid.ellipsoid.translation() = Eigen::Vector3f(2, 3, 4);
    const Vector3f eulerXYZ = Vector3f(-pi() / 4.0f, 2.0f * pi() / 4.0f, pi() / 3.0f);
    limit.data.ellipsoid.ellipsoid.linear() =
        eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
        Eigen::Scaling(2.0f, 0.4f, 0.6f);
    limit.data.ellipsoid.ellipsoidInv = limit.data.ellipsoid.ellipsoid.inverse();
    limit.data.ellipsoid.offset = Eigen::Vector3f(2, 1, 4);
    limit.data.ellipsoid.ellipsoidParent = 0;
    limit.data.ellipsoid.parent = 2;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::Linear;
    limit.weight = 2.0;
    limit.data.linear.scale = 3.0;
    limit.data.linear.offset = 2.0;
    limit.data.linear.referenceIndex = 2;
    limit.data.linear.targetIndex = 1;
    limits.push_back(limit);
  }

  // Piecewise linear limit:
  // We'll use this example from Wikipedia:
  //  f(x) = { -x - 3        if x < -3
  //           x + 3         if -3 <= x < 0
  //           -2x + 3       if 0 <= x < 3
  //           0.5x - 4.5    if x >= 3
  {
    ParameterLimit limit;
    limit.type = LimitType::Linear;
    limit.weight = 2.5;
    limit.data.linear.referenceIndex = 2;
    limit.data.linear.targetIndex = 1;

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = -1.0f;
      cur.data.linear.offset = 3.0f;
      cur.data.linear.rangeMin = -std::numeric_limits<float>::max();
      cur.data.linear.rangeMax = -3.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = 1.0f;
      cur.data.linear.offset = -3.0f;
      cur.data.linear.rangeMin = -3.0f;
      cur.data.linear.rangeMax = 0.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = -2.0f;
      cur.data.linear.offset = -3.0f;
      cur.data.linear.rangeMin = 0.0f;
      cur.data.linear.rangeMax = 3.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linear.scale = 0.5f;
      cur.data.linear.offset = 4.5f;
      cur.data.linear.rangeMin = 3.0f;
      cur.data.linear.rangeMax = std::numeric_limits<float>::max();
      limits.push_back(cur);
    }
  }

  {
    // Add one more limit affecting the same parameters:
    ParameterLimit limit;
    limit.type = LimitType::Linear;
    limit.weight = 2.5;
    limit.data.linear.referenceIndex = 2;
    limit.data.linear.targetIndex = 1;
    limit.data.linear.scale = 1.2f;
    limit.data.linear.offset = 0.3f;
    limits.push_back(limit);
  }

  {
    // Linear joint limit affecting joint parameter:
    ParameterLimit limit;
    limit.type = LimitType::LinearJoint;
    limit.weight = 2.5;
    limit.data.linearJoint.referenceJointIndex = 2;
    limit.data.linearJoint.referenceJointParameter = 3;
    limit.data.linearJoint.targetJointIndex = 1;
    limit.data.linearJoint.targetJointParameter = 0;

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = 1.0f;
      cur.data.linearJoint.offset = 0.0f;
      cur.data.linearJoint.rangeMin = -std::numeric_limits<float>::max();
      cur.data.linearJoint.rangeMax = 0;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = -1.0f;
      cur.data.linearJoint.offset = 0.0f;
      cur.data.linearJoint.rangeMin = 0.0f;
      cur.data.linearJoint.rangeMax = 2.0f;
      limits.push_back(cur);
    }

    {
      ParameterLimit cur = limit;
      cur.data.linearJoint.scale = 1.0f;
      cur.data.linearJoint.offset = 4.0f;
      cur.data.linearJoint.rangeMin = 2.0f;
      cur.data.linearJoint.rangeMax = std::numeric_limits<float>::max();
      limits.push_back(cur);
    }
  }

  {
    // Linear joint limit affecting joint parameter:
    ParameterLimit limit;
    limit.type = LimitType::LinearJoint;
    limit.weight = 2.5;
    limit.data.linearJoint.referenceJointIndex = 2;
    limit.data.linearJoint.referenceJointParameter = 3;
    limit.data.linearJoint.targetJointIndex = 1;
    limit.data.linearJoint.targetJointParameter = 0;
    limit.data.linearJoint.scale = 1.2f;
    limit.data.linearJoint.offset = 3.0f;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::HalfPlane;
    limit.weight = 3.5;
    limit.data.halfPlane.param1 = 1;
    limit.data.halfPlane.param2 = 3;
    limit.data.halfPlane.normal = Eigen::Vector2f(1, -1).normalized();
    limit.data.halfPlane.offset = 0.7f;
    limits.push_back(limit);
  }

  return {testCharacter.skeleton, testCharacter.parameterTransform, limits};
}

void validateParameterLimitsSame(const ParameterLimits& limits1, const ParameterLimits& limits2) {
  ASSERT_EQ(limits1.size(), limits2.size());

  for (size_t i = 0; i < limits1.size(); ++i) {
    const auto& l1 = limits1[i];
    const auto& l2 = limits2[i];

    EXPECT_NEAR(l1.weight, l2.weight, 1e-4f);
    EXPECT_EQ(l1.type, l2.type);

    switch (l1.type) {
      case LimitType::MinMax:
        EXPECT_NEAR(l1.data.minMax.limits.x(), l2.data.minMax.limits.x(), 1e-4f);
        EXPECT_NEAR(l1.data.minMax.limits.y(), l2.data.minMax.limits.y(), 1e-4f);
        EXPECT_EQ(l1.data.minMax.parameterIndex, l2.data.minMax.parameterIndex);
        break;
      case LimitType::MinMaxJoint:
      case LimitType::MinMaxJointPassive:
        EXPECT_NEAR(l1.data.minMaxJoint.limits.x(), l2.data.minMaxJoint.limits.x(), 1e-4f);
        EXPECT_NEAR(l1.data.minMaxJoint.limits.y(), l2.data.minMaxJoint.limits.y(), 1e-4f);
        EXPECT_EQ(l1.data.minMaxJoint.jointIndex, l2.data.minMaxJoint.jointIndex);
        EXPECT_EQ(l1.data.minMaxJoint.jointParameter, l2.data.minMaxJoint.jointParameter);
        break;
      case LimitType::Linear:
        EXPECT_NEAR(l1.data.linear.offset, l2.data.linear.offset, 1e-4f);
        EXPECT_NEAR(l1.data.linear.scale, l2.data.linear.scale, 1e-4f);
        EXPECT_EQ(l1.data.linear.targetIndex, l2.data.linear.targetIndex);
        EXPECT_EQ(l1.data.linear.referenceIndex, l2.data.linear.referenceIndex);

        if (l1.data.linear.rangeMin == 0.0f && l1.data.linear.rangeMax == 0.0f) {
          EXPECT_EQ(l2.data.linear.rangeMin, -std::numeric_limits<float>::max());
          EXPECT_EQ(l2.data.linear.rangeMax, std::numeric_limits<float>::max());
        } else {
          EXPECT_NEAR(l1.data.linear.rangeMin, l2.data.linear.rangeMin, 1e-4f);
          EXPECT_NEAR(l1.data.linear.rangeMax, l2.data.linear.rangeMax, 1e-4f);
        }
        break;
      case LimitType::LinearJoint:
        EXPECT_NEAR(l1.data.linearJoint.offset, l2.data.linearJoint.offset, 1e-4f);
        EXPECT_NEAR(l1.data.linearJoint.scale, l2.data.linearJoint.scale, 1e-4f);
        EXPECT_EQ(l1.data.linearJoint.targetJointIndex, l2.data.linearJoint.targetJointIndex);
        EXPECT_EQ(l1.data.linearJoint.referenceJointIndex, l2.data.linearJoint.referenceJointIndex);
        EXPECT_EQ(
            l1.data.linearJoint.targetJointParameter, l2.data.linearJoint.targetJointParameter);
        EXPECT_EQ(
            l1.data.linearJoint.referenceJointParameter,
            l2.data.linearJoint.referenceJointParameter);

        if (l1.data.linearJoint.rangeMin == 0.0f && l1.data.linearJoint.rangeMax == 0.0f) {
          EXPECT_EQ(l2.data.linearJoint.rangeMin, -std::numeric_limits<float>::max());
          EXPECT_EQ(l2.data.linearJoint.rangeMax, std::numeric_limits<float>::max());
        } else {
          EXPECT_NEAR(l1.data.linearJoint.rangeMin, l2.data.linearJoint.rangeMin, 1e-4f);
          EXPECT_NEAR(l1.data.linearJoint.rangeMax, l2.data.linearJoint.rangeMax, 1e-4f);
        }
        break;

      case LimitType::Ellipsoid:
        EXPECT_LE(
            (l1.data.ellipsoid.ellipsoid.matrix() - l2.data.ellipsoid.ellipsoid.matrix())
                .lpNorm<Eigen::Infinity>(),
            1e-4f);
        EXPECT_LE(
            (l1.data.ellipsoid.ellipsoidInv.matrix() - l2.data.ellipsoid.ellipsoidInv.matrix())
                .lpNorm<Eigen::Infinity>(),
            1e-4f);
        EXPECT_EQ(l1.data.ellipsoid.ellipsoidParent, l2.data.ellipsoid.ellipsoidParent);
        EXPECT_EQ(l1.data.ellipsoid.parent, l2.data.ellipsoid.parent);
        EXPECT_LE((l1.data.ellipsoid.offset - l2.data.ellipsoid.offset).norm(), 1e-4f);
        break;

      case LimitType::HalfPlane:
        EXPECT_LE((l1.data.halfPlane.normal - l2.data.halfPlane.normal).norm(), 1e-4f);
        EXPECT_NEAR(l1.data.halfPlane.offset, l2.data.halfPlane.offset, 1e-4f);
        EXPECT_EQ(l1.data.halfPlane.param1, l2.data.halfPlane.param1);
        EXPECT_EQ(l1.data.halfPlane.param2, l2.data.halfPlane.param2);
        break;
      case LimitType::LimitTypeCount:
        // do nothing
        break;
    }
  }
}

} // namespace

TEST(IoCharacterTest, ParameterLimits_RoundTrip) {
  const Character character = createCharacterWithLimits();

  const std::string limitsStr = writeParameterLimits(
      character.parameterLimits, character.skeleton, character.parameterTransform);
  std::cout << limitsStr << "\n";
  const auto limits2 =
      parseParameterLimits(limitsStr, character.skeleton, character.parameterTransform);
  validateParameterLimitsSame(character.parameterLimits, limits2);
}
