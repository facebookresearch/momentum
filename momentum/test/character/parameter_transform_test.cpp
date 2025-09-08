/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/inverse_parameter_transform.h"
#include "momentum/character/joint.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/constants.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct Momentum_ParameterTransformTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_ParameterTransformTest, Types);

TEST(Momentum_ParameterTransform, Definitions) {
  EXPECT_TRUE(kParametersPerJoint == 7);
}

TYPED_TEST(Momentum_ParameterTransformTest, Empty) {
  using T = typename TestFixture::Type;

  const size_t nJoints = 3;
  const ParameterTransformT<T> xf = ParameterTransformT<T>::empty(nJoints * kParametersPerJoint);
  const JointParametersT<T> params = xf.apply(Eigen::VectorX<T>::Zero(0));
  EXPECT_EQ(nJoints * kParametersPerJoint, params.size());
  EXPECT_TRUE(params.v.isZero());
}

TYPED_TEST(Momentum_ParameterTransformTest, Identity) {
  using T = typename TestFixture::Type;

  std::vector<std::string> jointNames = {"root", "joint1", "joint2"};
  const ParameterTransformT<T> xf = ParameterTransformT<T>::identity(jointNames);

  // Check dimensions
  EXPECT_EQ(xf.numJointParameters(), jointNames.size() * kParametersPerJoint);
  EXPECT_EQ(xf.numAllModelParameters(), jointNames.size() * kParametersPerJoint);

  // Check parameter names
  EXPECT_EQ(xf.name.size(), jointNames.size() * kParametersPerJoint);
  EXPECT_EQ(xf.name[0], "root_tx");
  EXPECT_EQ(xf.name[kParametersPerJoint], "joint1_tx");
  EXPECT_EQ(xf.name[2 * kParametersPerJoint], "joint2_tx");

  // Check transform is identity
  ModelParametersT<T> params = ModelParametersT<T>::Zero(xf.numAllModelParameters());
  params.v.setOnes();
  JointParametersT<T> jointParams = xf.apply(params);
  EXPECT_TRUE(jointParams.v.isApprox(params.v));

  // Check activeJointParams
  EXPECT_EQ(xf.activeJointParams.size(), jointNames.size() * kParametersPerJoint);
  for (int i = 0; i < xf.activeJointParams.size(); ++i) {
    EXPECT_TRUE(xf.activeJointParams[i]);
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, Functionality) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());

  JointParametersT<T> jointParameters = transform.apply(parameters);

  EXPECT_EQ(parameters.size(), transform.numAllModelParameters());
  EXPECT_EQ(jointParameters.size(), transform.numJointParameters());

  {
    SCOPED_TRACE("Checking Zero Parameters");
    EXPECT_EQ(jointParameters.v.norm(), 0);
  }

  {
    SCOPED_TRACE("Checking Other Parameters");
    parameters.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();
    jointParameters = transform.apply(parameters);
    EXPECT_EQ(jointParameters.size(), 21);

    VectorX<T> expectedJointParameters(21);
    expectedJointParameters << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, 0.0, 0.0, 0.0, pi<T>(),
        0.0, pi<T>() * 0.5, 0.0, 0.0, 0.0, 0.0, -pi<T>(), 0.0, pi<T>() * 0.5, 0.0;
    EXPECT_LE((jointParameters.v - expectedJointParameters).norm(), Eps<T>(1e-7f, 1e-15));
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, ApplyWithCharacterParameters) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Test with zero offsets
  {
    CharacterParametersT<T> charParams;
    charParams.pose = ModelParametersT<T>::Zero(transform.numAllModelParameters());
    charParams.pose.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();

    JointParametersT<T> jointParams = transform.apply(charParams);

    // Compare with apply(ModelParameters)
    JointParametersT<T> expectedParams = transform.apply(charParams.pose);
    EXPECT_TRUE(jointParams.v.isApprox(expectedParams.v));
  }

  // Test with non-zero offsets
  {
    CharacterParametersT<T> charParams;
    charParams.pose = ModelParametersT<T>::Zero(transform.numAllModelParameters());
    charParams.pose.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();
    charParams.offsets = JointParametersT<T>::Zero(transform.numJointParameters());
    charParams.offsets.v.setConstant(0.5);

    JointParametersT<T> jointParams = transform.apply(charParams);

    // Compare with apply(ModelParameters) + offsets
    JointParametersT<T> expectedParams = transform.apply(charParams.pose);
    expectedParams.v += charParams.offsets.v;
    EXPECT_TRUE(jointParams.v.isApprox(expectedParams.v));
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, ZeroAndBindPose) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Test zero() returns offsets
  {
    JointParametersT<T> zeroParams = transform.zero();
    EXPECT_TRUE(zeroParams.v.isApprox(transform.offsets));
  }

  // Test bindPose() returns zero vector
  {
    JointParametersT<T> bindPoseParams = transform.bindPose();
    EXPECT_EQ(bindPoseParams.size(), transform.numJointParameters());
    EXPECT_TRUE(bindPoseParams.v.isZero());
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, GetParameterIdByName) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Test finding existing parameters
  EXPECT_EQ(transform.getParameterIdByName("root_tx"), 0);
  EXPECT_EQ(transform.getParameterIdByName("root_ty"), 1);
  EXPECT_EQ(transform.getParameterIdByName("root_tz"), 2);

  // Test finding non-existent parameter
  EXPECT_EQ(transform.getParameterIdByName("non_existent_param"), kInvalidIndex);
}

TYPED_TEST(Momentum_ParameterTransformTest, ComputeActiveJointParams) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Test with all parameters
  {
    VectorX<bool> activeParams = transform.computeActiveJointParams();
    EXPECT_EQ(activeParams.size(), transform.numJointParameters());

    // Check that some parameters are active (exact values depend on test character)
    bool hasActiveParams = false;
    for (int i = 0; i < activeParams.size(); ++i) {
      if (activeParams[i]) {
        hasActiveParams = true;
        break;
      }
    }
    EXPECT_TRUE(hasActiveParams);
  }

  // Test with subset of parameters
  {
    ParameterSet paramSet;
    paramSet.set(0); // root_tx
    paramSet.set(3); // root_rx

    VectorX<bool> activeParams = transform.computeActiveJointParams(paramSet);
    EXPECT_EQ(activeParams.size(), transform.numJointParameters());

    // Only parameters affected by root_tx and root_rx should be active
    EXPECT_TRUE(activeParams[0]); // root_tx
    EXPECT_TRUE(activeParams[3]); // root_rx
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, GetParameterSets) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {
      "root_tx",
      "root_ty",
      "root_tz",
      "root_rx",
      "root_ry",
      "root_rz",
      "scale_global",
      "joint1_rx",
      "joint1_ry",
      "joint1_rz",
      "scale_joint1"};

  // Add parameter sets
  ParameterSet set1;
  set1.set(0); // root_tx
  set1.set(1); // root_ty
  transform.parameterSets["translation"] = set1;

  ParameterSet set2;
  set2.set(3); // root_rx
  set2.set(4); // root_ry
  set2.set(5); // root_rz
  transform.parameterSets["rotation"] = set2;

  // Test getParameterSet with existing set
  {
    ParameterSet result = transform.getParameterSet("translation");
    EXPECT_TRUE(result.test(0));
    EXPECT_TRUE(result.test(1));
    EXPECT_FALSE(result.test(2));
  }

  // Test getParameterSet with non-existent set (should throw)
  { EXPECT_THROW((void)transform.getParameterSet("non_existent"), std::runtime_error); }

  // Test getParameterSet with non-existent set and allowMissing=true
  {
    ParameterSet result = transform.getParameterSet("non_existent", true);
    EXPECT_TRUE(result.none());
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, GetScalingParameters) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {
      "root_tx",
      "root_ty",
      "root_tz",
      "root_rx",
      "root_ry",
      "root_rz",
      "scale_global",
      "joint1_rx",
      "joint1_ry",
      "joint1_rz",
      "scale_joint1"};

  ParameterSet scalingParams = transform.getScalingParameters();

  EXPECT_FALSE(scalingParams.test(0)); // root_tx
  EXPECT_FALSE(scalingParams.test(3)); // root_rx
  EXPECT_TRUE(scalingParams.test(6)); // scale_global
  EXPECT_TRUE(scalingParams.test(10)); // scale_joint1
}

TYPED_TEST(Momentum_ParameterTransformTest, GetRigidParameters) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {
      "root_tx",
      "root_ty",
      "root_tz",
      "root_rx",
      "root_ry",
      "root_rz",
      "scale_global",
      "joint1_rx",
      "joint1_ry",
      "joint1_rz",
      "hips_tx"};

  ParameterSet rigidParams = transform.getRigidParameters();

  EXPECT_TRUE(rigidParams.test(0)); // root_tx
  EXPECT_TRUE(rigidParams.test(3)); // root_rx
  EXPECT_FALSE(rigidParams.test(6)); // scale_global
  EXPECT_FALSE(rigidParams.test(7)); // joint1_rx
  EXPECT_TRUE(rigidParams.test(10)); // hips_tx
}

TYPED_TEST(Momentum_ParameterTransformTest, GetPoseParameters) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {
      "root_tx",
      "root_ty",
      "root_tz",
      "root_rx",
      "root_ry",
      "root_rz",
      "scale_global",
      "joint1_rx",
      "joint1_ry",
      "joint1_rz",
      "blend_0"};

  // Set up blend shape parameters
  transform.blendShapeParameters.resize(1);
  transform.blendShapeParameters[0] = 10; // blend_0

  ParameterSet poseParams = transform.getPoseParameters();

  EXPECT_TRUE(poseParams.test(0)); // root_tx
  EXPECT_TRUE(poseParams.test(3)); // root_rx
  EXPECT_FALSE(poseParams.test(6)); // scale_global
  EXPECT_TRUE(poseParams.test(7)); // joint1_rx
  EXPECT_FALSE(poseParams.test(10)); // blend_0
}

TYPED_TEST(Momentum_ParameterTransformTest, GetBlendShapeParameters) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "blend_0", "blend_1"};

  // Set up blend shape parameters
  transform.blendShapeParameters.resize(2);
  transform.blendShapeParameters[0] = 3; // blend_0
  transform.blendShapeParameters[1] = 4; // blend_1

  ParameterSet blendParams = transform.getBlendShapeParameters();

  EXPECT_FALSE(blendParams.test(0)); // root_tx
  EXPECT_TRUE(blendParams.test(3)); // blend_0
  EXPECT_TRUE(blendParams.test(4)); // blend_1

  // Test with invalid index
  transform.blendShapeParameters[0] = -1;
  blendParams = transform.getBlendShapeParameters();
  EXPECT_FALSE(blendParams.test(3)); // blend_0 (now invalid)
  EXPECT_TRUE(blendParams.test(4)); // blend_1
}

TYPED_TEST(Momentum_ParameterTransformTest, GetFaceExpressionParameters) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "face_expre_0", "face_expre_1"};

  // Set up face expression parameters
  transform.faceExpressionParameters.resize(2);
  transform.faceExpressionParameters[0] = 3; // face_expre_0
  transform.faceExpressionParameters[1] = 4; // face_expre_1

  ParameterSet faceParams = transform.getFaceExpressionParameters();

  EXPECT_FALSE(faceParams.test(0)); // root_tx
  EXPECT_TRUE(faceParams.test(3)); // face_expre_0
  EXPECT_TRUE(faceParams.test(4)); // face_expre_1

  // Test with invalid index
  transform.faceExpressionParameters[0] = -1;
  faceParams = transform.getFaceExpressionParameters();
  EXPECT_FALSE(faceParams.test(3)); // face_expre_0 (now invalid)
  EXPECT_TRUE(faceParams.test(4)); // face_expre_1
}

TYPED_TEST(Momentum_ParameterTransformTest, Cast) {
  using T = typename TestFixture::Type;
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;

  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // Cast to other type
  ParameterTransformT<OtherT> transformOther = transform.template cast<OtherT>();

  // Check that the cast worked correctly
  EXPECT_EQ(transform.name, transformOther.name);
  EXPECT_EQ(transform.transform.rows(), transformOther.transform.rows());
  EXPECT_EQ(transform.transform.cols(), transformOther.transform.cols());
  EXPECT_EQ(transform.offsets.size(), transformOther.offsets.size());
  EXPECT_EQ(transform.activeJointParams.size(), transformOther.activeJointParams.size());

  // Cast back to original type
  ParameterTransformT<T> transformBack = transformOther.template cast<T>();
  EXPECT_EQ(transform.name, transformBack.name);
}

// Test isApprox method
TEST(Momentum_ParameterTransform, IsApprox) {
  // Create a simple transform
  ParameterTransform transform1 = ParameterTransform::empty(21);
  transform1.name = {"root_tx", "root_ty", "root_tz"};
  transform1.transform.resize(21, 3);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f);
  triplets.emplace_back(1, 1, 1.0f);
  triplets.emplace_back(2, 2, 1.0f);
  transform1.transform.setFromTriplets(triplets.begin(), triplets.end());
  transform1.activeJointParams.setConstant(21, false);
  transform1.activeJointParams[0] = true;
  transform1.activeJointParams[1] = true;
  transform1.activeJointParams[2] = true;

  // Create an identical transform
  ParameterTransform transform2 = transform1;
  EXPECT_TRUE(transform1.isApprox(transform2));

  // Test with different name
  ParameterTransform transform3 = transform1;
  transform3.name[0] = "different_name";
  EXPECT_FALSE(transform1.isApprox(transform3));

  // Test with different transform matrix
  ParameterTransform transform4 = transform1;
  std::vector<Eigen::Triplet<float>> triplets2;
  triplets2.emplace_back(0, 0, 2.0f); // Different value
  triplets2.emplace_back(1, 1, 1.0f);
  triplets2.emplace_back(2, 2, 1.0f);
  transform4.transform.setFromTriplets(triplets2.begin(), triplets2.end());
  EXPECT_FALSE(transform1.isApprox(transform4));

  // Test with zero matrices
  ParameterTransform transform5 = ParameterTransform::empty(0);
  ParameterTransform transform6 = ParameterTransform::empty(0);
  EXPECT_TRUE(transform5.isApprox(transform6));
}

// Test PoseConstraint::operator== function
TEST(Momentum_ParameterTransform, PoseConstraintEquality) {
  // Create two identical pose constraints
  PoseConstraint pc1;
  pc1.parameterIdValue.emplace_back(0, 1.0f);
  pc1.parameterIdValue.emplace_back(1, 2.0f);

  PoseConstraint pc2;
  pc2.parameterIdValue.emplace_back(0, 1.0f);
  pc2.parameterIdValue.emplace_back(1, 2.0f);

  // Test equality
  EXPECT_TRUE(pc1 == pc2);

  // Test with different order (should still be equal since it's treated as a set)
  PoseConstraint pc3;
  pc3.parameterIdValue.emplace_back(1, 2.0f);
  pc3.parameterIdValue.emplace_back(0, 1.0f);
  EXPECT_TRUE(pc1 == pc3);

  // Test with different parameter index
  PoseConstraint pc4;
  pc4.parameterIdValue.emplace_back(0, 1.0f);
  pc4.parameterIdValue.emplace_back(2, 2.0f); // Different index
  EXPECT_FALSE(pc1 == pc4);

  // Test with different parameter value
  PoseConstraint pc5;
  pc5.parameterIdValue.emplace_back(0, 1.0f);
  pc5.parameterIdValue.emplace_back(1, 3.0f); // Different value
  EXPECT_FALSE(pc1 == pc5);

  // Test with approximately equal values (should be equal due to isApprox)
  PoseConstraint pc6;
  pc6.parameterIdValue.emplace_back(0, 1.0f);
  pc6.parameterIdValue.emplace_back(1, 2.0f + 1e-7f); // Very close to 2.0f
  EXPECT_TRUE(pc1 == pc6);

  // Test with different number of parameters
  PoseConstraint pc7;
  pc7.parameterIdValue.emplace_back(0, 1.0f);
  EXPECT_FALSE(pc1 == pc7);
}

// Test numBlendShapeParameters, numFaceExpressionParameters, and numSkeletonParameters methods
TEST(Momentum_ParameterTransform, ParameterCounts) {
  // Create a parameter transform
  ParameterTransform transform = ParameterTransform::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "blend_0", "blend_1", "face_expre_0"};
  transform.transform.resize(21, 6);

  // Initially no blend shape or face expression parameters
  EXPECT_EQ(transform.numBlendShapeParameters(), 0);
  EXPECT_EQ(transform.numFaceExpressionParameters(), 0);
  EXPECT_EQ(transform.numSkeletonParameters(), 6); // All parameters are skeleton parameters

  // Add blend shape parameters
  transform.blendShapeParameters.resize(2);
  transform.blendShapeParameters[0] = 3; // blend_0
  transform.blendShapeParameters[1] = 4; // blend_1

  EXPECT_EQ(transform.numBlendShapeParameters(), 2);
  EXPECT_EQ(transform.numFaceExpressionParameters(), 0);
  EXPECT_EQ(transform.numSkeletonParameters(), 4); // 6 total - 2 blend shape

  // Add face expression parameters
  transform.faceExpressionParameters.resize(1);
  transform.faceExpressionParameters[0] = 5; // face_expre_0

  EXPECT_EQ(transform.numBlendShapeParameters(), 2);
  EXPECT_EQ(transform.numFaceExpressionParameters(), 1);
  EXPECT_EQ(transform.numSkeletonParameters(), 3); // 6 total - 2 blend shape - 1 face expression
}

TYPED_TEST(Momentum_ParameterTransformTest, Simplify) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"};
  transform.transform.resize(21, 6);
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.push_back(Eigen::Triplet<T>(0, 0, 1.0));
  triplets.push_back(Eigen::Triplet<T>(1, 1, 1.0));
  triplets.push_back(Eigen::Triplet<T>(2, 2, 1.0));
  triplets.push_back(Eigen::Triplet<T>(3, 3, 1.0));
  triplets.push_back(Eigen::Triplet<T>(4, 4, 1.0));
  triplets.push_back(Eigen::Triplet<T>(5, 5, 1.0));
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Create a parameter set with only translation parameters
  ParameterSet translationOnly;
  translationOnly.set(0); // root_tx
  translationOnly.set(1); // root_ty
  translationOnly.set(2); // root_tz

  // Simplify the transform
  ParameterTransformT<T> simplified = transform.simplify(translationOnly);

  // Check that the simplified transform has only the selected parameters
  EXPECT_EQ(simplified.name.size(), 3);
  EXPECT_EQ(simplified.name[0], "root_tx");
  EXPECT_EQ(simplified.name[1], "root_ty");
  EXPECT_EQ(simplified.name[2], "root_tz");

  // Check that the transform matrix has the correct dimensions
  EXPECT_EQ(simplified.transform.rows(), 21);
  EXPECT_EQ(simplified.transform.cols(), 3);

  // Check that the transform matrix has the correct values
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(simplified.transform.coeff(i, i), 1.0, 1e-6);
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, SubsetParameterTransform) {
  using T = typename TestFixture::Type;

  // Create a parameter transform
  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"};
  transform.transform.resize(21, 6);
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.push_back(Eigen::Triplet<T>(0, 0, 1.0));
  triplets.push_back(Eigen::Triplet<T>(1, 1, 1.0));
  triplets.push_back(Eigen::Triplet<T>(2, 2, 1.0));
  triplets.push_back(Eigen::Triplet<T>(3, 3, 1.0));
  triplets.push_back(Eigen::Triplet<T>(4, 4, 1.0));
  triplets.push_back(Eigen::Triplet<T>(5, 5, 1.0));
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Create parameter limits
  ParameterLimits limits;

  // Add a MinMax limit for root_tx
  ParameterLimit limit1;
  limit1.type = MinMax;
  limit1.data.minMax.parameterIndex = 0;
  limit1.data.minMax.limits[0] = -1.0f;
  limit1.data.minMax.limits[1] = 1.0f;
  limits.push_back(limit1);

  // Add a MinMaxJoint limit
  ParameterLimit limitMinMaxJoint;
  limitMinMaxJoint.type = MinMaxJoint;
  limitMinMaxJoint.data.minMaxJoint.jointIndex = 0;
  limitMinMaxJoint.data.minMaxJoint.jointParameter = 0;
  limitMinMaxJoint.data.minMaxJoint.limits[0] = -1.0f;
  limitMinMaxJoint.data.minMaxJoint.limits[1] = 1.0f;
  limits.push_back(limitMinMaxJoint);

  // Add a Linear limit between root_tx and root_ty
  ParameterLimit limit2;
  limit2.type = Linear;
  limit2.data.linear.referenceIndex = 0;
  limit2.data.linear.targetIndex = 1;
  limit2.data.linear.scale = 2.0f;
  limit2.data.linear.offset = 0.0f;
  limits.push_back(limit2);

  // Add a Linear limit with invalid indices after subsetting
  ParameterLimit limitLinearInvalid;
  limitLinearInvalid.type = Linear;
  limitLinearInvalid.data.linear.referenceIndex = 3; // root_rx (will be invalid after subsetting)
  limitLinearInvalid.data.linear.targetIndex = 4; // root_ry (will be invalid after subsetting)
  limitLinearInvalid.data.linear.scale = 1.0f;
  limitLinearInvalid.data.linear.offset = 0.0f;
  limits.push_back(limitLinearInvalid);

  // Add a LinearJoint limit
  ParameterLimit limitLinearJoint;
  limitLinearJoint.type = LinearJoint;
  limitLinearJoint.data.linearJoint.referenceJointIndex = 0;
  limitLinearJoint.data.linearJoint.referenceJointParameter = 0;
  limitLinearJoint.data.linearJoint.targetJointIndex = 1;
  limitLinearJoint.data.linearJoint.targetJointParameter = 0;
  limitLinearJoint.data.linearJoint.scale = 1.0f;
  limitLinearJoint.data.linearJoint.offset = 0.0f;
  limits.push_back(limitLinearJoint);

  // Add a HalfPlane limit between root_rx and root_ry
  ParameterLimit limit3;
  limit3.type = HalfPlane;
  limit3.data.halfPlane.param1 = 3;
  limit3.data.halfPlane.param2 = 4;
  limit3.data.halfPlane.normal[0] = 1.0f;
  limit3.data.halfPlane.normal[1] = 1.0f;
  limit3.data.halfPlane.offset = 0.0f;
  limits.push_back(limit3);

  // Add a HalfPlane limit with invalid indices after subsetting
  ParameterLimit limitHalfPlaneInvalid;
  limitHalfPlaneInvalid.type = HalfPlane;
  limitHalfPlaneInvalid.data.halfPlane.param1 = 3; // root_rx (will be invalid after subsetting)
  limitHalfPlaneInvalid.data.halfPlane.param2 = 1; // root_ty (will be valid after subsetting)
  limitHalfPlaneInvalid.data.halfPlane.normal[0] = 1.0f;
  limitHalfPlaneInvalid.data.halfPlane.normal[1] = 1.0f;
  limitHalfPlaneInvalid.data.halfPlane.offset = 0.0f;
  limits.push_back(limitHalfPlaneInvalid);

  // Add a HalfPlane limit with valid indices after subsetting (to cover the break statement)
  ParameterLimit limitHalfPlaneValid;
  limitHalfPlaneValid.type = HalfPlane;
  limitHalfPlaneValid.data.halfPlane.param1 = 0; // root_tx (will be valid after subsetting)
  limitHalfPlaneValid.data.halfPlane.param2 = 1; // root_ty (will be valid after subsetting)
  limitHalfPlaneValid.data.halfPlane.normal[0] = 1.0f;
  limitHalfPlaneValid.data.halfPlane.normal[1] = 1.0f;
  limitHalfPlaneValid.data.halfPlane.offset = 0.0f;
  limits.push_back(limitHalfPlaneValid);

  // Add an Ellipsoid limit
  ParameterLimit limitEllipsoid;
  limitEllipsoid.type = Ellipsoid;
  limits.push_back(limitEllipsoid);

  // Add a MinMaxJointPassive limit
  ParameterLimit limitMinMaxJointPassive;
  limitMinMaxJointPassive.type = MinMaxJointPassive;
  limits.push_back(limitMinMaxJointPassive);

  // Add parameter sets
  ParameterSet set1;
  set1.set(0); // root_tx
  set1.set(1); // root_ty
  transform.parameterSets["translation"] = set1;

  // Add pose constraints
  PoseConstraint pc;
  pc.parameterIdValue.emplace_back(0, 1.0f);
  transform.poseConstraints["test"] = pc;

  // Add blend shape parameters directly to the transform
  transform.blendShapeParameters.resize(1);
  transform.blendShapeParameters[0] = 0; // root_tx

  // Add face expression parameters directly to the transform
  transform.faceExpressionParameters.resize(1);
  transform.faceExpressionParameters[0] = 1; // root_ty

  // Create a parameter set with only translation parameters
  ParameterSet translationOnly;
  translationOnly.set(0); // root_tx
  translationOnly.set(1); // root_ty

  // Subset the transform
  auto [subsetTransform, subsetLimits] =
      subsetParameterTransform(transform, limits, translationOnly);

  // Check that the subset transform has only the selected parameters
  EXPECT_EQ(subsetTransform.name.size(), 2);
  EXPECT_EQ(subsetTransform.name[0], "root_tx");
  EXPECT_EQ(subsetTransform.name[1], "root_ty");

  // Check that the transform matrix has the correct dimensions
  EXPECT_EQ(subsetTransform.transform.rows(), 21);
  EXPECT_EQ(subsetTransform.transform.cols(), 2);

  // Check that the transform matrix has the correct values
  EXPECT_NEAR(subsetTransform.transform.coeff(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(subsetTransform.transform.coeff(1, 1), 1.0, 1e-6);

  // Check that the parameter limits were correctly subset
  EXPECT_EQ(
      subsetLimits.size(),
      7); // MinMax, MinMaxJoint, Linear, LinearJoint, Ellipsoid, MinMaxJointPassive, and valid
          // HalfPlane should remain HalfPlane should be removed because its indices are invalid
          // after subsetting Linear with invalid indices should be removed HalfPlane with one
          // invalid index should be removed

  // Check that the parameter sets were correctly subset
  EXPECT_EQ(subsetTransform.parameterSets.size(), 1);
  EXPECT_TRUE(subsetTransform.parameterSets["translation"].test(0));
  EXPECT_TRUE(subsetTransform.parameterSets["translation"].test(1));

  // Check that the pose constraints were correctly subset
  EXPECT_EQ(subsetTransform.poseConstraints.size(), 1);
  EXPECT_EQ(subsetTransform.poseConstraints["test"].parameterIdValue.size(), 1);
  EXPECT_EQ(subsetTransform.poseConstraints["test"].parameterIdValue[0].first, 0);

  // Check that the blend shape parameters were correctly subset
  EXPECT_EQ(subsetTransform.blendShapeParameters.size(), 1);
  EXPECT_EQ(subsetTransform.blendShapeParameters[0], 0);

  // Check that the face expression parameters were correctly subset
  EXPECT_EQ(subsetTransform.faceExpressionParameters.size(), 1);
  EXPECT_EQ(subsetTransform.faceExpressionParameters[0], 1);
}

TYPED_TEST(Momentum_ParameterTransformTest, MapParameterTransformJoints) {
  using T = typename TestFixture::Type;

  // Create a parameter transform
  ParameterTransformT<T> transform = ParameterTransformT<T>::empty(3 * kParametersPerJoint);
  transform.name = {"root_tx", "root_ty", "root_tz", "joint1_rx", "joint1_ry", "joint1_rz"};
  transform.transform.resize(3 * kParametersPerJoint, 6);
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.push_back(Eigen::Triplet<T>(0, 0, 1.0)); // root_tx
  triplets.push_back(Eigen::Triplet<T>(1, 1, 1.0)); // root_ty
  triplets.push_back(Eigen::Triplet<T>(2, 2, 1.0)); // root_tz
  triplets.push_back(Eigen::Triplet<T>(kParametersPerJoint + 3, 3, 1.0)); // joint1_rx
  triplets.push_back(Eigen::Triplet<T>(kParametersPerJoint + 4, 4, 1.0)); // joint1_ry
  triplets.push_back(Eigen::Triplet<T>(kParametersPerJoint + 5, 5, 1.0)); // joint1_rz
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Set offsets
  transform.offsets.resize(3 * kParametersPerJoint);
  transform.offsets.setZero();
  transform.offsets[0] = 1.0;
  transform.offsets[kParametersPerJoint] = 2.0;
  transform.offsets[2 * kParametersPerJoint] = 3.0;

  // Set activeJointParams
  transform.activeJointParams.resize(3 * kParametersPerJoint);
  transform.activeJointParams.setConstant(false);
  transform.activeJointParams[0] = true;
  transform.activeJointParams[kParametersPerJoint + 3] = true;

  // Create joint mapping
  std::vector<size_t> jointMapping = {0, kInvalidIndex, 1};

  // Map the parameter transform
  ParameterTransformT<T> mappedTransform = mapParameterTransformJoints(transform, 2, jointMapping);

  // Check that the mapped transform has the correct dimensions
  EXPECT_EQ(mappedTransform.offsets.size(), 2 * kParametersPerJoint);
  EXPECT_EQ(mappedTransform.transform.rows(), 2 * kParametersPerJoint);
  EXPECT_EQ(mappedTransform.transform.cols(), 6);

  // Check that the offsets were correctly mapped
  EXPECT_EQ(mappedTransform.offsets[0], 1.0);
  EXPECT_EQ(mappedTransform.offsets[kParametersPerJoint], 3.0);

  // Check that the transform matrix was correctly mapped
  EXPECT_NEAR(mappedTransform.transform.coeff(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(
      mappedTransform.transform.coeff(kParametersPerJoint + 3, 3), 0.0, 1e-6); // joint1 was removed

  // Check that the activeJointParams were correctly mapped
  EXPECT_TRUE(mappedTransform.activeJointParams[0]);
  EXPECT_FALSE(mappedTransform.activeJointParams[kParametersPerJoint + 3]);

  // Check that the name was preserved
  EXPECT_EQ(mappedTransform.name, transform.name);
}

// Test addBlendShapeParameters function (only for float type)
TEST(Momentum_ParameterTransform, AddBlendShapeParameters) {
  // Create a parameter transform
  ParameterTransform transform = ParameterTransform::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz"};
  transform.transform.resize(21, 3);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f);
  triplets.emplace_back(1, 1, 1.0f);
  triplets.emplace_back(2, 2, 1.0f);
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Create parameter limits
  ParameterLimits limits;

  // Add blend shape parameters
  auto [transformWithBlendShapes, limitsWithBlendShapes] =
      addBlendShapeParameters(transform, limits, 2);

  // Check that the blend shape parameters were added
  EXPECT_EQ(transformWithBlendShapes.name.size(), 5);
  EXPECT_EQ(transformWithBlendShapes.name[3], "blend_0");
  EXPECT_EQ(transformWithBlendShapes.name[4], "blend_1");

  // Check that the transform matrix was resized
  EXPECT_EQ(transformWithBlendShapes.transform.rows(), 21);
  EXPECT_EQ(transformWithBlendShapes.transform.cols(), 5);

  // Check that the blend shape parameters were set
  EXPECT_EQ(transformWithBlendShapes.blendShapeParameters.size(), 2);
  EXPECT_EQ(transformWithBlendShapes.blendShapeParameters[0], 3);
  EXPECT_EQ(transformWithBlendShapes.blendShapeParameters[1], 4);

  // Test idempotence by adding blend shape parameters again
  auto [transformWithBlendShapes2, limitsWithBlendShapes2] =
      addBlendShapeParameters(transformWithBlendShapes, limitsWithBlendShapes, 2);

  // Check that the blend shape parameters were not duplicated
  EXPECT_EQ(transformWithBlendShapes2.name.size(), 5);
  EXPECT_EQ(transformWithBlendShapes2.blendShapeParameters.size(), 2);
}

// Test addFaceExpressionParameters function (only for float type)
TEST(Momentum_ParameterTransform, AddFaceExpressionParameters) {
  // Create a parameter transform
  ParameterTransform transform = ParameterTransform::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz"};
  transform.transform.resize(21, 3);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f);
  triplets.emplace_back(1, 1, 1.0f);
  triplets.emplace_back(2, 2, 1.0f);
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Create parameter limits
  ParameterLimits limits;

  // Add face expression parameters
  auto [transformWithFaceExpressions, limitsWithFaceExpressions] =
      addFaceExpressionParameters(transform, limits, 2);

  // Check that the face expression parameters were added
  EXPECT_EQ(transformWithFaceExpressions.name.size(), 5);
  EXPECT_EQ(transformWithFaceExpressions.name[3], "face_expre_0");
  EXPECT_EQ(transformWithFaceExpressions.name[4], "face_expre_1");

  // Check that the transform matrix was resized
  EXPECT_EQ(transformWithFaceExpressions.transform.rows(), 21);
  EXPECT_EQ(transformWithFaceExpressions.transform.cols(), 5);

  // Check that the face expression parameters were set
  EXPECT_EQ(transformWithFaceExpressions.faceExpressionParameters.size(), 2);
  EXPECT_EQ(transformWithFaceExpressions.faceExpressionParameters[0], 3);
  EXPECT_EQ(transformWithFaceExpressions.faceExpressionParameters[1], 4);

  // Test idempotence by adding face expression parameters again
  auto [transformWithFaceExpressions2, limitsWithFaceExpressions2] =
      addFaceExpressionParameters(transformWithFaceExpressions, limitsWithFaceExpressions, 2);

  // Check that the face expression parameters were not duplicated
  EXPECT_EQ(transformWithFaceExpressions2.name.size(), 5);
  EXPECT_EQ(transformWithFaceExpressions2.faceExpressionParameters.size(), 2);
}

// Test InverseParameterTransform
TEST(Momentum_InverseParameterTransform, Functionality) {
  // Create a parameter transform
  ParameterTransform transform = ParameterTransform::empty(21);
  transform.name = {"root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"};
  transform.transform.resize(21, 6);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.emplace_back(0, 0, 1.0f);
  triplets.emplace_back(1, 1, 1.0f);
  triplets.emplace_back(2, 2, 1.0f);
  triplets.emplace_back(3, 3, 1.0f);
  triplets.emplace_back(4, 4, 1.0f);
  triplets.emplace_back(5, 5, 1.0f);
  transform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Create an inverse parameter transform
  InverseParameterTransform inverseTransform(transform);

  EXPECT_EQ(inverseTransform.numAllModelParameters(), 6);
  EXPECT_EQ(inverseTransform.numJointParameters(), 21);

  // Test applying the inverse transform
  JointParameters jointParams = JointParameters::Zero(21);
  jointParams.v.segment(0, 6) << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3;

  CharacterParameters charParams = inverseTransform.apply(jointParams);
  ModelParameters modelParams = charParams.pose;

  // Check that the model parameters match the joint parameters
  EXPECT_EQ(modelParams.size(), 6);
  EXPECT_NEAR(modelParams.v[0], 1.0, 1e-6);
  EXPECT_NEAR(modelParams.v[1], 2.0, 1e-6);
  EXPECT_NEAR(modelParams.v[2], 3.0, 1e-6);
  EXPECT_NEAR(modelParams.v[3], 0.1, 1e-6);
  EXPECT_NEAR(modelParams.v[4], 0.2, 1e-6);
  EXPECT_NEAR(modelParams.v[5], 0.3, 1e-6);

  // Test round-trip conversion
  JointParameters roundTripJointParams = transform.apply(modelParams);
  EXPECT_TRUE(roundTripJointParams.v.segment(0, 6).isApprox(jointParams.v.segment(0, 6)));
}
