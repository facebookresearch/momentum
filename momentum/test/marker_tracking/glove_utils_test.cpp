/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/marker_tracking/glove_utils.h>
#include <momentum/math/utility.h>
#include <momentum/test/character/character_helpers.h>

#include <gtest/gtest.h>

using namespace momentum;

namespace {

// Create a test character with wrist joints named as expected by GloveConfig.
Character createCharacterWithWrists() {
  Character character = createTestCharacter(8);
  character.skeleton.joints[3].name = "l_wrist";
  character.skeleton.joints[5].name = "r_wrist";
  return character;
}

} // namespace

// Verify that createGloveCharacter correctly extends the skeleton: adds child
// joints of the wrists, wires up 6 DOFs each in the parameter transform, and
// registers a "gloves" parameter set that the solver can use to separate glove
// params from pose params.
TEST(GloveUtilsTest, CreateGloveCharacterExtendsSkeletonAndTransform) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;
  const auto gloveChar = createGloveCharacter(srcCharacter, cfg);

  // Two new joints added as children of the wrists
  ASSERT_EQ(gloveChar.skeleton.joints.size(), srcCharacter.skeleton.joints.size() + 2);
  const size_t leftGloveIdx = gloveChar.skeleton.getJointIdByName("glove_l_wrist");
  const size_t rightGloveIdx = gloveChar.skeleton.getJointIdByName("glove_r_wrist");
  ASSERT_NE(leftGloveIdx, kInvalidIndex);
  ASSERT_NE(rightGloveIdx, kInvalidIndex);
  EXPECT_EQ(
      gloveChar.skeleton.joints[leftGloveIdx].parent,
      gloveChar.skeleton.getJointIdByName("l_wrist"));
  EXPECT_EQ(
      gloveChar.skeleton.joints[rightGloveIdx].parent,
      gloveChar.skeleton.getJointIdByName("r_wrist"));

  // "gloves" parameter set should contain exactly 12 DOFs (2 hands x 6)
  // and the transform matrix should be consistently sized
  const auto gloveSet = gloveChar.parameterTransform.getParameterSet("gloves", true);
  EXPECT_EQ(gloveSet.count(), 12u);
  EXPECT_EQ(
      gloveChar.parameterTransform.numAllModelParameters(),
      srcCharacter.parameterTransform.numAllModelParameters() + 12);
  EXPECT_EQ(
      gloveChar.parameterTransform.transform.rows(),
      static_cast<Eigen::Index>(gloveChar.skeleton.joints.size()) * kParametersPerJoint);
}

// When wrist joints are missing from the skeleton, createGloveCharacter should
// gracefully skip them rather than crash or corrupt the character.
TEST(GloveUtilsTest, CreateGloveCharacterSkipsMissingWrists) {
  // No matching wrist names at all
  const auto srcCharacter = createTestCharacter(8);
  const GloveConfig cfg;
  const auto gloveChar = createGloveCharacter(srcCharacter, cfg);
  EXPECT_EQ(gloveChar.skeleton.joints.size(), srcCharacter.skeleton.joints.size());
  EXPECT_TRUE(gloveChar.parameterTransform.getParameterSet("gloves", true).none());

  // Only one matching wrist
  auto oneWristChar = createTestCharacter(8);
  oneWristChar.skeleton.joints[3].name = "l_wrist";
  const auto oneGloveChar = createGloveCharacter(oneWristChar, cfg);
  EXPECT_EQ(oneGloveChar.skeleton.joints.size(), oneWristChar.skeleton.joints.size() + 1);
  EXPECT_EQ(oneGloveChar.parameterTransform.getParameterSet("gloves", true).count(), 6u);
}

// End-to-end round-trip: set known model parameter values for glove DOFs,
// apply the parameter transform, and verify extractGloveOffsetsFromCharacter
// recovers them correctly. This validates that the transform matrix wiring,
// DOF ordering, and extraction logic are all consistent.
TEST(GloveUtilsTest, ParameterRoundTrip) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;
  const auto gloveChar = createGloveCharacter(srcCharacter, cfg);

  const auto numParams = gloveChar.parameterTransform.numAllModelParameters();
  ModelParameters pose = ModelParameters::Zero(numParams);

  // Collect glove parameter indices from the parameter set
  const auto gloveSet = gloveChar.parameterTransform.getParameterSet("gloves", true);
  std::vector<size_t> gloveParamIndices;
  for (size_t i = 0; i < gloveSet.size(); ++i) {
    if (gloveSet.test(i)) {
      gloveParamIndices.push_back(i);
    }
  }
  ASSERT_EQ(gloveParamIndices.size(), 12u);

  // Set left hand: translation = (1, 2, 3), rotation = (0.1, 0.2, 0.3)
  pose.v(gloveParamIndices[0]) = 1.0f;
  pose.v(gloveParamIndices[1]) = 2.0f;
  pose.v(gloveParamIndices[2]) = 3.0f;
  pose.v(gloveParamIndices[3]) = 0.1f;
  pose.v(gloveParamIndices[4]) = 0.2f;
  pose.v(gloveParamIndices[5]) = 0.3f;

  // Set right hand: translation = (4, 5, 6), rotation = (0.4, 0.5, 0.6)
  pose.v(gloveParamIndices[6]) = 4.0f;
  pose.v(gloveParamIndices[7]) = 5.0f;
  pose.v(gloveParamIndices[8]) = 6.0f;
  pose.v(gloveParamIndices[9]) = 0.4f;
  pose.v(gloveParamIndices[10]) = 0.5f;
  pose.v(gloveParamIndices[11]) = 0.6f;

  CharacterParameters params;
  params.pose = pose;
  params.offsets = JointParameters::Zero(gloveChar.skeleton.joints.size() * kParametersPerJoint);

  const auto offsets = extractGloveOffsetsFromCharacter(gloveChar, params, cfg);

  EXPECT_TRUE(offsets[0].translation.isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
  EXPECT_TRUE(offsets[0].rotationEulerXYZ.isApprox(Vector3f(0.1f, 0.2f, 0.3f)));
  EXPECT_TRUE(offsets[1].translation.isApprox(Vector3f(4.0f, 5.0f, 6.0f)));
  EXPECT_TRUE(offsets[1].rotationEulerXYZ.isApprox(Vector3f(0.4f, 0.5f, 0.6f)));
}
