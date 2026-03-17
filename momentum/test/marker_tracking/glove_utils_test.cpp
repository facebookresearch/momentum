/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/skeleton_solver_function.h>
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

// Verify that createGlovePositionConstraintData and
// createGloveOrientationConstraintData correctly convert glove observations
// into per-frame constraint data with the right joint indices, targets, and
// weights, and that invalid/unknown observations are filtered out.
TEST(GloveUtilsTest, CreateConstraintDataFromObservations) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;
  const auto gloveChar = createGloveCharacter(srcCharacter, cfg);

  // Use existing joint names from the test character for finger joints.
  // Joints are: root, joint1, joint2, l_wrist, joint4, r_wrist, joint6, joint7,
  //             glove_l_wrist, glove_r_wrist
  const std::string fingerJoint = "joint1";
  const size_t fingerJointIdx = gloveChar.skeleton.getJointIdByName(fingerJoint);
  ASSERT_NE(fingerJointIdx, kInvalidIndex);

  const size_t gloveBoneIdx = gloveChar.skeleton.getJointIdByName("glove_l_wrist");
  ASSERT_NE(gloveBoneIdx, kInvalidIndex);

  // Build 2 frames of glove data: frame 0 has a valid and an invalid obs,
  // frame 1 has a valid obs and one with an unknown joint name.
  std::vector<GloveFrameData> gloveData(2);

  GloveSensorObservation validObs;
  validObs.jointName = fingerJoint;
  validObs.position = Vector3f(1.0f, 2.0f, 3.0f);
  validObs.orientation = Eigen::Quaternionf(Eigen::AngleAxisf(0.5f, Vector3f::UnitZ()));
  validObs.valid = true;

  GloveSensorObservation invalidObs;
  invalidObs.jointName = fingerJoint;
  invalidObs.valid = false;

  GloveSensorObservation unknownObs;
  unknownObs.jointName = "nonexistent_joint";
  unknownObs.valid = true;

  gloveData[0] = {validObs, invalidObs};
  gloveData[1] = {validObs, unknownObs};

  // Position constraints
  const auto posData = createGlovePositionConstraintData(gloveData, gloveChar, cfg, 0);
  ASSERT_EQ(posData.size(), 2u);
  // Frame 0: only the valid obs should produce a constraint
  ASSERT_EQ(posData[0].size(), 1u);
  EXPECT_EQ(posData[0][0].sourceJoint, fingerJointIdx);
  EXPECT_EQ(posData[0][0].referenceJoint, gloveBoneIdx);
  EXPECT_TRUE(posData[0][0].target.isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
  EXPECT_FLOAT_EQ(posData[0][0].weight, 1.0f);

  // Frame 1: valid obs produces a constraint, unknown joint is skipped
  ASSERT_EQ(posData[1].size(), 1u);
  EXPECT_EQ(posData[1][0].sourceJoint, fingerJointIdx);

  // Orientation constraints
  const auto oriData = createGloveOrientationConstraintData(gloveData, gloveChar, cfg, 0);
  ASSERT_EQ(oriData.size(), 2u);
  ASSERT_EQ(oriData[0].size(), 1u);
  EXPECT_EQ(oriData[0][0].sourceJoint, fingerJointIdx);
  EXPECT_EQ(oriData[0][0].referenceJoint, gloveBoneIdx);
  EXPECT_TRUE(oriData[0][0].target.isApprox(validObs.orientation));
  EXPECT_FLOAT_EQ(oriData[0][0].weight, 1.0f);
  ASSERT_EQ(oriData[1].size(), 1u);

  // Right hand (hand_idx=1): should use glove_r_wrist as reference joint
  const size_t rGloveBoneIdx = gloveChar.skeleton.getJointIdByName("glove_r_wrist");
  ASSERT_NE(rGloveBoneIdx, kInvalidIndex);

  const auto rPosData = createGlovePositionConstraintData(gloveData, gloveChar, cfg, 1);
  ASSERT_EQ(rPosData.size(), 2u);
  ASSERT_EQ(rPosData[0].size(), 1u);
  EXPECT_EQ(rPosData[0][0].sourceJoint, fingerJointIdx);
  EXPECT_EQ(rPosData[0][0].referenceJoint, rGloveBoneIdx);

  const auto rOriData = createGloveOrientationConstraintData(gloveData, gloveChar, cfg, 1);
  ASSERT_EQ(rOriData.size(), 2u);
  ASSERT_EQ(rOriData[0].size(), 1u);
  EXPECT_EQ(rOriData[0][0].referenceJoint, rGloveBoneIdx);
}

// Verify that setupGloveErrorFunctions registers error functions with a
// SkeletonSolverFunction, and that updateGloveConstraintsForFrame correctly
// swaps per-frame constraint data so the solver sees different constraints
// for different frames.
TEST(GloveUtilsTest, SetupAndUpdatePerFrameErrorFunctions) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;
  const auto gloveChar = createGloveCharacter(srcCharacter, cfg);

  // Build 2 frames of glove data with different observations per frame
  std::vector<GloveFrameData> gloveData(2);

  GloveSensorObservation obs0;
  obs0.jointName = "joint1";
  obs0.position = Vector3f(1.0f, 0.0f, 0.0f);
  obs0.valid = true;

  GloveSensorObservation obs1a;
  obs1a.jointName = "joint1";
  obs1a.position = Vector3f(2.0f, 0.0f, 0.0f);
  obs1a.valid = true;

  GloveSensorObservation obs1b;
  obs1b.jointName = "joint2";
  obs1b.position = Vector3f(3.0f, 0.0f, 0.0f);
  obs1b.valid = true;

  gloveData[0] = {obs0};
  gloveData[1] = {obs1a, obs1b};

  // Create a solver function and set up glove error functions
  SkeletonSolverFunctionT<float> solverFunc(gloveChar, gloveChar.parameterTransform);

  auto funcs = setupGloveErrorFunctions(solverFunc, gloveChar, gloveData, cfg, 0);
  ASSERT_TRUE(funcs.has_value());

  // Swap to frame 0: should have 1 position constraint
  updateGloveConstraintsForFrame(*funcs, 0);
  EXPECT_EQ(funcs->posFunc->getNumConstraints(), 1u);
  EXPECT_EQ(funcs->oriFunc->numConstraints(), 1u);

  // Swap to frame 1: should have 2 position constraints
  updateGloveConstraintsForFrame(*funcs, 1);
  EXPECT_EQ(funcs->posFunc->getNumConstraints(), 2u);
  EXPECT_EQ(funcs->oriFunc->numConstraints(), 2u);

  // Swap back to frame 0: constraints should revert
  updateGloveConstraintsForFrame(*funcs, 0);
  EXPECT_EQ(funcs->posFunc->getNumConstraints(), 1u);

  // Empty glove data returns nullopt
  std::vector<GloveFrameData> emptyData;
  auto emptyFuncs = setupGloveErrorFunctions(solverFunc, gloveChar, emptyData, cfg, 0);
  EXPECT_FALSE(emptyFuncs.has_value());
}

// Verify that addGloveBones adds skeleton joints without model parameters,
// and addGloveCalibrationParameters adds model parameters for existing bones.
TEST(GloveUtilsTest, AddGloveBonesAndCalibrationParametersSeparately) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;

  // addGloveBones should add joints but no model parameters
  const auto bonesOnly = addGloveBones(srcCharacter, cfg);
  EXPECT_EQ(bonesOnly.skeleton.joints.size(), srcCharacter.skeleton.joints.size() + 2);
  EXPECT_NE(bonesOnly.skeleton.getJointIdByName("glove_l_wrist"), kInvalidIndex);
  EXPECT_NE(bonesOnly.skeleton.getJointIdByName("glove_r_wrist"), kInvalidIndex);
  EXPECT_EQ(
      bonesOnly.parameterTransform.numAllModelParameters(),
      srcCharacter.parameterTransform.numAllModelParameters());

  // addGloveCalibrationParameters should add model parameters for the existing bones
  const auto full = addGloveCalibrationParameters(bonesOnly, cfg);
  EXPECT_EQ(full.skeleton.joints.size(), bonesOnly.skeleton.joints.size());
  EXPECT_EQ(
      full.parameterTransform.numAllModelParameters(),
      bonesOnly.parameterTransform.numAllModelParameters() + 12);
  EXPECT_EQ(full.parameterTransform.getParameterSet("gloves", true).count(), 12u);

  // The composition should produce the same result as createGloveCharacter
  const auto wrapper = createGloveCharacter(srcCharacter, cfg);
  EXPECT_EQ(full.skeleton.joints.size(), wrapper.skeleton.joints.size());
  EXPECT_EQ(
      full.parameterTransform.numAllModelParameters(),
      wrapper.parameterTransform.numAllModelParameters());
}

// Verify that addGloveBones with non-zero offsets bakes the offset into the
// joint's translationOffset and preRotation, and that calling it twice
// does not duplicate the bones.
TEST(GloveUtilsTest, AddGloveBonesWithOffsets) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;

  std::array<GloveOffset, 2> offsets;
  offsets[0].translation = Vector3f(1.0f, 2.0f, 3.0f);
  offsets[0].rotationEulerXYZ = Vector3f(0.1f, 0.2f, 0.3f);
  offsets[1].translation = Vector3f(4.0f, 5.0f, 6.0f);
  offsets[1].rotationEulerXYZ = Vector3f(0.4f, 0.5f, 0.6f);

  const auto result = addGloveBones(srcCharacter, cfg, offsets);
  ASSERT_EQ(result.skeleton.joints.size(), srcCharacter.skeleton.joints.size() + 2);

  const size_t leftIdx = result.skeleton.getJointIdByName("glove_l_wrist");
  ASSERT_NE(leftIdx, kInvalidIndex);
  EXPECT_TRUE(
      result.skeleton.joints[leftIdx].translationOffset.isApprox(Vector3f(1.0f, 2.0f, 3.0f)));
  const auto expectedLeftRot =
      eulerToQuaternion<float>(Vector3f(0.1f, 0.2f, 0.3f), 0, 1, 2, EulerConvention::Intrinsic);
  EXPECT_TRUE(result.skeleton.joints[leftIdx].preRotation.isApprox(expectedLeftRot));

  const size_t rightIdx = result.skeleton.getJointIdByName("glove_r_wrist");
  ASSERT_NE(rightIdx, kInvalidIndex);
  EXPECT_TRUE(
      result.skeleton.joints[rightIdx].translationOffset.isApprox(Vector3f(4.0f, 5.0f, 6.0f)));
  const auto expectedRightRot =
      eulerToQuaternion<float>(Vector3f(0.4f, 0.5f, 0.6f), 0, 1, 2, EulerConvention::Intrinsic);
  EXPECT_TRUE(result.skeleton.joints[rightIdx].preRotation.isApprox(expectedRightRot));

  // Calling addGloveBones again should not add duplicate joints
  const auto again = addGloveBones(result, cfg, offsets);
  EXPECT_EQ(again.skeleton.joints.size(), result.skeleton.joints.size());
}

// Verify that bakeGloveOffsetsFromParams extracts solved offsets from a
// solvingCharacter and bakes them into the target character's skeleton.
TEST(GloveUtilsTest, BakeGloveOffsetsFromParams) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;

  // Create a solving character with glove bones and calibration parameters
  const auto solvingChar = createGloveCharacter(srcCharacter, cfg);

  // Build model parameters with known glove offsets
  const auto numParams = solvingChar.parameterTransform.numAllModelParameters();
  ModelParameters solvedParams = ModelParameters::Zero(numParams);

  const auto gloveSet = solvingChar.parameterTransform.getParameterSet("gloves", true);
  std::vector<size_t> gloveParamIndices;
  for (size_t i = 0; i < gloveSet.size(); ++i) {
    if (gloveSet.test(i)) {
      gloveParamIndices.push_back(i);
    }
  }
  ASSERT_EQ(gloveParamIndices.size(), 12u);

  // Left hand: tx=1, ty=2, tz=3, rx=0.1, ry=0.2, rz=0.3
  solvedParams.v(gloveParamIndices[0]) = 1.0f;
  solvedParams.v(gloveParamIndices[1]) = 2.0f;
  solvedParams.v(gloveParamIndices[2]) = 3.0f;
  solvedParams.v(gloveParamIndices[3]) = 0.1f;
  solvedParams.v(gloveParamIndices[4]) = 0.2f;
  solvedParams.v(gloveParamIndices[5]) = 0.3f;

  // Right hand: tx=4, ty=5, tz=6, rx=0.4, ry=0.5, rz=0.6
  solvedParams.v(gloveParamIndices[6]) = 4.0f;
  solvedParams.v(gloveParamIndices[7]) = 5.0f;
  solvedParams.v(gloveParamIndices[8]) = 6.0f;
  solvedParams.v(gloveParamIndices[9]) = 0.4f;
  solvedParams.v(gloveParamIndices[10]) = 0.5f;
  solvedParams.v(gloveParamIndices[11]) = 0.6f;

  // Bake into the original character (no glove bones yet)
  Character bakedChar = srcCharacter;
  bakeGloveOffsetsFromParams(bakedChar, solvedParams, solvingChar, cfg);

  // Should have glove bones with baked offsets but no extra model parameters
  ASSERT_EQ(bakedChar.skeleton.joints.size(), srcCharacter.skeleton.joints.size() + 2);
  EXPECT_EQ(
      bakedChar.parameterTransform.numAllModelParameters(),
      srcCharacter.parameterTransform.numAllModelParameters());

  const size_t leftIdx = bakedChar.skeleton.getJointIdByName("glove_l_wrist");
  ASSERT_NE(leftIdx, kInvalidIndex);
  EXPECT_TRUE(
      bakedChar.skeleton.joints[leftIdx].translationOffset.isApprox(Vector3f(1.0f, 2.0f, 3.0f)));

  const size_t rightIdx = bakedChar.skeleton.getJointIdByName("glove_r_wrist");
  ASSERT_NE(rightIdx, kInvalidIndex);
  EXPECT_TRUE(
      bakedChar.skeleton.joints[rightIdx].translationOffset.isApprox(Vector3f(4.0f, 5.0f, 6.0f)));
}

// Verify that constraint creation falls back to the wrist joint when the
// glove bone does not exist in the character.
TEST(GloveUtilsTest, ConstraintFallbackToWrist) {
  const auto srcCharacter = createCharacterWithWrists();
  const GloveConfig cfg;

  // No glove bones — constraint creation should fall back to wrist
  const size_t wristIdx = srcCharacter.skeleton.getJointIdByName("l_wrist");
  ASSERT_NE(wristIdx, kInvalidIndex);

  std::vector<GloveFrameData> gloveData(1);
  GloveSensorObservation obs;
  obs.jointName = "joint1";
  obs.position = Vector3f(1.0f, 2.0f, 3.0f);
  obs.valid = true;
  gloveData[0] = {obs};

  const auto posData = createGlovePositionConstraintData(gloveData, srcCharacter, cfg, 0);
  ASSERT_EQ(posData.size(), 1u);
  ASSERT_EQ(posData[0].size(), 1u);
  EXPECT_EQ(posData[0][0].referenceJoint, wristIdx);

  const auto oriData = createGloveOrientationConstraintData(gloveData, srcCharacter, cfg, 0);
  ASSERT_EQ(oriData.size(), 1u);
  ASSERT_EQ(oriData[0].size(), 1u);
  EXPECT_EQ(oriData[0][0].referenceJoint, wristIdx);

  // Right hand should also fall back to wrist
  const size_t rWristIdx = srcCharacter.skeleton.getJointIdByName("r_wrist");
  ASSERT_NE(rWristIdx, kInvalidIndex);

  const auto rPosData = createGlovePositionConstraintData(gloveData, srcCharacter, cfg, 1);
  ASSERT_EQ(rPosData.size(), 1u);
  ASSERT_EQ(rPosData[0].size(), 1u);
  EXPECT_EQ(rPosData[0][0].referenceJoint, rWristIdx);

  const auto rOriData = createGloveOrientationConstraintData(gloveData, srcCharacter, cfg, 1);
  ASSERT_EQ(rOriData.size(), 1u);
  ASSERT_EQ(rOriData[0].size(), 1u);
  EXPECT_EQ(rOriData[0][0].referenceJoint, rWristIdx);
}
