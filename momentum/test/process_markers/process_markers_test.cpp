/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/marker.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/marker_tracking/marker_tracker.h>
#include <momentum/marker_tracking/process_markers.h>
#include <momentum/test/character/character_helpers.h>

#include <gtest/gtest.h>

using namespace momentum;

namespace {

/// Helper to create a test character with locators and the corresponding identity parameters.
std::pair<Character, ModelParameters> createTestCharacterWithIdentity(size_t numJoints = 5) {
  Character character = createTestCharacter(numJoints);
  ModelParameters identity =
      ModelParameters::Zero(character.parameterTransform.numAllModelParameters());
  return {std::move(character), std::move(identity)};
}

/// Create marker data with the given number of frames, each containing markers matching the
/// character's locators. The marker positions are arbitrary (not based on actual locator positions)
/// and are only suitable for tests that verify frame counting, parameter bounds, and output
/// dimensions - not for tests that verify actual tracking accuracy.
std::vector<std::vector<Marker>> createMarkerData(const Character& character, size_t numFrames) {
  std::vector<std::vector<Marker>> markerData(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    std::vector<Marker> frameMarkers;
    for (const auto& locator : character.locators) {
      Marker m;
      m.name = locator.name;
      m.pos = Eigen::Vector3d(
          static_cast<double>(f), static_cast<double>(f) + 1.0, static_cast<double>(f) + 2.0);
      m.occluded = false;
      frameMarkers.push_back(m);
    }
    markerData[f] = std::move(frameMarkers);
  }
  return markerData;
}

/// Computes the world position of a locator given the skeleton state.
Eigen::Vector3f computeLocatorWorldPosition(
    const Locator& locator,
    const SkeletonState& skelState) {
  return skelState.jointState[locator.parent].transform.transformPoint(locator.offset);
}

/// Creates marker data from the character's locator positions at the identity pose.
std::vector<std::vector<Marker>> createIdentityPoseMarkerData(
    const Character& character,
    const ModelParameters& identity,
    size_t numFrames) {
  const JointParameters jointParams = character.parameterTransform.apply(identity);
  const SkeletonState skelState(jointParams, character.skeleton);

  std::vector<std::vector<Marker>> markerData(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    std::vector<Marker> frameMarkers;
    for (const auto& locator : character.locators) {
      Marker m;
      m.name = locator.name;
      m.pos = computeLocatorWorldPosition(locator, skelState).cast<double>();
      m.occluded = false;
      frameMarkers.push_back(m);
    }
    markerData[f] = std::move(frameMarkers);
  }
  return markerData;
}

/// Creates marker data where markers move linearly over frames.
/// At frame f, each marker is at its identity position plus (offset * f).
std::vector<std::vector<Marker>> createLinearlyMovingMarkerData(
    const Character& character,
    const ModelParameters& identity,
    size_t numFrames,
    const Eigen::Vector3d& offsetPerFrame) {
  const JointParameters jointParams = character.parameterTransform.apply(identity);
  const SkeletonState skelState(jointParams, character.skeleton);

  std::vector<std::vector<Marker>> markerData(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    std::vector<Marker> frameMarkers;
    const Eigen::Vector3d frameOffset = offsetPerFrame * static_cast<double>(f);
    for (const auto& locator : character.locators) {
      Marker m;
      m.name = locator.name;
      m.pos = computeLocatorWorldPosition(locator, skelState).cast<double>() + frameOffset;
      m.occluded = false;
      frameMarkers.push_back(m);
    }
    markerData[f] = std::move(frameMarkers);
  }
  return markerData;
}

} // namespace

// Tests that calibrateMarkers throws when firstFrame exceeds markerData size.
TEST(ProcessMarkersTest, CalibrateMarkers_FirstFrameExceedsSize_Throws) {
  auto [character, identity] = createTestCharacterWithIdentity();
  auto markerData = createMarkerData(character, 5);
  CalibrationConfig config;

  EXPECT_THROW(
      calibrateMarkers(character, identity, markerData, config, /*firstFrame=*/6), std::exception);
}

// Tests that processMarkers throws when firstFrame exceeds markerData size.
TEST(ProcessMarkersTest, ProcessMarkers_FirstFrameExceedsSize_Throws) {
  auto [character, identity] = createTestCharacterWithIdentity();
  auto markerData = createMarkerData(character, 5);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  EXPECT_THROW(
      processMarkers(
          character,
          identity,
          markerData,
          trackingConfig,
          calibrationConfig,
          /*calibrate=*/false,
          /*firstFrame=*/6),
      std::exception);
}

// Tests that processMarkers correctly limits the number of frames with maxFrames.
TEST(ProcessMarkersTest, ProcessMarkers_MaxFramesLimitsOutput) {
  auto [character, identity] = createTestCharacterWithIdentity();
  auto markerData = createMarkerData(character, 10);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf result = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false,
      /*firstFrame=*/0,
      /*maxFrames=*/3);

  EXPECT_EQ(result.cols(), 3);
}

// Tests that processMarkers uses all remaining frames when maxFrames is 0.
TEST(ProcessMarkersTest, ProcessMarkers_MaxFramesZeroUsesAllFrames) {
  auto [character, identity] = createTestCharacterWithIdentity();
  constexpr size_t kNumFrames = 5;
  auto markerData = createMarkerData(character, kNumFrames);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf result = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false,
      /*firstFrame=*/0,
      /*maxFrames=*/0);

  EXPECT_EQ(result.cols(), kNumFrames);
}

// Tests that processMarkers respects firstFrame offset.
TEST(ProcessMarkersTest, ProcessMarkers_FirstFrameOffsetReducesFrames) {
  auto [character, identity] = createTestCharacterWithIdentity();
  constexpr size_t kNumFrames = 10;
  auto markerData = createMarkerData(character, kNumFrames);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf result = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false,
      /*firstFrame=*/3,
      /*maxFrames=*/0);

  // With firstFrame=3 and maxFrames=0, should process frames 3..9 = 7 frames
  EXPECT_EQ(result.cols(), kNumFrames - 3);
}

// Tests that processMarkers clamps maxFrames to remaining data when it exceeds available frames.
TEST(ProcessMarkersTest, ProcessMarkers_MaxFramesExceedsRemaining_ClampedToEnd) {
  auto [character, identity] = createTestCharacterWithIdentity();
  constexpr size_t kNumFrames = 5;
  auto markerData = createMarkerData(character, kNumFrames);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf result = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false,
      /*firstFrame=*/2,
      /*maxFrames=*/100);

  // maxFrames=100 exceeds the remaining 3 frames, so output should be clamped to 3
  EXPECT_EQ(result.cols(), kNumFrames - 2);
}

// Tests that the output motion has the expected number of rows (model parameters).
TEST(ProcessMarkersTest, ProcessMarkers_OutputRowsMatchModelParameters) {
  auto [character, identity] = createTestCharacterWithIdentity();
  auto markerData = createMarkerData(character, 5);
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf result = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false);

  EXPECT_EQ(result.rows(), character.parameterTransform.numAllModelParameters());
}

// Tests that processMarkerFile handles an invalid output file extension gracefully.
TEST(ProcessMarkersTest, ProcessMarkerFile_InvalidExtension_NoThrow) {
  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;
  ModelOptions modelOptions;

  // processMarkerFile should log an error and return early for unsupported extensions.
  EXPECT_NO_THROW(processMarkerFile(
      "input.c3d",
      "output.txt",
      trackingConfig,
      calibrationConfig,
      modelOptions,
      /*calibrate=*/false));
}

// Tests that processing markers at identity positions results in locators at the expected
// positions.
TEST(ProcessMarkersTest, ProcessMarkers_IdentityPoseMarkers_LocatorsAtCorrectPosition) {
  constexpr size_t kNumJoints = 5;
  constexpr size_t kNumFrames = 3;
  auto [character, identity] = createTestCharacterWithIdentity(kNumJoints);

  auto markerData = createIdentityPoseMarkerData(character, identity, kNumFrames);

  TrackingConfig trackingConfig;
  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf motion = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false);

  ASSERT_EQ(motion.cols(), kNumFrames);

  const JointParameters identityJointParams = character.parameterTransform.apply(identity);
  const SkeletonState identitySkelState(identityJointParams, character.skeleton);

  constexpr float kTolerance = 1e-3f;

  for (size_t frame = 0; frame < kNumFrames; ++frame) {
    const ModelParameters frameParams = motion.col(frame);
    const JointParameters frameJointParams = character.parameterTransform.apply(frameParams);
    const SkeletonState frameSkelState(frameJointParams, character.skeleton);

    for (size_t i = 0; i < character.locators.size(); ++i) {
      const auto& locator = character.locators[i];
      const Eigen::Vector3f expectedPos = computeLocatorWorldPosition(locator, identitySkelState);
      const Eigen::Vector3f actualPos = computeLocatorWorldPosition(locator, frameSkelState);

      EXPECT_NEAR(actualPos.x(), expectedPos.x(), kTolerance)
          << "Locator '" << locator.name << "' x mismatch at frame " << frame;
      EXPECT_NEAR(actualPos.y(), expectedPos.y(), kTolerance)
          << "Locator '" << locator.name << "' y mismatch at frame " << frame;
      EXPECT_NEAR(actualPos.z(), expectedPos.z(), kTolerance)
          << "Locator '" << locator.name << "' z mismatch at frame " << frame;
    }
  }
}

// Tests that processing linearly moving markers results in locators following the linear motion.
// Since we apply a uniform translation to all markers, the solver should be able to recover
// this motion via root translation. We verify that the average locator position error is small.
TEST(ProcessMarkersTest, ProcessMarkers_LinearlyMovingMarkers_LocatorsFollowMotion) {
  constexpr size_t kNumJoints = 5;
  constexpr size_t kNumFrames = 5;
  const Eigen::Vector3d kOffsetPerFrame(0.1, 0.0, 0.0);

  auto [character, identity] = createTestCharacterWithIdentity(kNumJoints);

  auto markerData =
      createLinearlyMovingMarkerData(character, identity, kNumFrames, kOffsetPerFrame);

  TrackingConfig trackingConfig;
  // Use lower regularization to allow the solver to better fit the marker positions.
  trackingConfig.regularization = 0.0f;
  // Increase max iterations to ensure convergence.
  trackingConfig.maxIter = 50;

  CalibrationConfig calibrationConfig;

  const Eigen::MatrixXf motion = processMarkers(
      character,
      identity,
      markerData,
      trackingConfig,
      calibrationConfig,
      /*calibrate=*/false);

  ASSERT_EQ(motion.cols(), kNumFrames);

  const JointParameters identityJointParams = character.parameterTransform.apply(identity);
  const SkeletonState identitySkelState(identityJointParams, character.skeleton);

  // Per-locator tolerance may be slightly relaxed due to optimization tradeoffs,
  // but we also verify the average error is small. The error grows with displacement,
  // so we use a percentage of the total displacement as the maximum average error threshold.
  constexpr float kPerLocatorTolerance = 0.08f;
  constexpr float kMaxAverageErrorFraction = 0.10f; // 10% of displacement magnitude

  for (size_t frame = 0; frame < kNumFrames; ++frame) {
    const ModelParameters frameParams = motion.col(frame);
    const JointParameters frameJointParams = character.parameterTransform.apply(frameParams);
    const SkeletonState frameSkelState(frameJointParams, character.skeleton);

    const Eigen::Vector3f frameOffset = kOffsetPerFrame.cast<float>() * static_cast<float>(frame);
    const float displacementMagnitude = frameOffset.norm();
    // For frame 0 (no displacement), use a small absolute tolerance.
    const float maxAverageError =
        (frame == 0) ? 1e-3f : kMaxAverageErrorFraction * displacementMagnitude;

    float totalError = 0.0f;
    for (size_t i = 0; i < character.locators.size(); ++i) {
      const auto& locator = character.locators[i];
      const Eigen::Vector3f basePos = computeLocatorWorldPosition(locator, identitySkelState);
      const Eigen::Vector3f expectedPos = basePos + frameOffset;
      const Eigen::Vector3f actualPos = computeLocatorWorldPosition(locator, frameSkelState);

      const float locatorError = (actualPos - expectedPos).norm();
      totalError += locatorError;

      EXPECT_LT(locatorError, kPerLocatorTolerance)
          << "Locator '" << locator.name << "' error " << locatorError << " at frame " << frame;
    }

    const float avgError = totalError / static_cast<float>(character.locators.size());
    EXPECT_LT(avgError, maxAverageError) << "Average locator error at frame " << frame;
  }
}
