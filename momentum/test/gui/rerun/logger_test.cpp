/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/character_state.h>
#include <momentum/character/marker.h>
#include <momentum/gui/rerun/logger.h>
#include <momentum/gui/rerun/logging_redirect.h>
#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/helpers/sanitizer_utils.h>

#include <gtest/gtest.h>
#include <rerun.hpp>

using namespace momentum;

// Note on assertion level: The logger functions call rec_->log() / rec_->log_static()
// which is a fire-and-forget API — the rerun RecordingStream provides no inspection
// or readback mechanism for logged data. EXPECT_NO_THROW is the strongest assertion
// possible for these integration tests. The value assertions for data conversion
// correctness live in eigen_adapters_test.cpp and rerun_compat_test.cpp instead.

class RerunLoggerTest : public testing::Test {
 protected:
  void SetUp() override {
    // The rerun Rust FFI layer crashes under sanitizer builds due to a
    // null-pointer precondition in slice::from_raw_parts_mut inside
    // rr_recording_stream_log_impl. Skip all tests under sanitizers.
    if (isSanitizerBuild()) {
      GTEST_SKIP() << "Rerun SDK crashes under sanitizer builds (Rust FFI issue)";
    }
    rec_ = std::make_unique<rerun::RecordingStream>("logger_test");
    character_ = createTestCharacter(5);
    characterState_ = CharacterState(character_);
  }

  std::unique_ptr<rerun::RecordingStream> rec_;
  Character character_;
  CharacterState characterState_;
};

// --- logMesh tests ---

TEST_F(RerunLoggerTest, LogMesh) {
  ASSERT_NE(characterState_.meshState, nullptr);
  EXPECT_NO_THROW(logMesh(*rec_, "test/mesh", *characterState_.meshState));
}

TEST_F(RerunLoggerTest, LogMeshWithColor) {
  ASSERT_NE(characterState_.meshState, nullptr);
  rerun::Color color(200, 100, 50);
  EXPECT_NO_THROW(logMesh(*rec_, "test/mesh_colored", *characterState_.meshState, color));
}

// --- logJoints tests ---

TEST_F(RerunLoggerTest, LogJoints) {
  EXPECT_NO_THROW(logJoints(
      *rec_, "test/joints", character_.skeleton, characterState_.skeletonState.jointState));
}

// --- logMarkers tests ---

TEST_F(RerunLoggerTest, LogMarkers) {
  std::vector<Marker> markers = {
      {.name = "head", .pos = {0, 170, 0}, .occluded = false},
      {.name = "lhand", .pos = {-50, 100, 0}, .occluded = false},
      {.name = "rhand", .pos = {50, 100, 0}, .occluded = false},
  };
  EXPECT_NO_THROW(logMarkers(*rec_, "test/markers", markers));
}

TEST_F(RerunLoggerTest, LogMarkersWithOcclusion) {
  std::vector<Marker> markers = {
      {.name = "visible", .pos = {0, 170, 0}, .occluded = false},
      {.name = "hidden", .pos = {0, 0, 0}, .occluded = true},
  };
  EXPECT_NO_THROW(logMarkers(*rec_, "test/markers_occluded", markers));
}

TEST_F(RerunLoggerTest, LogMarkersEmpty) {
  std::vector<Marker> markers;
  EXPECT_NO_THROW(logMarkers(*rec_, "test/markers_empty", markers));
}

// --- logLocators tests ---

TEST_F(RerunLoggerTest, LogLocators) {
  EXPECT_NO_THROW(
      logLocators(*rec_, "test/locators", character_.locators, characterState_.locatorState));
}

TEST_F(RerunLoggerTest, LogLocatorsEmpty) {
  LocatorList emptyLocators;
  LocatorState emptyState;
  EXPECT_NO_THROW(logLocators(*rec_, "test/locators_empty", emptyLocators, emptyState));
}

// --- logMarkerLocatorCorrespondence tests ---

TEST_F(RerunLoggerTest, LogMarkerLocatorCorrespondenceEmpty) {
  std::map<std::string, size_t> emptyLookup;
  std::vector<Marker> markers;
  EXPECT_NO_THROW(logMarkerLocatorCorrespondence(
      *rec_, "test/correspondence", emptyLookup, characterState_.locatorState, markers, 10.0f));
}

TEST_F(RerunLoggerTest, LogMarkerLocatorCorrespondence) {
  // Build a lookup from locator names to their indices
  std::map<std::string, size_t> locatorLookup;
  for (size_t i = 0; i < character_.locators.size(); ++i) {
    locatorLookup[character_.locators[i].name] = i;
  }

  // Create markers that correspond to locator positions
  std::vector<Marker> markers;
  for (const auto& locator : character_.locators) {
    Marker m;
    m.name = locator.name;
    m.pos = characterState_.locatorState.position[locatorLookup[locator.name]].cast<double>();
    m.occluded = false;
    markers.push_back(m);
  }

  EXPECT_NO_THROW(logMarkerLocatorCorrespondence(
      *rec_, "test/correspondence", locatorLookup, characterState_.locatorState, markers, 10.0f));
}

// --- logCollisionGeometry tests ---

TEST_F(RerunLoggerTest, LogCollisionGeometry) {
  ASSERT_NE(character_.collision, nullptr);
  EXPECT_NO_THROW(logCollisionGeometry(
      *rec_, "test/collision", *character_.collision, characterState_.skeletonState));
}

// --- logBvh tests ---

TEST_F(RerunLoggerTest, LogBvh) {
  ASSERT_NE(character_.collision, nullptr);
  EXPECT_NO_THROW(logBvh(*rec_, "test/bvh", *character_.collision, characterState_.skeletonState));
}

// --- logCharacter tests ---

TEST_F(RerunLoggerTest, LogCharacter) {
  EXPECT_NO_THROW(logCharacter(*rec_, "test/character", character_, characterState_));
}

TEST_F(RerunLoggerTest, LogCharacterWithColor) {
  rerun::Color red(255, 0, 0);
  EXPECT_NO_THROW(logCharacter(*rec_, "test/character_red", character_, characterState_, red));
}

// --- logModelParams tests ---

TEST_F(RerunLoggerTest, LogModelParams) {
  const auto& paramNames = character_.parameterTransform.name;
  const Eigen::VectorXf params =
      Eigen::VectorXf::Zero(character_.parameterTransform.numAllModelParameters());
  EXPECT_NO_THROW(logModelParams(*rec_, "test/world", "test/pose", paramNames, params));
}

TEST_F(RerunLoggerTest, LogModelParamsWithValues) {
  const auto& paramNames = character_.parameterTransform.name;
  Eigen::VectorXf params =
      Eigen::VectorXf::Ones(character_.parameterTransform.numAllModelParameters());
  EXPECT_NO_THROW(logModelParams(*rec_, "test/world", "test/pose", paramNames, params));
}

// --- logJointParams tests ---

TEST_F(RerunLoggerTest, LogJointParams) {
  const auto jointNames = character_.skeleton.getJointNames();
  const Eigen::VectorXf params =
      Eigen::VectorXf::Zero(character_.skeleton.joints.size() * kParametersPerJoint);
  EXPECT_NO_THROW(logJointParams(*rec_, "test/world", "test/pose", jointNames, params));
}

// --- logModelParamNames tests ---

TEST_F(RerunLoggerTest, LogModelParamNames) {
  const auto& paramNames = character_.parameterTransform.name;
  EXPECT_NO_THROW(logModelParamNames(*rec_, "test/world", "test/pose", paramNames));
}

// --- logJointParamNames tests ---

TEST_F(RerunLoggerTest, LogJointParamNames) {
  const auto jointNames = character_.skeleton.getJointNames();
  EXPECT_NO_THROW(logJointParamNames(*rec_, "test/world", "test/pose", jointNames));
}

// --- logModelParamsColumns tests ---

TEST_F(RerunLoggerTest, LogModelParamsColumns) {
  const auto& paramNames = character_.parameterTransform.name;
  const Eigen::Index nParams = character_.parameterTransform.numAllModelParameters();
  const int nFrames = 3;

  Eigen::MatrixXf allParams = Eigen::MatrixXf::Random(nParams, nFrames);
  std::vector<int64_t> frameIndices = {0, 1, 2};
  std::vector<double> times = {0.0, 0.033, 0.066};

  EXPECT_NO_THROW(logModelParamsColumns(
      *rec_, "test/world", "test/pose", paramNames, allParams, frameIndices, times));
}

TEST_F(RerunLoggerTest, LogModelParamsColumnsEmpty) {
  const auto& paramNames = character_.parameterTransform.name;
  Eigen::MatrixXf emptyParams(paramNames.size(), 0);
  std::vector<int64_t> frameIndices;
  std::vector<double> times;

  EXPECT_NO_THROW(logModelParamsColumns(
      *rec_, "test/world", "test/pose", paramNames, emptyParams, frameIndices, times));
}

// --- logJointParamsColumns tests ---

TEST_F(RerunLoggerTest, LogJointParamsColumns) {
  const auto jointNames = character_.skeleton.getJointNames();
  const auto nJointParams =
      static_cast<Eigen::Index>(character_.skeleton.joints.size() * kParametersPerJoint);
  const int nFrames = 3;

  Eigen::MatrixXf allJointParams = Eigen::MatrixXf::Random(nJointParams, nFrames);
  std::vector<int64_t> frameIndices = {0, 1, 2};
  std::vector<double> times = {0.0, 0.033, 0.066};

  EXPECT_NO_THROW(logJointParamsColumns(
      *rec_, "test/world", "test/pose", jointNames, allJointParams, frameIndices, times));
}

TEST_F(RerunLoggerTest, LogJointParamsColumnsEmpty) {
  const auto jointNames = character_.skeleton.getJointNames();
  Eigen::MatrixXf emptyParams(jointNames.size() * kParametersPerJoint, 0);
  std::vector<int64_t> frameIndices;
  std::vector<double> times;

  EXPECT_NO_THROW(logJointParamsColumns(
      *rec_, "test/world", "test/pose", jointNames, emptyParams, frameIndices, times));
}

// --- redirectLogsToRerun tests ---

TEST_F(RerunLoggerTest, RedirectLogsToRerun) {
  // Without XR_LOGGER defined, this should return false
  const bool result = redirectLogsToRerun(*rec_);
#if !defined(MOMENTUM_WITH_XR_LOGGER)
  EXPECT_FALSE(result);
#else
  // With XR_LOGGER, the first call should succeed
  EXPECT_TRUE(result);
  // Subsequent calls should also return true (already redirected)
  EXPECT_TRUE(redirectLogsToRerun(*rec_));
#endif
}
