/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/marker.h>
#include <momentum/marker_tracking/marker_gap_fill.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace momentum;

namespace {

/// Create a single marker with the given properties.
Marker makeMarker(
    const std::string& name,
    const Eigen::Vector3d& pos,
    bool occluded = false,
    float weight = 1.0f) {
  Marker m;
  m.name = name;
  m.pos = pos;
  m.occluded = occluded;
  m.confidence = weight;
  return m;
}

/// Create marker data for a single marker that moves linearly along the x-axis
/// at 1 unit/frame, with specified frames occluded.
///
/// @param numFrames Total number of frames.
/// @param occludedFrames Set of frame indices where the marker is occluded.
std::vector<std::vector<Marker>> createLinearMotionData(
    size_t numFrames,
    const std::vector<size_t>& occludedFrames) {
  std::vector<std::vector<Marker>> data(numFrames);
  std::set<size_t> occSet(occludedFrames.begin(), occludedFrames.end());

  for (size_t f = 0; f < numFrames; ++f) {
    const bool occ = occSet.count(f) > 0;
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(static_cast<double>(f), 0.0, 0.0), occ));
  }
  return data;
}

} // namespace

class MarkerGapFillTest : public ::testing::Test {
 protected:
  GapFillConfig config;

  void SetUp() override {
    config.enabled = true;
    config.maxGapFrames = 10;
    // Disable displacement-adaptive gaps by default so existing tests
    // have deterministic behavior regardless of marker displacement.
    config.maxGapFramesStationary = 10;
    config.minVisibleFrames = 0; // disable suppression by default
    config.blendOffFrames = 5;
    config.velocityWindowFrames = 3;
  }
};

TEST_F(MarkerGapFillTest, EmptyData) {
  std::vector<std::vector<Marker>> data;
  preprocessMarkerGaps(data, config);
  EXPECT_TRUE(data.empty());
}

TEST_F(MarkerGapFillTest, NoGaps) {
  auto data = createLinearMotionData(10, {});
  auto original = data;
  preprocessMarkerGaps(data, config);

  // Nothing should change
  for (size_t f = 0; f < data.size(); ++f) {
    EXPECT_FALSE(data[f][0].occluded);
    EXPECT_FLOAT_EQ(data[f][0].confidence, 1.0f);
    EXPECT_EQ(data[f][0].pos, original[f][0].pos);
  }
}

TEST_F(MarkerGapFillTest, SingleFrameGapInterpolation) {
  // Linear motion: 0,1,2,_,4,5 — gap at frame 3
  auto data = createLinearMotionData(6, {3});
  preprocessMarkerGaps(data, config);

  EXPECT_FALSE(data[3][0].occluded);
  EXPECT_FLOAT_EQ(data[3][0].confidence, 1.0f);
  // For linear motion with cubic Hermite, the interpolated position should be
  // very close to the true linear value (3.0)
  EXPECT_NEAR(data[3][0].pos.x(), 3.0, 0.01);
}

TEST_F(MarkerGapFillTest, MultiFrameGapInterpolation) {
  // Linear motion: 0,1,_,_,_,5,6,7 — gap at frames 2,3,4
  auto data = createLinearMotionData(8, {2, 3, 4});
  preprocessMarkerGaps(data, config);

  for (size_t f = 2; f <= 4; ++f) {
    EXPECT_FALSE(data[f][0].occluded);
    EXPECT_FLOAT_EQ(data[f][0].confidence, 1.0f);
    EXPECT_NEAR(data[f][0].pos.x(), static_cast<double>(f), 0.1)
        << "Gap-filled frame " << f << " should be near the linear position";
  }
}

TEST_F(MarkerGapFillTest, GapExceedsMaxTreatedAsPermanent) {
  config.maxGapFrames = 2;
  config.maxGapFramesStationary = 2; // match maxGapFrames to disable adaptive
  config.blendOffFrames = 0; // disable blend-off to isolate the test

  // Gap of 3 frames exceeds maxGapFrames=2
  auto data = createLinearMotionData(8, {2, 3, 4});
  preprocessMarkerGaps(data, config);

  // Gap should NOT be interpolated (treated as permanent, blend-off disabled)
  for (size_t f = 2; f <= 4; ++f) {
    EXPECT_TRUE(data[f][0].occluded) << "Frame " << f << " should remain occluded";
  }
}

TEST_F(MarkerGapFillTest, LongGapBothSidesBlendedOff) {
  config.maxGapFrames = 2;
  config.maxGapFramesStationary = 2; // match maxGapFrames to disable adaptive
  config.blendOffFrames = 3;

  // Gap of 10 frames with visible on both sides: 0,1,_x10_,12,13
  auto data = createLinearMotionData(14, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  preprocessMarkerGaps(data, config);

  // First 3 frames of gap (2,3,4) should be blended off from the trailing side
  for (size_t f = 2; f <= 4; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be filled (trailing blend)";
    EXPECT_GT(data[f][0].confidence, 0.0f);
    EXPECT_LT(data[f][0].confidence, 1.0f);
  }
  EXPECT_GT(data[2][0].confidence, data[3][0].confidence);
  EXPECT_GT(data[3][0].confidence, data[4][0].confidence);

  // Last 3 frames of gap (9,10,11) should be blended on from the leading side
  for (size_t f = 9; f <= 11; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be filled (leading blend)";
    EXPECT_GT(data[f][0].confidence, 0.0f);
    EXPECT_LT(data[f][0].confidence, 1.0f);
  }
  EXPECT_LT(data[9][0].confidence, data[10][0].confidence);
  EXPECT_LT(data[10][0].confidence, data[11][0].confidence);

  // Middle frames (5,6,7,8) should remain occluded
  for (size_t f = 5; f <= 8; ++f) {
    EXPECT_TRUE(data[f][0].occluded) << "Frame " << f << " should remain occluded (middle)";
  }
}

TEST_F(MarkerGapFillTest, TrailingDropoutBlendOff) {
  config.blendOffFrames = 3;

  // Marker visible for frames 0-4, then permanently occluded 5-9
  auto data = createLinearMotionData(10, {5, 6, 7, 8, 9});
  preprocessMarkerGaps(data, config);

  // Frames 5,6,7 should be extrapolated with decreasing weight
  for (size_t f = 5; f <= 7; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be filled";
    EXPECT_GT(data[f][0].confidence, 0.0f) << "Frame " << f << " should have positive weight";
    EXPECT_LT(data[f][0].confidence, 1.0f) << "Frame " << f << " should have reduced weight";
    // Position should be extrapolated linearly
    EXPECT_NEAR(data[f][0].pos.x(), static_cast<double>(f), 0.1)
        << "Frame " << f << " extrapolated position";
  }

  // Weight should decrease monotonically
  EXPECT_GT(data[5][0].confidence, data[6][0].confidence);
  EXPECT_GT(data[6][0].confidence, data[7][0].confidence);

  // Frames beyond blend window should remain occluded
  EXPECT_TRUE(data[8][0].occluded);
  EXPECT_TRUE(data[9][0].occluded);
}

TEST_F(MarkerGapFillTest, LeadingDropoutBlendOff) {
  config.blendOffFrames = 3;

  // Marker occluded for frames 0-4, then visible 5-9
  auto data = createLinearMotionData(10, {0, 1, 2, 3, 4});
  preprocessMarkerGaps(data, config);

  // Frames 2,3,4 should be filled (last 3 of the 5-frame gap)
  for (size_t f = 2; f <= 4; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be filled";
    EXPECT_GT(data[f][0].confidence, 0.0f) << "Frame " << f << " should have positive weight";
    EXPECT_LT(data[f][0].confidence, 1.0f) << "Frame " << f << " should have reduced weight";
  }

  // Weight should increase toward the first visible frame
  EXPECT_LT(data[2][0].confidence, data[3][0].confidence);
  EXPECT_LT(data[3][0].confidence, data[4][0].confidence);

  // Frames beyond blend window should remain occluded
  EXPECT_TRUE(data[0][0].occluded);
  EXPECT_TRUE(data[1][0].occluded);
}

TEST_F(MarkerGapFillTest, AllFramesOccluded) {
  auto data = createLinearMotionData(5, {0, 1, 2, 3, 4});
  preprocessMarkerGaps(data, config);

  // Everything should stay occluded — no visible anchors to interpolate from
  for (size_t f = 0; f < 5; ++f) {
    EXPECT_TRUE(data[f][0].occluded) << "Frame " << f << " should remain occluded";
  }
}

TEST_F(MarkerGapFillTest, MultipleMarkersIndependentGaps) {
  // Create 10 frames with 2 markers
  std::vector<std::vector<Marker>> data(10);
  for (size_t f = 0; f < 10; ++f) {
    data[f].push_back(makeMarker("A", Eigen::Vector3d(static_cast<double>(f), 0, 0), false));
    data[f].push_back(makeMarker("B", Eigen::Vector3d(0, static_cast<double>(f), 0), false));
  }

  // Occlude marker A at frames 3-4 (temporary gap)
  data[3][0].occluded = true;
  data[4][0].occluded = true;

  // Occlude marker B at frames 7-9 (trailing dropout)
  data[7][1].occluded = true;
  data[8][1].occluded = true;
  data[9][1].occluded = true;

  preprocessMarkerGaps(data, config);

  // Marker A gap should be interpolated
  EXPECT_FALSE(data[3][0].occluded);
  EXPECT_FALSE(data[4][0].occluded);
  EXPECT_FLOAT_EQ(data[3][0].confidence, 1.0f);
  EXPECT_FLOAT_EQ(data[4][0].confidence, 1.0f);

  // Marker B should be blended off
  EXPECT_FALSE(data[7][1].occluded);
  EXPECT_GT(data[7][1].confidence, 0.0f);
  EXPECT_LT(data[7][1].confidence, 1.0f);

  // Verify A's Y and B's X are unchanged
  EXPECT_NEAR(data[3][0].pos.y(), 0.0, 0.01);
  EXPECT_NEAR(data[7][1].pos.x(), 0.0, 0.01);
}

TEST_F(MarkerGapFillTest, InterpolatedWeightIsOne) {
  auto data = createLinearMotionData(10, {4, 5});
  preprocessMarkerGaps(data, config);

  // Interpolated frames should have full weight
  EXPECT_FLOAT_EQ(data[4][0].confidence, 1.0f);
  EXPECT_FLOAT_EQ(data[5][0].confidence, 1.0f);
}

TEST_F(MarkerGapFillTest, DisabledGapFill) {
  config.maxGapFrames = 0;
  config.blendOffFrames = 0;

  auto data = createLinearMotionData(10, {3, 4, 5});
  preprocessMarkerGaps(data, config);

  // Nothing should change
  EXPECT_TRUE(data[3][0].occluded);
  EXPECT_TRUE(data[4][0].occluded);
  EXPECT_TRUE(data[5][0].occluded);
}

TEST_F(MarkerGapFillTest, StationaryMarkerGetsExtendedGap) {
  config.maxGapFrames = 5;
  config.maxGapFramesStationary = 20;
  config.maxGapDisplacement = 5.0; // 5cm
  config.blendOffFrames = 0; // disable blend-off to isolate

  // Marker at constant position (0,0,0), gap of 10 frames (> maxGapFrames=5
  // but < maxGapFramesStationary=20). Displacement = 0, so effective max gap
  // should be maxGapFramesStationary.
  const size_t numFrames = 20;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(0.0, 0.0, 0.0), f >= 5 && f <= 14));
  }

  preprocessMarkerGaps(data, config);

  // All gap frames should be interpolated (stationary → extended gap limit)
  for (size_t f = 5; f <= 14; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be interpolated";
    EXPECT_FLOAT_EQ(data[f][0].confidence, 1.0f);
  }
}

TEST_F(MarkerGapFillTest, LargeDisplacementUsesStandardMaxGap) {
  config.maxGapFrames = 5;
  config.maxGapFramesStationary = 20;
  config.maxGapDisplacement = 5.0; // 5cm
  config.blendOffFrames = 0; // disable blend-off to isolate

  // Marker moves 10cm during a 10-frame gap (> maxGapFrames=5).
  // Displacement = 10cm > threshold 5cm, so effective max gap = maxGapFrames.
  const size_t numFrames = 20;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    // Position: 0 for frames 0-4, then jump to 10cm for frames 15-19
    const double x = (f < 5) ? 0.0 : 10.0;
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(x, 0, 0), f >= 5 && f <= 14));
  }

  preprocessMarkerGaps(data, config);

  // Gap should NOT be interpolated (displacement too large, exceeds maxGapFrames)
  for (size_t f = 5; f <= 14; ++f) {
    EXPECT_TRUE(data[f][0].occluded) << "Frame " << f << " should remain occluded";
  }
}

TEST_F(MarkerGapFillTest, IntermediateDisplacementBlendsMaxGap) {
  config.maxGapFrames = 5;
  config.maxGapFramesStationary = 20;
  config.maxGapDisplacement = 10.0; // 10cm
  config.blendOffFrames = 0;

  // Marker moves 5cm during a 12-frame gap. Displacement = half of threshold,
  // so effective max gap ≈ 12.5 (midpoint between 20 and 5). A 12-frame gap
  // should just barely fit.
  const size_t numFrames = 25;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    const double x = (f < 5) ? 0.0 : 5.0;
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(x, 0, 0), f >= 5 && f <= 16));
  }

  preprocessMarkerGaps(data, config);

  // Effective max gap = 20 * 0.5 + 5 * 0.5 = 12.5 → rounds to 13.
  // Gap of 12 frames should be interpolated.
  for (size_t f = 5; f <= 16; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be interpolated";
    EXPECT_FLOAT_EQ(data[f][0].confidence, 1.0f);
  }
}

TEST_F(MarkerGapFillTest, ShortVisibleSegmentSuppressed) {
  config.maxGapFrames = 5;
  config.maxGapFramesStationary = 5;
  config.minVisibleFrames = 10;
  config.blendOffFrames = 0;

  // Marker: 20 vis, 15 occ, 5 vis, 15 occ, 20 vis.
  // Gap fill runs first but can't interpolate (gaps > maxGapFrames=5).
  // Then suppression fires: the 5-frame visible segment (frames 35-39)
  // is too short (< minVisibleFrames=10) and bounded by occluded on
  // both sides, so it gets suppressed.
  const size_t numFrames = 75;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    const bool occ = (f >= 20 && f < 35) || (f >= 40 && f < 55);
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(0.0, 0.0, 0.0), occ));
  }

  preprocessMarkerGaps(data, config);

  // The short visible segment should be suppressed (occluded + weight=0)
  for (size_t f = 35; f < 40; ++f) {
    EXPECT_TRUE(data[f][0].occluded) << "Frame " << f << " should be suppressed";
    EXPECT_FLOAT_EQ(data[f][0].confidence, 0.0f);
  }

  // Surrounding visible segments should be unchanged
  EXPECT_FALSE(data[0][0].occluded);
  EXPECT_FALSE(data[19][0].occluded);
  EXPECT_FALSE(data[55][0].occluded);
  EXPECT_FALSE(data[74][0].occluded);
}

TEST_F(MarkerGapFillTest, ShortVisibleSegmentKeptWhenGapsFilled) {
  config.maxGapFrames = 10;
  config.maxGapFramesStationary = 10;
  config.minVisibleFrames = 10;
  config.blendOffFrames = 0;

  // Marker: 10 vis, 3 occ, 5 vis, 3 occ, 10 vis.
  // Gap fill runs first: both 3-frame gaps are interpolated (< maxGapFrames=10).
  // After gap fill, the 5-frame segment is no longer between occluded regions,
  // so suppression doesn't fire.
  const size_t numFrames = 31;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    const bool occ = (f >= 10 && f < 13) || (f >= 18 && f < 21);
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(0.0, 0.0, 0.0), occ));
  }

  preprocessMarkerGaps(data, config);

  // Everything should be visible (gaps filled, no suppression needed)
  for (size_t f = 0; f < numFrames; ++f) {
    EXPECT_FALSE(data[f][0].occluded) << "Frame " << f << " should be visible";
  }
}

TEST_F(MarkerGapFillTest, HermitePreservesVelocityContinuity) {
  // Quadratic motion: pos(f) = (f^2, 0, 0)
  // This tests that Hermite interpolation with velocity estimation produces
  // smooth results (no kinks at boundaries).
  const size_t numFrames = 20;
  std::vector<std::vector<Marker>> data(numFrames);
  for (size_t f = 0; f < numFrames; ++f) {
    const auto x = static_cast<double>(f * f);
    data[f].push_back(makeMarker("M1", Eigen::Vector3d(x, 0, 0), false));
  }

  // Occlude frames 8-11 (gap of 4)
  for (size_t f = 8; f <= 11; ++f) {
    data[f][0].occluded = true;
  }

  preprocessMarkerGaps(data, config);

  // Check that the interpolated values are reasonable (within ~10% of the
  // quadratic since velocity estimation is approximate)
  for (size_t f = 8; f <= 11; ++f) {
    const auto expected = static_cast<double>(f * f);
    EXPECT_FALSE(data[f][0].occluded);
    EXPECT_NEAR(data[f][0].pos.x(), expected, expected * 0.15 + 1.0)
        << "Frame " << f << ": Hermite should approximate quadratic motion";
  }

  // Velocity continuity: the finite differences across the gap boundary
  // should be reasonably smooth (no huge jumps)
  const double velBefore = data[8][0].pos.x() - data[7][0].pos.x();
  const double velAt7 = data[7][0].pos.x() - data[6][0].pos.x();
  // The velocity change across the boundary shouldn't be too abrupt
  EXPECT_LT(std::abs(velBefore - velAt7), velAt7 * 0.5)
      << "Velocity should be continuous across gap boundary";
}
