/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/marker.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace momentum;

namespace {

Marker visible(const std::string& name, double x, float confidence = 1.0f) {
  return {.name = name, .pos = Vector3d(x, 0.0, 0.0), .occluded = false, .confidence = confidence};
}

Marker occluded(const std::string& name) {
  return {.name = name, .pos = Vector3d::Zero(), .occluded = true, .confidence = 1.0f};
}

} // namespace

TEST(MarkerTest, TracksHoldTheKeptObservationsOfEachMarkerInFirstSeenOrder) {
  const std::vector<std::vector<Marker>> frames = {
      {occluded("a"), visible("b", 1.0)},
      {visible("a", 2.0, 0.5f), visible("b", 3.0, 1e-4f)},
      {visible("b", 4.0), visible("a", 5.0)},
      {occluded("c")},
  };

  const std::vector<MarkerTrack> all = groupMarkerTracks(frames);
  ASSERT_EQ(all.size(), 2);
  EXPECT_EQ(all[0].name, "b");
  EXPECT_EQ(all[0].frames, (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(all[0].positions, (std::vector<Vector3d>{{1.0, 0, 0}, {3.0, 0, 0}, {4.0, 0, 0}}));
  EXPECT_EQ(all[0].confidences, (std::vector<float>{1.0f, 1e-4f, 1.0f}));
  EXPECT_EQ(all[1].name, "a");
  EXPECT_EQ(all[1].frames, (std::vector<size_t>{1, 2}));
  EXPECT_EQ(all[1].positions, (std::vector<Vector3d>{{2.0, 0, 0}, {5.0, 0, 0}}));
  EXPECT_EQ(all[1].confidences, (std::vector<float>{0.5f, 1.0f}));

  // A minimum confidence leaves out the low-confidence observation of "b".
  const std::vector<MarkerTrack> confident = groupMarkerTracks(frames, 1e-3f);
  ASSERT_EQ(confident.size(), 2);
  EXPECT_EQ(confident[0].frames, (std::vector<size_t>{0, 2}));
  EXPECT_EQ(confident[0].positions, (std::vector<Vector3d>{{1.0, 0, 0}, {4.0, 0, 0}}));
  EXPECT_EQ(confident[1].frames, all[1].frames);
}
