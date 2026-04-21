/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/filesystem.h"
#include "momentum/io/marker/marker_io.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace momentum;

namespace {

std::string getMarkerFile() {
  auto envVar = GetEnvVar("TEST_RESOURCES_PATH");
  const auto markerFilePath = filesystem::path(envVar.value()) / "markers.c3d";
  return markerFilePath.string();
}

std::vector<std::byte> readFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::istreambuf_iterator<char> begin(f);
  std::istreambuf_iterator<char> end;
  std::vector<char> chars(begin, end);
  std::vector<std::byte> bytes(chars.size());
  std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

void expectSequencesEqual(
    const std::vector<MarkerSequence>& a,
    const std::vector<MarkerSequence>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].name, b[i].name);
    EXPECT_FLOAT_EQ(a[i].fps, b[i].fps);
    ASSERT_EQ(a[i].frames.size(), b[i].frames.size());
    for (size_t f = 0; f < a[i].frames.size(); ++f) {
      ASSERT_EQ(a[i].frames[f].size(), b[i].frames[f].size());
      for (size_t m = 0; m < a[i].frames[f].size(); ++m) {
        EXPECT_EQ(a[i].frames[f][m].name, b[i].frames[f][m].name);
        EXPECT_EQ(a[i].frames[f][m].occluded, b[i].frames[f][m].occluded);
        if (!a[i].frames[f][m].occluded) {
          EXPECT_TRUE(a[i].frames[f][m].pos.isApprox(b[i].frames[f][m].pos));
        }
      }
    }
  }
}

TEST(MarkerIOTest, testLoadMarkers) {
  const std::string markerFile = getMarkerFile();
  const std::vector<MarkerSequence> actorSequences = loadMarkers(markerFile);
  // the file has one actor sequence with total 89 frames and 36 markers per frame
  EXPECT_EQ(actorSequences.size(), 1);
  EXPECT_EQ(actorSequences[0].frames.size(), 89);
  EXPECT_EQ(actorSequences[0].frames[0].size(), 36);
}

TEST(MarkerIOTest, testFindMainSubject) {
  // 4 visible markers
  std::vector<Marker> markersFrame0 = {
      {"RFT1", {0.0, 0.0, 0.0}, false},
      {"RFT2", {1.0, 0.0, 0.0}, false},
      {"RFT3", {1.0, 1.0, 0.0}, false},
      {"RFT4", {1.0, 1.0, 0.0}, false}};

  // 3 visible markers
  std::vector<Marker> markersFrame1 = {
      {"RFT1", {0.0, 0.0, 0.5}, false},
      {"RFT2", {1.0, 0.0, 0.5}, true},
      {"RFT3", {1.0, 1.0, 0.5}, false},
      {"RFT4", {1.0, 1.0, 0.5}, false}};

  // 1 visible markers
  std::vector<Marker> markersFrame2 = {
      {"RFT1", {0.0, 0.0, 0.5}, false},
      {"RFT2", {1.0, 0.0, 0.5}, true},
      {"RFT3", {1.0, 1.0, 0.5}, true},
      {"RFT4", {1.0, 1.0, 0.5}, true}};

  // actor-0 has more visible markers on the first frame, but actor-1 has more visible markers on
  // the second frame so it's the main subject.
  std::vector<MarkerSequence> actorSequences = {
      {
          "actor-0",
          {markersFrame1},
      },
      {
          "actor-1",
          {markersFrame2, markersFrame0},
      }};

  const int mainSubjectID = findMainSubjectIndex(actorSequences);
  EXPECT_EQ(mainSubjectID, 1);
}

TEST(MarkerIOTest, testLoadMarkersForMainSubject) {
  const std::string markerFile = getMarkerFile();
  std::optional<MarkerSequence> mainSubjectSequence = loadMarkersForMainSubject(markerFile);
  // the file has one actor sequence with total 89 frames and 36 markers per frame
  EXPECT_TRUE(mainSubjectSequence.has_value());
  EXPECT_EQ(mainSubjectSequence->frames.size(), 89);
  EXPECT_EQ(mainSubjectSequence->frames[0].size(), 36);
}

TEST(MarkerIOTest, testLoadMarkersEmpty) {
  const auto actorSequences = loadMarkers("");
  EXPECT_EQ(actorSequences.size(), 0);
}

// The c3d bytes path is gated on MOMENTUM_WITH_EZC3D_ISTREAM (only defined when
// the io_marker library is built against our patched ezc3d). In OSS builds the
// overload is a documented no-op, so file-vs-bytes equivalence tests are skipped.
#ifdef MOMENTUM_WITH_EZC3D_ISTREAM
TEST(MarkerIOTest, testLoadMarkersFromBytesC3d) {
  const std::string markerFile = getMarkerFile();
  const auto bytes = readFileBytes(markerFile);
  ASSERT_FALSE(bytes.empty()) << "fixture markers.c3d not found at " << markerFile;

  const auto fromFile = loadMarkers(markerFile);
  const auto fromBytes = loadMarkers(bytes, ".c3d");
  expectSequencesEqual(fromFile, fromBytes);
}

TEST(MarkerIOTest, testLoadMarkersForMainSubjectFromBytes) {
  const std::string markerFile = getMarkerFile();
  const auto bytes = readFileBytes(markerFile);
  ASSERT_FALSE(bytes.empty());

  const auto fromFile = loadMarkersForMainSubject(markerFile);
  const auto fromBytes = loadMarkersForMainSubject(bytes, ".c3d");
  ASSERT_TRUE(fromFile.has_value());
  ASSERT_TRUE(fromBytes.has_value());
  EXPECT_EQ(fromFile->frames.size(), fromBytes->frames.size());
  EXPECT_EQ(fromFile->frames[0].size(), fromBytes->frames[0].size());
}
#endif

TEST(MarkerIOTest, testLoadMarkersFromBytesUnknownFormat) {
  const std::vector<std::byte> bytes(16, std::byte{0});
  const auto sequences = loadMarkers(bytes, ".xyz");
  EXPECT_EQ(sequences.size(), 0);
}

TEST(MarkerIOTest, testLoadMarkersFromBytesEmpty) {
  const std::vector<std::byte> bytes;
  // empty buffer for any format -> graceful empty return, no crash
  EXPECT_EQ(loadMarkers(bytes, ".c3d").size(), 0);
  EXPECT_EQ(loadMarkers(bytes, ".trc").size(), 1); // trc returns one empty sequence
}

} // namespace
