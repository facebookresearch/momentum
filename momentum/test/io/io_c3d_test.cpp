/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/filesystem.h"
#include "momentum/io/marker/c3d_io.h"
#include "momentum/test/io/io_helpers.h"

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace momentum;

namespace {

constexpr auto kTolerance = std::numeric_limits<double>::epsilon();
constexpr auto kIntRealTolerance = 2e-4;

void loadC3dFiles(const filesystem::path& subgroup, std::vector<filesystem::path>& pathsOut) {
  auto envVar = GetEnvVar("TEST_RESOURCES_PATH");
  ASSERT_TRUE(envVar);

  const auto folderPath = filesystem::path(envVar.value()) / subgroup;
  constexpr auto kExt = ".c3d";

  ASSERT_TRUE(filesystem::exists(folderPath));
  ASSERT_TRUE(filesystem::is_directory(folderPath));
  pathsOut.clear();
  for (const auto& entry : filesystem::recursive_directory_iterator(folderPath)) {
    if (filesystem::is_regular_file(entry) && entry.path().extension() == kExt) {
      pathsOut.push_back(entry.path());
    }
  }
}

template <typename Type, int Size>
void checkVector(
    const Eigen::Matrix<Type, Size, 1>& a,
    const Eigen::Matrix<Type, Size, 1>& b,
    const double tolerance) {
  for (auto i = 0; i < Size; i++) {
    EXPECT_NEAR(a[i], b[i], tolerance);
  }
}

void checkMarkers(const Marker& refMarker, const Marker& testMarker, const double tolerance) {
  EXPECT_EQ(refMarker.name, testMarker.name);
  EXPECT_EQ(refMarker.occluded, testMarker.occluded);
  checkVector(refMarker.pos, testMarker.pos, tolerance);
}

void checkFrame(
    const std::vector<Marker>& refFrame,
    const std::vector<Marker>& testFrame,
    const double tolerance) {
  EXPECT_EQ(refFrame.size(), testFrame.size());
  for (auto i = 0; i < refFrame.size(); i++) {
    checkMarkers(refFrame[i], testFrame[i], tolerance);
  }
}

// Check if all the animation sequences in the provided paths are the same.
// Return the first sequence for further checking.
void checkAllTheSame(
    const std::vector<filesystem::path>& paths,
    const double tolerance,
    std::vector<MarkerSequence>& sequencesOut) {
  auto& firstSequences = sequencesOut;
  firstSequences.clear();

  for (const auto& path : paths) {
    std::vector<MarkerSequence> newSequences;
    EXPECT_NO_THROW(newSequences = loadC3d(filesystem::absolute(path).string()))
        << "load c3d from " << path.string();

    if (firstSequences.size() == 0) {
      firstSequences = newSequences;
    } else {
      ASSERT_EQ(firstSequences.size(), newSequences.size());

      for (auto actorId = 0; actorId < firstSequences.size(); actorId++) {
        const auto& firstSequence = firstSequences[actorId];
        const auto& sequenceName = firstSequence.name;
        auto newSequenceIter = std::find_if(
            newSequences.begin(),
            newSequences.end(),
            [sequenceName](const MarkerSequence& sequence) {
              return sequence.name == sequenceName;
            });
        ASSERT_TRUE(newSequenceIter != newSequences.end());
        ASSERT_EQ(newSequenceIter->fps, firstSequence.fps);
        ASSERT_EQ(newSequenceIter->frames.size(), firstSequence.frames.size());

        for (auto frameId = 0; frameId < newSequenceIter->frames.size(); frameId++) {
          checkFrame(firstSequence.frames[frameId], newSequenceIter->frames[frameId], tolerance);
        }
      }
    }
  }
}

} // namespace

TEST(C3DTest, ManufactureTest) {
  std::vector<filesystem::path> paths;
  loadC3dFiles(filesystem::path("manufacture"), paths);
  EXPECT_EQ(paths.size(), 4);

  for (const auto& path : paths) {
    std::vector<MarkerSequence> result;
    EXPECT_NO_THROW(result = loadC3d(path.string())) << "load c3d from " << path.string();
    EXPECT_EQ(result.size(), 1);
    auto frames = result[0].frames;
    auto fps = result[0].fps;
    ASSERT_GT(frames.size(), 0.0f);
    ASSERT_GT(fps, 0.0f);
  }
}

TEST(C3DTest, CompatibilityTest) {
  std::vector<filesystem::path> paths;
  loadC3dFiles(filesystem::path("compatibility_test"), paths);
  EXPECT_EQ(paths.size(), 5);

  std::vector<MarkerSequence> testSequences;
  checkAllTheSame(paths, kTolerance, testSequences);

  ASSERT_EQ(testSequences.size(), 1);
  const auto& firstSequence = testSequences[0];
  ASSERT_EQ(firstSequence.fps, 50.0f);
  ASSERT_EQ(firstSequence.frames.size(), 450);
  ASSERT_EQ(firstSequence.name, "");
  const auto& characterTemplate = firstSequence.frames[0];
  ASSERT_EQ(characterTemplate.size(), 26);
}

TEST(C3DTest, FileFormatVariantTest) {
  const auto kFileFormatFolder = filesystem::path("file_format_variant");

  std::vector<filesystem::path> intFilePaths;
  loadC3dFiles(kFileFormatFolder / "int", intFilePaths);
  EXPECT_EQ(intFilePaths.size(), 2);

  std::vector<MarkerSequence> testSequences;
  checkAllTheSame(intFilePaths, kTolerance, testSequences);

  std::vector<filesystem::path> realFilePaths;
  loadC3dFiles(kFileFormatFolder / "real", realFilePaths);
  EXPECT_EQ(realFilePaths.size(), 2);
  checkAllTheSame(realFilePaths, kTolerance, testSequences);

  checkAllTheSame({intFilePaths[0], realFilePaths[0]}, kIntRealTolerance, testSequences);

  ASSERT_EQ(testSequences.size(), 1);
  const auto& firstSequence = testSequences[0];
  ASSERT_EQ(firstSequence.fps, 50.0f);
  ASSERT_EQ(firstSequence.frames.size(), 450);
  ASSERT_EQ(firstSequence.name, "");
  const auto& characterTemplate = firstSequence.frames[0];
  ASSERT_EQ(characterTemplate.size(), 26);
}

TEST(C3DTest, DataFormatVariantTest) {
  const auto kDataFormatFolder = filesystem::path("data_format_variant");

  // The difference in this test data set is quite big. It is around 0.3.
  constexpr auto kIntTolerance = 3e-1;
  std::vector<MarkerSequence> testSequences;
  std::vector<filesystem::path> intFilePaths;
  loadC3dFiles(kDataFormatFolder / "int", intFilePaths);
  EXPECT_EQ(intFilePaths.size(), 2);
  checkAllTheSame(intFilePaths, kIntTolerance, testSequences);

  std::vector<filesystem::path> realFilePaths;
  loadC3dFiles(kDataFormatFolder / "real", realFilePaths);
  EXPECT_EQ(realFilePaths.size(), 2);
  checkAllTheSame(realFilePaths, kTolerance, testSequences);

  checkAllTheSame({intFilePaths[0], realFilePaths[0]}, kIntRealTolerance, testSequences);

  ASSERT_EQ(testSequences.size(), 1);
  const auto& firstSequence = testSequences[0];
  ASSERT_EQ(firstSequence.fps, 50.0f);
  ASSERT_EQ(firstSequence.frames.size(), 89);
  ASSERT_EQ(firstSequence.name, "");
  const auto& characterTemplate = firstSequence.frames[0];
  ASSERT_EQ(characterTemplate.size(), 36);
}

TEST(C3DTest, TestLimitedLoading) {
  std::vector<filesystem::path> paths;
  loadC3dFiles(filesystem::path("compatibility_test"), paths);
  EXPECT_EQ(paths.size(), 5);

  const std::vector<std::string> validMarkers = {
      "LTH1", "LTH2", "LTH3", "LTH4", "RTH1", "RTH2", "RTH3", "RTH4"};

  std::vector<MarkerSequence> result = loadC3d(paths[0].string());
  std::vector<MarkerSequence> limitedResult = loadC3d(paths[0].string(), UpVector::Y, validMarkers);

  ASSERT_LT(limitedResult[0].frames[0].size(), result[0].frames[0].size());
  ASSERT_EQ(limitedResult[0].frames[0].size(), 8);
}
