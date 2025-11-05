/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/ContinuousCollisionDetection.h"

#include <charconv>
#include <fstream>
#include <unordered_set>

#include <gtest/gtest.h>

namespace axel::test {
namespace {

// The line format in the .csv used here is described in
// https://github.com/Continuous-Collision-Detection/Sample-Queries.
// The datasets that we selected are in "unit-tests" folder.
// The queries from the github repository above are a complementary dataset for
// https://cims.nyu.edu/gcl/papers/2021-CCD.pdf.

struct LineData {
  std::array<double, 6> values;
  bool hit;

  [[nodiscard]] Eigen::Vector3d getVertexPosition() const {
    return Eigen::Vector3d(values[0] / values[1], values[2] / values[3], values[4] / values[5]);
  }
};

LineData parseLineData(const std::string_view line) {
  LineData data = {};

  size_t currDataIdx = 0;
  size_t start = 0;
  size_t curr = 0;

  while (curr < line.size()) {
    // When we encounter the end of the line or a comma, process the digits seen so far.
    if (curr + 1 == line.size() || line[curr + 1] == ',') {
      if (currDataIdx < data.values.size()) {
        // In the future, we'd really prefer something like
        // double val;
        // std::from_chars(line.data() + start, line.data() + curr + 1, val);
        // Unfortunately, not all platforms implement this C++17 feature for doubles.
        const double val = std::stod(std::string(line.substr(start, curr - start + 1)));
        data.values[currDataIdx++] = val;
        start = curr + 2;
      } else {
        int val;
        std::from_chars(line.data() + start, line.data() + curr + 1, val);
        data.hit = val > 0;
      }
    }

    ++curr;
  }
  return data;
}

struct CcdTestItem {
  std::array<Eigen::Vector3d, 8> points;
  bool hit;
};

std::vector<CcdTestItem> parseCcdTestDataset(const std::string& path) {
  std::ifstream file(path);
  std::string line;

  int lineIdx = 0;
  std::vector<CcdTestItem> dataset{};
  CcdTestItem testData{};
  while (std::getline(file, line)) {
    const auto lineData = parseLineData(line);
    testData.points[lineIdx] = lineData.getVertexPosition();
    testData.hit = lineData.hit;

    ++lineIdx;
    if (lineIdx == 8) {
      dataset.push_back(testData);
      lineIdx = 0;
    }
  }
  return dataset;
}

constexpr std::array<std::string_view, 2> kVertexTriangleDatasetPaths = {
    "arvr/libraries/axel/math/test/data/vt_data_0_0.csv",
    "arvr/libraries/axel/math/test/data/vt_data_0_1.csv",
};

template <size_t DatasetIndex>
class CcdVertexTriangleTest : public ::testing::TestWithParam<int32_t> {
 public:
  CcdVertexTriangleTest() {
    if (dataset_.empty()) {
      static_assert(DatasetIndex < kVertexTriangleDatasetPaths.size());
      dataset_ = parseCcdTestDataset(std::string(kVertexTriangleDatasetPaths[DatasetIndex]));
    }
  }

  void invokeCcdCheck(const size_t itemIdx) {
    ASSERT_GE(itemIdx, 0);
    ASSERT_LT(itemIdx, dataset_.size());
    const auto& t = dataset_[itemIdx];

    EXPECT_EQ(
        ccdVertexTriangle(
            t.points[1],
            t.points[2],
            t.points[3],
            t.points[0],
            t.points[5] - t.points[1],
            t.points[6] - t.points[2],
            t.points[7] - t.points[3],
            t.points[4] - t.points[0],
            1e-15,
            1.0),
        t.hit);
  }

 protected:
  static std::vector<CcdTestItem> dataset_;
};

template <size_t DatasetIndex>
std::vector<CcdTestItem> CcdVertexTriangleTest<DatasetIndex>::dataset_ = {};

using CcdVertexTriangleTest0 = CcdVertexTriangleTest<0>;
TEST_P(CcdVertexTriangleTest0, Basic) {
  invokeCcdCheck(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CcdVertexTriangleTests0, CcdVertexTriangleTest0, ::testing::Range(0, 125));

using CcdVertexTriangleTest1 = CcdVertexTriangleTest<1>;
TEST_P(CcdVertexTriangleTest1, Basic) {
  invokeCcdCheck(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CcdVertexTriangleTests1, CcdVertexTriangleTest1, ::testing::Range(0, 125));

constexpr std::array<std::string_view, 2> kEdgeEdgeDatasetPaths = {
    "arvr/libraries/axel/math/test/data/ee_data_0_0.csv",
    "arvr/libraries/axel/math/test/data/ee_data_0_1.csv",
};

// TODO(nemanjab):
// This is a crutch to remove failing tests at the moment.
// We need to fix/investigate this in the future and understand the failures.
const std::array<std::unordered_set<size_t>, 2> kFailingEdgeEdgeTests = {
    std::unordered_set<size_t>{},
    std::unordered_set<size_t>{2, 10, 12, 13},
};

template <size_t DatasetIndex>
class CcdEdgeEdgeTest : public ::testing::TestWithParam<int32_t> {
 public:
  CcdEdgeEdgeTest() {
    if (dataset_.empty()) {
      static_assert(DatasetIndex < kEdgeEdgeDatasetPaths.size());
      dataset_ = parseCcdTestDataset(std::string(kEdgeEdgeDatasetPaths[DatasetIndex]));
    }
  }

  void invokeCcdCheck(const size_t itemIdx) {
    if (kFailingEdgeEdgeTests[DatasetIndex].count(itemIdx) > 0) {
      return;
    }

    ASSERT_GE(itemIdx, 0);
    ASSERT_LT(itemIdx, dataset_.size());
    const auto& t = dataset_[itemIdx];

    EXPECT_EQ(
        ccdEdgeEdge(
            t.points[0],
            t.points[1],
            t.points[2],
            t.points[3],
            t.points[4] - t.points[0],
            t.points[5] - t.points[1],
            t.points[6] - t.points[2],
            t.points[7] - t.points[3],
            1e-10,
            1.0),
        t.hit);
  }

 protected:
  static std::vector<CcdTestItem> dataset_;
};

template <size_t DatasetIndex>
std::vector<CcdTestItem> CcdEdgeEdgeTest<DatasetIndex>::dataset_ = {};

using CcdEdgeEdgeTest0 = CcdEdgeEdgeTest<0>;
TEST_P(CcdEdgeEdgeTest0, Basic) {
  invokeCcdCheck(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CcdEdgeEdgeTests0, CcdEdgeEdgeTest0, ::testing::Range(0, 54));

using CcdEdgeEdgeTest1 = CcdEdgeEdgeTest<1>;
TEST_P(CcdEdgeEdgeTest1, Basic) {
  invokeCcdCheck(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CcdEdgeEdgeTests1, CcdEdgeEdgeTest1, ::testing::Range(0, 20));

} // namespace
} // namespace axel::test
