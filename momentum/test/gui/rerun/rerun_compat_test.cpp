/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/gui/rerun/rerun_compat.h>

#include <gtest/gtest.h>
#include <rerun.hpp>

using namespace momentum;

class RerunCompatTest : public testing::Test {
 protected:
  void SetUp() override {
    rec_ = std::make_unique<rerun::RecordingStream>("rerun_compat_test");
  }

  std::unique_ptr<rerun::RecordingStream> rec_;
};

TEST_F(RerunCompatTest, MakeScalar) {
  auto scalarFloat = makeScalar(1.5f);
  auto scalarDouble = makeScalar(2.0);
  auto scalarInt = makeScalar(42);

  ASSERT_TRUE(scalarFloat.scalars.has_value());
  ASSERT_TRUE(scalarDouble.scalars.has_value());
  ASSERT_TRUE(scalarInt.scalars.has_value());
  EXPECT_EQ(scalarFloat.scalars->length(), 1);
  EXPECT_EQ(scalarDouble.scalars->length(), 1);
  EXPECT_EQ(scalarInt.scalars->length(), 1);

  EXPECT_TRUE(hasComponentData(scalarFloat));
  rec_->log("test/scalar_float", scalarFloat);
  rec_->log("test/scalar_double", scalarDouble);
  rec_->log("test/scalar_int", scalarInt);
}

TEST_F(RerunCompatTest, MakeSeriesLineEmpty) {
  auto seriesLine = makeSeriesLine();
  EXPECT_FALSE(seriesLine.names.has_value());
  EXPECT_FALSE(seriesLine.colors.has_value());
  EXPECT_FALSE(hasComponentData(seriesLine));

  // safeLogStatic must not crash with an empty archetype
  EXPECT_NO_THROW(safeLogStatic(*rec_, "test/series", seriesLine));
}

TEST_F(RerunCompatTest, MakeSeriesLineWithName) {
  auto seriesLine = makeSeriesLineWithName("test_param");
  ASSERT_TRUE(seriesLine.names.has_value());
  EXPECT_EQ(seriesLine.names->length(), 1);
  EXPECT_TRUE(hasComponentData(seriesLine));

  rec_->log_static("test/named_series", seriesLine);
}

TEST_F(RerunCompatTest, SetTimeSeconds) {
  setTimeSeconds(*rec_, "log_time", 1.5);
  rec_->log("test/after_time", makeScalar(1.0f));
}

TEST_F(RerunCompatTest, MakeTransform3DEmpty) {
  auto transform = makeTransform3D();
  EXPECT_FALSE(transform.translation.has_value());
  EXPECT_FALSE(transform.rotation_axis_angle.has_value());
  EXPECT_FALSE(hasComponentData(transform));

  // safeLog must not crash with an empty archetype
  EXPECT_NO_THROW(safeLog(*rec_, "test/transform", transform));
}

TEST_F(RerunCompatTest, MakeTransform3DWithTranslation) {
  auto transform = makeTransform3D(rerun::components::Translation3D(1.0f, 2.0f, 3.0f));
  ASSERT_TRUE(transform.translation.has_value());
  EXPECT_EQ(transform.translation->length(), 1);
  EXPECT_TRUE(hasComponentData(transform));

  rec_->log("test/transform_translated", transform);
}

TEST_F(RerunCompatTest, LogAxes3D) {
  // Should not crash regardless of rerun version
  EXPECT_NO_THROW(logAxes3D(*rec_, "test/axes", 10.0f));
}

TEST_F(RerunCompatTest, MakeSequenceTimeColumn) {
  std::vector<int64_t> indices = {0, 1, 2, 3};
  auto column = makeSequenceTimeColumn(
      "frame_index",
      rerun::Collection<int64_t>::borrow(indices.data(), indices.size()),
      rerun::SortingStatus::Sorted);
  EXPECT_EQ(column.timeline.name, "frame_index");
  EXPECT_EQ(column.sorting_status, rerun::SortingStatus::Sorted);
  EXPECT_NE(column.array, nullptr);
}

TEST_F(RerunCompatTest, MakeDurationSecondsTimeColumn) {
  std::vector<double> times = {0.0, 0.033, 0.066, 0.1};
  auto column = makeDurationSecondsTimeColumn(
      "log_time",
      rerun::Collection<double>::borrow(times.data(), times.size()),
      rerun::SortingStatus::Sorted);
  EXPECT_EQ(column.timeline.name, "log_time");
  EXPECT_EQ(column.sorting_status, rerun::SortingStatus::Sorted);
  EXPECT_NE(column.array, nullptr);
}

TEST_F(RerunCompatTest, MakeScalarColumns) {
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
  auto columns = makeScalarColumns(std::move(values));
  EXPECT_GT(columns.size(), 0);
}

TEST_F(RerunCompatTest, MakeScalarColumnsEmpty) {
  std::vector<double> values;
  auto columns = makeScalarColumns(std::move(values));
  EXPECT_GT(columns.size(), 0);
}
