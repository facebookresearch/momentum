/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/test/helpers/sanitizer_utils.h>

#include <gtest/gtest.h>
#include <rerun.hpp>

using namespace momentum;

class RerunCompatTest : public testing::Test {
 protected:
  void SetUp() override {
    // The rerun Rust FFI layer crashes under sanitizer builds due to a
    // null-pointer precondition in slice::from_raw_parts_mut inside
    // rr_recording_stream_log_impl. Skip all tests under sanitizers.
    if (isSanitizerBuild()) {
      GTEST_SKIP() << "Rerun SDK crashes under sanitizer builds (Rust FFI issue)";
    }
    rec_ = std::make_unique<rerun::RecordingStream>("rerun_compat_test");
  }

  std::unique_ptr<rerun::RecordingStream> rec_;
};

TEST_F(RerunCompatTest, MakeScalar) {
  auto scalarFloat = makeScalar(1.5f);
  auto scalarDouble = makeScalar(2.0);
  auto scalarInt = makeScalar(42);

  // Verify the scalar component batch was populated
  EXPECT_TRUE(scalarFloat.scalars.has_value());
  EXPECT_TRUE(scalarDouble.scalars.has_value());
  EXPECT_TRUE(scalarInt.scalars.has_value());

  // Log to verify they are valid rerun archetypes (accepted by the stream)
  rec_->log("test/scalar_float", scalarFloat);
  rec_->log("test/scalar_double", scalarDouble);
  rec_->log("test/scalar_int", scalarInt);
}

TEST_F(RerunCompatTest, MakeSeriesLine) {
  auto seriesLine = makeSeriesLine();
  // Default-constructed SeriesLines has no optional fields set
  EXPECT_FALSE(seriesLine.names.has_value());
  EXPECT_FALSE(seriesLine.colors.has_value());
  rec_->log_static("test/series", seriesLine);
}

TEST_F(RerunCompatTest, MakeSeriesLineWithName) {
  auto seriesLine = makeSeriesLineWithName("test_param");
  // Verify the name was set on the archetype
  EXPECT_TRUE(seriesLine.names.has_value());
  rec_->log_static("test/named_series", seriesLine);
}

TEST_F(RerunCompatTest, SetTimeSeconds) {
  setTimeSeconds(*rec_, "log_time", 1.5);
  rec_->log("test/after_time", makeScalar(1.0f));
}

TEST_F(RerunCompatTest, MakeTransform3D) {
  auto transform = makeTransform3D();
  // Default-constructed Transform3D has no translation or rotation set
  EXPECT_FALSE(transform.translation.has_value());
  EXPECT_FALSE(transform.rotation_axis_angle.has_value());
  rec_->log("test/transform", transform);
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
