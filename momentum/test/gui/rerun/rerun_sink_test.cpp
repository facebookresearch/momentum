/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/gui/rerun/rerun_sink.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>
#include <rerun.hpp>

#include <filesystem>

using namespace momentum;

// The connect and spawn destinations need a running or installed viewer, so only the file
// destination is exercised here.

TEST(RerunSinkTest, SaveAndConnectExcludeEachOther) {
  CLI::App app;
  RerunSinkOptions options;
  addRerunSinkOptions(app, options);

  app.parse("--save recording.rrd", false);
  EXPECT_EQ(options.saveFile, "recording.rrd");
  EXPECT_TRUE(options.connectUrl.empty());

  CLI::App bothApp;
  RerunSinkOptions bothOptions;
  addRerunSinkOptions(bothApp, bothOptions);
  EXPECT_THROW(
      bothApp.parse("--save recording.rrd --connect rerun+http://localhost:9876/proxy", false),
      CLI::ExcludesError);
}

TEST(RerunSinkTest, SaveWritesTheRecordingToTheFile) {
  const auto file = temporaryFile("rerun_sink", "rrd");
  {
    const rerun::RecordingStream rec("rerun_sink_test");
    openRerunSink(rec, {.saveFile = file.path().string(), .connectUrl = ""});
    rec.log("points", rerun::Points3D({{1.0f, 2.0f, 3.0f}}));
    ASSERT_TRUE(rec.flush_blocking().is_ok());
  }

  ASSERT_TRUE(std::filesystem::exists(file.path()));
  EXPECT_GT(std::filesystem::file_size(file.path()), 0);
}
