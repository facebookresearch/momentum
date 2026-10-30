/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <string>

namespace momentum {

/// Output destination for a Rerun recording; a newly spawned viewer when both fields are empty.
struct RerunSinkOptions {
  /// Path of the .rrd recording to write, or empty.
  std::string saveFile;
  /// gRPC URL of a running viewer to stream to, or empty. Ignored when saveFile is set.
  std::string connectUrl;
};

/// Adds the mutually exclusive --save and --connect options to a CLI app.
///
/// @param[in,out] app The CLI app to add the options to.
/// @param[out] options The destination the parsed options are written to.
void addRerunSinkOptions(CLI::App& app, RerunSinkOptions& options);

/// Attaches a recording stream to its destination: an .rrd file, a running viewer, or a newly
/// spawned viewer.
///
/// Momentum logs are redirected into the recording when it goes to a viewer (see
/// @ref redirectLogsToRerun for the lifetime requirement) and stay on the console when it goes to a
/// file. Exits the process if the destination cannot be opened.
///
/// @param[in] rec The recording stream to attach.
/// @param[in] options The destination.
void openRerunSink(const rerun::RecordingStream& rec, const RerunSinkOptions& options);

} // namespace momentum
