/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/gui/rerun/rerun_sink.h"

#include "momentum/common/log.h"
#include "momentum/gui/rerun/logging_redirect.h"

namespace momentum {

void addRerunSinkOptions(CLI::App& app, RerunSinkOptions& options) {
  auto* saveOpt = app.add_option(
      "--save",
      options.saveFile,
      "Write the recording to this .rrd file instead of opening a viewer");
  app.add_option(
         "--connect",
         options.connectUrl,
         "Stream to a viewer already listening at this gRPC URL, e.g. "
         "rerun+http://localhost:9876/proxy, instead of opening one")
      ->excludes(saveOpt);
}

void openRerunSink(const rerun::RecordingStream& rec, const RerunSinkOptions& options) {
  if (!options.saveFile.empty()) {
    // A file recording leaves momentum logs on the console.
    MT_LOGI("Writing recording to {}", options.saveFile);
    rec.save(options.saveFile).exit_on_failure();
    return;
  }

  if (!options.connectUrl.empty()) {
    MT_LOGI("Streaming to viewer at {}", options.connectUrl);
    rec.connect_grpc(options.connectUrl).exit_on_failure();
  } else {
    rec.spawn().exit_on_failure();
  }
  redirectLogsToRerun(rec);
}

} // namespace momentum
