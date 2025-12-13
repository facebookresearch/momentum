/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/gui/rerun/logging_redirect.h"

#include "momentum/common/log.h"

#if defined(MOMENTUM_WITH_XR_LOGGER)
#include <logging/BrokerExtension.h>
#endif

#include <rerun.hpp>

#include <bit>

#if !defined(MOMENTUM_WITH_XR_LOGGER)
#include <iostream>
#endif

namespace momentum {

namespace {

#if defined(MOMENTUM_WITH_XR_LOGGER)

arvr::logging::LogResult callback(
    int logLevel,
    const char* /* channelName */,
    size_t /* channelNameSizeInBytes */,
    const char* message,
    size_t /* messageSizeInBytes */,
    bool /*isFatalLog*/,
    arvr::logging::CustomUserData userData) {
  const auto* rec = std::bit_cast<const rerun::RecordingStream*>(userData);

  rerun::TextLogLevel level;
  switch (logLevel) {
    case static_cast<int>(arvr::logging::Level::Error):
      level = rerun::TextLogLevel::Error;
      break;
    case static_cast<int>(arvr::logging::Level::Warning):
      level = rerun::TextLogLevel::Warning;
      break;
    case static_cast<int>(arvr::logging::Level::Info):
      level = rerun::TextLogLevel::Info;
      break;
    case static_cast<int>(arvr::logging::Level::Debug):
      level = rerun::TextLogLevel::Debug;
      break;
    case static_cast<int>(arvr::logging::Level::Trace):
      level = rerun::TextLogLevel::Trace;
      break;
    default: {
      return arvr::logging::LogResult::Filtered;
    }
  }

  // TODO: Make logging channel configurable
  rec->log("momentum", rerun::TextLog(message).with_level(level));

  return arvr::logging::LogResult::Accepted;
}

#endif

} // namespace

bool redirectLogsToRerun(const rerun::RecordingStream& rec) {
#if defined(MOMENTUM_WITH_XR_LOGGER)
  static arvr::logging::LogSinkHandle handle = 0;
  if (handle != 0) {
    // If handle is not 0, it means the log sink has already been added, so return true to indicate
    // that the log is already being redirected.
    return true;
  }

  handle = arvr::logging::addSink(
      callback, reinterpret_cast<arvr::logging::CustomUserData>(&rec), "Rerun logger");
  return (handle != 0);
#else
  (void)rec;
  MT_LOGW("Log redirection is not supported.\n");
  return false;
#endif
}

} // namespace momentum
