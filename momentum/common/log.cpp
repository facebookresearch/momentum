/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/log.h"
#include "momentum/common/exception.h"

#include <algorithm>

namespace momentum {

namespace {

#if defined(MOMENTUM_WITH_XR_LOGGER)

[[nodiscard]] arvr::logging::Level toArvrLogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::Disabled:
      return arvr::logging::Level::Disabled;
    case LogLevel::Error:
      return arvr::logging::Level::Error;
    case LogLevel::Warning:
      return arvr::logging::Level::Warning;
    case LogLevel::Info:
      return arvr::logging::Level::Info;
    case LogLevel::Debug:
      return arvr::logging::Level::Debug;
    case LogLevel::Trace:
      return arvr::logging::Level::Trace;
    default:
      MT_THROW("Unknown log level");
  }
}

#elif defined(MOMENTUM_WITH_SPDLOG)

[[nodiscard]] spdlog::level::level_enum toSpdlogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::Disabled:
      return spdlog::level::off;
    case LogLevel::Error:
      return spdlog::level::err;
    case LogLevel::Warning:
      return spdlog::level::warn;
    case LogLevel::Info:
      return spdlog::level::info;
    case LogLevel::Debug:
      return spdlog::level::debug;
    case LogLevel::Trace:
      return spdlog::level::trace;
    default:
      MT_THROW("Unknown log level");
  }
}

// Store the log level in our own static variable in addition to forwarding
// to spdlog. When spdlog is used in header-only mode and the common library
// is a shared library, spdlog's static globals may not be shared across
// shared-library boundaries. Using our own storage guarantees round-trip
// correctness for setLogLevel/getLogLevel.
LogLevel& spdlogLogLevel() {
  static LogLevel level = LogLevel::Info;
  return level;
}

#else

// Fallback: store the log level in a static variable when no logging backend
// is available.
LogLevel& fallbackLogLevel() {
  static LogLevel level = LogLevel::Info;
  return level;
}

#endif

[[nodiscard]] LogLevel stringToLogLevel(const std::string& logLevelStr) {
  std::string upperCaseLogLevelStr = logLevelStr;
  std::transform(
      upperCaseLogLevelStr.begin(),
      upperCaseLogLevelStr.end(),
      upperCaseLogLevelStr.begin(),
      ::toupper);
  if (upperCaseLogLevelStr == "TRACE") {
    return LogLevel::Trace;
  } else if (upperCaseLogLevelStr == "DEBUG") {
    return LogLevel::Debug;
  } else if (upperCaseLogLevelStr == "INFO") {
    return LogLevel::Info;
  } else if (upperCaseLogLevelStr == "WARNING") {
    return LogLevel::Warning;
  } else if (upperCaseLogLevelStr == "ERROR") {
    return LogLevel::Error;
  } else if (upperCaseLogLevelStr == "DISABLED") {
    return LogLevel::Disabled;
  } else {
    MT_THROW_T(std::invalid_argument, "Invalid log level: {}", logLevelStr);
  }
}

} // namespace

std::map<std::string, LogLevel> logLevelMap() {
  return {
      {"Disabled", LogLevel::Disabled},
      {"Error", LogLevel::Error},
      {"Warning", LogLevel::Warning},
      {"Info", LogLevel::Info},
      {"Debug", LogLevel::Debug},
      {"Trace", LogLevel::Trace}};
}

void setLogLevel(LogLevel level) {
#if defined(MOMENTUM_WITH_XR_LOGGER)
  arvr::logging::getChannel(DEFAULT_LOG_CHANNEL).setLevel(toArvrLogLevel(level));
#elif defined(MOMENTUM_WITH_SPDLOG)
  spdlogLogLevel() = level;
  spdlog::set_level(toSpdlogLevel(level));
#else
  fallbackLogLevel() = level;
#endif
}

void setLogLevel(const std::string& levelStr) {
  setLogLevel(stringToLogLevel(levelStr));
}

LogLevel getLogLevel() {
#if defined(MOMENTUM_WITH_XR_LOGGER)
  // Get the current log level from the XR logger
  int level = arvr::logging::getChannel(DEFAULT_LOG_CHANNEL).level();

  // Convert from arvr::logging::Level to momentum::LogLevel
  switch (static_cast<arvr::logging::Level>(level)) {
    case arvr::logging::Level::Disabled:
      return LogLevel::Disabled;
    case arvr::logging::Level::Error:
      return LogLevel::Error;
    case arvr::logging::Level::Warning:
      return LogLevel::Warning;
    case arvr::logging::Level::Info:
      return LogLevel::Info;
    case arvr::logging::Level::Debug:
      return LogLevel::Debug;
    case arvr::logging::Level::Trace:
      return LogLevel::Trace;
    default:
      // Default to Info level for unknown levels
      return LogLevel::Info;
  }
#elif defined(MOMENTUM_WITH_SPDLOG)
  return spdlogLogLevel();
#else
  return fallbackLogLevel();
#endif
}

} // namespace momentum
