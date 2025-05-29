/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_LOGGER)

// Allow DEFAULT_LOG_CHANNEL to be defined before including this file if you want to override the
// default channel name. To set your own channel name (the order matters):
//   #define DEFAULT_LOG_CHANNEL "my_custom_channel_name"
//   #include <momentum/common/logging.h>
#ifndef DEFAULT_LOG_CHANNEL
#include <momentum/common/log_channel.h>
#define DEFAULT_LOG_CHANNEL MOMENTUM_LOG_CHANNEL
#endif

#include <logging/Log.h>

#define MT_LOGT(...) XR_LOGT(__VA_ARGS__)
#define MT_LOGD(...) XR_LOGD(__VA_ARGS__)
#define MT_LOGI(...) XR_LOGI(__VA_ARGS__)
#define MT_LOGW(...) XR_LOGW(__VA_ARGS__)
#define MT_LOGE(...) XR_LOGE(__VA_ARGS__)

#define MT_LOGT_IF(condition, ...) XR_LOGT_IF(condition, __VA_ARGS__)
#define MT_LOGD_IF(condition, ...) XR_LOGD_IF(condition, __VA_ARGS__)
#define MT_LOGI_IF(condition, ...) XR_LOGI_IF(condition, __VA_ARGS__)
#define MT_LOGW_IF(condition, ...) XR_LOGW_IF(condition, __VA_ARGS__)
#define MT_LOGE_IF(condition, ...) XR_LOGE_IF(condition, __VA_ARGS__)

#define MT_LOGT_ONCE(...) XR_LOGT_ONCE(__VA_ARGS__)
#define MT_LOGD_ONCE(...) XR_LOGD_ONCE(__VA_ARGS__)
#define MT_LOGI_ONCE(...) XR_LOGI_ONCE(__VA_ARGS__)
#define MT_LOGW_ONCE(...) XR_LOGW_ONCE(__VA_ARGS__)
#define MT_LOGE_ONCE(...) XR_LOGE_ONCE(__VA_ARGS__)

#define MT_LOGT_ONCE_IF(condition, ...) XR_LOGT_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGD_ONCE_IF(condition, ...) XR_LOGD_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGI_ONCE_IF(condition, ...) XR_LOGI_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGW_ONCE_IF(condition, ...) XR_LOGW_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGE_ONCE_IF(condition, ...) XR_LOGE_ONCE_IF(condition, __VA_ARGS__)

#elif defined(MOMENTUM_WITH_SPDLOG)

#include <spdlog/spdlog.h>

#include <atomic>

#define MT_LOGT(...) ::spdlog::trace(__VA_ARGS__)
#define MT_LOGD(...) ::spdlog::debug(__VA_ARGS__)
#define MT_LOGI(...) ::spdlog::info(__VA_ARGS__)
#define MT_LOGW(...) ::spdlog::warn(__VA_ARGS__)
#define MT_LOGE(...) ::spdlog::error(__VA_ARGS__)

#define MT_LOGT_IF(condition, ...) \
  if (condition) {                 \
    MT_LOGT(__VA_ARGS__);          \
  }
#define MT_LOGD_IF(condition, ...) \
  if (condition) {                 \
    MT_LOGD(__VA_ARGS__);          \
  }
#define MT_LOGI_IF(condition, ...) \
  if (condition) {                 \
    MT_LOGI(__VA_ARGS__);          \
  }
#define MT_LOGW_IF(condition, ...) \
  if (condition) {                 \
    MT_LOGW(__VA_ARGS__);          \
  }
#define MT_LOGE_IF(condition, ...) \
  if (condition) {                 \
    MT_LOGE(__VA_ARGS__);          \
  }

// This function is designed to limit the number of times an error is logged.
// Please note that in a multi-threaded context, its behavior may not be guaranteed.
#define _MT_RUN_ONCE(runcode)                                \
  {                                                          \
    static std::atomic<bool> codeRan(false);                 \
    if (!codeRan) {                                          \
      bool expected = false;                                 \
      if (codeRan.compare_exchange_strong(expected, true)) { \
        runcode;                                             \
      }                                                      \
    }                                                        \
  }

#define MT_LOGT_ONCE(...) _MT_RUN_ONCE(MT_LOGT(__VA_ARGS__))
#define MT_LOGD_ONCE(...) _MT_RUN_ONCE(MT_LOGD(__VA_ARGS__))
#define MT_LOGI_ONCE(...) _MT_RUN_ONCE(MT_LOGI(__VA_ARGS__))
#define MT_LOGW_ONCE(...) _MT_RUN_ONCE(MT_LOGW(__VA_ARGS__))
#define MT_LOGE_ONCE(...) _MT_RUN_ONCE(MT_LOGE(__VA_ARGS__))

#define MT_LOGT_ONCE_IF(condition, ...) \
  if (condition) {                      \
    MT_LOGT_ONCE(__VA_ARGS__);          \
  }
#define MT_LOGD_ONCE_IF(condition, ...) \
  if (condition) {                      \
    MT_LOGD_ONCE(__VA_ARGS__);          \
  }
#define MT_LOGI_ONCE_IF(condition, ...) \
  if (condition) {                      \
    MT_LOGI_ONCE(__VA_ARGS__);          \
  }
#define MT_LOGW_ONCE_IF(condition, ...) \
  if (condition) {                      \
    MT_LOGW_ONCE(__VA_ARGS__);          \
  }
#define MT_LOGE_ONCE_IF(condition, ...) \
  if (condition) {                      \
    MT_LOGE_ONCE(__VA_ARGS__);          \
  }

#else

#define MT_LOGT(...)
#define MT_LOGD(...)
#define MT_LOGI(...)
#define MT_LOGW(...)
#define MT_LOGE(...)

#define MT_LOGT_IF(condition, ...)
#define MT_LOGD_IF(condition, ...)
#define MT_LOGI_IF(condition, ...)
#define MT_LOGW_IF(condition, ...)
#define MT_LOGE_IF(condition, ...)

#define MT_LOGT_ONCE(...)
#define MT_LOGD_ONCE(...)
#define MT_LOGI_ONCE(...)
#define MT_LOGW_ONCE(...)
#define MT_LOGE_ONCE(...)

#define MT_LOGT_ONCE_IF(condition, ...)
#define MT_LOGD_ONCE_IF(condition, ...)
#define MT_LOGI_ONCE_IF(condition, ...)
#define MT_LOGW_ONCE_IF(condition, ...)
#define MT_LOGE_ONCE_IF(condition, ...)

#endif

#include <momentum/common/checks.h>

#include <map>
#include <string>

namespace momentum {

/// Logging levels in descending order of verbosity
enum class LogLevel {
  Disabled = 0,
  Error,
  Warning,
  Info,
  Debug,
  Trace,
};

[[nodiscard]] std::map<std::string, LogLevel> logLevelMap();

void setLogLevel(LogLevel level);

void setLogLevel(const std::string& levelStr);

[[nodiscard]] LogLevel getLogLevel();

} // namespace momentum
