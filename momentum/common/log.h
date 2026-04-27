/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_LOGGER)

/// To override the default log channel, define `DEFAULT_LOG_CHANNEL` before including this header
/// (order matters):
///   #define DEFAULT_LOG_CHANNEL "my_custom_channel_name"
///   #include <momentum/common/log.h>
#ifndef DEFAULT_LOG_CHANNEL
#include <momentum/common/log_channel.h>
#define DEFAULT_LOG_CHANNEL MOMENTUM_LOG_CHANNEL
#endif

#include <logging/Log.h>

/// @{
/// Log a message at the corresponding severity (T=Trace, D=Debug, I=Info, W=Warning, E=Error).
/// Uses fmt-style formatting via the underlying logger backend.
#define MT_LOGT(...) XR_LOGT(__VA_ARGS__)
#define MT_LOGD(...) XR_LOGD(__VA_ARGS__)
#define MT_LOGI(...) XR_LOGI(__VA_ARGS__)
#define MT_LOGW(...) XR_LOGW(__VA_ARGS__)
#define MT_LOGE(...) XR_LOGE(__VA_ARGS__)
/// @}

/// @{
/// Log only if `condition` evaluates to true at runtime.
#define MT_LOGT_IF(condition, ...) XR_LOGT_IF(condition, __VA_ARGS__)
#define MT_LOGD_IF(condition, ...) XR_LOGD_IF(condition, __VA_ARGS__)
#define MT_LOGI_IF(condition, ...) XR_LOGI_IF(condition, __VA_ARGS__)
#define MT_LOGW_IF(condition, ...) XR_LOGW_IF(condition, __VA_ARGS__)
#define MT_LOGE_IF(condition, ...) XR_LOGE_IF(condition, __VA_ARGS__)
/// @}

/// @{
/// Log at most once per process lifetime, regardless of how often the call site is reached.
#define MT_LOGT_ONCE(...) XR_LOGT_ONCE(__VA_ARGS__)
#define MT_LOGD_ONCE(...) XR_LOGD_ONCE(__VA_ARGS__)
#define MT_LOGI_ONCE(...) XR_LOGI_ONCE(__VA_ARGS__)
#define MT_LOGW_ONCE(...) XR_LOGW_ONCE(__VA_ARGS__)
#define MT_LOGE_ONCE(...) XR_LOGE_ONCE(__VA_ARGS__)
/// @}

/// @{
/// Log at most once per process lifetime, and only if `condition` is true on that first reach.
#define MT_LOGT_ONCE_IF(condition, ...) XR_LOGT_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGD_ONCE_IF(condition, ...) XR_LOGD_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGI_ONCE_IF(condition, ...) XR_LOGI_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGW_ONCE_IF(condition, ...) XR_LOGW_ONCE_IF(condition, __VA_ARGS__)
#define MT_LOGE_ONCE_IF(condition, ...) XR_LOGE_ONCE_IF(condition, __VA_ARGS__)
/// @}

#elif defined(MOMENTUM_WITH_SPDLOG)

#include <spdlog/spdlog.h>

#include <atomic>

// TODO: The MT_LOG*_IF macros below expand to a bare `if (condition) { ... }`. When used inside an
// unbraced `if`/`else` chain at the call site, this can change control flow (dangling-else) or
// silently swallow a following `else`. Wrap in `do { ... } while (0)` for hygienic expansion.

/// @{
/// Log a message at the corresponding severity (T=Trace, D=Debug, I=Info, W=Warning, E=Error).
/// Forwards to spdlog with fmt-style formatting.
#define MT_LOGT(...) ::spdlog::trace(__VA_ARGS__)
#define MT_LOGD(...) ::spdlog::debug(__VA_ARGS__)
#define MT_LOGI(...) ::spdlog::info(__VA_ARGS__)
#define MT_LOGW(...) ::spdlog::warn(__VA_ARGS__)
#define MT_LOGE(...) ::spdlog::error(__VA_ARGS__)
/// @}

/// @{
/// Log only if `condition` evaluates to true at runtime.
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
/// @}

// TODO: The leading underscore in `_MT_RUN_ONCE` puts the identifier in a name reserved by the
// C++ standard at file scope (any identifier with a leading underscore in the global namespace).
// Rename to e.g. `MT_RUN_ONCE_DETAIL` or `MT_DETAIL_RUN_ONCE`.

/// Execute `runcode` at most once per process lifetime.
///
/// Implementation uses an atomic compare-exchange so concurrent first reaches are serialized;
/// `runcode` runs exactly once even under contention. Subsequent calls become a single relaxed
/// atomic load.
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

/// @{
/// Log at most once per process lifetime, regardless of how often the call site is reached.
#define MT_LOGT_ONCE(...) _MT_RUN_ONCE(MT_LOGT(__VA_ARGS__))
#define MT_LOGD_ONCE(...) _MT_RUN_ONCE(MT_LOGD(__VA_ARGS__))
#define MT_LOGI_ONCE(...) _MT_RUN_ONCE(MT_LOGI(__VA_ARGS__))
#define MT_LOGW_ONCE(...) _MT_RUN_ONCE(MT_LOGW(__VA_ARGS__))
#define MT_LOGE_ONCE(...) _MT_RUN_ONCE(MT_LOGE(__VA_ARGS__))
/// @}

/// @{
/// Log at most once per process lifetime, gated by `condition`.
///
/// @note The "once" latch is set on the first reach where `condition` is true; reaches where the
/// condition is false do not consume the latch.
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
/// @}

#else

// Fallback: no logging backend is configured. All MT_LOG* macros expand to nothing, so call-site
// arguments are not evaluated. Callers must not rely on side effects in log arguments.

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

/// Logging levels in ascending order of verbosity. `Disabled` suppresses all output; each
/// successive level enables strictly more messages (Error < Warning < Info < Debug < Trace).
enum class LogLevel {
  Disabled = 0,
  Error,
  Warning,
  Info,
  Debug,
  Trace,
};

/// Returns the canonical mapping from human-readable level names ("Disabled", "Error",
/// "Warning", "Info", "Debug", "Trace") to `LogLevel` values. Useful for building CLI argument
/// parsers or config validators.
[[nodiscard]] std::map<std::string, LogLevel> logLevelMap();

/// Sets the global log threshold; messages below `level` are filtered out.
///
/// Forwards to the active backend (XR logger / spdlog). When no backend is compiled in, the value
/// is recorded in a process-local fallback so that `getLogLevel` round-trips.
void setLogLevel(LogLevel level);

/// Parses `levelStr` (case-insensitive: "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "DISABLED")
/// and forwards to `setLogLevel(LogLevel)`.
///
/// @throws std::invalid_argument if `levelStr` is not one of the recognized names.
void setLogLevel(const std::string& levelStr);

/// Returns the current global log threshold. Mirrors what was last set via `setLogLevel`; defaults
/// to `LogLevel::Info` if never set.
[[nodiscard]] LogLevel getLogLevel();

} // namespace momentum
