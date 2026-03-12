/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/logging_pybind.h"

#include <momentum/common/log.h>

#if defined(MOMENTUM_WITH_XR_LOGGER)
#include <logging/BrokerExtension.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace pymomentum {

namespace {

#if defined(MOMENTUM_WITH_XR_LOGGER)

arvr::logging::LogResult pythonLogCallback(
    int logLevel,
    const char* /* channelName */,
    size_t /* channelNameSizeInBytes */,
    const char* message,
    size_t /* messageSizeInBytes */,
    bool /* isFatalLog */,
    arvr::logging::CustomUserData /* userData */) {
  // Map the log level to a human-readable prefix:
  const char* prefix = "";
  switch (logLevel) {
    case static_cast<int>(arvr::logging::Level::Error):
      prefix = "ERROR: ";
      break;
    case static_cast<int>(arvr::logging::Level::Warning):
      prefix = "WARNING: ";
      break;
    case static_cast<int>(arvr::logging::Level::Info):
      prefix = "";
      break;
    case static_cast<int>(arvr::logging::Level::Debug):
      prefix = "DEBUG: ";
      break;
    case static_cast<int>(arvr::logging::Level::Trace):
      prefix = "TRACE: ";
      break;
    default:
      return arvr::logging::LogResult::Filtered;
  }

  // Acquire the GIL before calling into Python:
  py::gil_scoped_acquire gil;
  py::print(std::string(prefix) + message);

  return arvr::logging::LogResult::Accepted;
}

#endif

} // namespace

void registerLoggingBindings(py::module& m) {
  // Expose LogLevel enum:
  py::enum_<momentum::LogLevel>(m, "LogLevel", "Log level for the momentum library.")
      .value("Disabled", momentum::LogLevel::Disabled)
      .value("Error", momentum::LogLevel::Error)
      .value("Warning", momentum::LogLevel::Warning)
      .value("Info", momentum::LogLevel::Info)
      .value("Debug", momentum::LogLevel::Debug)
      .value("Trace", momentum::LogLevel::Trace);

  // Expose setLogLevel and getLogLevel:
  m.def(
      "set_log_level",
      py::overload_cast<momentum::LogLevel>(&momentum::setLogLevel),
      R"(Sets the log level for the momentum library.

All log messages with a level less verbose than the specified level will be suppressed.
For example, setting the level to :attr:`LogLevel.Warning` will suppress :attr:`LogLevel.Info`,
:attr:`LogLevel.Debug`, and :attr:`LogLevel.Trace` messages.

:param level: The desired log level.
      )",
      py::arg("level"));

  m.def(
      "set_log_level",
      py::overload_cast<const std::string&>(&momentum::setLogLevel),
      R"(Sets the log level for the momentum library using a string name.

Valid values: "Disabled", "Error", "Warning", "Info", "Debug", "Trace" (case-insensitive).

:param level: The desired log level as a string.
      )",
      py::arg("level"));

  m.def(
      "get_log_level",
      &momentum::getLogLevel,
      R"(Returns the current log level for the momentum library.

:return: The current :class:`LogLevel`.
      )");

  m.def(
      "redirect_logs_to_python",
      []() -> bool {
#if defined(MOMENTUM_WITH_XR_LOGGER)
        static arvr::logging::LogSinkHandle handle = 0;
        if (handle != 0) {
          return true;
        }

        handle = arvr::logging::addSink(pythonLogCallback, 0, "Python logger");
        return (handle != 0);
#else
        MT_LOGW("Log redirection to Python is not supported without XR_LOGGER.");
        return false;
#endif
      },
      R"(Redirects C++ log messages from the momentum library to Python's stdout via :func:`print`.

This allows C++ log output (from :func:`MT_LOG` macros) to appear alongside Python output,
which is useful when running momentum from a Python script or Jupyter notebook.

The redirection is performed at most once; subsequent calls have no effect and return ``True``.

:return: ``True`` if the redirection was successfully set up (or was already active).
      )");

  // Test helper: fire MT_LOGW from multiple C++ threads concurrently.
  // This is used to verify that the Python logging callback acquires the GIL
  // correctly and does not crash under concurrent access.
  m.def(
      "_test_multithreaded_logging",
      [](int numThreads, int numMessagesPerThread) {
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t) {
          threads.emplace_back([t, numMessagesPerThread]() {
            // Spawned C++ threads do not hold the GIL. Each MT_LOGW call
            // triggers the logging callback which must acquire the GIL
            // before calling py::print:
            for (int i = 0; i < numMessagesPerThread; ++i) {
              MT_LOGW("Multithreaded logging test: thread {} message {}", t, i);
            }
          });
        }
        // Release the GIL while waiting for the threads, so that the logging
        // callbacks (which need the GIL) can proceed without deadlocking:
        py::gil_scoped_release release;
        for (auto& thread : threads) {
          thread.join();
        }
      },
      R"(Test helper: fire :func:`MT_LOGW` from multiple C++ threads concurrently.

This is used internally to verify that the Python logging callback handles
the GIL correctly and does not crash under concurrent access.

:param num_threads: Number of threads to spawn.
:param num_messages_per_thread: Number of log messages each thread emits.
      )",
      py::arg("num_threads"),
      py::arg("num_messages_per_thread"));
}

} // namespace pymomentum
