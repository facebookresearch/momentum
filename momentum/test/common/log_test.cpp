/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/log.h"

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using namespace momentum;

// Test the logLevelMap function
TEST(LogTest, LogLevelMap) {
  auto map = logLevelMap();

  // Check that the map contains all expected log levels
  EXPECT_EQ(map.size(), 6);
  EXPECT_EQ(map["Disabled"], LogLevel::Disabled);
  EXPECT_EQ(map["Error"], LogLevel::Error);
  EXPECT_EQ(map["Warning"], LogLevel::Warning);
  EXPECT_EQ(map["Info"], LogLevel::Info);
  EXPECT_EQ(map["Debug"], LogLevel::Debug);
  EXPECT_EQ(map["Trace"], LogLevel::Trace);
}

// Test setting log level with enum
TEST(LogTest, SetLogLevelEnum) {
  // Save the original log level to restore it later
  LogLevel originalLevel = getLogLevel();

  // Test setting different log levels
  setLogLevel(LogLevel::Info);
#if defined(MOMENTUM_WITH_XR_LOGGER)
  EXPECT_EQ(getLogLevel(), LogLevel::Info);

  setLogLevel(LogLevel::Debug);
  EXPECT_EQ(getLogLevel(), LogLevel::Debug);

  setLogLevel(LogLevel::Error);
  EXPECT_EQ(getLogLevel(), LogLevel::Error);

  setLogLevel(LogLevel::Warning);
  EXPECT_EQ(getLogLevel(), LogLevel::Warning);

  setLogLevel(LogLevel::Trace);
  EXPECT_EQ(getLogLevel(), LogLevel::Trace);

  setLogLevel(LogLevel::Disabled);
  EXPECT_EQ(getLogLevel(), LogLevel::Disabled);
#else
  // With SPDLOG, setting log level is not supported, so we just verify that the function doesn't
  // crash
  setLogLevel(LogLevel::Debug);
  setLogLevel(LogLevel::Error);
  setLogLevel(LogLevel::Warning);
  setLogLevel(LogLevel::Trace);
  setLogLevel(LogLevel::Disabled);
#endif

  // Restore the original log level
  setLogLevel(originalLevel);
}

// Test setting log level with string
TEST(LogTest, SetLogLevelString) {
  // Save the original log level to restore it later
  LogLevel originalLevel = getLogLevel();

  // Test valid log level strings (case insensitive)
  setLogLevel("Trace");
#if defined(MOMENTUM_WITH_XR_LOGGER)
  EXPECT_EQ(getLogLevel(), LogLevel::Trace);

  setLogLevel("DEBUG");
  EXPECT_EQ(getLogLevel(), LogLevel::Debug);

  setLogLevel("info");
  EXPECT_EQ(getLogLevel(), LogLevel::Info);

  setLogLevel("Warning");
  EXPECT_EQ(getLogLevel(), LogLevel::Warning);

  setLogLevel("ERROR");
  EXPECT_EQ(getLogLevel(), LogLevel::Error);

  setLogLevel("disabled");
  EXPECT_EQ(getLogLevel(), LogLevel::Disabled);
#else
  // With SPDLOG, setting log level is not supported, so we just verify that the function doesn't
  // crash
  setLogLevel("DEBUG");
  setLogLevel("info");
  setLogLevel("Warning");
  setLogLevel("ERROR");
  setLogLevel("disabled");
#endif

  // Test invalid log level string
  EXPECT_THROW(setLogLevel("InvalidLevel"), std::invalid_argument);

  // Restore the original log level
  setLogLevel(originalLevel);
}

// Test the getLogLevel function
TEST(LogTest, GetLogLevel) {
  // Save the original log level to restore it later
  LogLevel originalLevel = getLogLevel();

  // Set a specific log level
  setLogLevel(LogLevel::Debug);

#if defined(MOMENTUM_WITH_XR_LOGGER)
  // Verify that getLogLevel returns the correct level
  EXPECT_EQ(getLogLevel(), LogLevel::Debug);
#else
  // With SPDLOG, setting log level is not supported, so we just verify that getLogLevel returns a
  // valid level
  EXPECT_GE(static_cast<int>(getLogLevel()), static_cast<int>(LogLevel::Disabled));
  EXPECT_LE(static_cast<int>(getLogLevel()), static_cast<int>(LogLevel::Trace));
#endif

  // Restore the original log level
  setLogLevel(originalLevel);
}

// Test the ordering of log levels
TEST(LogTest, LogLevelOrdering) {
  // Verify the ordering of log levels
  EXPECT_LT(static_cast<int>(LogLevel::Disabled), static_cast<int>(LogLevel::Error));
  EXPECT_LT(static_cast<int>(LogLevel::Error), static_cast<int>(LogLevel::Warning));
  EXPECT_LT(static_cast<int>(LogLevel::Warning), static_cast<int>(LogLevel::Info));
  EXPECT_LT(static_cast<int>(LogLevel::Info), static_cast<int>(LogLevel::Debug));
  EXPECT_LT(static_cast<int>(LogLevel::Debug), static_cast<int>(LogLevel::Trace));
}

// Test basic logging macros
// Note: These tests only verify that the macros don't crash
// They don't verify the actual output since that depends on the logging implementation
TEST(LogTest, BasicLoggingMacros) {
  MT_LOGT("Trace message");
  MT_LOGD("Debug message");
  MT_LOGI("Info message");
  MT_LOGW("Warning message");
  MT_LOGE("Error message");

  // Test with formatting
  MT_LOGT("Trace message with {} and {}", "arg1", 42);
  MT_LOGD("Debug message with {} and {}", "arg1", 42);
  MT_LOGI("Info message with {} and {}", "arg1", 42);
  MT_LOGW("Warning message with {} and {}", "arg1", 42);
  MT_LOGE("Error message with {} and {}", "arg1", 42);
}

// Test conditional logging macros
TEST(LogTest, ConditionalLoggingMacros) {
  bool condition = true;
  MT_LOGT_IF(condition, "Conditional trace message");
  MT_LOGD_IF(condition, "Conditional debug message");
  MT_LOGI_IF(condition, "Conditional info message");
  MT_LOGW_IF(condition, "Conditional warning message");
  MT_LOGE_IF(condition, "Conditional error message");

  condition = false;
  MT_LOGT_IF(condition, "This should not be logged");
  MT_LOGD_IF(condition, "This should not be logged");
  MT_LOGI_IF(condition, "This should not be logged");
  MT_LOGW_IF(condition, "This should not be logged");
  MT_LOGE_IF(condition, "This should not be logged");

  // Test with formatting
  condition = true;
  MT_LOGT_IF(condition, "Conditional trace message with {} and {}", "arg1", 42);
  MT_LOGD_IF(condition, "Conditional debug message with {} and {}", "arg1", 42);
  MT_LOGI_IF(condition, "Conditional info message with {} and {}", "arg1", 42);
  MT_LOGW_IF(condition, "Conditional warning message with {} and {}", "arg1", 42);
  MT_LOGE_IF(condition, "Conditional error message with {} and {}", "arg1", 42);
}

// Test "once" logging macros
TEST(LogTest, OnceLoggingMacros) {
  // These should log only once per test run
  for (int i = 0; i < 3; i++) {
    MT_LOGT_ONCE("Once trace message {}", i);
    MT_LOGD_ONCE("Once debug message {}", i);
    MT_LOGI_ONCE("Once info message {}", i);
    MT_LOGW_ONCE("Once warning message {}", i);
    MT_LOGE_ONCE("Once error message {}", i);
  }

  // Test with formatting
  for (int i = 0; i < 3; i++) {
    MT_LOGT_ONCE("Once trace message with {} and {}", "arg1", i);
    MT_LOGD_ONCE("Once debug message with {} and {}", "arg1", i);
    MT_LOGI_ONCE("Once info message with {} and {}", "arg1", i);
    MT_LOGW_ONCE("Once warning message with {} and {}", "arg1", i);
    MT_LOGE_ONCE("Once error message with {} and {}", "arg1", i);
  }

  // Test conditional once logging
  bool condition = true;
  for (int i = 0; i < 3; i++) {
    MT_LOGT_ONCE_IF(condition, "Conditional once trace message {}", i);
    MT_LOGD_ONCE_IF(condition, "Conditional once debug message {}", i);
    MT_LOGI_ONCE_IF(condition, "Conditional once info message {}", i);
    MT_LOGW_ONCE_IF(condition, "Conditional once warning message {}", i);
    MT_LOGE_ONCE_IF(condition, "Conditional once error message {}", i);
  }

  condition = false;
  for (int i = 0; i < 3; i++) {
    MT_LOGT_ONCE_IF(condition, "This should not be logged");
    MT_LOGD_ONCE_IF(condition, "This should not be logged");
    MT_LOGI_ONCE_IF(condition, "This should not be logged");
    MT_LOGW_ONCE_IF(condition, "This should not be logged");
    MT_LOGE_ONCE_IF(condition, "This should not be logged");
  }
}

// Test complex conditions with conditional logging
TEST(LogTest, ComplexConditions) {
  int value = 42;
  MT_LOGT_IF(value > 40, "Value is greater than 40: {}", value);
  MT_LOGD_IF(value % 2 == 0, "Value is even: {}", value);
  MT_LOGI_IF(value >= 0 && value <= 100, "Value is between 0 and 100: {}", value);
  MT_LOGW_IF(value != 0, "Value is non-zero: {}", value);
  MT_LOGE_IF(value == 42, "Value is 42: {}", value);

  value = 39;
  MT_LOGT_IF(value > 40, "This should not be logged");
  MT_LOGD_IF(value % 2 == 0, "This should not be logged");
  MT_LOGI_IF(value >= 0 && value <= 100, "Value is between 0 and 100: {}", value);
  MT_LOGW_IF(value != 0, "Value is non-zero: {}", value);
  MT_LOGE_IF(value == 42, "This should not be logged");
}

// Test with custom types that can be converted to string
class CustomType {
 public:
  explicit CustomType(const std::string& value) : value_(value) {}

  friend std::ostream& operator<<(std::ostream& os, const CustomType& obj) {
    os << "CustomType(" << obj.value_ << ")";
    return os;
  }

  [[nodiscard]] const std::string& getValue() const {
    return value_;
  }

 private:
  std::string value_;
};

// Add formatter specialization for CustomType
namespace fmt {
template <>
struct formatter<CustomType> : formatter<std::string> {
  auto format(const CustomType& obj, format_context& ctx) const {
    return formatter<std::string>::format("CustomType(" + obj.getValue() + ")", ctx);
  }
};
} // namespace fmt

TEST(LogTest, CustomTypes) {
  CustomType obj("test");
  MT_LOGT("Custom type: {}", obj);
  MT_LOGD("Custom type: {}", obj);
  MT_LOGI("Custom type: {}", obj);
  MT_LOGW("Custom type: {}", obj);
  MT_LOGE("Custom type: {}", obj);
}
