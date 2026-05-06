/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

/// Throws an exception of a specified type with a formatted message.
/// @param Exception The type of exception to throw.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_T(std::invalid_argument, "Invalid argument: {}", arg);
///   MT_THROW_T(std::bad_array_new_length); // No message
/// @endcode
#define MT_THROW_T(Exception, ...) ::momentum::detail::throwImpl<Exception>(__VA_ARGS__)

/// Throws a std::runtime_error with a formatted message.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW("Error occurred: {}", errorDetail);
/// @endcode
#define MT_THROW(...) MT_THROW_T(::momentum::detail::DefaultException, ##__VA_ARGS__)

/// Conditionally throws an exception of a specified type with a formatted message if the condition
/// is true.
/// @param Condition The condition to evaluate.
/// @param Exception The type of exception to throw.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_IF_T(x > y, std::out_of_range, "x ({}) is greater than y ({})", x, y);
///   MT_THROW_IF_T(x > y, std::bad_array_new_length); // No message
/// @endcode
#define MT_THROW_IF_T(Condition, Exception, ...) \
  do {                                           \
    if (Condition) {                             \
      MT_THROW_T(Exception, ##__VA_ARGS__);      \
    }                                            \
  } while (0)

/// Conditionally throws a std::runtime_error with a formatted message if the condition is true.
/// @param Condition The condition to evaluate.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_IF(x == nullptr, "x cannot be nullptr");
/// @endcode
#define MT_THROW_IF(Condition, ...) \
  MT_THROW_IF_T(Condition, ::momentum::detail::DefaultException, ##__VA_ARGS__)

namespace momentum::detail {

using DefaultException = std::runtime_error;

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)

template <typename Exception = DefaultException, typename... Args>
[[noreturn]] void throwImpl(fmt::format_string<Args...> format, Args&&... args) {
  throw Exception{fmt::format(format, std::forward<Args>(args)...)};
}

/// No-message overload. Selects a default-constructed exception when possible (e.g.
/// `std::bad_array_new_length`); otherwise falls back to constructing from an empty string,
/// which is required for exceptions like `std::runtime_error`/`std::logic_error` that have
/// no default constructor.
template <typename Exception>
[[noreturn]] void throwImpl() {
  if constexpr (std::is_constructible_v<Exception>) {
    throw Exception{};
  } else {
    throw Exception{""};
  }
}

#else

/// Fallback when exceptions are disabled: write the formatted message to stderr and abort.
template <typename Exception = DefaultException, typename... Args>
[[noreturn]] void throwImpl(fmt::format_string<Args...> format, Args&&... args) {
  std::fprintf(
      stderr, "FATAL ERROR: %s\n", fmt::format(format, std::forward<Args>(args)...).c_str());
  std::abort();
}

template <typename Exception>
[[noreturn]] void throwImpl() {
  std::fputs("FATAL ERROR (no message)\n", stderr);
  std::abort();
}

#endif

} // namespace momentum::detail
