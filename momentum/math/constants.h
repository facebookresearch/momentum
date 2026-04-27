/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>
#include <type_traits>

namespace momentum {

/// Returns a quiet NaN value of type T.
template <typename T = float>
[[nodiscard]] constexpr T nan() noexcept {
  return std::numeric_limits<T>::quiet_NaN();
}

/// Returns a type-appropriate tolerance/epsilon value.
///
/// @param FloatEps Tolerance returned when T is `float` (default: 1e-7).
/// @param DoubleEps Tolerance returned when T is `double` (default: 1e-16).
/// @return The selected tolerance for T. Only `float` and `double` are supported;
///         other types fall through and produce an undefined return value.
// TODO: Add a static_assert or constexpr error path for unsupported T to avoid
// silent undefined-return-value behavior for non-float/double types.
template <typename T>
[[nodiscard]] constexpr T Eps(T FloatEps = T(1e-7), T DoubleEps = T(1e-16)) {
  if constexpr (std::is_same_v<T, float>) {
    return FloatEps;
  } else if constexpr (std::is_same_v<T, double>) {
    return DoubleEps;
  }
}

/// Returns the natural logarithm of 2 (log_e 2).
template <typename T = float>
[[nodiscard]] constexpr T ln2() noexcept {
  return 0.69314718055994530942;
}

/// Returns the mathematical constant pi.
///
/// @note Use this instead of `M_PI`. Windows MSVC does not define `M_PI` by
///       default (it requires `_USE_MATH_DEFINES` before `<cmath>`). This
///       function is always available on all platforms.
template <typename T = float>
[[nodiscard]] constexpr T pi() noexcept {
  return 3.14159265358979323846;
}

/// Returns 2 * pi.
///
/// @note Use this instead of `2 * M_PI`. Windows MSVC does not define `M_PI`
///       by default (it requires `_USE_MATH_DEFINES` before `<cmath>`). This
///       function is always available on all platforms.
template <typename T = float>
[[nodiscard]] constexpr T twopi() noexcept {
  return T(2) * pi<T>();
}

/// Converts the given angle x (in degrees) to radians.
///
/// If called without an argument, it returns the conversion factor from degrees to radians.
template <typename T = float>
[[nodiscard]] constexpr T toRad(T x = T(1)) noexcept {
  static_assert(std::is_floating_point_v<T>, "toRad requires a floating point argument.");
  return x * pi<T>() / T(180);
}
// TODO: Support Eigen types to provide consistent conversion function usage with a single argument.

/// Converts the given angle x (in radians) to degrees.
///
/// If called without an argument, it returns the conversion factor from radians to degrees.
template <typename T = float>
[[nodiscard]] constexpr T toDeg(T x = T(1)) noexcept {
  static_assert(std::is_floating_point_v<T>, "toDeg requires a floating point argument.");
  return x * T(180) / pi<T>();
}

/// Converts the given length x (in centimeters) to meters.
///
/// If called without an argument, it returns the conversion factor from centimeters to meters.
template <typename T = float>
[[nodiscard]] constexpr T toM(T x = T(1)) noexcept {
  static_assert(std::is_floating_point_v<T>, "toM requires a floating point argument.");
  return x * T(0.01);
}

/// Converts the given length x (in meters) to centimeters.
///
/// If called without an argument, it returns the conversion factor from meters to centimeters.
template <typename T = float>
[[nodiscard]] constexpr T toCm(T x = T(1)) noexcept {
  static_assert(std::is_floating_point_v<T>, "toCm requires a floating point argument.");
  return x * T(100);
}

} // namespace momentum
