/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gsl/gsl>

namespace momentum {

/// Casts a span of bytes to a span of a different type
///
/// This function takes a span of bytes and returns a span of a different type, by reinterpreting
/// the bytes as the new type. This function is useful when working with serialized data or when
/// dealing with raw memory. The function assumes that the size of the new type is an exact multiple
/// of the size of the original span.
///
/// @tparam T The type to cast the span to
/// @param bs The original span of bytes to cast
///
/// @return A span of the new type, with the same number of elements as the original span
template <typename T>
[[nodiscard]] std::span<T> cast_span(std::span<std::byte> bs) {
  auto ptr = reinterpret_cast<T*>(bs.data());
  auto tsize = gsl::narrow<decltype(bs)::size_type>(sizeof(T));
  return {ptr, bs.size_bytes() / tsize};
}

/// Casts a span of const bytes to a span of a different type
///
/// This function takes a span of const bytes and returns a span of a different type, by
/// reinterpreting the bytes as the new type. This function is useful when working with serialized
/// data or when dealing with raw memory. The function assumes that the size of the new type is an
/// exact multiple of the size of the original span.
///
/// @tparam T The type to cast the span to
/// @param bs The original span of const bytes to cast
///
/// @return A span of the new type, with the same number of elements as the original span
template <typename T>
[[nodiscard]] std::span<const T> cast_span(std::span<const std::byte> bs) {
  auto ptr = reinterpret_cast<const T*>(bs.data());
  auto tsize = gsl::narrow<decltype(bs)::size_type>(sizeof(T));
  return {ptr, bs.size_bytes() / tsize};
}

} // namespace momentum
