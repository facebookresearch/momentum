/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// @file span_compat.h
/// @brief Provides a C++17-compatible span type alias.
///
/// This header provides `momentum::span<T>` which aliases to `std::span<T>` when
/// C++20 is available, and falls back to `gsl::span<T>` for C++17 compilers.
/// This enables cross-platform compatibility for platforms that don't yet support C++20.

// Detect C++20 support
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
#define MOMENTUM_HAS_CPP20_SPAN 1
#else
#define MOMENTUM_HAS_CPP20_SPAN 0
#endif

#if MOMENTUM_HAS_CPP20_SPAN
#include <span>
#else
#include <gsl/span>
#endif

namespace momentum {

#if MOMENTUM_HAS_CPP20_SPAN

/// Span type alias using std::span (C++20).
template <typename T>
using span = std::span<T>;

#else

/// Span type alias using gsl::span (C++17 fallback).
template <typename T>
using span = gsl::span<T>;

#endif

} // namespace momentum
