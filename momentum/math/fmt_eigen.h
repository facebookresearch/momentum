/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/ostream.h>
#include <Eigen/Core>

/// @file
/// @brief libfmt formatter support for Eigen types.
///
/// Provides an `fmt::formatter` specialization for any type derived from
/// `Eigen::DenseBase` (matrices, vectors, and expression templates). This
/// makes Eigen objects directly usable with `fmt::format`, `fmt::print`, and
/// related APIs by delegating to Eigen's `operator<<` via `ostream_formatter`.
///
/// Usage: simply `#include "momentum/math/fmt_eigen.h"` and format Eigen
/// values normally, e.g. `fmt::format("{}", matrix)` or
/// `fmt::print("v = {}\n", vector)`.
///
/// Only enabled for libfmt >= 10.0.0, which introduced `ostream_formatter`.
/// Note: `Eigen::Quaternion` is not derived from `Eigen::DenseBase` and is
/// therefore not covered by this formatter.

#if FMT_VERSION > 100000

// See https://stackoverflow.com/a/73755864 for the SFINAE pattern.
#ifndef MOMENTUM_FMT_EIGEN_SHARED_PTR_FORMATTER
#define MOMENTUM_FMT_EIGEN_SHARED_PTR_FORMATTER
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
    : ostream_formatter {};
#endif

#endif
