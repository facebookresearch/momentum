/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/types.h>

namespace momentum {

inline constexpr float kDefaultExtrapolateFactor = 0.8f;
inline constexpr float kDefaultExtrapolateMaxDelta = 0.4f; // ~23 degrees

/// Extrapolates model parameters by first clamping the difference between current and previous
/// parameters to the range [-maxDelta, maxDelta], and then scaling this clamped difference by
/// `factor`. Returns the current parameters when sizes mismatch.
[[nodiscard]] ModelParameters extrapolateModelParameters(
    const ModelParameters& previous,
    const ModelParameters& current,
    float factor = kDefaultExtrapolateFactor,
    float maxDelta = kDefaultExtrapolateMaxDelta);

/// Extrapolates model parameters considering active parameters. The extrapolation is done by first
/// clamping the difference between current and previous parameters to the range [-maxDelta,
/// maxDelta] for each active parameter, and then scaling this clamped difference by `factor`.
/// Returns the current parameters when sizes mismatch.
[[nodiscard]] ModelParameters extrapolateModelParameters(
    const ModelParameters& previous,
    const ModelParameters& current,
    const ParameterSet& activeParams,
    float factor = kDefaultExtrapolateFactor,
    float maxDelta = kDefaultExtrapolateMaxDelta);

} // namespace momentum
