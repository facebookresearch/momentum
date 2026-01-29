/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/math/fwd.h>

#include <pybind11/numpy.h>

#include <optional>
#include <tuple>

namespace pymomentum {

/// Convert an MPPCA model to numpy arrays.
/// Returns (pi, mu, W, sigma, parameter_indices) where:
/// - pi: [nMixtures] mixture weights
/// - mu: [nMixtures, dimension] means
/// - W: [nMixtures, rank, dimension] PCA basis vectors
/// - sigma: [nMixtures] noise standard deviations
/// - parameter_indices: [dimension] mapping to parameter transform indices (-1 if not found)
std::tuple<
    pybind11::array_t<float>,
    pybind11::array_t<float>,
    pybind11::array_t<float>,
    pybind11::array_t<float>,
    pybind11::array_t<int>>
mppcaToArrays(
    const momentum::Mppca& mppca,
    std::optional<const momentum::ParameterTransform*> paramTransform);

} // namespace pymomentum
