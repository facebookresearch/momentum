/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skeleton_utility.h>
#include <momentum/common/profile.h>

namespace momentum {

ModelParameters extrapolateModelParameters(
    const ModelParameters& previous,
    const ModelParameters& current,
    const float factor,
    const float maxDelta) {
  MT_PROFILE_FUNCTION();
  // check if we have any reasonable data
  if (current.size() != previous.size()) {
    return current;
  }

  // yes, perform extrapolation
  const ModelParameters result =
      current.v + (current.v - previous.v).cwiseMin(maxDelta).cwiseMax(-maxDelta) * factor;

  return result;
}

ModelParameters extrapolateModelParameters(
    const ModelParameters& previous,
    const ModelParameters& current,
    const ParameterSet& activeParams,
    const float factor,
    const float maxDelta) {
  MT_PROFILE_FUNCTION();
  if (current.size() != previous.size()) {
    return current;
  }

  // yes, perform extrapolation
  // Cap the maximum extrapolation angle to avoid issues where the extrapolation grows over time.

  const auto sz = current.size();
  ModelParameters result = current;
  for (Eigen::Index i = 0; i < sz; ++i) {
    if (activeParams.test(i)) {
      result(i) = current(i) + factor * std::clamp(current(i) - previous(i), -maxDelta, maxDelta);
    }
  }

  return result;
}

} // namespace momentum
