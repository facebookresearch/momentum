/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>

namespace momentum {

/// Return a name that is unique within a set of existing names.
///
/// If @p requestedName is not already present in @p existingNames it is returned unchanged.
/// Otherwise a numeric suffix is appended ("name_1", "name_2", ...) until an unused name is found.
///
/// @param requestedName The desired name.
/// @param existingNames Any container with a `find()` member returning an end-comparable iterator
///   (e.g. std::unordered_map / std::set keyed by name).
/// @return The first name not present in @p existingNames.
template <typename T>
std::string makeUniqueName(const std::string& requestedName, const T& existingNames) {
  if (existingNames.find(requestedName) == existingNames.end()) {
    return requestedName;
  }

  for (std::size_t suffix = 1;; ++suffix) {
    std::string candidate = requestedName + "_" + std::to_string(suffix);
    if (existingNames.find(candidate) == existingNames.end()) {
      return candidate;
    }
  }
}

} // namespace momentum
