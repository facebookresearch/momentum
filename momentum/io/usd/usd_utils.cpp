/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_utils.h"

#include <cctype>

namespace momentum {

std::string sanitizePrimName(const std::string& name) {
  std::string result = name;
  for (auto& c : result) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }
  // Prim names must start with a letter or underscore, not a digit
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result.insert(0, "_");
  }
  if (result.empty()) {
    result = "_";
  }
  return result;
}

} // namespace momentum
