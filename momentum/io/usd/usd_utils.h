/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace momentum {

/// Sanitize a string for use as a USD SDF prim name.
///
/// USD prim names must match [a-zA-Z_][a-zA-Z0-9_]*. This function replaces
/// invalid characters with underscores and prepends an underscore if the name
/// starts with a digit. Empty names become "_".
///
/// @param[in] name The original name to sanitize.
/// @return A valid USD prim name.
std::string sanitizePrimName(const std::string& name);

} // namespace momentum
