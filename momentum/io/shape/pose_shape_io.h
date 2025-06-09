/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/pose_shape.h>

namespace momentum {

/// Loads pose shape data from a binary file.
///
/// @param filename Path to the pose shape data file
/// @param character Character containing skeleton and mesh for validation and joint mapping
/// @return PoseShape with loaded data, or empty PoseShape if file cannot be opened
PoseShape loadPoseShape(const std::string& filename, const Character& character);

} // namespace momentum
