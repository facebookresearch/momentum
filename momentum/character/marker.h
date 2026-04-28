/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <string>
#include <vector>

namespace momentum {

/// A single motion capture marker observation (position, occlusion, confidence).
struct Marker {
  /// The name of the marker.
  std::string name = "Undefined";

  /// The 3D position of the marker, in centimeters.
  Vector3d pos = {0.0, 0.0, 0.0};

  /// True if the marker is occluded in this frame, false if visible.
  bool occluded = true;

  /// Trust weight for this observation (e.g. a keypoint score). Conventionally in [0, 1],
  /// but the range is not enforced.
  float confidence = 1.0;
};

/// All frames from a single motion capture sequence for one subject (actor).
///
/// @invariant Every inner `frames[i]` vector must have the same size — markers are
/// indexed positionally and the marker count is fixed across the sequence.
struct MarkerSequence {
  /// Identifier for the actor or subject this sequence belongs to.
  std::string name;

  /// All marker observations across the sequence. Layout: `[numFrames][numMarkers]`.
  std::vector<std::vector<Marker>> frames;

  /// Frame rate of the capture, in frames per second.
  float fps = 30.0f;
};

} // namespace momentum
