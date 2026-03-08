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

/// Marker represents a motion capture marker that defines an active marker during motion capture.
struct Marker {
  /// The name of the marker, default is "Undefined".
  std::string name = "Undefined";

  /// The 3D position of the marker as a Vector3d object, where the unit is assumed to be in
  /// centimeters.
  Vector3d pos = {0.0, 0.0, 0.0};

  /// The occlusion status of the marker, true if occluded, false otherwise.
  bool occluded = true;

  /// How much to trust this observation, eg. a keypoint. The value should be in [0, 1] but not
  /// enforced.
  float confidence = 1.0;
};

/// MarkerSequence stores all the frames from a single capture sequence for one subject (motion
/// capture actor).
///
/// Each frame is a std::vector<Marker> containing the position and occlusion status of
/// all the markers placed on the subject.
/// The size of the std::vector<Marker> must be consistent for all frames.
struct MarkerSequence {
  /// Name of the actor sequence (typically a unique subject name or ID)
  std::string name;

  /// A 2D vector that specifies the Marker (name/position/occlusion) for all markers
  /// throughout all captured frames. Size: [numFrames][numMarkers]
  std::vector<std::vector<Marker>> frames;

  /// The frame rate of the motion capture sequence in frames per second (default is 30.0)
  float fps = 30.0f;
};

} // namespace momentum
