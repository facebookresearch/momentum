/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <limits>
#include <span>
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

/// The kept observations of one marker in a marker sequence, in frame order.
struct MarkerTrack {
  /// The name of the marker.
  std::string name;

  /// The frame index of each observation.
  std::vector<size_t> frames;

  /// The position of the marker in each observation, in centimeters.
  std::vector<Vector3d> positions;

  /// The confidence of each observation.
  std::vector<float> confidences;
};

/// Group the observations of a marker sequence into one track per marker name, in the order of
/// each name's first kept observation.
///
/// Occluded observations, and observations whose confidence is below @p minConfidence, are left
/// out; a marker without any kept observation gets no track.
///
/// @param frames The marker observations of every frame (see MarkerSequence::frames).
/// @param minConfidence The confidence below which an observation is left out; by default none is.
/// @return The tracks.
[[nodiscard]] std::vector<MarkerTrack> groupMarkerTracks(
    std::span<const std::vector<Marker>> frames,
    float minConfidence = -std::numeric_limits<float>::infinity());

} // namespace momentum
