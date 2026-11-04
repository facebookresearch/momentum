/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/marker.h"

#include <unordered_map>

namespace momentum {

std::vector<MarkerTrack> groupMarkerTracks(
    std::span<const std::vector<Marker>> frames,
    float minConfidence) {
  std::vector<MarkerTrack> tracks;
  std::unordered_map<std::string, size_t> trackByName;
  for (size_t frame = 0; frame < frames.size(); ++frame) {
    for (const Marker& marker : frames[frame]) {
      if (marker.occluded || marker.confidence < minConfidence) {
        continue;
      }
      const auto [entry, added] = trackByName.try_emplace(marker.name, tracks.size());
      if (added) {
        tracks.emplace_back().name = marker.name;
      }
      MarkerTrack& track = tracks.at(entry->second);
      track.frames.push_back(frame);
      track.positions.push_back(marker.pos);
      track.confidences.push_back(marker.confidence);
    }
  }
  return tracks;
}

} // namespace momentum
