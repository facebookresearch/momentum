/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/marker_io.h"

#include "momentum/character/marker.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/fbx/fbx_io.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/marker/c3d_io.h"
#include "momentum/io/marker/trc_io.h"

namespace momentum {

namespace {
size_t markerCount(const std::vector<std::vector<Marker>>& frames) {
  size_t maxCountMarkers = 0;
  for (const auto& fr : frames) {
    const size_t markerCount =
        std::count_if(fr.begin(), fr.end(), [](const Marker& marker) { return !marker.occluded; });
    maxCountMarkers = std::max(markerCount, maxCountMarkers);
  }
  return maxCountMarkers;
}
} // namespace

std::vector<MarkerSequence> loadMarkers(
    const std::string& filename,
    UpVector up,
    std::span<const std::string> validMarkerNames) {
  const std::string ext = filesystem::path(filename).extension().string();
  try {
    if (ext == ".c3d") {
      return loadC3d(filename, up, validMarkerNames);
    } else if (ext == ".trc") {
      return {loadTrc(filename, up)};
    } else if (ext == ".glb") {
      return {loadMarkerSequence(filename)};
    } else if (ext == ".fbx") {
      return {loadFbxMarkerSequence(filename)};
    } else {
      MT_LOGE("{} Unknown marker file type {}", __func__, filename);
      return {};
    }
  } catch (...) {
    MT_LOGE("{}: Unknown error in {}", __func__, filename);
    return {};
  }
}

std::optional<MarkerSequence> loadMarkersForMainSubject(
    const std::string& filename,
    UpVector up,
    std::span<const std::string> validMarkerNames) {
  const std::vector<MarkerSequence> markerSequences = loadMarkers(filename, up, validMarkerNames);
  const int subjectID = findMainSubjectIndex(markerSequences);
  if (subjectID < 0) {
    return {};
  } else {
    return {markerSequences.at(subjectID)};
  }
}

int findMainSubjectIndex(std::span<const MarkerSequence> markerSequences) {
  if (markerSequences.empty()) {
    return -1;
  }

  // We define the main subject as one with the most markers. Ideally, the main subject is the one
  // with a proper name, because unlabeled markers are usually the largest number, except when there
  // is only one actor.
  // XXX Note that the logic could still break, eg. when a person is unnamed but there are named
  // objects.
  //
  // We will go through the sequence to find the max number of visible markers. This is quite slow
  // but more robust.
  const size_t numActors = markerSequences.size();
  std::vector<int> maxMarkers(numActors, 0);
  // count markers for each actor
  for (size_t iActor = 0; iActor < numActors; ++iActor) {
    maxMarkers.at(iActor) = markerCount(markerSequences[iActor].frames);
  }

  // Special case when there's only one actor.
  if (numActors == 1) {
    if (maxMarkers.at(0) > 0) {
      return 0;
    } else {
      return -1;
    }
  }

  // find the max marker count with a name
  size_t maxCount = 0;
  size_t actorID = 0; // default to be the first actor
  for (size_t iActor = 0; iActor < numActors; ++iActor) {
    const size_t count = maxMarkers.at(iActor);
    auto actorName = markerSequences[iActor].name;
    std::transform(actorName.begin(), actorName.end(), actorName.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    if (count > maxCount && !actorName.empty() && actorName != "unlabeled") {
      maxCount = count;
      actorID = iActor;
    }
  }

  if (maxCount > 0) {
    return actorID;
  } else {
    return -1;
  }
}
} // namespace momentum
