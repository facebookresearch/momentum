/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/c3d_io.h"

#include "momentum/character/marker.h"
#include "momentum/common/log.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/marker/conversions.h"

#include <ezc3d/ezc3d_all.h>

#include <optional>
#include <set>
#include <unordered_map>

namespace momentum {

namespace {

// Parse markerLabelIn and find the actor name string.
// For a named actor, the marker label format can be actor:marker or actor_marker (could have more
// possibilities); for an unnamed actor, it's simply the marker name. There could be multiple named
// actor in a session and/or an unamed actor.
std::string findSubjectName(const std::string& markerLabelIn) {
  constexpr auto kNamespaceSep = ':';
  auto sepPos = markerLabelIn.find(kNamespaceSep);
  if (sepPos == std::string::npos) {
    constexpr auto kUnderscoreSep = "_";
    sepPos = markerLabelIn.find_last_of(kUnderscoreSep);
    if (sepPos != std::string::npos) {
      // If candidate name is side indicator, do not use as subject name.
      auto candidateName = markerLabelIn.substr(0, sepPos);
      std::transform(candidateName.begin(), candidateName.end(), candidateName.begin(), ::tolower);
      const std::set<std::string> kSideStrSet = {"l", "r", "left", "right"};
      if (kSideStrSet.find(candidateName) != kSideStrSet.end()) {
        return {};
      }
    } else {
      return {};
    }
  }

  return markerLabelIn.substr(0, sepPos);
}

bool isMarkerValid(const std::string_view label, std::span<const std::string> validMarkerNames) {
  // there might be dangling markers with suffix, eg.
  // an official label "T10", but also "T10-1", "T10-2" for a few stray frames
  // We will just ignore these because they probably have been interpolated to the official
  // label #TODO: see if we still want to keep this
  if ((label.size() >= 2 && label[label.size() - 2] == '-') ||
      (label.size() >= 3 && label[label.size() - 3] == '-')) {
    return false;
  }

  // check if the label is in the valid list
  if (!validMarkerNames.empty()) {
    if (std::find(validMarkerNames.begin(), validMarkerNames.end(), label) ==
        validMarkerNames.end()) {
      return false;
    }
  }

  return true;
}

} // namespace

namespace {

// Returns the validated unit string from the c3d POINT group, or std::nullopt on error.
std::optional<std::string> getValidatedUnit(const ezc3d::ParametersNS::GroupNS::Group& pointGroup) {
  constexpr auto kUnitStr = "UNITS";
  const auto& unit = pointGroup.parameter(kUnitStr).valuesAsString();
  if (unit.size() != 1) {
    MT_LOGE("{}: Invalid c3d file: no unit information found!", __func__);
    return std::nullopt;
  }
  const auto& unitStr = unit[0];
  if (unitStr != "mm" && unitStr != "m" && unitStr != "cm" && unitStr != "dm") {
    MT_LOGE("{}: Unknown unit string '{}' found in the file.", __func__, unitStr);
    return std::nullopt;
  }
  MT_LOGW_IF(
      unitStr != "mm",
      "{}: Unit '{}' is not mm. Translating a C3D file that contains 3D point data stored in any units other than millimeters is extremely complex and any mistakes will render the file invalid.",
      __func__,
      unitStr);
  return unitStr;
}

// Load all point labels from POINT.LABELS plus the LABEL2/LABEL3/... overflow sections.
// The c3d format limits LABELS to 255 entries per section.
// Returns an empty vector if the total label count is inconsistent with the header.
std::vector<std::string> loadPointLabels(
    const ezc3d::ParametersNS::GroupNS::Group& pointGroup,
    size_t kPointsPerFrame) {
  constexpr auto kLabelStr = "LABELS";
  constexpr auto kNumLabelsPerSection = 255;
  constexpr size_t kAdditionSectionStartIndex = 2;

  auto pointLabels = pointGroup.parameter(kLabelStr).valuesAsString();
  const size_t numOfExtraSections = kPointsPerFrame / kNumLabelsPerSection;
  pointLabels.reserve(kPointsPerFrame);
  for (size_t sectionId = kAdditionSectionStartIndex;
       sectionId < kAdditionSectionStartIndex + numOfExtraSections;
       sectionId++) {
    const auto labelStr = kLabelStr + std::to_string(sectionId);
    const auto& additionalPointLabels = pointGroup.parameter(labelStr).valuesAsString();
    pointLabels.insert(
        pointLabels.end(), additionalPointLabels.begin(), additionalPointLabels.end());
  }
  if (pointLabels.size() < kPointsPerFrame) {
    MT_LOGE("{}: Number of point labels loaded isn't consistent with the header! ", __func__);
    return {};
  }
  pointLabels.resize(kPointsPerFrame);
  return pointLabels;
}

// Group point indices by their parsed subject name.
// A Subject is a collection of points that should be grouped together to represent an object.
std::unordered_map<std::string, std::vector<size_t>> groupPointsBySubject(
    std::span<const std::string> pointLabels) {
  std::unordered_map<std::string, std::vector<size_t>> subjectNameMap;
  subjectNameMap.reserve(pointLabels.size());
  for (size_t iLabel = 0; iLabel < pointLabels.size(); ++iLabel) {
    subjectNameMap[findSubjectName(pointLabels[iLabel])].push_back(iLabel);
  }
  return subjectNameMap;
}

// Initialize per-actor MarkerSequence/template-actor arrays and return the mapping from
// each c3d point index to (actorId, markerId). Indices not mapped to any marker stay (-1, -1).
// Resizes ``resultAnim`` and ``templateActors`` to ``subjectNameMap.size()``.
std::vector<std::pair<int, int>> buildSubjectStructures(
    const std::unordered_map<std::string, std::vector<size_t>>& subjectNameMap,
    std::span<const std::string> pointLabels,
    std::span<const std::string> validMarkerNames,
    float fps,
    std::vector<MarkerSequence>& resultAnim,
    std::vector<std::vector<Marker>>& templateActors) {
  resultAnim.assign(subjectNameMap.size(), {});
  templateActors.assign(subjectNameMap.size(), {});
  std::vector<std::pair<int, int>> markerMap(pointLabels.size(), {-1, -1});
  size_t actorId = 0;
  for (const auto& [subjectName, pointIndices] : subjectNameMap) {
    int markerCount = 0;
    // ``actorId`` is bounded by ``subjectNameMap.size()``, which we just used
    // to size both vectors above; the analyzer cannot trace this through the
    // range-based for loop counter.
    auto& actorSequence = resultAnim.at(actorId);
    actorSequence.fps = fps;
    actorSequence.name = subjectName;
    auto& templateActor = templateActors.at(actorId);
    for (const auto kPointIdx : pointIndices) {
      if (kPointIdx >= pointLabels.size()) {
        continue;
      }
      auto pointLabel = pointLabels[kPointIdx];
      if (!subjectName.empty()) {
        pointLabel = pointLabel.substr(subjectName.size() + 1);
      }
      if (!isMarkerValid(pointLabel, validMarkerNames)) {
        continue;
      }
      // ``kPointIdx`` is checked above against ``pointLabels.size()``, which
      // also sizes ``markerMap``.
      markerMap.at(kPointIdx) = {static_cast<int>(actorId), markerCount};
      Marker marker;
      marker.name = pointLabel;
      marker.occluded = true;
      templateActor.push_back(marker);
      ++markerCount;
    }
    ++actorId;
  }
  return markerMap;
}

// Reset all template markers to occluded, then update them from one frame's c3d points.
// Returns true if any marker was visible (residual >= 0, non-NaN, non-zero) in this frame.
bool updateFrameMarkers(
    const ezc3d::DataNS::Frame& frameData,
    std::span<const std::pair<int, int>> markerMap,
    UpVector up,
    const std::string& unitStr,
    std::vector<std::vector<Marker>>& templateActors) {
  for (auto& templateActor : templateActors) {
    for (auto& marker : templateActor) {
      marker.occluded = true;
    }
  }

  bool hasVisible = false;
  const auto& framePoints = frameData.points().points();
  for (size_t pointId = 0; pointId < markerMap.size(); ++pointId) {
    const auto [kActorId, kMarkerId] = markerMap[pointId];
    if (kActorId == -1 || kMarkerId == -1) {
      continue;
    }
    const auto& pointData = framePoints[pointId];
    Vector3d pos{pointData.x(), pointData.y(), pointData.z()};
    if (std::isnan(pos[0]) || std::isnan(pos[1]) || std::isnan(pos[2])) {
      continue;
    }
    if (pos == Eigen::Vector3d::Zero()) {
      continue;
    }
    // residual < 0 indicates that the data is invalid;
    // residual == 0 indicates that the data is generated.
    if (pointData.residual() < 0) {
      continue;
    }
    // ``kActorId`` / ``kMarkerId`` come from ``markerMap`` populated by
    // ``buildSubjectStructures`` and are guaranteed to address the matching
    // ``templateActors`` entry; bounds-checked accessors guard against any
    // future divergence.
    if (static_cast<size_t>(kActorId) >= templateActors.size() ||
        static_cast<size_t>(kMarkerId) >= templateActors.at(kActorId).size()) {
      continue;
    }
    auto& marker = templateActors.at(kActorId).at(kMarkerId);
    marker.occluded = false;
    marker.pos = toMomentumVector3(pos, up, unitStr);
    hasVisible = true;
  }
  return hasVisible;
}

// Shared parsing logic, agnostic to whether the c3d came from a file path or an istream.
std::vector<MarkerSequence>
loadC3dFromObject(ezc3d::c3d& c3dFile, UpVector up, std::span<const std::string> validMarkerNames) {
  try {
    const auto& header = c3dFile.header();
    const size_t kPointsPerFrame = header.nb3dPoints();
    const auto kFrameCount = header.nbFrames();
    const float fps = header.frameRate();
    MT_LOGD("{}: Found {} frames", __func__, kFrameCount);
    MT_LOGD("{}: Frame rate {}", __func__, fps);

    constexpr auto kPointGroupStr = "POINT";
    const auto& pointGroup = c3dFile.parameters().group(kPointGroupStr);

    const auto unitStr = getValidatedUnit(pointGroup);
    if (!unitStr.has_value()) {
      return {};
    }

    const auto pointLabels = loadPointLabels(pointGroup, kPointsPerFrame);
    if (pointLabels.empty()) {
      return {};
    }

    const auto subjectNameMap = groupPointsBySubject(pointLabels);
    const size_t kNumOfSubjects = subjectNameMap.size();
    std::vector<MarkerSequence> resultAnim(kNumOfSubjects);
    std::vector<std::vector<Marker>> templateActors(kNumOfSubjects);
    const auto markerMap = buildSubjectStructures(
        subjectNameMap, pointLabels, validMarkerNames, fps, resultAnim, templateActors);

    bool hasAnim = false;
    const auto& frames = c3dFile.data().frames();
    for (int frameId = 0; frameId < kFrameCount; ++frameId) {
      hasAnim |= updateFrameMarkers(frames[frameId], markerMap, up, *unitStr, templateActors);
      for (size_t actorId = 0; actorId < templateActors.size(); ++actorId) {
        resultAnim[actorId].frames.push_back(templateActors[actorId]);
      }
    }

    if (!hasAnim) {
      return {};
    }
    return resultAnim;
  } catch (std::exception& e) {
    MT_LOGE("{}: Exception: {}", __func__, e.what());
    return {};
  } catch (...) {
    MT_LOGE("{}: Unknown c3d reading error", __func__);
    return {};
  }
}

} // namespace

std::vector<MarkerSequence>
loadC3d(const std::string& filename, UpVector up, std::span<const std::string> validMarkerNames) {
  try {
    ezc3d::c3d c3dFile(filename);
    return loadC3dFromObject(c3dFile, up, validMarkerNames);
  } catch (const std::exception& e) {
    MT_LOGE("{}: failed to open {}: {}", __func__, filename, e.what());
    return {};
  }
}

#ifdef MOMENTUM_WITH_EZC3D_ISTREAM
std::vector<MarkerSequence> loadC3d(
    std::span<const std::byte> bytes,
    UpVector up,
    std::span<const std::string> validMarkerNames) {
  try {
    ispanstream stream(bytes);
    ezc3d::c3d c3dFile(static_cast<std::istream&>(stream));
    return loadC3dFromObject(c3dFile, up, validMarkerNames);
  } catch (const std::exception& e) {
    MT_LOGE("{}: failed to read c3d from buffer ({} bytes): {}", __func__, bytes.size(), e.what());
    return {};
  }
}
#endif

} // namespace momentum
