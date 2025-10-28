/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/c3d_io.h"

#include "momentum/character/marker.h"
#include "momentum/common/log.h"
#include "momentum/io/marker/conversions.h"

#include <ezc3d/ezc3d_all.h>

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

std::vector<MarkerSequence>
loadC3d(const std::string& filename, UpVector up, std::span<const std::string> validMarkerNames) {
  std::vector<MarkerSequence> resultAnim;
  float fps = 0.0f;

  bool hasAnim = false;

  try {
    // Load animation Info from the header
    ezc3d::c3d c3dFile(filename);
    const auto& header = c3dFile.header();
    const auto kPointsPerFrame = header.nb3dPoints();
    const auto kFrameCount = header.nbFrames();
    fps = header.frameRate();
    MT_LOGD("{}: Found {} frames", __func__, kFrameCount);
    MT_LOGD("{}: Frame rate {}", __func__, fps);

    // Load Point Labels from Parameter Section
    constexpr auto kPointGroupStr = "POINT";
    constexpr auto kLabelStr = "LABELS";
    const auto& parameters = c3dFile.parameters();
    const auto& pointGroup = parameters.group(kPointGroupStr);

    // Load Unit
    constexpr auto kUnitStr = "UNITS";
    const auto& unit = pointGroup.parameter(kUnitStr).valuesAsString();
    if (unit.size() != 1) {
      MT_LOGE("{}: Invalid c3d file: no unit information found!", __func__);
      return {};
    }

    const auto& unitStr = unit[0];
    if (unitStr != "mm" && unitStr != "m" && unitStr != "cm" && unitStr != "dm") {
      MT_LOGE("{}: Unknown unit string '{}' found in the file.", __func__, unitStr);
      return {};
    }
    MT_LOGW_IF(
        unitStr != "mm",
        "{}: Unit '{}' is not mm. Translating a C3D file that contains 3D point data stored in any units other than millimeters is extremely complex and any mistakes will render the file invalid.",
        __func__,
        unitStr);

    // Point:Labels section can store max 255 labels. For the other labels, find them in section
    // LABEL2, LABEL3.. Each additional section can contain max 255 labels. The file may contain
    // more labels than kPointsPerFrame since some points may be invalid.
    constexpr auto kNumLabelsPerSection = 255;
    auto pointLabels = pointGroup.parameter(kLabelStr).valuesAsString();
    size_t numOfExtraSections = kPointsPerFrame / kNumLabelsPerSection;

    constexpr size_t kAdditionSectionStartIndex = 2;
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
    // Only the first kPointsPerFrame points need to be loaded
    pointLabels.resize(kPointsPerFrame);

    // Go through the labels to find all the subjects and their markers.
    // A Subject is a collection of points that should be grouped together to represent an object.
    std::unordered_map<std::string, std::vector<int>> subjectNameMap;
    subjectNameMap.reserve(kPointsPerFrame);
    for (size_t iLabel = 0; iLabel < kPointsPerFrame; ++iLabel) {
      const std::string& name = pointLabels[iLabel];
      const std::string subjectName = findSubjectName(name);
      subjectNameMap[subjectName].push_back(iLabel);
    }

    // Set up a look up table for each point on which actor it belongs to
    const auto kNumOfSubjects = subjectNameMap.size();
    std::vector<std::pair<int, int>> markerMap(kPointsPerFrame, {-1, -1});
    resultAnim = std::vector<MarkerSequence>(kNumOfSubjects);
    auto templateActors = std::vector<std::vector<Marker>>(kNumOfSubjects);
    auto actorId = 0;
    for (const auto& namePointsPair : subjectNameMap) {
      const auto& subjectName = namePointsPair.first;
      auto markerCount = 0;
      auto& actorSequence = resultAnim[actorId];
      actorSequence.fps = fps;
      auto& templateActor = templateActors[actorId];
      actorSequence.name = subjectName;
      for (const auto kPointIdx : namePointsPair.second) {
        auto pointLabel = pointLabels[kPointIdx];
        if (!subjectName.empty()) {
          pointLabel = pointLabel.substr(subjectName.size() + 1);
        }

        if (isMarkerValid(pointLabel, validMarkerNames)) {
          markerMap[kPointIdx] = std::make_pair(actorId, markerCount);

          Marker marker;
          marker.name = pointLabel;
          marker.occluded = true;
          templateActor.push_back(marker);
          markerCount++;
        }
      }
      actorId++;
    }

    // Go through each frame to save the data
    const auto& frames = c3dFile.data().frames();
    for (int frameId = 0; frameId < kFrameCount; frameId++) {
      // Set all markers to be occluded
      for (auto& templateActor : templateActors) {
        for (auto& marker : templateActor) {
          marker.occluded = true;
        }
      }
      const auto& frameData = frames[frameId];
      const auto& framePoints = frameData.points().points();
      for (int pointId = 0; pointId < kPointsPerFrame; ++pointId) {
        const auto& markerMapEntry = markerMap[pointId];
        const auto kActorId = markerMapEntry.first;
        const auto kMarkerId = markerMapEntry.second;
        // not a marker of interest
        if ((kActorId == -1) || (kMarkerId == -1)) {
          continue;
        }

        auto& marker = templateActors[kActorId][kMarkerId];
        const auto& pointData = framePoints[pointId];
        Vector3d pos{pointData.x(), pointData.y(), pointData.z()};
        if (std::isnan(pos[0]) || std::isnan(pos[1]) || std::isnan(pos[2])) {
          continue;
        }
        if (pos == Eigen::Vector3d::Zero()) {
          continue;
        }
        pos = toMomentumVector3(pos, up, unitStr);

        const auto& residual = pointData.residual();
        // residual < 0 indicates that the data is invalid
        // residual == 0 indicates that the data is generated. Can see if the want to keep the =
        // here or not
        if (residual >= 0) {
          marker.occluded = false;
          marker.pos = pos;
          hasAnim = true;
        }
      }
      for (actorId = 0; actorId < templateActors.size(); actorId++) {
        resultAnim[actorId].frames.push_back(templateActors[actorId]);
      }
    }
  } catch (std::exception& e) {
    MT_LOGE("{}: Exception: {}", e.what(), __func__);
    return {};
  } catch (...) {
    MT_LOGE("{}: Unknown c3d reading error", __func__);
    return {};
  }

  if (!hasAnim) {
    resultAnim.clear();
  }

  return resultAnim;
}

} // namespace momentum
