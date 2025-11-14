/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/locator_io.h"

#include "momentum/character/locator_state.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/utility.h"

#include <nlohmann/json.hpp>
#include <gsl/gsl>

#include <fstream>
#include <unordered_set>

namespace momentum {

namespace {

std::string firstDuplicate(const LocatorList& locators) {
  std::unordered_set<std::string> locatorNames;
  for (const auto& l : locators) {
    auto itr = locatorNames.find(l.name);
    if (itr != locatorNames.end()) {
      return l.name;
    }
    locatorNames.insert(itr, l.name);
  }
  return "";
}

// Convert byte index to line and column numbers
std::pair<size_t, size_t> byteIndexToLineColumn(std::string_view input, size_t byteIndex) {
  size_t line = 1;
  size_t column = 1;

  for (size_t i = 0; i < byteIndex && i < input.size(); ++i) {
    if (input[i] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
  }

  return {line, column};
}

} // namespace

LocatorList loadLocators(
    const filesystem::path& filename,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  std::ifstream instream(filename, std::ios::binary);
  MT_THROW_IF(!instream.is_open(), "Cannot find file {}", filename.string());
  instream.seekg(0, std::ios::end);
  auto length = instream.tellg();
  instream.seekg(0, std::ios::beg);

  auto buffer = std::make_unique<char[]>(length);
  instream.read((char*)buffer.get(), length);

  return loadLocatorsFromBuffer(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(buffer.get()), length),
      skeleton,
      parameterTransform);
}

LocatorList loadLocatorsFromBuffer(
    std::span<const std::byte> rawData,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  const std::string_view input(reinterpret_cast<const char*>(rawData.data()), rawData.size());

  LocatorList res;

  try {
    const nlohmann::json j = nlohmann::json::parse(input);

    if (!j.contains("locators") || !j["locators"].is_array()) {
      return res;
    }

    const SkeletonState state(parameterTransform.bindPose(), skeleton);

    for (const auto& locatorJson : j["locators"]) {
      Locator l;

      // Read lock fields
      l.locked.x() = locatorJson.value("lockX", 0);
      l.locked.y() = locatorJson.value("lockY", 0);
      l.locked.z() = locatorJson.value("lockZ", 0);

      // Read name
      l.name = locatorJson.value("name", std::string());

      // Read parent index or parent name
      l.parent = locatorJson.value("parent", kInvalidIndex);
      if (locatorJson.contains("parentName")) {
        const std::string parentName = locatorJson["parentName"].get<std::string>();
        for (size_t k = 0; k < skeleton.joints.size(); k++) {
          if (skeleton.joints[k].name == parentName) {
            l.parent = static_cast<int>(k);
            break;
          }
        }
        if (l.parent == kInvalidIndex) {
          MT_LOGW("Invalid parent name: {}", parentName);
        }
      }

      // Skip locator if it has no parent
      if (l.parent == kInvalidIndex) {
        MT_LOGW(
            "Skipping locator '{}' because it has no valid parent",
            l.name.empty() ? "(unnamed)" : l.name);
        continue;
      }

      // Read weight
      l.weight = locatorJson.value("weight", 1.0f);

      // Read offset
      const bool hasOffsetX = locatorJson.contains("offsetX");
      const bool hasOffsetY = locatorJson.contains("offsetY");
      const bool hasOffsetZ = locatorJson.contains("offsetZ");
      l.offset.x() = locatorJson.value("offsetX", 0.0f);
      l.offset.y() = locatorJson.value("offsetY", 0.0f);
      l.offset.z() = locatorJson.value("offsetZ", 0.0f);

      // Check for partial offset specification
      const bool hasAnyOffset = hasOffsetX || hasOffsetY || hasOffsetZ;
      const bool hasAllOffset = hasOffsetX && hasOffsetY && hasOffsetZ;
      if (hasAnyOffset && !hasAllOffset) {
        MT_LOGW(
            "Locator '{}' has partial offset specification (offsetX: {}, offsetY: {}, offsetZ: {}). "
            "If any offset component is specified, all three should be provided.",
            l.name.empty() ? "(unnamed)" : l.name,
            hasOffsetX,
            hasOffsetY,
            hasOffsetZ);
      }

      // Read global coordinates and convert to local offset if present
      const bool hasGlobalX = locatorJson.contains("globalX");
      const bool hasGlobalY = locatorJson.contains("globalY");
      const bool hasGlobalZ = locatorJson.contains("globalZ");
      const bool hasAnyGlobal = hasGlobalX || hasGlobalY || hasGlobalZ;
      const bool hasAllGlobal = hasGlobalX && hasGlobalY && hasGlobalZ;

      // Check for partial global specification
      if (hasAnyGlobal && !hasAllGlobal) {
        MT_LOGW(
            "Locator '{}' has partial global specification (globalX: {}, globalY: {}, globalZ: {}). "
            "If any global component is specified, all three should be provided.",
            l.name.empty() ? "(unnamed)" : l.name,
            hasGlobalX,
            hasGlobalY,
            hasGlobalZ);
      }

      // Check for both global and local offset specification
      if (hasAnyGlobal && hasAnyOffset) {
        MT_LOGW(
            "Locator '{}' has both global (globalX/Y/Z) and local (offsetX/Y/Z) offset specification. "
            "Only one type of offset should be specified. Global offset will take precedence.",
            l.name.empty() ? "(unnamed)" : l.name);
      }

      if (hasAnyGlobal) {
        Vector3f global;
        global.x() = locatorJson.value("globalX", 0.0f);
        global.y() = locatorJson.value("globalY", 0.0f);
        global.z() = locatorJson.value("globalZ", 0.0f);
        l.offset = state.jointState[l.parent].transform.inverse() * global;
      }

      // Read limit weights
      l.limitWeight[0] = locatorJson.value("limitWeightX", 0.0f);
      l.limitWeight[1] = locatorJson.value("limitWeightY", 0.0f);
      l.limitWeight[2] = locatorJson.value("limitWeightZ", 0.0f);

      // Read skin attachment fields
      l.attachedToSkin = (locatorJson.value("attachedToSkin", 0) != 0);
      l.skinOffset = locatorJson.value("skinOffset", 0.0f);

      l.limitOrigin = l.offset;
      res.push_back(l);
    }
  } catch (const nlohmann::json::parse_error& e) {
    const auto [line, column] = byteIndexToLineColumn(input, e.byte);
    MT_THROW("Failed to parse locators JSON at line {}, column {}: {}", line, column, e.what());
  }

  std::string dup = firstDuplicate(res);
  MT_THROW_IF(!dup.empty(), "Duplicated locator {} found", dup);

  return res;
}

void saveLocators(
    const filesystem::path& filename,
    const LocatorList& locators,
    const Skeleton& skeleton,
    const LocatorSpace space) {
  const SkeletonState state(VectorXf::Zero(skeleton.joints.size() * kParametersPerJoint), skeleton);
  const LocatorState lstate(state, locators);

  std::ofstream o(filename);
  o << "{\n\t\"locators\":[\n";
  for (size_t i = 0; i < locators.size(); i++) {
    o << "\t\t{\n";
    o << "\t\t\t\"name\":\"" << locators[i].name << "\",\n";
    if (space == LocatorSpace::Global) {
      o << "\t\t\t\"globalX\":" << lstate.position[i].x() << ",\n";
      o << "\t\t\t\"globalY\":" << lstate.position[i].y() << ",\n";
      o << "\t\t\t\"globalZ\":" << lstate.position[i].z() << ",\n";
    } else if (space == LocatorSpace::Local) {
      o << "\t\t\t\"offsetX\":" << locators[i].offset.x() << ",\n";
      o << "\t\t\t\"offsetY\":" << locators[i].offset.y() << ",\n";
      o << "\t\t\t\"offsetZ\":" << locators[i].offset.z() << ",\n";
    }
    o << "\t\t\t\"lockX\":" << locators[i].locked.x() << ",\n";
    o << "\t\t\t\"lockY\":" << locators[i].locked.y() << ",\n";
    o << "\t\t\t\"lockZ\":" << locators[i].locked.z() << ",\n";
    o << "\t\t\t\"weight\":" << locators[i].weight << ",\n";
    if (locators[i].limitWeight[0] != 0.0f) {
      o << "\t\t\t\"limitWeightX\":" << locators[i].limitWeight.x() << ",\n";
    }
    if (locators[i].limitWeight[1] != 0.0f) {
      o << "\t\t\t\"limitWeightY\":" << locators[i].limitWeight.y() << ",\n";
    }
    if (locators[i].limitWeight[2] != 0.0f) {
      o << "\t\t\t\"limitWeightZ\":" << locators[i].limitWeight.z() << ",\n";
    }
    if (locators[i].attachedToSkin) {
      o << "\t\t\t\"attachedToSkin\":1,\n";
    }
    if (locators[i].skinOffset != 0.0f) {
      o << "\t\t\t\"skinOffset\":" << locators[i].skinOffset << ",\n";
    }
    MT_CHECK(
        0 <= locators[i].parent && locators[i].parent < skeleton.joints.size(),
        "Invalid joint index");
    if (!skeleton.joints.empty() && locators[i].parent < skeleton.joints.size()) {
      o << "\t\t\t\"parentName\":\"" << skeleton.joints[locators[i].parent].name << "\"\n";
    }
    if (i < locators.size() - 1) {
      o << "\t\t},\n";
    } else {
      o << "\t\t}\n";
    }
  }
  o << "\t]\n}\n";
}

} // namespace momentum
