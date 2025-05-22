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
#include "momentum/common/string.h"
#include "momentum/math/utility.h"

#include <gsl/gsl>

#include <charconv>
#include <fstream>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace momentum {

namespace {

// Function to convert a string_view to a numeric value.
//
// It removes leading whitespaces, and then uses std::from_chars to perform the conversion.
// The base parameter is optional, and is used when converting to an integral type.
template <typename T>
[[nodiscard]] T from_chars_wrapping(std::string_view s, std::optional<int> base = std::nullopt) {
  s.remove_prefix(std::min(s.find_first_not_of(" \t\n\r\f\v"), s.size()));
  T value;

  if constexpr (std::is_integral_v<T>) {
    if (base.has_value()) {
      std::from_chars(s.data(), s.data() + s.size(), value, base.value());
    } else {
      std::from_chars(s.data(), s.data() + s.size(), value);
    }
  } else {
    value = std::stof(std::string(s));
  }

  return value;
}

// Function to convert a string_view to a numeric value of type T.
template <typename T>
[[nodiscard]] T svto(const std::string_view input, std::optional<int> base = std::nullopt) {
  try {
    return from_chars_wrapping<T>(input, base);
  } catch (...) {
    MT_THROW("Error parsing string to number");
  }
}

// Convert a string_view to an int. Throws a runtime error if the conversion fails.
[[nodiscard]] int svtoi(const std::string_view input) {
  return svto<int>(input);
}

// Convert a string_view to a float. Throws a runtime error if the conversion fails.
[[nodiscard]] float svtof(const std::string_view input) {
  return svto<float>(input);
}

std::string firstDuplicate(const LocatorList& locators) {
  std::unordered_set<std::string> locatorNames;
  for (const auto& l : locators) {
    auto itr = locatorNames.find(l.name);
    if (itr != locatorNames.end())
      return l.name;
    locatorNames.insert(itr, l.name);
  }
  return "";
}

} // namespace

LocatorList loadLocators(
    const filesystem::path& filename,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  std::ifstream instream(filename, std::ios::binary);
  MT_THROW_IF(!instream.is_open(), "Cannot find file {}", filename.string());

  instream.seekg(0, instream.end);
  auto length = instream.tellg();
  instream.seekg(0, instream.beg);

  auto buffer = std::make_unique<char[]>(length);
  instream.read((char*)buffer.get(), length);

  return loadLocatorsFromBuffer(
      gsl::make_span<const std::byte>(reinterpret_cast<const std::byte*>(buffer.get()), length),
      skeleton,
      parameterTransform);
}

LocatorList loadLocatorsFromBuffer(
    gsl::span<const std::byte> rawData,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  const std::string_view input(reinterpret_cast<const char*>(rawData.data()), rawData.size());

  LocatorList res;

  const SkeletonState state(parameterTransform.bindPose(), skeleton);

  const auto lines = tokenize(input, "\r\n");
  if (lines.size() < 4 || lines[0] != "{" ||
      trim(lines[1]).find("\"locators\":") == std::string::npos) {
    return res;
  }
  for (size_t line = 2; line < lines.size(); line++) {
    std::string_view current = trim(lines[line]);
    if (current != "{")
      continue;

    // start of locator
    Locator l;
    Vector3f global;
    bool haveGlobal = false;
    do {
      current = trim(lines.at(++line));
      if (current[0] == '}')
        break;
      if (current[0] == '/')
        continue;
      const auto tokens = tokenize(current, ":");
      if (tokens.size() != 2)
        continue;
      if (tokens[0] == "\"lockX\"")
        l.locked.x() = svtoi(tokens[1]);
      else if (tokens[0] == "\"lockY\"")
        l.locked.y() = svtoi(tokens[1]);
      else if (tokens[0] == "\"lockZ\"")
        l.locked.z() = svtoi(tokens[1]);
      else if (tokens[0] == "\"name\"") {
        const auto first = tokens[1].find_first_of('"') + 1;
        const auto last = tokens[1].find_last_of('"') - 1;
        l.name = tokens[1].substr(first, last - first + 1);
      } else if (tokens[0] == "\"parent\"")
        l.parent = svtoi(tokens[1]);
      else if (tokens[0] == "\"parentName\"") {
        const auto first = tokens[1].find_first_of('"') + 1;
        const auto last = tokens[1].find_last_of('"') - 1;
        const std::string_view parent = tokens[1].substr(first, last - first + 1);
        for (size_t k = 0; k < skeleton.joints.size(); k++) {
          if (skeleton.joints[k].name == parent) {
            l.parent = static_cast<int>(k);
            break;
          }
        }
        // invalid joint name
        if (l.parent == kInvalidIndex) {
          MT_LOGW("Invalid parent name: {}", parent);
        }
      } else if (tokens[0] == "\"weight\"")
        l.weight = svtof(tokens[1]);
      else if (tokens[0] == "\"offsetX\"")
        l.offset.x() = svtof(tokens[1]);
      else if (tokens[0] == "\"offsetY\"")
        l.offset.y() = svtof(tokens[1]);
      else if (tokens[0] == "\"offsetZ\"")
        l.offset.z() = svtof(tokens[1]);
      else if (tokens[0] == "\"globalX\"") {
        global.x() = svtof(tokens[1]);
        haveGlobal = true;
      } else if (tokens[0] == "\"globalY\"") {
        global.y() = svtof(tokens[1]);
        haveGlobal = true;
      } else if (tokens[0] == "\"globalZ\"") {
        global.z() = svtof(tokens[1]);
        haveGlobal = true;
      } else if (tokens[0] == "\"limitWeightX\"")
        l.limitWeight[0] = svtof(tokens[1]);
      else if (tokens[0] == "\"limitWeightY\"")
        l.limitWeight[1] = svtof(tokens[1]);
      else if (tokens[0] == "\"limitWeightZ\"")
        l.limitWeight[2] = svtof(tokens[1]);
    } while (line < lines.size() - 1 && (current != "}," || current != "}"));

    // skip locator if it has no parent
    if (l.parent == kInvalidIndex) {
      continue;
    }

    if (haveGlobal) {
      l.offset = state.jointState[l.parent].transformation.inverse() * (global);
    }
    l.limitOrigin = l.offset;
    res.push_back(l);
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
    if (locators[i].limitWeight[0] != 0.0f)
      o << "\t\t\t\"limitWeightX\":" << locators[i].limitWeight.x() << ",\n";
    if (locators[i].limitWeight[1] != 0.0f)
      o << "\t\t\t\"limitWeightY\":" << locators[i].limitWeight.y() << ",\n";
    if (locators[i].limitWeight[2] != 0.0f)
      o << "\t\t\t\"limitWeightZ\":" << locators[i].limitWeight.z() << ",\n";
    MT_CHECK(
        0 <= locators[i].parent && locators[i].parent < skeleton.joints.size(),
        "Invalid joint index");
    o << "\t\t\t\"parentName\":\"" << skeleton.joints[locators[i].parent].name << "\"\n";
    if (i < locators.size() - 1)
      o << "\t\t},\n";
    else
      o << "\t\t}\n";
  }
  o << "\t]\n}\n";
}

} // namespace momentum
