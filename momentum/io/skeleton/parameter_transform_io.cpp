/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/parameter_transform_io.h"

#include "momentum/character/skeleton.h"
#include "momentum/common/log.h"
#include "momentum/common/string.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/utility.h"
#include "momentum/math/utility.h"

#include <re2/re2.h>

#include <fstream>

namespace momentum {

namespace {

using io_detail::SectionContent;

// Section name constants to avoid typos
constexpr const char* kParameterTransformSection = "ParameterTransform";
constexpr const char* kParameterSetsSection = "ParameterSets";
constexpr const char* kPoseConstraintsSection = "PoseConstraints";
constexpr const char* kParameterLimitsSection = "Limits";

bool isKnownSection(const std::string& sectionName) {
  return sectionName == kParameterTransformSection || sectionName == kParameterSetsSection ||
      sectionName == kPoseConstraintsSection || sectionName == kParameterLimitsSection;
}

std::unordered_map<std::string, SectionContent> loadMomentumModelCommon(std::istream& input) {
  if (!input) {
    MT_LOGW("Unable to read parameter transform data.");
    return {};
  }

  std::unordered_map<std::string, SectionContent> result;
  std::string sectionName;
  std::string sectionContent;
  size_t sectionStartLine = 0;
  size_t linesNotInSectionCount = 0;
  constexpr size_t kMaxLinesNotInSectionWarnings = 5;

  std::string line;
  size_t lineNumber = 0;
  while (GetLineCrossPlatform(input, line)) {
    ++lineNumber;
    line = trim(line.substr(0, line.find_first_of('#')));
    if (line.empty()) {
      // Skip leading comments
      continue;
    }

    if (line == "Momentum Model Definition V1.0") {
      break;
    }

    MT_THROW(
        "Invalid model definition file; expected 'Momentum Model Definition V1.0', got {}", line);
  }

  while (GetLineCrossPlatform(input, line)) {
    ++lineNumber;
    // erase all comments
    line = trim(line.substr(0, line.find_first_of('#')));

    // skip empty lines or comment lines
    if (line.empty()) {
      continue;
    }

    // look for new section
    static const re2::RE2 reg(R"(\[(\w+)\])");
    std::string newSectionName;
    if (re2::RE2::FullMatch(line, reg, &newSectionName)) {
      // new section, store old section
      if (!sectionName.empty() && !sectionContent.empty()) {
        if (isKnownSection(sectionName)) {
          result[sectionName].addSegment(sectionContent, sectionStartLine);
        }
      }

      // Check if this is a known section type
      if (!isKnownSection(newSectionName)) {
        MT_LOGW("Unexpected section found at line {}: [{}]", lineNumber, newSectionName);
      }

      // start new section
      sectionName = newSectionName;
      sectionContent.clear();
      sectionStartLine = lineNumber + 1; // Content starts on the next line after section header
    } else if (!sectionName.empty()) {
      // Only accumulate content for known sections
      if (isKnownSection(sectionName)) {
        sectionContent += line + "\n";
      }
    } else {
      // Line is not in a section and is not a comment
      if (linesNotInSectionCount < kMaxLinesNotInSectionWarnings) {
        MT_LOGW("Line {} is not in a section and is not a comment: {}", lineNumber, line);
        ++linesNotInSectionCount;
        if (linesNotInSectionCount == kMaxLinesNotInSectionWarnings) {
          MT_LOGW("Suppressing further warnings about lines not in sections");
        }
      }
    }
  }

  // store last section
  if (!sectionName.empty() && !sectionContent.empty()) {
    if (isKnownSection(sectionName)) {
      result[sectionName].addSegment(sectionContent, sectionStartLine);
    }
  }

  return result;
}

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinitionFromStream(
    std::istream& instream,
    const Skeleton& skeleton) {
  MT_THROW_IF(!instream, "Unable to read parameter transform data.");

  const auto sections = loadMomentumModelCommon(instream);

  std::tuple<ParameterTransform, ParameterLimits> res;
  ParameterTransform& pt = std::get<0>(res);
  ParameterLimits& pl = std::get<1>(res);

  const auto ptIt = sections.find(kParameterTransformSection);
  if (ptIt == sections.end()) {
    pt = parseParameterTransform(momentum::io_detail::SectionContent{}, skeleton);
  } else {
    pt = parseParameterTransform(ptIt->second, skeleton);
  }

  const auto psIt = sections.find(kParameterSetsSection);
  if (psIt != sections.end()) {
    pt.parameterSets = parseParameterSets(psIt->second, pt);
  }

  const auto pcIt = sections.find(kPoseConstraintsSection);
  if (pcIt != sections.end()) {
    pt.poseConstraints = parsePoseConstraints(pcIt->second, pt);
  }

  const auto plIt = sections.find(kParameterLimitsSection);
  if (plIt != sections.end()) {
    pl = parseParameterLimits(plIt->second, skeleton, pt);
  }

  return res;
}

void parseParameter(
    std::vector<Eigen::Triplet<float>>& triplets,
    ParameterTransform& pt,
    const std::vector<std::string>& pTokens,
    const Skeleton& skeleton,
    size_t jointIndex,
    size_t attributeIndex,
    const std::string& line) {
  // split the parameter names
  const auto dtokens = tokenize(pTokens[1], "+");
  for (const auto& dtoken : dtokens) {
    // tokenize each subtoken
    const auto stokens = tokenize(dtoken, "*");

    if (stokens.size() != 2) {
      if (stokens.size() != 1) {
        continue;
      }

      // additional weight
      const Eigen::Index pindex = jointIndex * kParametersPerJoint + attributeIndex;
      const float weight = gsl::narrow_cast<float>(std::stod(stokens[0]));
      pt.offsets(pindex) = weight;
      continue;
    }

    // first should be the weight
    const float weight = gsl::narrow_cast<float>(std::stod(stokens[0]));

    const std::string& parameterName = trim(stokens[1]);
    size_t parameterIndex = kInvalidIndex;
    for (size_t d = 0; d < pt.name.size(); d++) {
      if (pt.name[d] == parameterName) {
        parameterIndex = static_cast<int>(d);
        break;
      }
    }

    // check if the first token is actually a joint name as well
    const size_t refJointIndex =
        skeleton.getJointIdByName(parameterName.substr(0, parameterName.find_first_of('.')));
    size_t refJointParameter = kInvalidIndex;
    if (refJointIndex != kInvalidIndex) {
      const std::string jpString = parameterName.substr(parameterName.find_first_of('.') + 1);
      for (size_t l = 0; l < kParametersPerJoint; l++) {
        if (jpString == kJointParameterNames[l]) {
          refJointParameter = l;
          break;
        }
      }
    }

    if (parameterIndex == kInvalidIndex && refJointIndex == kInvalidIndex) {
      // no reference transform, so create new parameter
      parameterIndex = static_cast<int>(pt.name.size());
      pt.name.push_back(parameterName);

      // add triplet
      triplets.push_back(
          Eigen::Triplet<float>(
              static_cast<int>(jointIndex) * kParametersPerJoint + static_cast<int>(attributeIndex),
              static_cast<int>(parameterIndex),
              weight));
    } else if (
        parameterIndex == kInvalidIndex && refJointIndex != kInvalidIndex &&
        refJointParameter != kInvalidIndex) {
      // we actually reference a joint that is in the list earlier, copy over all parameters
      // defining this joint and multiply them
      const size_t refJointId = refJointIndex * kParametersPerJoint + refJointParameter;
      for (size_t tr = 0; tr < triplets.size(); tr++) {
        const auto& t = triplets[tr];
        if (static_cast<size_t>(t.row()) == refJointId) {
          triplets.push_back(
              Eigen::Triplet<float>(
                  static_cast<int>(jointIndex) * kParametersPerJoint +
                      static_cast<int>(attributeIndex),
                  static_cast<int>(t.col()),
                  t.value() * weight));
        }
      }
    } else if (parameterIndex != kInvalidIndex) {
      // add triplet
      triplets.push_back(
          Eigen::Triplet<float>(
              static_cast<int>(jointIndex) * kParametersPerJoint + static_cast<int>(attributeIndex),
              static_cast<int>(parameterIndex),
              weight));
    } else {
      MT_THROW("Could not parse channel expression : {}", line);
    }
  }
}

} // namespace

std::unordered_map<std::string, std::string> loadMomentumModel(const filesystem::path& filename) {
  if (filename.empty()) {
    return {};
  }

  std::ifstream infile(filename);
  MT_THROW_IF(!infile.is_open(), "Cannot find file {}", filename.string());
  const auto sections = loadMomentumModelCommon(infile);

  std::unordered_map<std::string, std::string> result;
  for (const auto& [sectionName, content] : sections) {
    result[sectionName] = content.toString();
  }
  return result;
}

std::unordered_map<std::string, std::string> loadMomentumModelFromBuffer(
    std::span<const std::byte> buffer) {
  if (buffer.empty()) {
    return {};
  }

  ispanstream inputStream(buffer);
  const auto sections = loadMomentumModelCommon(inputStream);

  std::unordered_map<std::string, std::string> result;
  for (const auto& [sectionName, content] : sections) {
    result[sectionName] = content.toString();
  }
  return result;
}

// Internal overload that accepts SectionContent
ParameterTransform parseParameterTransform(
    const SectionContent& content,
    const Skeleton& skeleton) {
  ParameterTransform pt;
  pt.activeJointParams.setConstant(skeleton.joints.size() * kParametersPerJoint, false);
  pt.offsets.setZero(skeleton.joints.size() * kParametersPerJoint);

  // triplet list
  std::vector<Eigen::Triplet<float>> triplets;

  auto iterator = content.begin();
  std::string line;
  while (iterator.getline(line)) {
    const size_t lineIndex = iterator.currentLine();

    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // Skip empty lines
    if (trim(line).empty()) {
      continue;
    }

    // ------------------------------------------------
    //  parse parameter vector
    // ------------------------------------------------
    const auto pTokens = tokenize(line, "=");
    if (pTokens.size() != 2) {
      MT_LOGW(
          "Ignoring invalid line under [ParameterTransform] section at line {}: {}",
          lineIndex,
          line);
      continue;
    }

    // split pToken[0] into joint name and attribute name
    const auto aTokens = tokenize(pTokens[0], ".");
    MT_THROW_IF(
        aTokens.size() != 2, "Unknown joint name in expression at line {}: {}", lineIndex, line);

    // get the right joint to modify
    const size_t jointIndex = skeleton.getJointIdByName(trim(aTokens[0]));
    MT_THROW_IF(
        jointIndex == kInvalidIndex,
        "Unknown joint name in expression at line {}: {}",
        lineIndex,
        line);

    // the first pToken is the name of the joint and it's attribute type
    size_t attributeIndex = kInvalidIndex;
    const auto trimName = trim(aTokens[1]);
    for (size_t j = 0; j < kParametersPerJoint; j++) {
      if (trimName == kJointParameterNames[j]) {
        attributeIndex = j;
        break;
      }
    }

    // if we didn't find a right name exit with an error
    MT_THROW_IF(
        attributeIndex == kInvalidIndex,
        "Unknown channel name in expression at line {}: {}",
        lineIndex,
        line);

    // enable the attribute in the skeleton if we have a parameter controlling it
    pt.activeJointParams[jointIndex * kParametersPerJoint + attributeIndex] = 1;

    // split the parameter names
    parseParameter(triplets, pt, pTokens, skeleton, jointIndex, attributeIndex, line);
  }

  // resize the Transform matrix to the correct size
  pt.transform.resize(
      static_cast<int>(skeleton.joints.size()) * kParametersPerJoint,
      static_cast<int>(pt.name.size()));

  // finish parameter setup by creating sparse matrix
  triplets.erase(
      std::remove_if(
          std::begin(triplets),
          std::end(triplets),
          [](const Eigen::Triplet<float>& t) { return t.value() == 0.0f; }),
      std::end(triplets));

  pt.transform.setFromTriplets(triplets.begin(), triplets.end());

  return pt;
}

// Public API wrapper for backward compatibility
ParameterTransform
parseParameterTransform(const std::string& data, const Skeleton& skeleton, size_t lineOffset) {
  SectionContent content;
  if (!data.empty()) {
    content.addSegment(data, lineOffset);
  }
  return parseParameterTransform(content, skeleton);
}

// Internal overload that accepts SectionContent
ParameterSets parseParameterSets(const SectionContent& content, const ParameterTransform& pt) {
  ParameterSets result;

  auto iterator = content.begin();
  std::string line;
  while (iterator.getline(line)) {
    const size_t lineIndex = iterator.currentLine();

    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // Skip empty lines
    if (trim(line).empty()) {
      continue;
    }

    // Skip if not parameterset definitions
    if (line.find("parameterset") != 0) {
      MT_LOGW(
          "Ignoring invalid line under [ParameterSets] section at line {}: {}", lineIndex, line);
      continue;
    }

    // parse parameterset
    const auto pTokens = tokenize(line, " \t\r\n");
    MT_THROW_IF(
        pTokens.size() < 2,
        "Could not parse parameterset line in parameter configuration at line {}: {}",
        lineIndex,
        line);

    ParameterSet ps;
    for (size_t i = 2; i < pTokens.size(); i++) {
      const std::string& parameterName = trim(pTokens[i]);
      size_t parameterIndex = kInvalidIndex;
      for (size_t d = 0; d < pt.name.size(); d++) {
        if (pt.name[d] == parameterName) {
          parameterIndex = d;
          break;
        }
      }

      MT_THROW_IF(
          parameterIndex == kInvalidIndex,
          "Could not parse parameterset line in parameter configuration at line {}: {}. Found unknown parameter name {}.",
          lineIndex,
          line,
          parameterName);

      ps.set(parameterIndex, true);
    }

    result[pTokens[1]] = ps;
  }

  return result;
}

// Public API wrapper for backward compatibility
ParameterSets
parseParameterSets(const std::string& data, const ParameterTransform& pt, size_t lineOffset) {
  SectionContent content;
  if (!data.empty()) {
    content.addSegment(data, lineOffset);
  }
  return parseParameterSets(content, pt);
}

// Internal overload that accepts SectionContent
PoseConstraints parsePoseConstraints(const SectionContent& content, const ParameterTransform& pt) {
  PoseConstraints result;

  auto iterator = content.begin();
  std::string line;
  while (iterator.getline(line)) {
    const size_t lineIndex = iterator.currentLine();

    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // Skip empty lines
    if (trim(line).empty()) {
      continue;
    }

    // Skip if not poseconstraints definitions
    if (line.find("poseconstraints") != 0) {
      MT_LOGW(
          "Ignoring invalid line under [PoseConstraints] section at line {}: {}", lineIndex, line);
      continue;
    }

    // parse parameterset
    const auto pTokens = tokenize(line, " \t\r\n");
    MT_THROW_IF(
        pTokens.size() < 2,
        "Could not parse 'poseconstraints' line in parameter configuration at line {}: {}",
        lineIndex,
        line);

    PoseConstraint ps;
    for (size_t i = 2; i < pTokens.size(); i++) {
      const std::string& item = trim(pTokens[i]);

      const auto cTokens = tokenize(item, "=");
      if (cTokens.size() != 2) {
        continue;
      }

      size_t parameterIndex = kInvalidIndex;
      for (size_t d = 0; d < pt.name.size(); d++) {
        if (pt.name[d] == cTokens[0]) {
          parameterIndex = d;
          break;
        }
      }

      MT_THROW_IF(
          parameterIndex == kInvalidIndex,
          "Could not parse 'poseconstraints' line in parameter configuration at line {}: {}",
          lineIndex,
          line);

      ps.parameterIdValue.emplace_back(parameterIndex, std::stof(cTokens[1]));
    }

    result[pTokens[1]] = ps;
  }

  return result;
}

// Public API wrapper for backward compatibility
PoseConstraints
parsePoseConstraints(const std::string& data, const ParameterTransform& pt, size_t lineOffset) {
  SectionContent content;
  if (!data.empty()) {
    content.addSegment(data, lineOffset);
  }
  return parsePoseConstraints(content, pt);
}

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    const filesystem::path& filename,
    const Skeleton& skeleton) {
  std::ifstream infile(filename);
  MT_THROW_IF(
      !infile.is_open(),
      "Unable to open parameter transform file '{}' for reading.",
      filename.string());

  return loadModelDefinitionFromStream(infile, skeleton);
}

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    std::span<const std::byte> rawData,
    const Skeleton& skeleton) {
  if (rawData.empty()) {
    return {};
  }

  ispanstream inputStream(rawData);

  MT_THROW_IF(!inputStream, "Unable to read parameter transform data.");

  return loadModelDefinitionFromStream(inputStream, skeleton);
}

std::string writeParameterTransform(
    const ParameterTransform& parameterTransform,
    const Skeleton& skeleton) {
  std::ostringstream oss;

  // Write each joint parameter definition
  for (size_t iJoint = 0; iJoint < skeleton.joints.size(); ++iJoint) {
    const auto& joint = skeleton.joints[iJoint];

    for (size_t iParam = 0; iParam < kParametersPerJoint; ++iParam) {
      const size_t jointParamIndex = iJoint * kParametersPerJoint + iParam;

      // Skip inactive joint parameters
      if (!parameterTransform.activeJointParams[jointParamIndex]) {
        continue;
      }

      // Write joint parameter name
      oss << joint.name << "." << kJointParameterNames[iParam] << " = ";

      // Collect all model parameters that influence this joint parameter
      std::vector<std::pair<size_t, float>> influences;
      for (Eigen::Index iModelParam = 0; iModelParam < parameterTransform.transform.cols();
           ++iModelParam) {
        const float weight = parameterTransform.transform.coeff(jointParamIndex, iModelParam);
        if (weight != 0.0f) {
          influences.emplace_back(iModelParam, weight);
        }
      }

      // Write the offset if it exists
      const float offset = parameterTransform.offsets[jointParamIndex];
      bool needsPlus = false;

      if (!influences.empty()) {
        bool firstTerm = true;
        for (const auto& [modelParamIndex, weight] : influences) {
          MT_THROW_IF(
              modelParamIndex >= parameterTransform.name.size(),
              "Model parameter index {} is out of bounds (name size: {})",
              modelParamIndex,
              parameterTransform.name.size());
          if (!firstTerm) {
            oss << " + ";
          }
          oss << weight << "*" << parameterTransform.name.at(modelParamIndex);
          firstTerm = false;
          needsPlus = true;
        }
      }

      if (offset != 0.0f) {
        if (needsPlus) {
          oss << " + ";
        }
        oss << offset;
      }

      oss << "\n";
    }
  }

  return oss.str();
}

std::string writeParameterSets(const ParameterSets& parameterSets) {
  std::ostringstream oss;

  for (const auto& [name, paramSet] : parameterSets) {
    oss << "parameterset " << name;

    // Write all active parameters in the set
    for (size_t i = 0; i < paramSet.size(); ++i) {
      if (paramSet.test(i)) {
        // We need the parameter name, but we don't have access to parameterTransform here
        // This is a limitation - we'll need to pass it in
        oss << " param_" << i;
      }
    }
    oss << "\n";
  }

  return oss.str();
}

std::string writePoseConstraints(const PoseConstraints& poseConstraints) {
  std::ostringstream oss;

  for (const auto& [name, constraint] : poseConstraints) {
    oss << "poseconstraints " << name;

    // Write all parameter=value pairs
    for (const auto& [paramIndex, value] : constraint.parameterIdValue) {
      // We need the parameter name, but we don't have access to parameterTransform here
      oss << " param_" << paramIndex << "=" << value;
    }
    oss << "\n";
  }

  return oss.str();
}

std::string writeModelDefinition(
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const ParameterLimits& parameterLimits) {
  std::ostringstream oss;

  // Write header
  oss << "Momentum Model Definition V1.0\n\n";

  // Write ParameterTransform section
  if (!parameterTransform.name.empty()) {
    oss << "[ParameterTransform]\n";
    oss << writeParameterTransform(parameterTransform, skeleton);
    oss << "\n";
  }

  // Write ParameterSets section
  if (!parameterTransform.parameterSets.empty()) {
    oss << "[ParameterSets]\n";
    for (const auto& [name, paramSet] : parameterTransform.parameterSets) {
      oss << "parameterset " << name;

      // Write all active parameters in the set
      for (size_t i = 0; i < paramSet.size() && i < parameterTransform.name.size(); ++i) {
        if (paramSet.test(i)) {
          oss << " " << parameterTransform.name[i];
        }
      }
      oss << "\n";
    }
    oss << "\n";
  }

  // Write PoseConstraints section
  if (!parameterTransform.poseConstraints.empty()) {
    oss << "[PoseConstraints]\n";
    for (const auto& [name, constraint] : parameterTransform.poseConstraints) {
      oss << "poseconstraints " << name;

      // Write all parameter=value pairs
      for (const auto& [paramIndex, value] : constraint.parameterIdValue) {
        if (paramIndex < parameterTransform.name.size()) {
          oss << " " << parameterTransform.name[paramIndex] << "=" << value;
        }
      }
      oss << "\n";
    }
    oss << "\n";
  }

  // Write ParameterLimits section using existing function
  if (!parameterLimits.empty()) {
    oss << "[" << kParameterLimitsSection << "]\n";
    oss << writeParameterLimits(parameterLimits, skeleton, parameterTransform);
  }

  return oss.str();
}

} // namespace momentum
