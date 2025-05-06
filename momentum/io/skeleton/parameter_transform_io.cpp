/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/parameter_transform_io.h"

#include "momentum/character/skeleton.h"
#include "momentum/common/string.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/math/utility.h"

#include <re2/re2.h>

#include <fstream>

namespace momentum {

namespace {

std::unordered_map<std::string, std::string> loadMomentumModelCommon(std::istream& input) {
  if (!input)
    return {};

  std::unordered_map<std::string, std::string> result;
  std::string sectionName;
  std::string sectionContent;

  std::string line;
  GetLineCrossPlatform(input, line);
  if (trim(line) != "Momentum Model Definition V1.0") {
    return result;
  }

  while (GetLineCrossPlatform(input, line)) {
    // erase all comments
    line = trim(line.substr(0, line.find_first_of('#')));

    // skip empty lines or comment lines
    if (line.empty())
      continue;

    // look for new section
    static const re2::RE2 reg("\\[(\\w+)\\]");
    std::string newSectionName;
    if (re2::RE2::FullMatch(line, reg, &newSectionName)) {
      // new section, store old section
      if (!sectionName.empty())
        result[sectionName] = sectionContent;

      // start new section
      sectionName = newSectionName;
      sectionContent.clear();
    } else if (!sectionName.empty())
      sectionContent += line + "\n";
  }

  // store last section
  if (!sectionName.empty())
    result[sectionName] = sectionContent;

  return result;
}

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinitionFromStream(
    std::istream& instream,
    const Skeleton& skeleton) {
  MT_THROW_IF(!instream, "Unable to read parameter transform data.");

  std::string data;
  std::string line;
  GetLineCrossPlatform(instream, line);

  while (GetLineCrossPlatform(instream, line)) {
    // erase all comments
    line = trim(line.substr(0, line.find_first_of('#')));

    // skip empty lines or comment lines
    if (line.empty())
      continue;

    data += line + "\n";
  }

  std::tuple<ParameterTransform, ParameterLimits> res;
  ParameterTransform& pt = std::get<0>(res);
  ParameterLimits& pl = std::get<1>(res);

  pt = parseParameterTransform(data, skeleton);
  pt.parameterSets = parseParameterSets(data, pt);
  pl = parseParameterLimits(data, skeleton, pt);

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
      if (stokens.size() != 1)
        continue;

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
      triplets.push_back(Eigen::Triplet<float>(
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
          triplets.push_back(Eigen::Triplet<float>(
              static_cast<int>(jointIndex) * kParametersPerJoint + static_cast<int>(attributeIndex),
              static_cast<int>(t.col()),
              t.value() * weight));
        }
      }
    } else if (parameterIndex != kInvalidIndex) {
      // add triplet
      triplets.push_back(Eigen::Triplet<float>(
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
  if (filename.empty())
    return {};

  std::ifstream infile(filename);
  MT_THROW_IF(!infile.is_open(), "Cannot find file {}", filename.string());
  return loadMomentumModelCommon(infile);
}

std::unordered_map<std::string, std::string> loadMomentumModelFromBuffer(
    gsl::span<const std::byte> buffer) {
  if (buffer.empty())
    return {};

  ispanstream inputStream(buffer);
  return loadMomentumModelCommon(inputStream);
}

ParameterTransform parseParameterTransform(const std::string& data, const Skeleton& skeleton) {
  ParameterTransform pt;
  pt.activeJointParams.setConstant(skeleton.joints.size() * kParametersPerJoint, false);
  pt.offsets.setZero(skeleton.joints.size() * kParametersPerJoint);

  // triplet list
  std::vector<Eigen::Triplet<float>> triplets;

  std::istringstream f(data);
  std::string line;
  while (std::getline(f, line)) {
    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // ignore limit lines
    if (line.find("limit") == 0)
      continue;

    // load parameterset definitions
    if (line.find("parameterset") == 0)
      continue;

    // load poseconstraints
    if (line.find("poseconstraints") == 0)
      continue;

    // ------------------------------------------------
    //  parse parameter vector
    // ------------------------------------------------
    const auto pTokens = tokenize(line, "=");
    if (pTokens.size() != 2)
      continue;

    // split pToken[0] into joint name and attribute name
    const auto aTokens = tokenize(pTokens[0], ".");
    MT_THROW_IF(aTokens.size() != 2, "Unknown joint name in expression : {}", line);

    // get the right joint to modify
    const size_t jointIndex = skeleton.getJointIdByName(trim(aTokens[0]));
    MT_THROW_IF(jointIndex == kInvalidIndex, "Unknown joint name in expression : {}", line);

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
    MT_THROW_IF(attributeIndex == kInvalidIndex, "Unknown channel name in expression : {}", line);

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

ParameterSets parseParameterSets(const std::string& data, const ParameterTransform& pt) {
  ParameterSets result;

  std::istringstream f(data);
  std::string line;
  while (std::getline(f, line)) {
    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // Skip if not parameterset definitions
    if (line.find("parameterset") != 0)
      continue;

    // parse parameterset
    const auto pTokens = tokenize(line, " \t\r\n");
    MT_THROW_IF(
        pTokens.size() < 2,
        "Could not parse parameterset line in parameter configuration : {}",
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
          "Could not parse parameterset line in parameter configuration : {}. Found unknown parameter name {}.",
          line,
          parameterName);

      ps.set(parameterIndex, true);
    }

    result[pTokens[1]] = ps;
  }

  return result;
}

PoseConstraints parsePoseConstraints(const std::string& data, const ParameterTransform& pt) {
  PoseConstraints result;

  std::istringstream f(data);
  std::string line;
  while (std::getline(f, line)) {
    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // load parameterset definitions
    if (line.find("poseconstraints") != 0)
      continue;

    // parse parameterset
    const auto pTokens = tokenize(line, " \t\r\n");
    MT_THROW_IF(
        pTokens.size() < 2,
        "Could not parse 'poseconstraints' line in parameter configuration : {}",
        line);

    PoseConstraint ps;
    for (size_t i = 2; i < pTokens.size(); i++) {
      const std::string& item = trim(pTokens[i]);

      const auto cTokens = tokenize(item, "=");
      if (cTokens.size() != 2)
        continue;

      size_t parameterIndex = kInvalidIndex;
      for (size_t d = 0; d < pt.name.size(); d++) {
        if (pt.name[d] == cTokens[0]) {
          parameterIndex = d;
          break;
        }
      }

      MT_THROW_IF(
          parameterIndex == kInvalidIndex,
          "Could not parse 'poseconstraints' line in parameter configuration : {}",
          line);

      ps.parameterIdValue.emplace_back(parameterIndex, std::stof(cTokens[1]));
    }

    result[pTokens[1]] = ps;
  }

  return result;
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
    gsl::span<const std::byte> rawData,
    const Skeleton& skeleton) {
  if (rawData.empty())
    return {};

  ispanstream inputStream(rawData);

  MT_THROW_IF(!inputStream, "Unable to read parameter transform data.");

  return loadModelDefinitionFromStream(inputStream, skeleton);
}

} // namespace momentum
