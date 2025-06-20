/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/motion/mmo_io.h"

#include "momentum/character/character.h"
#include "momentum/common/log.h"

#include <fstream>

namespace momentum {

void saveMmo(
    const std::string& filename,
    gsl::span<const VectorXf> poses,
    const VectorXf& scale,
    const Character& character) {
  if (poses.empty() ||
      poses[0].size() !=
          gsl::narrow<Eigen::Index>(character.parameterTransform.numAllModelParameters())) {
    MT_LOGW("{}: Cannot save data: motion data is either empty or of wrong dimension", __func__);
    return;
  }

  // make sure all poses have the right size
  MatrixXf data(poses[0].size(), poses.size());
  for (size_t i = 0; i < poses.size(); i++) {
    if (poses[i].size() != poses[0].size()) {
      MT_LOGW("{}: Cannot save data: not all poses are of the same dimension", __func__);
      return;
    }

    data.col(i) = poses[i];
  }

  saveMmo(filename, data, scale, character);
}

void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    const Character& character,
    const MatrixXf& additionalParameters,
    gsl::span<const std::string> additionalParameterNames) {
  if (poses.rows() !=
      gsl::narrow<Eigen::Index>(character.parameterTransform.numAllModelParameters())) {
    MT_LOGW(
        "{}: Cannot save data: poses are of wrong dimension. Expected: {}, actual: {}",
        __func__,
        character.parameterTransform.numAllModelParameters(),
        poses.rows());
    return;
  }
  if (scale.size() !=
      gsl::narrow<Eigen::Index>(character.skeleton.joints.size() * kParametersPerJoint)) {
    MT_LOGW(
        "{}: Cannot save data: model scale is of wrong dimension. Expected: {}, actual: {}",
        __func__,
        character.skeleton.joints.size() * kParametersPerJoint,
        scale.size());
    return;
  }
  if (additionalParameters.rows() != gsl::narrow<Eigen::Index>(additionalParameterNames.size())) {
    MT_LOGW(
        "{}: Cannot save data: number of additional parameters don't match. Names: {}, data: {}",
        __func__,
        additionalParameterNames.size(),
        additionalParameters.rows());
    return;
  }
  if (additionalParameters.cols() != 0 && additionalParameters.cols() != poses.cols()) {
    MT_LOGW(
        "{}: Cannot save data: additional parameters don't match pose dimensions. Expected: {}, actual {}",
        __func__,
        poses.cols(),
        additionalParameters.cols());
    return;
  }

  // create enhance parameter list;
  std::vector<std::string> parameterNames;
  parameterNames.reserve(
      character.parameterTransform.name.size() + additionalParameterNames.size());
  for (const auto& name : character.parameterTransform.name) {
    parameterNames.emplace_back(name);
  }
  for (const auto& name : additionalParameterNames) {
    parameterNames.emplace_back("__" + name + "__");
  }

  // write out joint names
  std::vector<std::string> jointNames;
  for (const auto& joint : character.skeleton.joints) {
    jointNames.emplace_back(joint.name);
  }

  // create joined pose matrix
  MatrixXf pc = MatrixXf(poses.rows() + additionalParameters.rows(), poses.cols());
  pc.topRows(poses.rows()) = poses;

  if (additionalParameters.rows() > 0) {
    pc.bottomRows(additionalParameters.rows()) = additionalParameters;
  }

  saveMmo(filename, pc, scale, parameterNames, jointNames);
}

void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    gsl::span<const std::string> parameterNames,
    gsl::span<const std::string> jointNames) {
  if (poses.rows() != gsl::narrow<Eigen::Index>(parameterNames.size())) {
    MT_LOGW(
        "{}: Cannot save data: poses are of wrong dimension. Expected: {}, actual: {}",
        __func__,
        parameterNames.size(),
        poses.rows());
    return;
  }
  if (scale.size() != gsl::narrow<Eigen::Index>(jointNames.size() * kParametersPerJoint)) {
    MT_LOGW(
        "{}: Cannot save data: model scale is of wrong dimension. Expected: {}, actual: {}",
        __func__,
        jointNames.size() * kParametersPerJoint,
        scale.size());
    return;
  }

  std::ofstream fs(filename, std::ios::out | std::ios::binary);
  // write out sizes
  size_t data = 0;
  data = parameterNames.size(); // number of parameters
  fs.write((const char*)&data, sizeof(size_t));

  data = jointNames.size(); // number of joints
  fs.write((const char*)&data, sizeof(size_t));

  data = poses.cols(); // number of frames
  fs.write((const char*)&data, sizeof(size_t));

  // write out parameters names
  for (const auto& name : parameterNames) {
    data = name.size();
    fs.write((const char*)&data, sizeof(size_t));
    fs.write((const char*)name.data(), data);
  }

  // write out joint names
  for (const auto& name : jointNames) {
    data = name.size();
    fs.write((const char*)&data, sizeof(size_t));
    fs.write((const char*)name.data(), data);
  }

  // write out offsets
  fs.write((const char*)scale.data(), scale.size() * sizeof(float));

  // write out frames
  fs.write((const char*)poses.data(), poses.size() * sizeof(float));
}

std::tuple<MatrixXf, VectorXf> mapMotionToCharacter(
    const MatrixXf& poses,
    const VectorXf& offsets,
    gsl::span<const std::string> parameterNames,
    gsl::span<const std::string> jointNames,
    const Character& character) {
  std::tuple<MatrixXf, VectorXf> result;
  auto& outPoses = std::get<0>(result);
  auto& outOffset = std::get<1>(result);

  // create mapping for parameters
  std::vector<size_t> pMap;
  for (const auto& name : parameterNames) {
    auto iter = std::find(
        character.parameterTransform.name.begin(), character.parameterTransform.name.end(), name);
    size_t index = std::distance(character.parameterTransform.name.begin(), iter);
    if (index >= character.parameterTransform.name.size()) {
      index = kInvalidIndex;
    }
    pMap.push_back(index);
  }

  // create mapping for joints
  std::vector<size_t> jMap;
  jMap.reserve(jointNames.size());
  for (const auto& name : jointNames) {
    jMap.push_back(character.skeleton.getJointIdByName(name));
  }

  // read offsets and re-arrange them according to the map
  outOffset.setZero(character.skeleton.joints.size() * kParametersPerJoint);
  for (size_t i = 0; i < jMap.size(); i++) {
    if (jMap[i] != kInvalidIndex) {
      outOffset.middleRows<kParametersPerJoint>(jMap[i] * kParametersPerJoint) =
          offsets.middleRows<kParametersPerJoint>(i * kParametersPerJoint);
    }
  }

  // read frames and re-arrange them according to the map
  outPoses.setZero(character.parameterTransform.numAllModelParameters(), poses.cols());
  for (size_t i = 0; i < pMap.size(); i++) {
    if (pMap[i] != kInvalidIndex) {
      outPoses.row(pMap[i]) = poses.row(i);
    }
  }

  return result;
}

std::tuple<MatrixXf, VectorXf> loadMmo(const std::string& filename, const Character& character) {
  const auto [iposes, iscale, pNames, jNames] = loadMmo(filename);
  return mapMotionToCharacter(iposes, iscale, pNames, jNames, character);
}

// get auxilary data from motion
std::tuple<MatrixXf, std::vector<std::string>> getAuxilaryDataFromMotion(
    const MatrixXf& poses,
    gsl::span<const std::string> parameterNames) {
  std::tuple<MatrixXf, std::vector<std::string>> result;
  auto& values = std::get<0>(result);
  auto& names = std::get<1>(result);

  std::vector<size_t> map;
  for (size_t i = 0; i < parameterNames.size(); i++) {
    const auto& n = parameterNames[i];
    const auto l = n.length();
    if (n.size() > 4 && n[0] == '_' && n[1] == '_' && n[l - 1] == '_' && n[l - 2] == '_') {
      map.push_back(i);
      names.push_back(n.substr(2, l - 4));
    }
  }

  values.resize(gsl::narrow<Eigen::Index>(names.size()), poses.cols());
  for (size_t i = 0; i < map.size(); i++) {
    values.row(i) = poses.row(map[i]);
  }

  return result;
}

std::tuple<MatrixXf, VectorXf, std::vector<std::string>, std::vector<std::string>> loadMmo(
    const std::string& filename) {
  std::tuple<MatrixXf, VectorXf, std::vector<std::string>, std::vector<std::string>> result;
  auto& poses = std::get<0>(result);
  auto& scale = std::get<1>(result);
  auto& parameterNames = std::get<2>(result);
  auto& jointNames = std::get<3>(result);

  std::ifstream fs(filename, std::ios::in | std::ios::binary);

  if (!fs.is_open()) {
    MT_LOGW("{}: Failed to open file {}", __func__, filename);
    return result;
  }

  // read total number frames/joints/offsets
  size_t nParams = 0;
  size_t nJoints = 0;
  size_t nFrames = 0;
  fs.read((char*)&nParams, sizeof(size_t));
  fs.read((char*)&nJoints, sizeof(size_t));
  fs.read((char*)&nFrames, sizeof(size_t));

  // load parameter names
  for (size_t i = 0; i < nParams; i++) {
    // read parameter name
    size_t count = 0;
    fs.read((char*)&count, sizeof(size_t));
    std::string name;
    name.resize(count);
    fs.read((char*)name.data(), count);
    parameterNames.push_back(name);
  }

  // load joint names
  for (size_t i = 0; i < nJoints; i++) {
    // read parameter name
    size_t count = 0;
    fs.read((char*)&count, sizeof(size_t));
    std::string name;
    name.resize(count);
    fs.read((char*)name.data(), count);
    jointNames.push_back(name);
  }

  // read offsets and re-arrange them according to the map
  scale = VectorXf(nJoints * kParametersPerJoint);
  fs.read((char*)scale.data(), scale.size() * sizeof(float));

  // read frames and re-arrange them according to the map
  poses = MatrixXf(nParams, nFrames);
  fs.read((char*)poses.data(), poses.size() * sizeof(float));

  return result;
}

} // namespace momentum
