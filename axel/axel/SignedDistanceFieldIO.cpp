/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/SignedDistanceFieldIO.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace axel {

namespace {

template <typename ScalarType>
nlohmann::json sdfToJsonObject(const SignedDistanceField<ScalarType>& sdf) {
  const auto& bounds = sdf.bounds();
  const auto& res = sdf.resolution();
  const auto& data = sdf.data();

  nlohmann::json j;
  j["bounds_min"] = {bounds.min().x(), bounds.min().y(), bounds.min().z()};
  j["bounds_max"] = {bounds.max().x(), bounds.max().y(), bounds.max().z()};
  j["resolution"] = {res.x(), res.y(), res.z()};

  const auto* bytePtr = reinterpret_cast<const uint8_t*>(data.data());
  const size_t byteCount = data.size() * sizeof(ScalarType);
  j["data"] = nlohmann::json::binary({bytePtr, bytePtr + byteCount});

  return j;
}

template <typename ScalarType>
SignedDistanceField<ScalarType> jsonObjectToSdf(const nlohmann::json& j) {
  const auto& boundsMin = j.at("bounds_min");
  const auto& boundsMax = j.at("bounds_max");
  const auto& res = j.at("resolution");

  const BoundingBox<ScalarType> bounds(
      Eigen::Vector3<ScalarType>(
          boundsMin[0].get<ScalarType>(),
          boundsMin[1].get<ScalarType>(),
          boundsMin[2].get<ScalarType>()),
      Eigen::Vector3<ScalarType>(
          boundsMax[0].get<ScalarType>(),
          boundsMax[1].get<ScalarType>(),
          boundsMax[2].get<ScalarType>()));

  const Eigen::Vector3<Index> resolution(
      res[0].get<Index>(), res[1].get<Index>(), res[2].get<Index>());

  const auto& bin = j.at("data").get_binary();
  const size_t numElements = bin.size() / sizeof(ScalarType);
  std::vector<ScalarType> data(numElements);
  const auto* src = bin.data();
  std::copy(src, src + numElements * sizeof(ScalarType), reinterpret_cast<uint8_t*>(data.data()));

  return SignedDistanceField<ScalarType>(bounds, resolution, std::move(data));
}

void writeMsgpackToFile(const nlohmann::json& j, const std::string& path) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }
  const auto bytes = nlohmann::json::to_msgpack(j);
  file.write(
      reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file.good()) {
    throw std::runtime_error("Failed to write msgpack data to: " + path);
  }
}

nlohmann::json readMsgpackFromFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " + path);
  }
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  if (!file.good()) {
    throw std::runtime_error("Failed to read msgpack data from: " + path);
  }
  return nlohmann::json::from_msgpack(buffer);
}

} // namespace

template <typename ScalarType>
void saveSdfToMsgpack(const SignedDistanceField<ScalarType>& sdf, const std::string& path) {
  writeMsgpackToFile(sdfToJsonObject(sdf), path);
}

template <typename ScalarType>
SignedDistanceField<ScalarType> loadSdfFromMsgpack(const std::string& path) {
  return jsonObjectToSdf<ScalarType>(readMsgpackFromFile(path));
}

template <typename ScalarType>
void saveSdfsToMsgpack(
    const std::unordered_map<std::string, SignedDistanceField<ScalarType>>& sdfs,
    const std::string& path) {
  nlohmann::json j;
  for (const auto& [name, sdf] : sdfs) {
    nlohmann::json entry;
    entry["sdf"] = sdfToJsonObject(sdf);
    if (!sdf.parentJoint().empty()) {
      entry["parent_joint"] = sdf.parentJoint();
    }
    j[name] = std::move(entry);
  }
  writeMsgpackToFile(j, path);
}

template <typename ScalarType>
std::unordered_map<std::string, SignedDistanceField<ScalarType>> loadSdfsFromMsgpack(
    const std::string& path) {
  const auto j = readMsgpackFromFile(path);
  std::unordered_map<std::string, SignedDistanceField<ScalarType>> result;
  for (const auto& [name, entry] : j.items()) {
    auto sdf = jsonObjectToSdf<ScalarType>(entry.at("sdf"));
    if (entry.contains("parent_joint")) {
      sdf.setParentJoint(entry.at("parent_joint").get<std::string>());
    }
    result.emplace(name, std::move(sdf));
  }
  return result;
}

template void saveSdfToMsgpack<float>(
    const SignedDistanceField<float>& sdf,
    const std::string& path);
template void saveSdfToMsgpack<double>(
    const SignedDistanceField<double>& sdf,
    const std::string& path);

template SignedDistanceField<float> loadSdfFromMsgpack<float>(const std::string& path);
template SignedDistanceField<double> loadSdfFromMsgpack<double>(const std::string& path);

template void saveSdfsToMsgpack<float>(
    const std::unordered_map<std::string, SignedDistanceField<float>>& sdfs,
    const std::string& path);
template void saveSdfsToMsgpack<double>(
    const std::unordered_map<std::string, SignedDistanceField<double>>& sdfs,
    const std::string& path);

template std::unordered_map<std::string, SignedDistanceField<float>> loadSdfsFromMsgpack<float>(
    const std::string& path);
template std::unordered_map<std::string, SignedDistanceField<double>> loadSdfsFromMsgpack<double>(
    const std::string& path);

} // namespace axel
