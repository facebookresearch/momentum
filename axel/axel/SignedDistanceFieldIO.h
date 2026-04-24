/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include "axel/SignedDistanceField.h"

namespace axel {

/// Save a signed distance field to a msgpack file.
///
/// @param sdf The signed distance field to save
/// @param path Output file path
template <typename ScalarType>
void saveSdfToMsgpack(const SignedDistanceField<ScalarType>& sdf, const std::string& path);

/// Load a signed distance field from a msgpack file.
///
/// @param path Input file path
/// @return The loaded signed distance field
/// @throws std::runtime_error if the file cannot be read or parsed
template <typename ScalarType>
SignedDistanceField<ScalarType> loadSdfFromMsgpack(const std::string& path);

/// Save multiple named signed distance fields to a single msgpack file.
/// Each entry is stored as {"sdf": {...}, "parent_joint": "..."} where
/// parent_joint is serialized from each SDF's parentJoint() field (omitted if empty).
///
/// @param sdfs Map of name to SDF
/// @param path Output file path
template <typename ScalarType>
void saveSdfsToMsgpack(
    const std::unordered_map<std::string, SignedDistanceField<ScalarType>>& sdfs,
    const std::string& path);

/// Load multiple named signed distance fields from a single msgpack file.
/// The parent_joint field (if present) is restored onto each SDF via setParentJoint().
///
/// @param path Input file path
/// @return Map of name to SDF
/// @throws std::runtime_error if the file cannot be read or parsed
template <typename ScalarType>
std::unordered_map<std::string, SignedDistanceField<ScalarType>> loadSdfsFromMsgpack(
    const std::string& path);

extern template void saveSdfToMsgpack<float>(
    const SignedDistanceField<float>& sdf,
    const std::string& path);
extern template void saveSdfToMsgpack<double>(
    const SignedDistanceField<double>& sdf,
    const std::string& path);

extern template SignedDistanceField<float> loadSdfFromMsgpack<float>(const std::string& path);
extern template SignedDistanceField<double> loadSdfFromMsgpack<double>(const std::string& path);

extern template void saveSdfsToMsgpack<float>(
    const std::unordered_map<std::string, SignedDistanceField<float>>& sdfs,
    const std::string& path);
extern template void saveSdfsToMsgpack<double>(
    const std::unordered_map<std::string, SignedDistanceField<double>>& sdfs,
    const std::string& path);

extern template std::unordered_map<std::string, SignedDistanceField<float>>
loadSdfsFromMsgpack<float>(const std::string& path);
extern template std::unordered_map<std::string, SignedDistanceField<double>>
loadSdfsFromMsgpack<double>(const std::string& path);

} // namespace axel
