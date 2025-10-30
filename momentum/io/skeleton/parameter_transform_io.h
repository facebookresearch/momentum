/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_limits.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/common/filesystem.h>

#include <span>

#include <string>
#include <unordered_map>

namespace momentum {

std::unordered_map<std::string, std::string> loadMomentumModel(const filesystem::path& filename);

std::unordered_map<std::string, std::string> loadMomentumModelFromBuffer(
    std::span<const std::byte> buffer);

ParameterTransform parseParameterTransform(const std::string& data, const Skeleton& skeleton);

ParameterSets parseParameterSets(const std::string& data, const ParameterTransform& pt);

PoseConstraints parsePoseConstraints(const std::string& data, const ParameterTransform& pt);

// load transform definition from file
std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    const filesystem::path& filename,
    const Skeleton& skeleton);

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    std::span<const std::byte> rawData,
    const Skeleton& skeleton);

} // namespace momentum
