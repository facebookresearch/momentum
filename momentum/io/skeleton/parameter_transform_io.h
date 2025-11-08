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

namespace io_detail {
// Forward declaration
class SectionContent;
} // namespace io_detail

std::unordered_map<std::string, std::string> loadMomentumModel(const filesystem::path& filename);

std::unordered_map<std::string, std::string> loadMomentumModelFromBuffer(
    std::span<const std::byte> buffer);

// Internal overloads for use within momentum parsing
ParameterTransform parseParameterTransform(
    const io_detail::SectionContent& content,
    const Skeleton& skeleton);
ParameterSets parseParameterSets(
    const io_detail::SectionContent& content,
    const ParameterTransform& pt);
PoseConstraints parsePoseConstraints(
    const io_detail::SectionContent& content,
    const ParameterTransform& pt);

// Public APIs for external use
ParameterTransform
parseParameterTransform(const std::string& data, const Skeleton& skeleton, size_t lineOffset = 0);

ParameterSets
parseParameterSets(const std::string& data, const ParameterTransform& pt, size_t lineOffset = 0);

PoseConstraints
parsePoseConstraints(const std::string& data, const ParameterTransform& pt, size_t lineOffset = 0);

// load transform definition from file
std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    const filesystem::path& filename,
    const Skeleton& skeleton);

std::tuple<ParameterTransform, ParameterLimits> loadModelDefinition(
    std::span<const std::byte> rawData,
    const Skeleton& skeleton);

// Write functions to serialize model definition components
std::string writeParameterTransform(
    const ParameterTransform& parameterTransform,
    const Skeleton& skeleton);

std::string writeParameterSets(const ParameterSets& parameterSets);

std::string writePoseConstraints(const PoseConstraints& poseConstraints);

/// Write complete model definition file
/// @param skeleton The character's skeletal structure
/// @param parameterTransform Maps model parameters to joint parameters
/// @param parameterLimits Constraints on model parameters (can be empty)
/// @return String containing the complete model definition
std::string writeModelDefinition(
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const ParameterLimits& parameterLimits);

} // namespace momentum
