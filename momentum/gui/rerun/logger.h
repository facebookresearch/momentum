/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/character/locator_state.h>
#include <momentum/character/marker.h>

#include <rerun.hpp>

#include <cstdint>
#include <map>
#include <string>

namespace momentum {

/// @param[in] (Optional) The color to use for the mesh. If not provided, the colors stored in the
/// mesh are used.
void logMesh(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const Mesh& mesh,
    std::optional<rerun::Color> color = std::nullopt);

void logJoints(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const Skeleton& skeleton,
    const JointStateList& jointStates);

void logMarkers(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    std::span<const Marker> markers);

void logLocators(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const LocatorList& locators,
    const LocatorState& locatorState);

void logMarkerLocatorCorrespondence(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const std::map<std::string, size_t>& locatorLookup,
    const LocatorState& locatorState,
    std::span<const Marker> markers,
    float kPositionErrorThreshold);

void logBvh(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState);

void logCollisionGeometry(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState);

void logCharacter(
    const rerun::RecordingStream& rec,
    const std::string& charStreamName,
    const Character& character,
    const CharacterState& characterState,
    const rerun::Color& color = rerun::Color(200, 200, 200));

// Separate logs for world parameters vs pose parameters because they are in different scale
void logModelParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::VectorXf& params);

void logJointParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::VectorXf& params);

void logModelParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names);

void logJointParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names);

/// Log model parameters for all frames at once using send_columns.
///
/// This is significantly faster than calling logModelParams per frame because it batches
/// all time points into a single send_columns call per parameter.
///
/// @param rec The recording stream.
/// @param worldPrefix The stream prefix for world (root) parameters.
/// @param posePrefix The stream prefix for pose parameters.
/// @param names The parameter names.
/// @param allParams Matrix where each column is a frame's parameter vector (nParams x nFrames).
/// @param frameIndices The frame index for each column.
/// @param times The time in seconds for each column.
void logModelParamsColumns(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::MatrixXf& allParams,
    std::span<const int64_t> frameIndices,
    std::span<const double> times);

/// Log joint parameters for all frames at once using send_columns.
///
/// This is significantly faster than calling logJointParams per frame because it batches
/// all time points into a single send_columns call per parameter channel.
///
/// @param rec The recording stream.
/// @param worldPrefix The stream prefix for world/root joint parameters.
/// @param posePrefix The stream prefix for pose joint parameters.
/// @param names The joint names.
/// @param allJointParams Matrix where each column is a frame's joint parameter vector
///                       (nJoints * kParametersPerJoint x nFrames).
/// @param frameIndices The frame index for each column.
/// @param times The time in seconds for each column.
void logJointParamsColumns(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    std::span<const std::string> names,
    const Eigen::MatrixXf& allJointParams,
    std::span<const int64_t> frameIndices,
    std::span<const double> times);

} // namespace momentum
