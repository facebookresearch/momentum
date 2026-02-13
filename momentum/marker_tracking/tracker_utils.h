/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/locator.h>
#include <momentum/character/marker.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skinned_locator_triangle_error_function.h>

namespace momentum {

std::vector<std::vector<momentum::PositionData>> createConstraintData(
    std::span<const std::vector<momentum::Marker>> markerData,
    const momentum::LocatorList& locators);

std::vector<std::vector<momentum::SkinnedLocatorConstraint>> createSkinnedConstraintData(
    std::span<const std::vector<momentum::Marker>> markerData,
    const momentum::SkinnedLocatorList& locators);

// TODO: remove the one in momentum

// Create a LocatorCharacter where each locator is a bone in its skeleton. This character is used
// for calibrating locator offsets (as bone offset parameters).
momentum::Character createLocatorCharacter(
    const momentum::Character& sourceCharacter,
    const std::string& prefix);

/// Convert locators to skinned locators by finding the closest point on the mesh surface that
/// matches the correct bone index and using the skinned weights from that point.  Does not add
/// parameters for the skinned locators, however, that should be a separate step if you are planning
/// to solve for their locations.
/// @param sourceCharacter Character with locators to convert
/// @param maxDistance Maximum distance to search for the closest point on the mesh surface.  If the
/// locator is further than this distance, it will not be converted.
/// @param minSkinWeight Minimum skin weight threshold for considering a mesh triangle as belonging
/// to the same bone as the locator.
/// @param verbose If true, print diagnostic messages about locators that could not be converted.
momentum::Character locatorsToSkinnedLocators(
    const momentum::Character& sourceCharacter,
    float maxDistance = 3.0f,
    float minSkinWeight = 0.03f,
    bool verbose = false);

/// Convert skinned locators to regular locators by selecting the bone with the highest skin weight.
/// This is useful when exporting to file formats that don't support skinned locators (e.g., Maya).
/// Each skinned locator will be converted to a regular locator attached to the bone with the
/// highest weight. The position will be transformed from the rest pose space to the local space
/// of the selected bone.
/// @param sourceCharacter Character with skinned locators to convert
/// @return Character with skinned locators converted to regular locators
momentum::Character skinnedLocatorsToLocators(const momentum::Character& sourceCharacter);

std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> createSkinnedLocatorMeshConstraints(
    const momentum::Character& character,
    float targetDepth = 1.0f,
    float maxSearchDistanceCm = 3.0f,
    float maxNormalAngleDeg = 30.0f);

/// Build triangle adjacency: for each triangle, find triangles that share an edge.
///
/// @param mesh The mesh to build adjacency for
/// @return Vector where result[i] contains indices of triangles adjacent to triangle i
std::vector<std::vector<size_t>> buildTriangleAdjacency(const momentum::Mesh& mesh);

/// Find candidate triangles by walking along the mesh surface using DFS.
///
/// Starting from a triangle, performs a depth-first search along the mesh surface,
/// collecting triangles that are within the search distance, have compatible normals,
/// and have sufficient skin weight for the specified parent joint.
///
/// @param mesh The mesh to search
/// @param skin The skin weights
/// @param skeleton The skeleton for joint hierarchy
/// @param adjacency Pre-computed triangle adjacency (from buildTriangleAdjacency)
/// @param startTriangleIdx Index of the starting triangle
/// @param initialPoint The initial locator position (for distance calculations)
/// @param referenceNormal The normal of the starting triangle (for angle filtering)
/// @param parentJointIdx The parent joint index for skin weight filtering
/// @param maxSearchDistanceCm Maximum distance from initialPoint to include triangles
/// @param maxNormalAngleDeg Maximum angle (in degrees) between triangle normal and reference normal
/// @return Vector of candidate triangles that pass all filters
std::vector<momentum::CandidateTriangle> findCandidateTrianglesDfs(
    const momentum::Mesh& mesh,
    const momentum::SkinWeights& skin,
    const momentum::Skeleton& skeleton,
    std::span<const std::vector<size_t>> adjacency,
    size_t startTriangleIdx,
    const Eigen::Vector3f& initialPoint,
    const Eigen::Vector3f& referenceNormal,
    uint32_t parentJointIdx,
    float maxSearchDistanceCm,
    float maxNormalAngleDeg = 30.0f);

// Extract locator offsets from a LocatorCharacter for a normal Character given input calibrated
// parameters
momentum::LocatorList extractLocatorsFromCharacter(
    const momentum::Character& locatorCharacter,
    const momentum::CharacterParameters& calibParams);

// TODO: move to momentum proper
momentum::ModelParameters extractParameters(
    const momentum::ModelParameters& params,
    const momentum::ParameterSet& parameterSet);

std::tuple<Eigen::VectorXf, momentum::LocatorList, momentum::SkinnedLocatorList>
extractIdAndLocatorsFromParams(
    const momentum::ModelParameters& param,
    const momentum::Character& sourceCharacter,
    const momentum::Character& targetCharacter);

Mesh extractBlendShapeFromParams(
    const momentum::ModelParameters& param,
    const momentum::Character& sourceCharacter);

void fillIdentity(
    const momentum::ParameterSet& idSet,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& motion);

// convert from joint to model parameters using only the parameters active in idSet
ModelParameters jointIdentityToModelIdentity(
    const Character& c,
    const ParameterSet& idSet,
    const JointParameters& jointIdentity);

void removeIdentity(
    const momentum::ParameterSet& idSet,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& motion);

std::vector<std::vector<momentum::Marker>> extractMarkersFromMotion(
    const momentum::Character& character,
    const Eigen::MatrixXf& motion);

} // namespace momentum
