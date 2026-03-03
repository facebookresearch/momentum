/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/joint.h>
#include <momentum/character/locator.h>
#include <momentum/character/types.h>

#include <fx/gltf.h>
#include <nlohmann/json.hpp>

#include <tuple>
#include <vector>

namespace momentum {

/// Create a collision capsule from a glTF node.
///
/// @param[in] node The glTF node containing the collision capsule data.
/// @param[in] extension The FB_momentum extension JSON data.
/// @return The created TaperedCapsule object.
TaperedCapsule createCollisionCapsule(const fx::gltf::Node& node, const nlohmann::json& extension);

/// Create a locator from a glTF node.
///
/// @param[in] node The glTF node containing the locator data.
/// @param[in] extension The FB_momentum extension JSON data.
/// @return The created Locator object.
Locator createLocator(const fx::gltf::Node& node, const nlohmann::json& extension);

/// Create a joint from a glTF node.
///
/// @param[in] node The glTF node containing the joint data.
/// @return The created Joint object.
Joint createJoint(const fx::gltf::Node& node);

/// Check if a node is allowed in the skeleton hierarchy.
///
/// The skinned mesh should not exist in the same hierarchy as the skeleton.
/// Note that markers have mesh, so we are allowing the nodes with mesh in the hierarchy.
///
/// @param[in] node The glTF node to check.
/// @return True if the node is a hierarchy node, false otherwise.
[[nodiscard]] bool isHierarchyNode(const fx::gltf::Node& node);

/// Gather skeleton roots from the glTF model.
///
/// Return a vector of roots for each skeleton. Each item represents a skeleton, and for each
/// skeleton multiple root nodes are allowed.
///
/// @param[in] model The glTF document.
/// @return A vector of vectors containing root node indices for each skeleton.
[[nodiscard]] std::vector<std::vector<uint32_t>> gatherSkeletonRoots(
    const fx::gltf::Document& model);

/// Load the hierarchy from the model.
///
/// This includes skeleton, collision geometry, locators and a mapping between the nodes and the
/// objects.
///
/// @param[in] model The glTF document.
/// @return A tuple containing the name, joints, collision geometry, locators, and node-to-object
/// map.
std::tuple<std::string, JointList, CollisionGeometry_u, LocatorList, std::vector<size_t>>
loadHierarchy(const fx::gltf::Document& model);

/// Create a skeleton with only a root joint.
///
/// @return A JointList containing a single root joint.
JointList createSkeletonWithOnlyRoot();

} // namespace momentum
