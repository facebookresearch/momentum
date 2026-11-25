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

#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

namespace momentum {

/// The node index that marks "no node", for example the parent of a scene root.
inline constexpr uint32_t kInvalidNodeId = std::numeric_limits<uint32_t>::max();

/// Create a collision capsule from a glTF node.
///
/// @param[in] node The glTF node containing the collision capsule data.
/// @param[in] extension The FB_momentum extension JSON data.
/// @return The created TaperedCapsule object.
TaperedCapsule createCollisionCapsule(const fx::gltf::Node& node, const nlohmann::json& extension);

/// Create a collision ellipsoid from a glTF node.
///
/// @param[in] node The glTF node containing the collision ellipsoid data.
/// @param[in] extension The FB_momentum extension JSON data.
/// @return The created CollisionEllipsoid object.
CollisionEllipsoid createCollisionEllipsoid(
    const fx::gltf::Node& node,
    const nlohmann::json& extension);

/// Create a collision box from a glTF node.
///
/// @param[in] node The glTF node containing the collision box data.
/// @param[in] extension The FB_momentum extension JSON data.
/// @return The created CollisionBox object.
CollisionBox createCollisionBox(const fx::gltf::Node& node, const nlohmann::json& extension);

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

/// Map each node index to its parent node index, walking the default scene breadth-first.
///
/// Each node gets the parent that reaches it first, so the result is a forest even for a malformed
/// document whose nodes have several parents or form a cycle. Scene roots and nodes not reachable
/// from the scene map to kInvalidNodeId.
///
/// @param[in] model The glTF document.
/// @return A vector of size `model.nodes.size()` mapping each node index to its parent node index.
[[nodiscard]] std::vector<uint32_t> buildSceneParentMap(const fx::gltf::Document& model);

/// The scene root above @p nodeId (or @p nodeId itself if it is a scene root).
///
/// For a character written by GltfBuilder this is the character's root node, which carries its
/// name.
///
/// @param[in] nodeId The node index.
/// @param[in] parentMap The parent map from @ref buildSceneParentMap.
/// @return The node index, or kInvalidNodeId if @p nodeId is out of range.
[[nodiscard]] uint32_t getTopLevelAncestor(uint32_t nodeId, std::span<const uint32_t> parentMap);

/// Gather skeleton roots from the glTF model.
///
/// Return a vector of roots for each skeleton. Each item represents a skeleton, and for each
/// skeleton multiple root nodes are allowed. When the document marks its joints with the momentum
/// extension (as GltfBuilder does), each group is one scene root with a marked joint below it, so a
/// character is found whether or not it has a skinned mesh. Otherwise skeletons are found from the
/// skins.
///
/// @param[in] model The glTF document.
/// @param[in] sceneParentMap The parent map from @ref buildSceneParentMap.
/// @return A vector of vectors containing root node indices for each skeleton.
[[nodiscard]] std::vector<std::vector<uint32_t>> gatherSkeletonRoots(
    const fx::gltf::Document& model,
    std::span<const uint32_t> sceneParentMap);

/// Load the hierarchy of one skeleton returned by @ref gatherSkeletonRoots.
///
/// This includes the joints, collision geometry and locators below @p skeletonRoots, and a mapping
/// from every node index to this character's joint, collision or locator index (kInvalidIndex for
/// nodes that do not belong to it). Each node is loaded at most once, so a malformed document whose
/// nodes form a cycle or have several parents still loads.
///
/// @param[in] model The glTF document.
/// @param[in] skeletonRoots The root node indices of one skeleton.
/// @param[in] characterRootNode The node whose name is the character's name, usually
///            getTopLevelAncestor(skeletonRoots.front(), ...). An out-of-range index gives an empty
///            name.
/// @return A tuple containing the name, joints, collision geometry, locators, physical properties,
/// and node-to-object map.
std::tuple<
    std::string,
    JointList,
    CollisionGeometry_u,
    LocatorList,
    PhysicalProperties,
    std::vector<size_t>>
loadHierarchyForRoots(
    const fx::gltf::Document& model,
    std::span<const uint32_t> skeletonRoots,
    uint32_t characterRootNode);

/// Create a skeleton with only a root joint.
///
/// @return A JointList containing a single root joint.
JointList createSkeletonWithOnlyRoot();

} // namespace momentum
