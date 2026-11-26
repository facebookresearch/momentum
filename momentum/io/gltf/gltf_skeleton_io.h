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
#include <momentum/character/skeleton.h>
#include <momentum/character/types.h>
#include <momentum/math/transform.h>

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

/// Whether the document marks its joints with the momentum extension, as GltfBuilder does.
///
/// A document without marks comes from another tool: every node without a mesh in a skeleton's
/// hierarchy loads as a joint, with the transforms that the glTF format gives it (see
/// @ref jointFileTransforms).
///
/// @param[in] model The glTF document.
/// @return True if a node of the document is a marked joint.
[[nodiscard]] bool hasMarkedJoints(const fx::gltf::Document& model);

/// The local transform of a glTF node in momentum units: its matrix, or its translation, rotation
/// and scale.
///
/// Momentum transforms have a uniform scale, so a non-uniform scale keeps the scale of the node's
/// x axis.
///
/// @param[in] node The glTF node.
/// @return The node's transform relative to its parent node.
[[nodiscard]] Transform nodeLocalTransform(const fx::gltf::Node& node);

/// How the local transform of a joint of a document without joint marks maps to momentum.
///
/// A glTF joint is placed by the transforms of all nodes above it, scale included, but a momentum
/// joint has no rest scale and is placed only by the joints above it. So the momentum local
/// transform of a joint is `pre * fileLocal * post`, where `fileLocal` is the local transform of
/// the joint's node: `pre` maps from the frame of the parent joint without its scale (the world for
/// a root joint) to the frame of the node's parent, and `post` removes the node's accumulated
/// scale. The world transform of a glTF joint is then the momentum world transform times
/// `post.inverse()`.
struct JointFileTransform {
  /// The joint's node, or kInvalidNodeId for a joint without one.
  uint32_t node = kInvalidNodeId;
  /// From the parent joint's frame without its scale to the frame of the node's parent.
  Transform pre;
  /// The inverse of the node's accumulated scale.
  Transform post;
};

/// The @ref JointFileTransform of each joint of a character of a document without joint marks.
///
/// The transforms use the rest transforms of the nodes above each joint; animation channels of
/// nodes that are not joints do not change them. A skeleton outside the scene that the loader reads
/// is placed by the nodes above it in its own tree.
///
/// @param[in] model The glTF document.
/// @param[in] skeleton The character's skeleton.
/// @param[in] nodeToObjectMap The character's node map (see @ref loadHierarchyForRoots).
/// @return One entry per joint of @p skeleton.
[[nodiscard]] std::vector<JointFileTransform> jointFileTransforms(
    const fx::gltf::Document& model,
    const Skeleton& skeleton,
    std::span<const size_t> nodeToObjectMap);

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
