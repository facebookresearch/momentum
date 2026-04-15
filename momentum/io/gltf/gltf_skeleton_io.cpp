/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_skeleton_io.h"

#include "momentum/character/joint.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/io/common/json_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"

#include <algorithm>
#include <limits>

namespace momentum {

namespace {

constexpr auto kInvalidNodeId = std::numeric_limits<uint32_t>::max();

bool hasMomentumExtension(const fx::gltf::Document& model) {
  bool hasExtension = model.extensionsAndExtras.count("extensions") != 0 &&
      model.extensionsAndExtras["extensions"].count("FB_momentum") != 0;
  return hasExtension;
}

nlohmann::json getMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") != 0 &&
      extensionsAndExtras["extensions"].count("FB_momentum") != 0) {
    return extensionsAndExtras["extensions"]["FB_momentum"];
  } else {
    return nlohmann::json::object();
  }
}

[[nodiscard]] std::vector<uint32_t> getAncestorsIncludingSelf(
    uint32_t nodeId,
    const std::vector<uint32_t>& parents) {
  auto ancestors = std::vector<uint32_t>();
  auto curNode = nodeId;
  while (curNode != kInvalidNodeId) {
    ancestors.push_back(curNode);
    curNode = parents[curNode];
  }
  std::reverse(ancestors.begin(), ancestors.end());
  return ancestors;
}

[[nodiscard]] uint32_t findClosestCommonAncestor(
    const uint32_t node1,
    const uint32_t node2,
    const std::vector<uint32_t>& parents) {
  const auto ancestors1 = getAncestorsIncludingSelf(node1, parents);
  const auto ancestors2 = getAncestorsIncludingSelf(node2, parents);
  const auto maxNumIter = std::min(ancestors1.size(), ancestors2.size());
  auto commonAncestor = kInvalidNodeId;
  for (auto i = 0; i < maxNumIter; i++) {
    if (ancestors1[i] == ancestors2[i]) {
      commonAncestor = ancestors1[i];
    } else {
      break;
    }
  }
  return commonAncestor;
}

// DFS the hierarchy graph to reconstruct it.
// Along the way, save the joints, collision geometry and the locators, as well as a
// nodeToObjectMap for query. nodeToObjectMap stores mapping from node index to joint index,
// if the node is a joint. If it's a collision capsule or locator,
// stores the index in the collision volume array or locator array.
// If no extension is present in the glb file, right now we will load every not-skinned node as a
// joint. #TODO: correctly load hierarchy from the skinning information if no momentum extension is
// in the file.
void loadHierarchyRecursive(
    const fx::gltf::Document& model,
    const int nodeId,
    size_t parentJointId,
    JointList& joints,
    CollisionGeometry_u& collision,
    LocatorList& locators,
    std::vector<size_t>& nodeToObjectMap,
    bool useExtension) {
  MT_THROW_IF(
      (nodeId < 0) || (nodeId >= model.nodes.size()),
      "Invalid node id found in the gltf hierarchy: {}",
      nodeId);
  const auto& node = model.nodes[nodeId];
  if (!isHierarchyNode(node)) {
    return;
  }

  const auto& extension = getMomentumExtension(node.extensionsAndExtras);
  const std::string type = extension.value("type", "");
  MT_CHECK(nodeId < nodeToObjectMap.size(), "id: {}, size: {}", nodeId, nodeToObjectMap.size());
  MT_CHECK(
      nodeToObjectMap.size() == model.nodes.size(),
      "NodeMap: {}, nodes: {}",
      nodeToObjectMap.size(),
      model.nodes.size());
  auto newParentJointId = parentJointId;
  if (type == "collision_capsule") {
    // Found collision geometry, should be the end node
    MT_THROW_IF(
        parentJointId == kInvalidIndex,
        "Invalid collision capsule without a parent joint: {}",
        node.name);
    auto capsule = createCollisionCapsule(node, extension);
    capsule.parent = parentJointId;
    collision->push_back(capsule);
    nodeToObjectMap[nodeId] = collision->size() - 1;
  } else if (type == "locator") {
    // Found locator, should be the end node
    MT_THROW_IF(
        parentJointId == kInvalidIndex, "Invalid locator without a parent joint: {}", node.name);
    Locator loc = createLocator(node, extension);
    loc.parent = parentJointId;
    locators.push_back(loc);
    nodeToObjectMap[nodeId] = locators.size() - 1;
  } else if ((!useExtension && (node.mesh < 0)) || type == "skeleton_joint") {
    Joint joint = createJoint(node);
    joint.parent = parentJointId;
    joints.push_back(joint);
    nodeToObjectMap[nodeId] = joints.size() - 1;
    newParentJointId = joints.size() - 1;
  }
  // #TODO: log skipped nodes

  MT_CHECK(!model.nodes.empty());
  for (auto childId : model.nodes[nodeId].children) {
    MT_THROW_IF(
        (childId < 0) || (childId >= model.nodes.size()),
        "Invalid node id found in the gltf hierarchy: {}",
        childId);

    loadHierarchyRecursive(
        model,
        childId,
        newParentJointId,
        joints,
        collision,
        locators,
        nodeToObjectMap,
        useExtension);
  }
}

} // namespace

TaperedCapsule createCollisionCapsule(const fx::gltf::Node& node, const nlohmann::json& extension) {
  TaperedCapsule tc;
  tc.parent = kInvalidIndex;
  tc.transformation = Transform();
  tc.transformation.rotation = toMomentumQuaternionf(node.rotation);
  tc.transformation.translation = toMomentumVec3f(node.translation);

  try {
    tc.length = extension["length"];
    tc.radius = fromJson<Vector2f>(extension["radius"]);
  } catch (const std::exception&) {
    MT_THROW(
        "Fail to parse json {} for collision capsule {} when loading character.",
        node.name,
        extension.dump());
  }
  return tc;
}

Locator createLocator(const fx::gltf::Node& node, const nlohmann::json& extension) {
  Locator loc;
  loc.parent = kInvalidIndex;
  loc.name = node.name;
  loc.offset = toMomentumVec3f(node.translation);

  try {
    loc.weight = extension["weight"];
    if (extension.contains("limitOrigin")) {
      loc.limitOrigin = fromJson<Vector3f>(extension["limitOrigin"]);
    } else {
      loc.limitOrigin = loc.offset;
    }
    loc.limitWeight = fromJson<Vector3f>(extension["limitWeight"]);
    loc.locked = fromJson<Vector3i>(extension["locked"]);
    if (extension.contains("attachedToSkin")) {
      loc.attachedToSkin = extension["attachedToSkin"];
    }
    if (extension.contains("skinOffset")) {
      loc.skinOffset = extension["skinOffset"];
    }
  } catch (const std::exception&) {
    MT_THROW(
        "Fail to parse json {} for locator {} when loading character.",
        extension.dump(),
        node.name);
  }
  return loc;
}

Joint createJoint(const fx::gltf::Node& node) {
  Joint joint;
  joint.name = node.name;
  joint.parent = kInvalidIndex;
  joint.preRotation = toMomentumQuaternionf(node.rotation);
  joint.translationOffset = toMomentumVec3f(node.translation);
  return joint;
}

bool isHierarchyNode(const fx::gltf::Node& node) {
  return (node.skin < 0) && (node.camera < 0);
}

std::vector<std::vector<uint32_t>> gatherSkeletonRoots(const fx::gltf::Document& model) {
  MT_CHECK(model.scene < model.scenes.size(), "{}: {}", model.scene, model.scenes.size());
  // If no skinned mesh is present, consider the scene root nodes as skeleton roots.
  if (model.meshes.empty() || model.skins.empty()) {
    std::vector<uint32_t> roots = model.scenes[model.scene].nodes;
    return {roots};
  }

  // Build a parent map for ancestor query
  auto parentMap = std::vector<uint32_t>(model.nodes.size(), kInvalidNodeId);
  auto nodeStack = model.scenes[model.scene].nodes;
  while (!nodeStack.empty()) {
    std::vector<uint32_t> nextStack;
    for (const auto& nodeId : nodeStack) {
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      const auto& node = model.nodes[nodeId];
      nextStack.reserve(nextStack.size() + node.children.size());
      nextStack.insert(nextStack.end(), node.children.begin(), node.children.end());
      for (auto child : model.nodes[nodeId].children) {
        parentMap[child] = nodeId;
      }
    }
    nodeStack.swap(nextStack);
  }

  // Assumption: all the joints in the skin are directly connected (i.e. no non-joint node in
  // between two joints) Merge skeleton within the same skin. If they are in the same hierarchy,
  // they can be merged under their closest common ancestor.
  auto skinSkeletonRoots =
      std::vector<std::vector<uint32_t>>(model.skins.size(), std::vector<uint32_t>());
  for (auto skinId = 0; skinId < model.skins.size(); skinId++) {
    const auto& skin = model.skins[skinId];
    auto& roots = skinSkeletonRoots[skinId];
    for (const auto jointId : skin.joints) {
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
      const auto parentId = parentMap[jointId];
      auto parentIter = std::find(skin.joints.begin(), skin.joints.end(), parentId);
      if (parentIter == skin.joints.end()) {
        bool foundSibling = false;
        for (auto& root : roots) {
          auto commonAncestor = findClosestCommonAncestor(jointId, root, parentMap);
          if (commonAncestor != kInvalidNodeId) {
            root = commonAncestor;
            foundSibling = true;
            break;
          }
        }
        if (!foundSibling) {
          skinSkeletonRoots[skinId].push_back(jointId);
        }
      }
    }
  }

  auto result = std::vector<std::vector<uint32_t>>(model.skins.size());
  // Merge skeleton between different skins. If the skeleton root of one is an ancestor of another,
  // the other one can be merged to the same root.
  for (auto skinId = 0; skinId < model.skins.size(); skinId++) {
    MT_CHECK(
        !skinSkeletonRoots[skinId].empty(),
        "id: {}, size: {}",
        skinId,
        skinSkeletonRoots[skinId].size());
    if (skinSkeletonRoots[skinId].size() > 1) {
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
      auto& roots = result[skinId];
      roots = skinSkeletonRoots[skinId];
    } else {
      auto ancestors = getAncestorsIncludingSelf(skinSkeletonRoots[skinId][0], parentMap);
      bool isIndependent = true;
      for (auto checkId = 0; checkId < model.skins.size(); checkId++) {
        MT_CHECK(
            !skinSkeletonRoots[checkId].empty(),
            "id: {}, size: {}",
            checkId,
            skinSkeletonRoots[checkId].size());
        if ((checkId == skinId) || skinSkeletonRoots[checkId].size() > 1) {
          continue;
        }
        if (std::find(ancestors.begin(), ancestors.end(), skinSkeletonRoots[checkId][0]) !=
            ancestors.end()) {
          isIndependent = false;
          break;
        }
      }
      if (isIndependent) {
        // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
        auto& roots = result[skinId];
        roots.push_back(skinSkeletonRoots[skinId][0]);
      }
    }
  }
  // Remove the empty skeletons that were skipped
  result.erase(
      std::remove_if(
          result.begin(),
          result.end(),
          [](const std::vector<uint32_t>& roots) { return roots.empty(); }),
      result.end());
  return result;
}

std::tuple<std::string, JointList, CollisionGeometry_u, LocatorList, std::vector<size_t>>
loadHierarchy(const fx::gltf::Document& model) {
  JointList joints;
  auto collision = std::make_unique<CollisionGeometry>();
  LocatorList locators;
  auto nodeToObjectMap = std::vector<size_t>(model.nodes.size(), kInvalidIndex);
  const bool useExtension = hasMomentumExtension(model);

  // Currently only the first found skeleton is used
  // #TODO: support multiple characters in the file, and use the skin information stored in
  // CharacterInfo.
  auto skeletonRoots = gatherSkeletonRoots(model);
  MT_CHECK(skeletonRoots.size() == 1, "{}", skeletonRoots.size());
  for (const auto& rootNode : skeletonRoots[0]) {
    loadHierarchyRecursive(
        model, rootNode, kInvalidIndex, joints, collision, locators, nodeToObjectMap, useExtension);
    if (!joints.empty()) {
      break;
    }
  }
  MT_THROW_IF(model.nodes.empty(), "No valid node found in the gltf file.");
  std::string name = model.nodes.at(0).name;

  std::sort(
      collision->begin(), collision->end(), [](const TaperedCapsule& c1, const TaperedCapsule& c2) {
        int c1Parent = c1.parent == kInvalidIndex ? -1 : static_cast<int>(c1.parent);
        int c2Parent = c2.parent == kInvalidIndex ? -1 : static_cast<int>(c2.parent);
        return c1Parent < c2Parent;
      });
  std::sort(locators.begin(), locators.end(), [](const Locator& l1, const Locator& l2) {
    int l1Parent = l1.parent == kInvalidIndex ? -1 : static_cast<int>(l1.parent);
    int l2Parent = l2.parent == kInvalidIndex ? -1 : static_cast<int>(l2.parent);
    return l1Parent < l2Parent;
  });
  // remove the collision geometry if it was empty to begin with
  if (collision->empty()) {
    collision.reset();
  }
  return std::make_tuple(name, joints, std::move(collision), locators, nodeToObjectMap);
}

JointList createSkeletonWithOnlyRoot() {
  JointList joints;
  joints.push_back(Joint());
  MT_CHECK(joints.size() == 1, "{}", joints.size());
  joints[0].name = "root";
  return joints;
}

} // namespace momentum
