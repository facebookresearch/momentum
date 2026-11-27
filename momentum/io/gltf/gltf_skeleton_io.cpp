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
#include "momentum/common/log.h"
#include "momentum/io/common/json_utils.h"
#include "momentum/io/common/stable_sort_order.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/io/gltf/utils/momentum_extension.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <ranges>
#include <span>

namespace momentum {

namespace {

// Whether `node` is a joint that GltfBuilder marked with the momentum extension.
[[nodiscard]] bool isMarkedJoint(const fx::gltf::Node& node) {
  return findMomentumExtension(node.extensionsAndExtras).value("type", std::string()) ==
      "skeleton_joint";
}

[[nodiscard]] std::vector<uint32_t> getAncestorsIncludingSelf(
    uint32_t nodeId,
    std::span<const uint32_t> parents) {
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
    std::span<const uint32_t> parents) {
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
// Along the way, save the joints, collision geometry and the locators, the node of each collision
// shape and locator, as well as a nodeToObjectMap for query. nodeToObjectMap stores mapping from
// node index to joint index, if the node is a joint. If it's a collision capsule or locator,
// stores the index in the collision volume array or locator array.
// If no node of the document is a marked joint, every node without a mesh, and every node that a
// skin lists (`skinJoints`), is loaded as a joint. A node with a skin or a camera, and the nodes
// below it, are not part of the hierarchy (see isHierarchyNode()), even when a skin lists them.
// #TODO: correctly load hierarchy from the skinning information if no momentum extension is in the
// file.
void loadHierarchyRecursive(
    const fx::gltf::Document& model,
    const int nodeId,
    size_t parentJointId,
    JointList& joints,
    CollisionGeometry_u& collision,
    std::vector<size_t>& collisionNodes,
    LocatorList& locators,
    std::vector<size_t>& locatorNodes,
    PhysicalProperties& physicalProperties,
    std::vector<size_t>& nodeToObjectMap,
    bool useExtension,
    bool meshNodesAreJoints,
    const std::vector<bool>& skinJoints,
    std::vector<bool>& visited) {
  MT_THROW_IF(
      (nodeId < 0) || (nodeId >= model.nodes.size()),
      "Invalid node id found in the gltf hierarchy: {}",
      nodeId);
  // A malformed document can link a node from several parents or in a cycle; load it only once.
  if (visited.at(nodeId)) {
    return;
  }
  visited.at(nodeId) = true;
  const auto& node = model.nodes[nodeId];
  if (!isHierarchyNode(node)) {
    return;
  }

  const auto& extension = findMomentumExtension(node.extensionsAndExtras);
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
    collisionNodes.push_back(nodeId);
    nodeToObjectMap[nodeId] = collision->size() - 1;
  } else if (type == "collision_ellipsoid") {
    // Found collision geometry, should be the end node
    MT_THROW_IF(
        parentJointId == kInvalidIndex,
        "Invalid collision ellipsoid without a parent joint: {}",
        node.name);
    auto ellipsoid = createCollisionEllipsoid(node, extension);
    ellipsoid.parent = parentJointId;
    collision->push_back(ellipsoid);
    collisionNodes.push_back(nodeId);
    nodeToObjectMap[nodeId] = collision->size() - 1;
  } else if (type == "collision_box") {
    // Found collision geometry, should be the end node
    MT_THROW_IF(
        parentJointId == kInvalidIndex,
        "Invalid collision box without a parent joint: {}",
        node.name);
    auto box = createCollisionBox(node, extension);
    box.parent = parentJointId;
    collision->push_back(box);
    collisionNodes.push_back(nodeId);
    nodeToObjectMap[nodeId] = collision->size() - 1;
  } else if (type == "locator") {
    // Found locator, should be the end node
    MT_THROW_IF(
        parentJointId == kInvalidIndex, "Invalid locator without a parent joint: {}", node.name);
    Locator loc = createLocator(node, extension);
    loc.parent = parentJointId;
    locators.push_back(loc);
    locatorNodes.push_back(nodeId);
    nodeToObjectMap[nodeId] = locators.size() - 1;
  } else if (
      (!useExtension && (node.mesh < 0 || meshNodesAreJoints || skinJoints[nodeId])) ||
      type == "skeleton_joint") {
    Joint joint = createJoint(node);
    joint.parent = parentJointId;
    joints.push_back(joint);
    nodeToObjectMap[nodeId] = joints.size() - 1;
    newParentJointId = joints.size() - 1;
    if (extension.contains("physicalProperties")) {
      physicalProperties.push_back(jointPhysicalPropertiesFromJson(
          extension.at("physicalProperties"), joint.name, newParentJointId));
    }
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
        collisionNodes,
        locators,
        locatorNodes,
        physicalProperties,
        nodeToObjectMap,
        useExtension,
        meshNodesAreJoints,
        skinJoints,
        visited);
  }
}

// Orders the collision shapes or locators `items` by parent joint, keeping their order within a
// joint, and points the entry of each one's node (`itemNodes`) in `nodeToObjectMap` at its new
// index.
template <typename T>
void sortByParentJoint(
    std::vector<T>& items,
    std::span<const size_t> itemNodes,
    std::span<size_t> nodeToObjectMap) {
  std::vector<size_t> parents;
  parents.reserve(items.size());
  for (const T& item : items) {
    parents.push_back(item.parent);
  }
  const std::vector<size_t> order = stableSortOrder(parents);
  for (size_t position = 0; position < order.size(); ++position) {
    nodeToObjectMap[itemNodes[order[position]]] = position;
  }
  applyOrder(items, order);
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

CollisionEllipsoid createCollisionEllipsoid(
    const fx::gltf::Node& node,
    const nlohmann::json& extension) {
  CollisionEllipsoid ellipsoid;
  ellipsoid.parent = kInvalidIndex;
  ellipsoid.transformation = Transform();
  ellipsoid.transformation.rotation = toMomentumQuaternionf(node.rotation);
  ellipsoid.transformation.translation = toMomentumVec3f(node.translation);

  try {
    ellipsoid.radii = fromJson<Vector3f>(extension["radii"]);
  } catch (const std::exception&) {
    MT_THROW(
        "Fail to parse json {} for collision ellipsoid {} when loading character.",
        extension.dump(),
        node.name);
  }
  return ellipsoid;
}

CollisionBox createCollisionBox(const fx::gltf::Node& node, const nlohmann::json& extension) {
  CollisionBox box;
  box.parent = kInvalidIndex;
  box.transformation = Transform();
  box.transformation.rotation = toMomentumQuaternionf(node.rotation);
  box.transformation.translation = toMomentumVec3f(node.translation);

  try {
    box.halfExtents = fromJson<Vector3f>(extension["halfExtents"]);
  } catch (const std::exception&) {
    MT_THROW(
        "Fail to parse json {} for collision box {} when loading character.",
        extension.dump(),
        node.name);
  }
  return box;
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

bool hasMomentumType(const fx::gltf::Node& node) {
  return !findMomentumExtension(node.extensionsAndExtras).value("type", std::string()).empty();
}

bool hasMarkedJoints(const fx::gltf::Document& model) {
  return std::ranges::any_of(model.nodes, isMarkedJoint);
}

std::vector<bool> skinJointNodes(const fx::gltf::Document& model) {
  std::vector<bool> result(model.nodes.size(), false);
  for (const auto& skin : model.skins) {
    for (const uint32_t node : skin.joints) {
      if (node < result.size()) {
        result[node] = true;
      }
    }
  }
  return result;
}

Transform nodeLocalTransform(const fx::gltf::Node& node) {
  if (node.matrix != fx::gltf::defaults::IdentityMatrix) {
    Matrix4f matrix = Map<const Matrix4f>(node.matrix.data());
    matrix.topRightCorner<3, 1>() *= toCm();
    return Transform::fromMatrix(matrix);
  }
  return Transform(
      toMomentumVec3f(node.translation), toMomentumQuaternionf(node.rotation), node.scale[0]);
}

namespace {

// Whether loadHierarchyRecursive can load `node` as a joint of a document without joint marks: a
// node of a skeleton without a mesh or that a skin lists, or any node of an object (see
// gatherObjectRoots()).
[[nodiscard]] bool isUnmarkedJointNode(const fx::gltf::Node& node) {
  if (!isHierarchyNode(node)) {
    return false;
  }
  const std::string type = findMomentumExtension(node.extensionsAndExtras).value("type", "");
  return type != "collision_capsule" && type != "collision_ellipsoid" && type != "collision_box" &&
      type != "locator";
}

// Whether the scale of `node` differs between its axes.
[[nodiscard]] bool hasNonUniformScale(const fx::gltf::Node& node) {
  Vector3f scale = Map<const Vector3f>(node.scale.data());
  if (node.matrix != fx::gltf::defaults::IdentityMatrix) {
    scale = Map<const Matrix4f>(node.matrix.data()).topLeftCorner<3, 3>().colwise().norm();
  }
  constexpr float kTolerance = 1e-4f;
  return (scale.array() - scale.x()).abs().maxCoeff() > kTolerance * std::abs(scale.x());
}

// Walks breadth first from `roots`: each node that is not yet queued gets the node that lists it as
// a child as its parent, and is queued. Each node is queued at most once, after its parent, so
// following parents always ends at a root.
void addParentsBreadthFirst(
    const fx::gltf::Document& model,
    std::span<const uint32_t> roots,
    std::vector<bool>& queued,
    std::vector<uint32_t>& parentMap) {
  std::vector<uint32_t> queue;
  for (const uint32_t root : roots) {
    if (root < queued.size() && !queued.at(root)) {
      queued.at(root) = true;
      queue.push_back(root);
    }
  }
  for (size_t i = 0; i < queue.size(); ++i) {
    const uint32_t parent = queue.at(i);
    for (const int32_t child : model.nodes.at(parent).children) {
      if (child < 0 || static_cast<size_t>(child) >= queued.size() || queued.at(child)) {
        continue;
      }
      queued.at(child) = true;
      parentMap.at(child) = parent;
      queue.push_back(static_cast<uint32_t>(child));
    }
  }
}

// The root nodes of the scene that the loader reads, or none.
std::span<const uint32_t> loadedSceneRoots(const fx::gltf::Document& model) {
  const int32_t scene = sceneToLoad(model);
  if (scene < 0 || static_cast<size_t>(scene) >= model.scenes.size()) {
    return {};
  }
  return model.scenes.at(scene).nodes;
}

// The parent of each node, for placing joints and for the skeleton of each skin: the parent in the
// loaded scene (see
// buildSceneParentMap()) for the nodes that the scene reaches, and for the other nodes the parent
// in their own tree, from the nodes that no node lists as a child. So a skeleton outside the loaded
// scene keeps its hierarchy.
std::vector<uint32_t> buildNodeParentMap(const fx::gltf::Document& model) {
  const size_t numNodes = model.nodes.size();
  auto parentMap = std::vector<uint32_t>(numNodes, kInvalidNodeId);
  std::vector<bool> queued(numNodes, false);
  addParentsBreadthFirst(model, loadedSceneRoots(model), queued, parentMap);
  std::vector<bool> isChild(numNodes, false);
  for (const auto& node : model.nodes) {
    for (const int32_t child : node.children) {
      if (child >= 0 && static_cast<size_t>(child) < numNodes) {
        isChild.at(child) = true;
      }
    }
  }
  std::vector<uint32_t> roots;
  for (uint32_t nodeId = 0; nodeId < numNodes; ++nodeId) {
    if (!isChild.at(nodeId)) {
      roots.push_back(nodeId);
    }
  }
  addParentsBreadthFirst(model, roots, queued, parentMap);
  return parentMap;
}

} // namespace

std::vector<JointFileTransform> jointFileTransforms(
    const fx::gltf::Document& model,
    const Skeleton& skeleton,
    std::span<const size_t> nodeToObjectMap) {
  const size_t numJoints = skeleton.joints.size();
  const bool marked = hasMarkedJoints(model);
  std::vector<JointFileTransform> result(numJoints);
  for (size_t nodeId = 0; nodeId < model.nodes.size() && nodeId < nodeToObjectMap.size();
       ++nodeId) {
    const size_t jointId = nodeToObjectMap[nodeId];
    const auto& node = model.nodes.at(nodeId);
    // The node map shares its indices between joints, locators and collision shapes.
    if (jointId < numJoints && (marked ? isMarkedJoint(node) : isUnmarkedJointNode(node)) &&
        node.name == skeleton.joints[jointId].name) {
      result[jointId].node = static_cast<uint32_t>(nodeId);
    }
  }
  // The loader places marked joints by their own transforms.
  if (marked) {
    return result;
  }

  const auto parentMap = buildNodeParentMap(model);
  std::vector<std::optional<Transform>> worlds(model.nodes.size());
  const auto world = [&](uint32_t nodeId) {
    std::vector<uint32_t> chain;
    for (uint32_t id = nodeId; id != kInvalidNodeId && !worlds[id].has_value();
         id = parentMap[id]) {
      chain.push_back(id);
    }
    for (const uint32_t id : std::views::reverse(chain)) {
      const auto& node = model.nodes.at(id);
      // Called outside MT_LOGW_IF: builds without logging do not evaluate its condition.
      if (hasNonUniformScale(node)) {
        MT_LOGW(
            "glTF node '{}' has a non-uniform scale; momentum uses the scale of its x axis.",
            node.name);
      }
      const uint32_t parent = parentMap[id];
      worlds[id] = parent == kInvalidNodeId ? nodeLocalTransform(node)
                                            : *worlds[parent] * nodeLocalTransform(node);
    }
    return *worlds[nodeId];
  };

  for (size_t jointId = 0; jointId < numJoints; ++jointId) {
    auto& entry = result[jointId];
    if (entry.node == kInvalidNodeId) {
      continue;
    }
    const uint32_t parentNode = parentMap[entry.node];
    const Transform parentWorld = parentNode == kInvalidNodeId ? Transform() : world(parentNode);
    Transform parentJointFrame;
    const size_t parentJoint = skeleton.joints[jointId].parent;
    if (parentJoint < numJoints && result[parentJoint].node != kInvalidNodeId) {
      const Transform parentJointWorld = world(result[parentJoint].node);
      parentJointFrame = Transform(parentJointWorld.translation, parentJointWorld.rotation);
    }
    entry.pre = parentJointFrame.inverse() * parentWorld;
    const float scale = world(entry.node).scale;
    if (std::abs(scale) > std::numeric_limits<float>::epsilon()) {
      entry.post = Transform::makeScale(1.0f / scale);
    } else {
      MT_LOGW(
          "glTF joint '{}' has a rest scale of zero; its scale is not removed from its rest pose.",
          model.nodes.at(entry.node).name);
    }
  }
  return result;
}

int32_t sceneToLoad(const fx::gltf::Document& model) {
  return model.scene < 0 && !model.scenes.empty() ? 0 : model.scene;
}

std::vector<uint32_t> buildSceneParentMap(const fx::gltf::Document& model) {
  auto parentMap = std::vector<uint32_t>(model.nodes.size(), kInvalidNodeId);
  std::vector<bool> queued(model.nodes.size(), false);
  addParentsBreadthFirst(model, loadedSceneRoots(model), queued, parentMap);
  return parentMap;
}

uint32_t getTopLevelAncestor(uint32_t nodeId, std::span<const uint32_t> parentMap) {
  if (nodeId >= parentMap.size()) {
    return kInvalidNodeId;
  }
  while (parentMap[nodeId] != kInvalidNodeId) {
    nodeId = parentMap[nodeId];
  }
  return nodeId;
}

std::vector<std::vector<uint32_t>> gatherSkeletonRoots(
    const fx::gltf::Document& model,
    std::span<const uint32_t> sceneParentMap) {
  const int32_t scene = sceneToLoad(model);
  MT_THROW_IF(
      scene < 0 || static_cast<size_t>(scene) >= model.scenes.size(),
      "Invalid glTF scene index {} for {} scene(s)",
      scene,
      model.scenes.size());
  const auto& sceneRoots = model.scenes.at(scene).nodes;

  // GltfBuilder marks joints with the momentum extension and puts each character under its own
  // scene root, so every scene root with a marked joint below it is one character.
  if (hasMomentumExtension(model.extensionsAndExtras)) {
    std::vector<bool> hasMarkedJoint(model.nodes.size(), false);
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId) {
      if (isMarkedJoint(model.nodes[nodeId])) {
        const uint32_t root = getTopLevelAncestor(static_cast<uint32_t>(nodeId), sceneParentMap);
        hasMarkedJoint.at(root) = true;
      }
    }
    std::vector<std::vector<uint32_t>> characterRoots;
    for (const uint32_t root : sceneRoots) {
      if (root < hasMarkedJoint.size() && hasMarkedJoint.at(root)) {
        hasMarkedJoint.at(root) = false;
        characterRoots.push_back({root});
      }
    }
    if (!characterRoots.empty()) {
      return characterRoots;
    }
  }

  // A document without meshes or without skins gives the scene roots as skeleton roots.
  if (model.meshes.empty() || model.skins.empty()) {
    return {sceneRoots};
  }

  // The glTF format defines the joint hierarchy of a skin by the node hierarchy, whichever scene
  // holds it, and requires the joints of a skin to have a common root. So the skeleton of a skin
  // starts at the closest common ancestor of its joints in the node hierarchy.
  const auto parentMap = buildNodeParentMap(model);
  auto skinSkeletonRoots =
      std::vector<std::vector<uint32_t>>(model.skins.size(), std::vector<uint32_t>());
  for (auto skinId = 0; skinId < model.skins.size(); skinId++) {
    const auto& skin = model.skins[skinId];
    auto& roots = skinSkeletonRoots[skinId];
    for (const auto jointId : skin.joints) {
      if (jointId >= parentMap.size()) {
        // Malformed glTF: the skin references a node that does not exist.
        MT_LOGW("Skipping out-of-range joint {} in glTF skin {}.", jointId, skinId);
        continue;
      }
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

  for (auto skinId = 0; skinId < model.skins.size(); skinId++) {
    MT_THROW_IF(
        skinSkeletonRoots[skinId].size() > 1,
        "glTF skin {} ('{}') has joints without a common root, which the glTF format does not "
        "allow.",
        skinId,
        model.skins[skinId].name);
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
    auto ancestors = getAncestorsIncludingSelf(skinSkeletonRoots[skinId][0], parentMap);
    bool isIndependent = true;
    for (auto checkId = 0; checkId < model.skins.size(); checkId++) {
      MT_CHECK(
          !skinSkeletonRoots[checkId].empty(),
          "id: {}, size: {}",
          checkId,
          skinSkeletonRoots[checkId].size());
      if (checkId == skinId) {
        continue;
      }
      // Skins with the same root share one skeleton, which the first of them keeps.
      const uint32_t checkRoot = skinSkeletonRoots[checkId][0];
      const bool dependent = checkRoot == skinSkeletonRoots[skinId][0]
          ? checkId < skinId
          : std::ranges::find(ancestors, checkRoot) != ancestors.end();
      if (dependent) {
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
  // Remove the empty skeletons that were skipped
  result.erase(
      std::remove_if(
          result.begin(),
          result.end(),
          [](const std::vector<uint32_t>& roots) { return roots.empty(); }),
      result.end());
  return result;
}

std::vector<uint32_t> gatherObjectRoots(
    const fx::gltf::Document& model,
    std::span<const uint32_t> sceneParentMap,
    std::span<const std::vector<uint32_t>> skeletonRoots) {
  const size_t numNodes = model.nodes.size();
  // The nodes of the skeletons, and the nodes above them.
  std::vector<bool> inSkeleton(numNodes, false);
  std::vector<bool> aboveSkeleton(numNodes, false);
  for (const auto& roots : skeletonRoots) {
    for (const uint32_t root : roots) {
      if (root >= numNodes) {
        continue;
      }
      for (uint32_t node = sceneParentMap[root]; node != kInvalidNodeId;
           node = sceneParentMap[node]) {
        aboveSkeleton[node] = true;
      }
      std::vector<uint32_t> stack = {root};
      while (!stack.empty()) {
        const uint32_t node = stack.back();
        stack.pop_back();
        if (node >= numNodes || inSkeleton[node]) {
          continue;
        }
        inSkeleton[node] = true;
        for (const int32_t child : model.nodes[node].children) {
          stack.push_back(static_cast<uint32_t>(child));
        }
      }
    }
  }

  // Whether the hierarchy below `node` (the node included) has a mesh without a skin.
  const auto hasUnskinnedMesh = [&](uint32_t start) {
    std::vector<bool> visited(numNodes, false);
    std::vector<uint32_t> stack = {start};
    while (!stack.empty()) {
      const uint32_t node = stack.back();
      stack.pop_back();
      if (node >= numNodes || visited[node] || !isHierarchyNode(model.nodes[node]) ||
          hasMomentumType(model.nodes[node])) {
        continue;
      }
      visited[node] = true;
      if (model.nodes[node].mesh >= 0) {
        return true;
      }
      for (const int32_t child : model.nodes[node].children) {
        stack.push_back(static_cast<uint32_t>(child));
      }
    }
    return false;
  };

  // Each highest node that is neither in nor above a skeleton, in the order of the scene.
  std::vector<uint32_t> result;
  const int32_t scene = sceneToLoad(model);
  if (scene < 0 || static_cast<size_t>(scene) >= model.scenes.size()) {
    return result;
  }
  std::vector<bool> visited(numNodes, false);
  const auto& sceneNodes = model.scenes.at(scene).nodes;
  std::vector<uint32_t> stack(sceneNodes.rbegin(), sceneNodes.rend());
  while (!stack.empty()) {
    const uint32_t node = stack.back();
    stack.pop_back();
    if (node >= numNodes || visited[node] || inSkeleton[node] ||
        !isHierarchyNode(model.nodes[node]) || hasMomentumType(model.nodes[node])) {
      continue;
    }
    visited[node] = true;
    if (aboveSkeleton[node]) {
      for (const int32_t child : std::views::reverse(model.nodes[node].children)) {
        stack.push_back(static_cast<uint32_t>(child));
      }
    } else if (hasUnskinnedMesh(node)) {
      result.push_back(node);
    }
  }
  return result;
}

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
    uint32_t characterRootNode,
    bool meshNodesAreJoints) {
  JointList joints;
  auto collision = std::make_unique<CollisionGeometry>();
  std::vector<size_t> collisionNodes;
  LocatorList locators;
  std::vector<size_t> locatorNodes;
  PhysicalProperties physicalProperties;
  auto nodeToObjectMap = std::vector<size_t>(model.nodes.size(), kInvalidIndex);
  // GltfBuilder marks every joint when it writes the momentum extension, and writes a
  // document-level extension (the frame rate of a motion) even without it, so the marks decide.
  const bool useExtension = hasMarkedJoints(model);
  std::vector<bool> visited(model.nodes.size(), false);
  // Without joint marks, a node that a skin lists is a joint, even when it has a mesh without a
  // skin.
  const std::vector<bool> skinJoints = skinJointNodes(model);

  for (const auto& rootNode : skeletonRoots) {
    loadHierarchyRecursive(
        model,
        rootNode,
        kInvalidIndex,
        joints,
        collision,
        collisionNodes,
        locators,
        locatorNodes,
        physicalProperties,
        nodeToObjectMap,
        useExtension,
        meshNodesAreJoints,
        skinJoints,
        visited);
    if (!joints.empty()) {
      break;
    }
  }
  MT_THROW_IF(model.nodes.empty(), "No valid node found in the gltf file.");
  std::string name =
      characterRootNode < model.nodes.size() ? model.nodes[characterRootNode].name : std::string();

  // Order the collision geometry and the locators by parent joint, keeping the file order within a
  // joint, as the FBX loader does. Every one has a parent joint (see loadHierarchyRecursive).
  sortByParentJoint(*collision, collisionNodes, nodeToObjectMap);
  sortByParentJoint(locators, locatorNodes, nodeToObjectMap);
  // remove the collision geometry if it was empty to begin with
  if (collision->empty()) {
    collision.reset();
  }
  return std::make_tuple(
      name, joints, std::move(collision), locators, physicalProperties, nodeToObjectMap);
}

JointList createSkeletonWithOnlyRoot() {
  JointList joints;
  joints.push_back(Joint());
  MT_CHECK(joints.size() == 1, "{}", joints.size());
  joints[0].name = "root";
  return joints;
}

} // namespace momentum
