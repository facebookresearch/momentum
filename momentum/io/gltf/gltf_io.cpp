/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_io.h"

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/joint.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/gltf/gltf_builder.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/io/gltf/utils/json_utils.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/mppca.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <variant>

namespace {

using namespace momentum;

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

// Return whether or not the node is allowed in the skeleton hierarchy, which includes the
// joints, locators and collision geometries.
// The skinned mesh should not exist in the same hierarchy as the skeleton.
// Note that markers have mesh, so we are allowing the nodes with mesh in the hierarchy
inline bool isHierarchyNode(const fx::gltf::Node& node) {
  return (node.skin < 0) && (node.camera < 0);
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

// Return a vector of roots for each skeleton. Each item represents a skeleton, and for each
// skeleton multiple root nodes are allowed.
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
        for (auto i = 0; i < roots.size(); i++) {
          auto commonAncestor = findClosestCommonAncestor(jointId, roots[i], parentMap);
          if (commonAncestor != kInvalidNodeId) {
            roots[i] = commonAncestor;
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
      (nodeId < 0) || (nodeId > model.nodes.size()),
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
        (childId < 0) || (childId > model.nodes.size()),
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

// Load the hierarchy from the file, including skeleton, collision geometry, locators and a mapping
// between the nodes and the objects.
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
  return std::make_tuple(name, joints, std::move(collision), locators, nodeToObjectMap);
}

// Return the number of vertices newly loaded
size_t
addMesh(const fx::gltf::Document& model, const fx::gltf::Primitive& primitive, Mesh_u& mesh) {
  if (primitive.mode != fx::gltf::Primitive::Mode::Triangles) {
    MT_LOGW("Mesh not loaded because it's not a triangle mesh");
    return 0;
  }

  // load index buffer
  auto idxDense = copyAccessorBuffer<uint32_t>(model, primitive.indices);
  if (idxDense.empty()) {
    // Try fallback with short indices.
    auto a = copyAccessorBuffer<uint16_t>(model, primitive.indices);
    for (const auto& ae : a) {
      idxDense.push_back(ae);
    }
  }
  MT_CHECK(idxDense.size() % 3 == 0, "{} % 3 = {}", idxDense.size(), idxDense.size() % 3);
  std::vector<Vector3i> idx(idxDense.size() / 3);
  std::copy_n(idxDense.data(), idxDense.size(), &idx[0][0]);

  // load vertex position buffer
  auto pos = copyAccessorBuffer<Vector3f>(model, primitive.attributes.at("POSITION"));
  toMomentumVec3f(pos);

  // if we have no points or indices, skip loading
  if (idx.empty() || pos.empty()) {
    return 0;
  }

  // load optional normal buffer
  std::vector<Vector3f> nml;
  const auto normId = primitive.attributes.find("NORMAL");
  if (normId != primitive.attributes.end()) {
    nml = copyAccessorBuffer<Vector3f>(model, normId->second);
  }
  MT_CHECK(nml.empty() || nml.size() == pos.size(), "nml: {}, pos: {}", nml.size(), pos.size());
  MT_LOGT_IF(nml.empty(), "no vertex normal found");

  // load optional color buffer
  std::vector<Vector3b> col;
  const auto colorId = primitive.attributes.find("COLOR_0");
  if (colorId != primitive.attributes.end()) {
    col = copyAccessorBuffer<Vector3b>(model, colorId->second);
  }
  MT_CHECK(col.empty() || col.size() == pos.size(), "col: {}, pos: {}", col.size(), pos.size());
  MT_LOGT_IF(col.empty(), "no vertex color found");

  // NOTE: gltf does not support multiple texcoords per vertex, so we only load the GLTF texcoords
  //       if we don't have our own custom extension
  const auto& extension = getMomentumExtension(primitive);
  std::vector<Vector2f> texcoord;
  std::vector<Vector3i> texfaces;
  if (extension.count("texcoords") > 0 && extension.count("texfaces") > 0) {
    texcoord = copyAccessorBuffer<Vector2f>(model, extension.at("texcoords"));

    // load "face index buffer
    auto fidxDense = copyAccessorBuffer<uint32_t>(model, extension.at("texfaces"));
    MT_CHECK(fidxDense.size() % 3 == 0, "{} % 3 = {}", fidxDense.size(), fidxDense.size() % 3);
    texfaces.resize(fidxDense.size() / 3);
    if (fidxDense.size() > 0) {
      std::copy_n(fidxDense.data(), fidxDense.size(), &texfaces[0][0]);
    }
  } else {
    // load optional standard GLTF texcoord buffer
    auto texcoordId = primitive.attributes.find("TEXCOORD_0");
    if (texcoordId != primitive.attributes.end()) {
      texcoord = copyAccessorBuffer<Vector2f>(model, texcoordId->second);
    }
    const auto kVertexOffset = mesh->vertices.size();
    // Update vertex indices of the faces!!!
    if (kVertexOffset > 0) {
      for (auto&& iFace : idx) {
        iFace[0] += kVertexOffset;
        iFace[1] += kVertexOffset;
        iFace[2] += kVertexOffset;
      }
    }
    if (texcoord.size() == pos.size()) {
      texfaces = idx;
    }
    MT_CHECK(
        texcoord.empty() || texcoord.size() == pos.size(),
        "texcoord: {}, pos: {}",
        texcoord.size(),
        pos.size());
  }
  MT_LOGT_IF(texcoord.empty(), "no texture coords found");

  // append new faces
  mesh->faces.insert(mesh->faces.end(), idx.begin(), idx.end());
  mesh->vertices.insert(mesh->vertices.end(), pos.begin(), pos.end());
  mesh->normals.insert(mesh->normals.end(), nml.begin(), nml.end());
  mesh->colors.insert(mesh->colors.end(), col.begin(), col.end());
  mesh->texcoords.insert(mesh->texcoords.end(), texcoord.begin(), texcoord.end());
  mesh->texcoord_faces.insert(mesh->texcoord_faces.end(), texfaces.begin(), texfaces.end());

  // make sure we have enough normals and colors
  mesh->normals.resize(mesh->vertices.size(), Vector3f::Zero());
  mesh->colors.resize(mesh->vertices.size(), Vector3b::Zero());
  mesh->confidence.resize(mesh->vertices.size(), 1.0f);
  return pos.size();
}

void addSkinWeights(
    const fx::gltf::Document& model,
    const fx::gltf::Skin& skin,
    const fx::gltf::Primitive& primitive,
    const std::vector<size_t>& nodeToObjectMap,
    const size_t kNumVertices,
    SkinWeights_u& skinWeights) {
  const auto kVertexOffset = skinWeights->index.rows();
  skinWeights->index.conservativeResize(kVertexOffset + kNumVertices, Eigen::NoChange);
  skinWeights->weight.conservativeResize(kVertexOffset + kNumVertices, Eigen::NoChange);
  skinWeights->index.bottomRows(kNumVertices).setZero();
  skinWeights->weight.bottomRows(kNumVertices).setZero();

  // load skinning in batches of 4 indices/weights at a time (up to 8)
  for (size_t i = 0; i < 2; i++) {
    // load skinning index buffer
    auto jointAttribute = primitive.attributes.find(std::string("JOINTS_") + std::to_string(i));
    std::vector<Vector4s> jointIndices;
    if (jointAttribute != primitive.attributes.end()) {
      jointIndices = copyAccessorBuffer<Vector4s>(model, jointAttribute->second);
      if (jointIndices.empty()) {
        // Try fallback with short indices.
        auto jointIndicesShort = copyAccessorBuffer<Vector4b>(model, jointAttribute->second);
        for (const auto& value : jointIndicesShort) {
          jointIndices.push_back(value.cast<uint16_t>());
        }
      }
    }

    // load skinning weight buffer
    auto weightsAttribute = primitive.attributes.find(std::string("WEIGHTS_") + std::to_string(i));
    std::vector<Vector4f> weightsData;
    if (weightsAttribute != primitive.attributes.end()) {
      weightsData = copyAlignedAccessorBuffer<Vector4f>(model, weightsAttribute->second);
      if (weightsData.empty()) {
        MT_LOGW("No skinning weights read");
        return;
      }
    } else {
      MT_LOGW("No skinning weights stored on primitive");
      return;
    }

    if (jointIndices.empty() || weightsData.empty() || jointIndices.size() != weightsData.size() ||
        jointIndices.size() != kNumVertices) {
      MT_LOGW_IF(jointIndices.empty() || weightsData.empty(), "Mesh is not skinned to any joint");
      MT_LOGW_IF(
          !jointIndices.empty() && !weightsData.empty() &&
              jointIndices.size() != weightsData.size(),
          "Inconsistent data: {} vertices are skinned but {} weights found",
          jointIndices.size(),
          weightsData.size());
      MT_LOGW_IF(
          jointIndices.size() != kNumVertices,
          "Inconsistent data: {} vertices are skinned but mesh has {} vertices",
          jointIndices.size(),
          kNumVertices);
      return;
    }

    // copy indices/vertices into our buffer
    for (Eigen::Index v = 0; v < static_cast<int>(kNumVertices); v++) {
      for (size_t d = 0; d < 4; d++) {
        const auto& skinWeight = weightsData[v][d];
        const auto& jointIdxInSkin = jointIndices[v][d];
        MT_CHECK(
            jointIdxInSkin >= 0 && jointIdxInSkin < skin.joints.size(),
            "{}: {}",
            jointIdxInSkin,
            skin.joints.size());
        const auto& nodeIdx = skin.joints[jointIdxInSkin];
        const auto& jointIndex = nodeToObjectMap[nodeIdx];
        if (jointIndex != kInvalidIndex) {
          skinWeights->index(kVertexOffset + v, i * 4 + d) = (uint32_t)jointIndex;
          skinWeights->weight(kVertexOffset + v, i * 4 + d) = skinWeight;
        } else {
          MT_LOGW("Invalid joint index encountered when reading skinning weights");
        }
      }
    }
  }
}

// Load mesh and skinning information together so we can validate correctness
std::tuple<Mesh_u, SkinWeights_u> loadSkinnedMesh(
    const fx::gltf::Document& model,
    const std::vector<size_t>& meshNodes,
    const std::vector<size_t>& nodeToObjectMap) {
  std::vector<size_t> skinnedNodes;
  skinnedNodes.reserve(meshNodes.size());
  std::vector<size_t> unskinnedNodes;
  unskinnedNodes.reserve(meshNodes.size());
  for (auto nodeId : meshNodes) {
    MT_CHECK(nodeId >= 0 && nodeId < model.nodes.size(), "{}: {}", nodeId, model.nodes.size());
    const auto& node = model.nodes[nodeId];
    if ((node.mesh >= 0) && (node.skin >= 0)) {
      MT_CHECK(node.mesh < model.meshes.size(), "{}: {}", node.mesh, model.meshes.size());
      MT_CHECK(node.skin < model.skins.size(), "{}: {}", node.skin, model.skins.size());
      skinnedNodes.push_back(nodeId);
    } else if (node.mesh >= 0) {
      MT_CHECK(node.mesh < model.meshes.size(), "{}: {}", node.mesh, model.meshes.size());
      unskinnedNodes.push_back(nodeId);
    }
  }

  MT_LOGW_IF(
      !skinnedNodes.empty() && !unskinnedNodes.empty(),
      "Found both skinned meshes {} and meshes without skinning {}. Unskinned meshes will be ignored.",
      skinnedNodes.size(),
      unskinnedNodes.size());
  auto resultMesh = std::make_unique<Mesh>();
  auto skinWeights = std::make_unique<SkinWeights>();
  if (!skinnedNodes.empty()) {
    for (auto nodeId : skinnedNodes) {
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      const auto& node = model.nodes[nodeId];
      const auto& mesh = model.meshes[node.mesh];
      const auto& skin = model.skins[node.skin];
      for (const auto& primitive : mesh.primitives) {
        const auto kNumVertices = addMesh(model, primitive, resultMesh);
        addSkinWeights(model, skin, primitive, nodeToObjectMap, kNumVertices, skinWeights);
        MT_CHECK(
            resultMesh->vertices.size() == skinWeights->index.rows(),
            "vertices: {}, skinWeights: {}",
            resultMesh->vertices.size(),
            skinWeights->index.rows());
      }
    }
  } else if (!unskinnedNodes.empty()) {
    for (auto nodeId : unskinnedNodes) {
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      const auto& node = model.nodes[nodeId];
      const auto& mesh = model.meshes[node.mesh];
      for (const auto& primitive : mesh.primitives) {
        addMesh(model, primitive, resultMesh);
      }
    }
    return {std::move(resultMesh), nullptr};
  } else {
    return {};
  }
  return {std::move(resultMesh), std::move(skinWeights)};
}

JointList createSkeletonWithOnlyRoot() {
  JointList joints;
  joints.push_back(Joint());
  MT_CHECK(joints.size() == 1, "{}", joints.size());
  joints[0].name = "root";
  return joints;
}

SkinWeights_u bindMeshToJoint(const Mesh_u& mesh, size_t jointId) {
  auto skinWeights = std::make_unique<SkinWeights>();
  const auto kNumVertices = mesh->vertices.size();
  skinWeights->index.conservativeResize(kNumVertices, Eigen::NoChange);
  skinWeights->weight.conservativeResize(kNumVertices, Eigen::NoChange);
  skinWeights->index.bottomRows(kNumVertices).setZero();
  skinWeights->weight.bottomRows(kNumVertices).setZero();
  for (auto vertId = 0; vertId < kNumVertices; vertId++) {
    skinWeights->index(vertId, 0) = jointId;
    skinWeights->weight(vertId, 0) = 1.0f;
  }

  return skinWeights;
}

void loadGlobalExtensions(const fx::gltf::Document& model, Character& character) {
  try {
    // load additional momentum data if present
    const auto& def = getMomentumExtension(model.extensionsAndExtras);

    if (def.count("transform") > 0) {
      character.parameterTransform = parameterTransformFromJson(character, def["transform"]);
    } else {
      character.parameterTransform =
          ParameterTransform::empty(character.skeleton.joints.size() * kParametersPerJoint);
    }
    if (def.count("parameterSet") > 0) {
      character.parameterTransform.parameterSets =
          parameterSetsFromJson(character, def["parameterSet"]);
    }
    if (def.count("poseConstraints") > 0) {
      character.parameterTransform.poseConstraints =
          poseConstraintsFromJson(character, def["poseConstraints"]);
    }
    if (def.count("parameterLimits") > 0) {
      character.parameterLimits = parameterLimitsFromJson(character, def["parameterLimits"]);
    }
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf : {}", err.what());
  }
}

Character populateCharacterFromModel(
    const fx::gltf::Document& model,
    std::vector<size_t>& nodeToObjectMap) {
  // #TODO: set character name
  Character result;

  // ---------------------------------------------
  // load the joints, collision geometry and locators
  // ---------------------------------------------
  std::tie(
      result.name, result.skeleton.joints, result.collision, result.locators, nodeToObjectMap) =
      loadHierarchy(model);
  MT_CHECK(
      nodeToObjectMap.size() == model.nodes.size(),
      "nodeMap: {}, nodes: {}",
      nodeToObjectMap.size(),
      model.nodes.size());

  // ---------------------------------------------
  // load the mesh -- load from the nodes not in the hierarchy, and combine all meshes into one
  // ---------------------------------------------
  std::vector<size_t> remainingNodes;
  remainingNodes.reserve(model.nodes.size());
  for (auto nodeId = 0; nodeId < nodeToObjectMap.size(); nodeId++) {
    if (nodeToObjectMap[nodeId] == kInvalidIndex) {
      remainingNodes.push_back(nodeId);
    }
  }

  std::tie(result.mesh, result.skinWeights) =
      loadSkinnedMesh(model, remainingNodes, nodeToObjectMap);
  // No skinning nodes found but there are meshes in the scene. We will create a joint to parent
  // them.
  if (result.mesh != nullptr && result.skinWeights == nullptr) {
    result.skeleton.joints = createSkeletonWithOnlyRoot();
    for (auto& locator : result.locators) {
      locator.parent = 0;
    }
    result.skinWeights = bindMeshToJoint(result.mesh, 0);
  }

  loadGlobalExtensions(model, result);

  // Finalize jointMap and inverseBind pose
  result.resetJointMap();
  result.initInverseBindPose();

  return result;
}

std::tuple<MotionParameters, IdentityParameters> getMotionFromModel(fx::gltf::Document& model) {
  const auto& def = getMomentumExtension(model.extensionsAndExtras);
  const auto motion = def.value("motion", nlohmann::json::object());

  const size_t nframes = motion.value("nframes", 0);
  const int32_t poseBuffer = motion.value("poses", int32_t(-1));
  const int32_t offsetBuffer = motion.value("offsets", int32_t(-1));

  if (nframes == 0 || (poseBuffer < 0 && offsetBuffer < 0)) {
    return {};
  }

  // get values from gltf binary buffers
  MatrixXf storedMotion;
  VectorXf storedIdentity;
  std::vector<std::string> parameterNames;
  std::vector<std::string> jointNames;
  if (poseBuffer >= 0) {
    const auto poseValues = copyAccessorBuffer<float>(model, poseBuffer);
    parameterNames = motion.value("parameterNames", std::vector<std::string>());

    if (poseValues.size() != parameterNames.size() * nframes) {
      return {};
    }

    storedMotion = Map<const MatrixXf>(poseValues.data(), parameterNames.size(), nframes);
  }
  if (offsetBuffer >= 0) {
    const auto offsetValues = copyAccessorBuffer<float>(model, offsetBuffer);
    jointNames = motion.value("jointNames", std::vector<std::string>());

    if (offsetValues.size() != jointNames.size() * kParametersPerJoint) {
      return {};
    }

    storedIdentity =
        Map<const VectorXf>(offsetValues.data(), jointNames.size() * kParametersPerJoint);
  }

  return {{parameterNames, storedMotion}, {jointNames, storedIdentity}};
}

fx::gltf::Document loadModel(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input) {
  fx::gltf::Document model;
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  constexpr fx::gltf::ReadQuotas kQuotas = {8, kMax, kMax};

  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, filesystem::path>) {
          // Read first 4 chars for file sensing (https://docs.fileformat.com/3d/glb/)
          // It is ASCII string "glTF", and can be used to identify data as Binary glTF
          std::string magic;
          std::ifstream infile(arg);
          MT_THROW_IF(!infile.is_open(), "Unable to open: {}", arg.string());
          std::copy_n(std::istreambuf_iterator<char>(infile.rdbuf()), 4, std::back_inserter(magic));
          if (magic == "glTF") {
            model = fx::gltf::LoadFromBinary(arg.string(), kQuotas);
          } else {
            model = fx::gltf::LoadFromText(arg.string(), kQuotas);
          }
        } else if constexpr (std::is_same_v<T, gsl::span<const std::byte>>) {
          ispanstream inputStream(arg);
          model = fx::gltf::LoadFromBinary(inputStream, "", kQuotas);
        }
      },
      input);

  return model;
}

Character loadModelAndCharacter(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input) {
  Character result;
  try {
    // Set maximum filesize to 4 gigabyte (glb hard limit due to uint32).
    // Need to fork gltf load to support dynamic loading at some point if files get too big
    fx::gltf::Document model = loadModel(input);
    result = loadGltfCharacter(model);
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf : {}", err.what());
  }

  return result;
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(fx::gltf::Document& model) {
  // parse motion data
  try {
    // load motion
    const auto& def = getMomentumExtension(model.extensionsAndExtras);
    const float fps = def.value("fps", 120.0f);
    const auto& [motion, identity] = getMotionFromModel(model);
    return {motion, identity, fps};
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to parse motion data  : {}", err.what());
  }

  return {};
}

std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotionCommon(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input) {
  // ---------------------------------------------
  // load Skeleton and Mesh
  // ---------------------------------------------
  try {
    // Set maximum filesize to 4 gigabyte (glb hard limit due to uint32).
    // Need to fork gltf load to support dynamic loading at some point if files get too big
    fx::gltf::Document model = loadModel(input);
    Character character = loadGltfCharacter(model);

    // load motion
    auto [motion, identity, fps] = loadMotion(model);

    // check that there is actual motion data within the glb / gltf
    MT_LOGW_IF(std::get<1>(motion).cols() == 0, "No motion data found in gltf file");
    return {character, std::get<1>(motion), std::get<1>(identity), fps};
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf : {}", err.what());
  }

  return {};
}

std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacterCommon(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input,
    const Character& character) {
  const auto [loadedChar, motion, identity, fps] = loadCharacterWithMotionCommon(input);
  return {
      mapMotionToCharacter({loadedChar.parameterTransform.name, motion}, character),
      mapIdentityToCharacter({loadedChar.skeleton.getJointNames(), identity}, character),
      fps};
}

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    const float fps,
    gsl::span<const SkeletonState> skeletonStates,
    const std::vector<std::vector<Marker>>& markerSequence,
    bool embedResource) {
  GltfBuilder fileBuilder;
  constexpr auto kAddExtensions = true;
  constexpr auto kAddCollision = true;
  constexpr auto kAddLocators = true;

  const auto kCharacterIsEmpty = character.skeleton.joints.empty() && character.mesh == nullptr;
  if (!kCharacterIsEmpty) {
    const auto kAddMesh = character.mesh != nullptr;
    fileBuilder.addCharacter(
        character,
        Vector3f::Zero(),
        Quaternionf::Identity(),
        kAddExtensions,
        kAddCollision,
        kAddLocators,
        kAddMesh);
  }
  // Add potential motion or offsets, even if the character is empty
  // (it could be a motion database for example)
  if (!skeletonStates.empty()) {
    fileBuilder.addSkeletonStates(character, fps, skeletonStates);
  }
  if (!markerSequence.empty()) {
    MT_THROW_IF(
        !skeletonStates.empty() && (skeletonStates.size() != markerSequence.size()),
        "Size of skeleton states vector {} does not correspond to size of marker sequence vector {}",
        skeletonStates.size(),
        markerSequence.size());

    fileBuilder.addMarkerSequence(fps, markerSequence);
  }
  if (embedResource) {
    fileBuilder.forceEmbedResources();
  }

  return fileBuilder.getDocument();
}

Character loadGltfCharacterInternal(
    const fx::gltf::Document& model,
    std::vector<size_t>& nodeToObjectMap) {
  // ---------------------------------------------
  // load Skeleton and Mesh
  // ---------------------------------------------
  Character result;
  try {
    result = populateCharacterFromModel(model, nodeToObjectMap);
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf : {}", err.what());
  }

  return result;
}

// Function to load a glTF file and extract motion data stored in the native GLTF format
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStatesCommon(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input) {
  const fx::gltf::Document model = loadModel(input);
  std::vector<size_t> nodeToObjectMap;
  Character character = loadGltfCharacterInternal(model, nodeToObjectMap);

  // Iterate over each animation in the glTF document
  std::vector<SkeletonState> skeletonStates;

  if (model.animations.empty()) {
    MT_LOGW("No animation found in gltf file");
    return {character, skeletonStates, {}};
  } else if (model.animations.size() > 1) {
    MT_LOGW("Multiple animations found in gltf file. Only the first one will be used.");
  }

  const auto& animation = model.animations.front();
  std::vector<float> timestamps;

  std::vector<SkeletonState> states;

  // Iterate over each channel in the animation
  for (const auto& channel : animation.channels) {
    // Get the target node for the channel
    const auto& node = model.nodes[channel.target.node];
    const auto& sampler = animation.samplers[channel.sampler];

    if (channel.target.node >= nodeToObjectMap.size() ||
        nodeToObjectMap.at(channel.target.node) == kInvalidIndex) {
      continue;
    }

    const auto boneIdx = nodeToObjectMap[channel.target.node];
    MT_THROW_IF(
        boneIdx >= character.skeleton.joints.size(),
        "Bone index {} is larger than the number of joints in the skeleton ({})",
        boneIdx,
        character.skeleton.joints.size());

    MT_THROW_IF(
        node.name != character.skeleton.joints[boneIdx].name,
        "Bone mismatch between gltf and character; GLTF: {}, Character: {}",
        node.name,
        character.skeleton.joints[boneIdx].name);

    // Check the type of transformation (translation, rotation, scale)
    std::vector<Vector3f> translations_m;
    std::vector<Quaternionf> rotations;
    std::vector<Vector3f> scales;
    if (channel.target.path == "translation") {
      // Handle translation data
      translations_m = copyAccessorBuffer<Vector3f>(model, sampler.output);
      // Apply translations to the node or character skeleton
    } else if (channel.target.path == "rotation") {
      // Handle rotation data
      rotations = copyAccessorBuffer<Quaternionf>(model, sampler.output);
      // Apply rotations to the node or character skeleton
    } else if (channel.target.path == "scale") {
      // Handle scale data
      scales = copyAccessorBuffer<Vector3f>(model, sampler.output);
      // Apply scales to the node or character skeleton
    } else {
      continue;
    }

    auto timestampsCur = copyAccessorBuffer<float>(model, sampler.input);
    if (timestamps.empty()) {
      timestamps = timestampsCur;
    } else {
      if (timestamps.size() != timestampsCur.size()) {
        MT_THROW(
            "Timestamps do not match between channels, got {} and {}",
            timestamps.size(),
            timestampsCur.size());
      }
    }

    if (states.empty()) {
      states.resize(
          timestamps.size(),
          SkeletonState(
              JointParameters::Zero(character.parameterTransform.numJointParameters()),
              character.skeleton));
    }

    MT_THROW_IF(states.size() != timestamps.size(), "States and timestamps do not match in size");

    if (translations_m.size() == states.size()) {
      for (size_t i = 0; i < states.size(); ++i) {
        states.at(i).jointState.at(boneIdx).localTransform.translation = toCm() * translations_m[i];
      }
    }
    if (rotations.size() == states.size()) {
      for (size_t i = 0; i < states.size(); ++i) {
        states.at(i).jointState.at(boneIdx).localTransform.rotation = rotations[i];
      }
    }
    if (scales.size() == states.size()) {
      for (size_t i = 0; i < states.size(); ++i) {
        states.at(i).jointState.at(boneIdx).localTransform.scale = scales[i].x();
      }
    }
  }

  for (auto& s : states) {
    for (size_t iJoint = 0; iJoint < character.skeleton.joints.size(); ++iJoint) {
      Transform parentTransform;
      const auto parentJoint = character.skeleton.joints[iJoint].parent;
      if (parentJoint != kInvalidIndex) {
        parentTransform = s.jointState[parentJoint].transform;
      }
      s.jointState[iJoint].transform = parentTransform * s.jointState[iJoint].localTransform;
    }
  }

  return {character, states, timestamps};
}

} // namespace

namespace momentum {

Character loadGltfCharacter(const fx::gltf::Document& model) {
  std::vector<size_t> nodeToObjectMap;
  return loadGltfCharacterInternal(model, nodeToObjectMap);
}

Character loadGltfCharacter(const filesystem::path& gltfFilename) {
  return loadModelAndCharacter(gltfFilename);
}

Character loadGltfCharacter(gsl::span<const std::byte> byteSpan) {
  return loadModelAndCharacter(byteSpan);
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename) {
  // ---------------------------------------------
  // load model, parse motion data
  // ---------------------------------------------
  try {
    fx::gltf::Document model = loadModel(gltfFilename);
    return ::loadMotion(model);
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf from file '{}'. Error: {}", gltfFilename.string(), err.what());
  }

  return {};
}

std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    const filesystem::path& gltfFilename) {
  return loadCharacterWithMotionCommon(gltfFilename);
}

std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    gsl::span<const std::byte> byteSpan) {
  return loadCharacterWithMotionCommon(byteSpan);
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(gsl::span<const std::byte> byteSpan) {
  return loadCharacterWithSkeletonStatesCommon(byteSpan);
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(const filesystem::path& gltfFilename) {
  return loadCharacterWithSkeletonStatesCommon(gltfFilename);
}

std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    const filesystem::path& gltfFilename,
    const Character& character) {
  return loadMotionOnCharacterCommon(gltfFilename, character);
}

std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    const gsl::span<const std::byte> byteSpan,
    const Character& character) {
  return loadMotionOnCharacterCommon(byteSpan, character);
}

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    const float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    const std::vector<std::vector<Marker>>& markerSequence,
    bool embedResource) {
  GltfBuilder fileBuilder;
  constexpr auto kAddExtensions = true;
  constexpr auto kAddCollision = true;
  constexpr auto kAddLocators = true;

  const auto kCharacterIsEmpty = character.skeleton.joints.empty() && character.mesh == nullptr;
  if (!kCharacterIsEmpty) {
    const auto kAddMesh = character.mesh != nullptr;
    fileBuilder.addCharacter(
        character,
        Vector3f::Zero(),
        Quaternionf::Identity(),
        kAddExtensions,
        kAddCollision,
        kAddLocators,
        kAddMesh);
  }
  // Add potential motion or offsets, even if the character is empty
  // (it could be a motion database for example)
  if ((!std::get<0>(motion).empty()) || (!std::get<0>(offsets).empty())) {
    fileBuilder.addMotion(character, fps, motion, offsets, kAddExtensions);
  }
  if (!markerSequence.empty()) {
    fileBuilder.addMarkerSequence(fps, markerSequence);
  }
  if (embedResource) {
    fileBuilder.forceEmbedResources();
  }

  return fileBuilder.getDocument();
}

MarkerSequence loadMarkerSequence(const filesystem::path& filename) {
  MarkerSequence result;

  fx::gltf::Document model = loadModel(filename);
  if (model.animations.empty()) {
    return result;
  }

  const auto& def = getMomentumExtension(model.extensionsAndExtras);
  const float fps = def.value("fps", 120.0f);
  result.fps = fps;
  // TODO: result.name = model.name;

  const auto& animation = model.animations.front();

  for (const auto& channel : animation.channels) {
    // make sure we're on a translation channel
    if (channel.target.path != "translation") {
      continue;
    }

    // get the node for the channel
    const auto& node = model.nodes[channel.target.node];

    // ignore skins and cameras; don't ignore meshes because a marker node may have a mesh for viz
    if (node.skin >= 0 || node.camera >= 0) {
      continue;
    }

    const auto& extension = getMomentumExtension(node.extensionsAndExtras);
    const std::string type = extension.value("type", "");
    if (type == "marker") {
      // found a marker, get the sampler
      const auto& sampler = animation.samplers[channel.sampler];

      // get the data from the sampler
      const auto timestamps = copyAccessorBuffer<float>(model, sampler.input);
      auto positions = copyAccessorBuffer<Vector3f>(model, sampler.output);
      toMomentumVec3f(positions);
      MT_CHECK(
          timestamps.size() == positions.size(),
          "timestamps: {}, positions: {}",
          timestamps.size(),
          positions.size());

      // resize the output array if necessary
      const size_t length = static_cast<size_t>(timestamps.back() * fps + 0.5f) + 1;
      if (length > result.frames.size()) {
        result.frames.resize(length);
      }

      // go over all data and enter into the output array
      for (size_t i = 0; i < timestamps.size(); i++) {
        const size_t index = static_cast<size_t>(timestamps[i] * fps + 0.5f);
        result.frames[index].emplace_back();
        auto& marker = result.frames[index].back();

        marker.name = node.name;
        marker.occluded = false;
        marker.pos = positions[i].cast<double>();
      }
    }
  }

  return result;
}

void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    const std::vector<std::vector<Marker>>& markerSequence,
    const GltfFileFormat fileFormat) {
  constexpr auto kEmbedResources = false; // Don't embed resource for saving glb
  // create new model
  fx::gltf::Document model =
      makeCharacterDocument(character, fps, motion, offsets, markerSequence, kEmbedResources);

  GltfBuilder::save(model, filename, fileFormat, kEmbedResources);
}

void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    gsl::span<const SkeletonState> skeletonStates,
    const std::vector<std::vector<Marker>>& markerSequence,
    const GltfFileFormat fileFormat) {
  constexpr auto kEmbedResources = false; // Don't embed resource for saving glb
  // create new model
  fx::gltf::Document model =
      ::makeCharacterDocument(character, fps, skeletonStates, markerSequence, kEmbedResources);

  GltfBuilder::save(model, filename, fileFormat, kEmbedResources);
}

std::vector<std::byte> saveCharacterToBytes(
    const Character& character,
    float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    const std::vector<std::vector<Marker>>& markerSequence) {
  constexpr auto kEmbedResources = false; // Don't embed resource for saving glb
  fx::gltf::Document model =
      makeCharacterDocument(character, fps, motion, offsets, markerSequence, kEmbedResources);

  std::ostringstream output(std::ios::binary | std::ios::out);
  fx::gltf::Save(model, output, {}, true);
  const auto view = output.str();
  const auto n = view.size();
  std::vector<std::byte> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = static_cast<std::byte>(view[i]);
  }
  return result;
}

} // namespace momentum
