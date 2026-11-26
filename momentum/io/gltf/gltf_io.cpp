/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/inverse_parameter_transform.h"
#include "momentum/character/joint.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/common/json_utils.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/gltf/gltf_animation_io.h"
#include "momentum/io/gltf/gltf_builder.h"
#include "momentum/io/gltf/gltf_io_internal.h"
#include "momentum/io/gltf/gltf_mesh_io.h"
#include "momentum/io/gltf/gltf_skeleton_io.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/io/gltf/utils/momentum_extension.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/mppca.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <iterator>
#include <limits>
#include <ranges>
#include <set>
#include <string_view>
#include <variant>

namespace {

using namespace momentum;

// Load the rig (parameter transform, limits, parameter sets, pose constraints and metadata) from
// the character's root node, where GltfBuilder stores each character's own copy, or else from the
// document, which holds a single copy.
void loadCharacterExtensions(
    const fx::gltf::Document& model,
    uint32_t rootNode,
    Character& character) {
  try {
    const nlohmann::json& globalDef = findMomentumExtension(model.extensionsAndExtras);
    const nlohmann::json& nodeDef = rootNode < model.nodes.size()
        ? findMomentumExtension(model.nodes[rootNode].extensionsAndExtras)
        : globalDef;
    const nlohmann::json& def = nodeDef.contains("transform") ? nodeDef : globalDef;

    rigFromJson(def, character);
    if (def.count("metadata") > 0) {
      character.metadata = metadataFromFile(def.value<std::string>("metadata", {}), "glTF");
    }
  } catch (std::runtime_error& err) {
    MT_THROW("{}", gltfLoadErrorMessage(err.what()));
  }
}

// The mesh nodes of the character whose node map is `nodeToObjectMap`: each skinned mesh whose skin
// starts at one of its joints (GltfBuilder writes the skeleton root first), and, when
// `includeUnskinnedMeshes`, the unskinned meshes, which no particular character owns.
std::vector<size_t> characterMeshNodes(
    const fx::gltf::Document& model,
    std::span<const size_t> nodeToObjectMap,
    bool includeUnskinnedMeshes) {
  std::vector<size_t> meshNodes;
  for (size_t nodeId = 0; nodeId < model.nodes.size(); nodeId++) {
    const auto& node = model.nodes[nodeId];
    if (node.mesh < 0 || nodeToObjectMap[nodeId] != kInvalidIndex) {
      continue;
    }
    if (node.skin < 0) {
      if (includeUnskinnedMeshes) {
        meshNodes.push_back(nodeId);
      }
      continue;
    }
    const auto skinId = static_cast<size_t>(node.skin);
    if (skinId >= model.skins.size() || model.skins[skinId].joints.empty()) {
      continue;
    }
    const uint32_t firstJoint = model.skins[skinId].joints.front();
    if (firstJoint < nodeToObjectMap.size() && nodeToObjectMap[firstJoint] != kInvalidIndex) {
      meshNodes.push_back(nodeId);
    }
  }
  return meshNodes;
}

// Whether two inverse bind poses are the same up to rounding: their linear parts relative to their
// size, and their translations relative to the larger of their lengths and 1 cm. Compared as one
// matrix, a difference in rotation would hide behind a large translation, although it moves the
// vertices far from the origin.
[[nodiscard]] bool sameUpToRounding(const Affine3f& a, const Affine3f& b) {
  constexpr float kRounding = 1e-5f;
  const float length = std::max({a.translation().norm(), b.translation().norm(), 1.0f});
  return a.linear().isApprox(b.linear(), kRounding) &&
      (a.translation() - b.translation()).norm() <= kRounding * length;
}

// Gives the joints of the skins of `meshNodes` the skins' inverse bind matrices, which the glTF
// format uses to skin the meshes, in momentum's joint frames: momentum joints have no scale, so
// each matrix also gets its joint's accumulated rest scale (see JointFileTransform). Other joints
// keep the inverse of their rest pose. With `onlyWhereTheyDiffer`, a joint also keeps it when the
// file's matrix is the same up to rounding (see sameUpToRounding()), so that a file momentum wrote
// loads exactly as before.
void applySkinInverseBindMatrices(
    const fx::gltf::Document& model,
    std::span<const size_t> meshNodes,
    std::span<const JointFileTransform> fileJoints,
    std::span<const size_t> nodeToObjectMap,
    bool onlyWhereTheyDiffer,
    TransformationList& inverseBindPose) {
  std::vector<bool> assigned(inverseBindPose.size(), false);
  for (const size_t meshNode : meshNodes) {
    const auto& node = model.nodes.at(meshNode);
    if (node.skin < 0 || static_cast<size_t>(node.skin) >= model.skins.size()) {
      continue;
    }
    const auto& skin = model.skins.at(node.skin);
    // Without the matrices, the glTF format uses identity matrices.
    std::vector<Matrix4f> matrices(skin.joints.size(), Matrix4f::Identity());
    if (skin.inverseBindMatrices >= 0) {
      matrices = copyAccessorBuffer<Matrix4f>(model, skin.inverseBindMatrices);
    }
    // The glTF format requires at least one matrix for each joint; the first ones are used.
    if (matrices.size() < skin.joints.size()) {
      MT_LOGW(
          "glTF skin '{}' has {} inverse bind matrices for {} joints; its joints keep the inverse of "
          "their rest pose.",
          skin.name,
          matrices.size(),
          skin.joints.size());
      continue;
    }
    for (size_t i = 0; i < skin.joints.size(); ++i) {
      const uint32_t jointNode = skin.joints[i];
      const size_t jointId =
          jointNode < nodeToObjectMap.size() ? nodeToObjectMap[jointNode] : kInvalidIndex;
      if (jointId >= fileJoints.size() || fileJoints[jointId].node != jointNode) {
        continue;
      }
      Affine3f inverseBind;
      inverseBind.matrix() = matrices.at(i);
      inverseBind.translation() *= toCm();
      inverseBind = fileJoints[jointId].post.inverse().toAffine3() * inverseBind;
      if (!assigned.at(jointId)) {
        if (!onlyWhereTheyDiffer || !sameUpToRounding(inverseBind, inverseBindPose.at(jointId))) {
          inverseBindPose.at(jointId) = inverseBind;
        }
        assigned.at(jointId) = true;
      } else if (!inverseBind.isApprox(inverseBindPose.at(jointId), 1e-4f)) {
        MT_LOGW(
            "glTF joint '{}' has different inverse bind matrices in two skins; using the first.",
            model.nodes.at(jointNode).name);
      }
    }
  }
}

// Places the joints of a document from another tool with the transforms of all nodes above them,
// as the glTF format does (see JointFileTransform), and returns these transforms. A document with
// marked joints keeps the joints as loaded.
std::vector<JointFileTransform> placeJointsAsInFile(
    const fx::gltf::Document& model,
    std::span<const size_t> nodeToObjectMap,
    Skeleton& skeleton) {
  auto fileJoints = jointFileTransforms(model, skeleton, nodeToObjectMap);
  if (hasMarkedJoints(model)) {
    return fileJoints;
  }
  for (size_t jointId = 0; jointId < fileJoints.size(); ++jointId) {
    const auto& fileJoint = fileJoints[jointId];
    if (fileJoint.node == kInvalidNodeId) {
      continue;
    }
    const Transform local =
        fileJoint.pre * nodeLocalTransform(model.nodes.at(fileJoint.node)) * fileJoint.post;
    skeleton.joints.at(jointId).translationOffset = local.translation;
    skeleton.joints.at(jointId).preRotation = local.rotation;
  }
  return fileJoints;
}

// Build the character of the object whose root node is `objectRoot` (see gatherObjectRoots()): a
// joint for each node of its hierarchy, and the meshes of its nodes, each bound rigidly to the
// joint of its node.
CharacterWithNodeMap buildObjectCharacter(const fx::gltf::Document& model, uint32_t objectRoot) {
  CharacterWithNodeMap entry;
  entry.rootNode = objectRoot;
  entry.isObject = true;
  Character& result = entry.character;
  const std::array<uint32_t, 1> roots = {objectRoot};
  std::tie(
      result.name,
      result.skeleton.joints,
      result.collision,
      result.locators,
      result.physicalProperties,
      entry.nodeToObjectMap) =
      loadHierarchyForRoots(model, roots, objectRoot, /*meshNodesAreJoints=*/true);
  std::vector<size_t> meshNodes;
  for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId) {
    const auto& node = model.nodes[nodeId];
    // A node with a momentum type, such as a locator, maps to an index of its type, not to a joint.
    if (node.mesh >= 0 && node.skin < 0 && !hasMomentumType(node) &&
        entry.nodeToObjectMap.at(nodeId) < result.skeleton.joints.size()) {
      meshNodes.push_back(nodeId);
    }
  }
  BlendShape_u blendShape;
  std::tie(result.mesh, result.skinWeights, blendShape) =
      loadRigidMeshes(model, meshNodes, entry.nodeToObjectMap);
  if (blendShape != nullptr) {
    result.blendShape = std::move(blendShape);
  }
  // An object has no rig of its own: the rig of the document belongs to its skinned characters.
  result.parameterTransform =
      ParameterTransform::empty(result.skeleton.joints.size() * kParametersPerJoint);
  const auto fileJoints = placeJointsAsInFile(model, entry.nodeToObjectMap, result.skeleton);
  result.resetJointMap();
  result.initInverseBindPose();
  // A mesh without a skin lies in its node's frame, with the node's accumulated scale.
  for (size_t jointId = 0; jointId < fileJoints.size(); ++jointId) {
    result.inverseBindPose.at(jointId) = fileJoints[jointId].post.inverse().toAffine3();
  }
  return entry;
}

// Warns about the meshes without a skin of a document from another tool that no character loads:
// those inside a skeleton, and those of an object that cannot be read.
void warnAboutUnloadedMeshes(
    const fx::gltf::Document& model,
    std::span<const CharacterWithNodeMap> characters) {
  std::string names;
  size_t count = 0;
  for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId) {
    const auto& node = model.nodes[nodeId];
    // A node with a momentum type, such as a marker with a cube mesh, is read as that type.
    if (node.mesh < 0 || node.skin >= 0 || hasMomentumType(node)) {
      continue;
    }
    const bool loaded =
        std::ranges::any_of(characters, [nodeId](const CharacterWithNodeMap& entry) {
          return nodeId < entry.nodeToObjectMap.size() &&
              entry.nodeToObjectMap[nodeId] != kInvalidIndex;
        });
    if (!loaded) {
      names += (count++ == 0 ? "'" : ", '") + node.name + "'";
    }
  }
  MT_LOGW_IF(
      count > 0, "glTF file has {} meshes without a skin that are not loaded: {}", count, names);
}

// The name of the skeleton that starts at `skeletonRoots`: the name of a skin that has its root as
// a joint, or else the name of its root node.
std::string skeletonName(const fx::gltf::Document& model, std::span<const uint32_t> skeletonRoots) {
  if (skeletonRoots.empty()) {
    return {};
  }
  const uint32_t root = skeletonRoots.front();
  for (const auto& skin : model.skins) {
    if (!skin.name.empty() && std::ranges::find(skin.joints, root) != skin.joints.end()) {
      return skin.name;
    }
  }
  return root < model.nodes.size() ? model.nodes[root].name : std::string();
}

// In a document from another tool, a top-level node can hold several characters, such as two hands
// and an object below one world node. A skinned character there is named after its skeleton (see
// skeletonName()) instead of after the top-level node.
void nameCharactersThatShareATopLevelNode(
    const fx::gltf::Document& model,
    std::span<const uint32_t> sceneParentMap,
    std::span<const std::vector<uint32_t>> skeletonRoots,
    std::vector<CharacterWithNodeMap>& characters) {
  // The skinned characters come first, one per skeleton, then the objects that loaded, each named
  // after its own root node.
  std::vector<uint32_t> topLevelNodes;
  topLevelNodes.reserve(characters.size());
  std::ranges::transform(
      characters | std::views::take(skeletonRoots.size()),
      std::back_inserter(topLevelNodes),
      &CharacterWithNodeMap::rootNode);
  std::ranges::transform(
      characters | std::views::drop(skeletonRoots.size()),
      std::back_inserter(topLevelNodes),
      [&](const CharacterWithNodeMap& object) {
        return getTopLevelAncestor(object.rootNode, sceneParentMap);
      });
  size_t index = 0;
  for (const auto& roots : skeletonRoots) {
    const uint32_t topLevelNode = topLevelNodes.at(index);
    if (topLevelNode != kInvalidNodeId && std::ranges::count(topLevelNodes, topLevelNode) > 1) {
      // A skeleton without a name keeps the name of the top-level node.
      std::string name = skeletonName(model, roots);
      if (!name.empty()) {
        characters.at(index).character.name = std::move(name);
      }
    }
    ++index;
  }
}

// The functions below that list the parts of a document that momentum does not read are called
// only in log macros, which do not evaluate their arguments in builds without logging, so these
// builds, such as production builds, do not include them.

// Throws when the document requires an extension whose data momentum cannot read.
void checkRequiredExtensions(const fx::gltf::Document& model) {
  constexpr std::array<std::string_view, 2> kCompressedMeshes = {
      "KHR_draco_mesh_compression", "EXT_meshopt_compression"};
  for (const std::string& extension : model.extensionsRequired) {
    MT_THROW_IF(
        std::ranges::find(kCompressedMeshes, extension) != kCompressedMeshes.end(),
        "The glTF file requires {}, and momentum cannot read compressed meshes; save the file without mesh compression.",
        extension);
  }
}

// `names`, separated by commas.
template <typename Names>
std::string joinNames(const Names& names) {
  std::string result;
  for (const auto& name : names) {
    if (!result.empty()) {
      result += ", ";
    }
    result += name;
  }
  return result;
}

// The required extensions of the document that momentum does not read, separated by commas.
[[maybe_unused]] std::string unreadRequiredExtensions(const fx::gltf::Document& model) {
  // Momentum reads the quantized attributes of KHR_mesh_quantization as other accessors.
  constexpr std::array<std::string_view, 2> kRead = {"FB_momentum", "KHR_mesh_quantization"};
  std::vector<std::string_view> notRead;
  std::ranges::copy_if(
      model.extensionsRequired, std::back_inserter(notRead), [&](const std::string& extension) {
        return std::ranges::find(kRead, extension) == kRead.end();
      });
  return joinNames(notRead);
}

// The vertex attributes of the document that momentum does not read, separated by commas: other
// sets of texture coordinates and colors than the first, more than eight joint influences, and the
// normals and tangents of morph targets.
[[maybe_unused]] std::string unreadAttributes(const fx::gltf::Document& model) {
  std::set<std::string> notRead;
  // The set index of attribute `name` with `prefix`, such as 1 for TEXCOORD_1, or -1.
  const auto setIndex = [](std::string_view name, std::string_view prefix) {
    int index = -1;
    if (name.starts_with(prefix)) {
      std::from_chars(name.data() + prefix.size(), name.data() + name.size(), index);
    }
    return index;
  };
  for (const auto& mesh : model.meshes) {
    for (const auto& primitive : mesh.primitives) {
      for (const auto& [name, accessor] : primitive.attributes) {
        if (setIndex(name, "TEXCOORD_") > 0 || setIndex(name, "COLOR_") > 0 ||
            setIndex(name, "JOINTS_") > 1 || setIndex(name, "WEIGHTS_") > 1) {
          notRead.insert(name);
        }
      }
      for (const auto& target : primitive.targets) {
        for (const auto& [name, accessor] : target) {
          if (name != "POSITION") {
            notRead.insert("morph target " + name);
          }
        }
      }
    }
  }
  return joinNames(notRead);
}

// The mesh nodes of the document whose skin momentum cannot read, separated by commas: nodes with a
// skin index outside the skins, or with a skin without joints. No character loads their meshes.
[[maybe_unused]] std::string meshesWithUnreadableSkins(const fx::gltf::Document& model) {
  std::vector<std::string> names;
  for (const auto& node : model.nodes) {
    if (node.mesh >= 0 && node.skin >= 0 &&
        (static_cast<size_t>(node.skin) >= model.skins.size() ||
         model.skins.at(node.skin).joints.empty())) {
      names.push_back("'" + node.name + "'");
    }
  }
  return joinNames(names);
}

// The animation channels of a document from another tool that no character reads, separated by
// commas: the channels of nodes that are not joints of a loaded character, such as nodes above a
// skeleton, and the channels of morph target weights. Empty for a document with marked joints.
[[maybe_unused]] std::string unreadChannels(
    const fx::gltf::Document& model,
    std::span<const CharacterWithNodeMap> characters) {
  if (hasMarkedJoints(model)) {
    return {};
  }
  std::set<std::string> notRead;
  for (const auto& animation : model.animations) {
    for (const auto& channel : animation.channels) {
      // A channel without a target node targets what an extension defines.
      const bool hasNode =
          channel.target.node >= 0 && static_cast<size_t>(channel.target.node) < model.nodes.size();
      const auto node = hasNode ? static_cast<size_t>(channel.target.node) : 0;
      const bool read = hasNode && channel.target.path != "weights" &&
          std::ranges::any_of(characters, [node](const CharacterWithNodeMap& entry) {
                          return node < entry.nodeToObjectMap.size() &&
                              entry.nodeToObjectMap[node] != kInvalidIndex;
                        });
      if (!read) {
        const std::string target =
            hasNode ? "'" + model.nodes.at(node).name + "'" : std::string("(no node)");
        notRead.insert(target + " " + channel.target.path);
      }
    }
  }
  return joinNames(notRead);
}

// Build the character whose skeleton starts at `skeletonRoots`, named after `rootNode`.
Character buildCharacter(
    const fx::gltf::Document& model,
    std::span<const uint32_t> skeletonRoots,
    uint32_t rootNode,
    bool includeUnskinnedMeshes,
    std::vector<size_t>& nodeToObjectMap) {
  Character result;

  // ---------------------------------------------
  // load the joints, collision geometry and locators
  // ---------------------------------------------
  std::tie(
      result.name,
      result.skeleton.joints,
      result.collision,
      result.locators,
      result.physicalProperties,
      nodeToObjectMap) = loadHierarchyForRoots(model, skeletonRoots, rootNode);
  MT_CHECK(
      nodeToObjectMap.size() == model.nodes.size(),
      "nodeMap: {}, nodes: {}",
      nodeToObjectMap.size(),
      model.nodes.size());

  // ---------------------------------------------
  // load the mesh -- combine all meshes of this character into one
  // ---------------------------------------------
  BlendShape_u loadedBlendShape;
  const auto meshNodes = characterMeshNodes(model, nodeToObjectMap, includeUnskinnedMeshes);
  std::tie(result.mesh, result.skinWeights, loadedBlendShape) =
      loadSkinnedMesh(model, meshNodes, nodeToObjectMap);

  // Set the blendShape if we loaded any
  if (loadedBlendShape != nullptr) {
    result.blendShape = std::move(loadedBlendShape);
  }

  // Check if there's no skeleton and no skinning nodes found
  // but there are meshes in the scene. We will create a joint to parent
  // them.
  if (result.skeleton.joints.empty() && result.mesh != nullptr && result.skinWeights == nullptr) {
    result.skeleton.joints = createSkeletonWithOnlyRoot();
    for (auto& locator : result.locators) {
      locator.parent = 0;
    }
    result.skinWeights = bindMeshToJoint(result.mesh, 0);
  }

  loadCharacterExtensions(model, rootNode, result);

  // A document from another tool places its joints with the transforms of all nodes above them.
  // Every document skins its meshes with its inverse bind matrices, as the glTF format defines.
  const bool useFileTransforms = !hasMarkedJoints(model);
  const std::vector<JointFileTransform> fileJoints =
      placeJointsAsInFile(model, nodeToObjectMap, result.skeleton);

  // Finalize jointMap and inverseBind pose
  result.resetJointMap();
  result.initInverseBindPose();
  // The inverse bind matrices of a file momentum wrote are the inverse of its rest pose, unless the
  // character was bound in another pose.
  applySkinInverseBindMatrices(
      model,
      meshNodes,
      fileJoints,
      nodeToObjectMap,
      /*onlyWhereTheyDiffer=*/!useFileTransforms,
      result.inverseBindPose);

  return result;
}

Character loadModelAndCharacter(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  return loadFromInput(
      input, [](const fx::gltf::Document& model) { return loadGltfCharacter(model); });
}

std::vector<Character> loadModelAndCharacters(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  return loadFromInput(
      input, [](const fx::gltf::Document& model) { return loadGltfCharacters(model); });
}

std::tuple<std::vector<CharacterMotion>, float> loadModelAndCharactersWithMotion(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  return loadFromInput(
      input, [](const fx::gltf::Document& model) { return loadGltfCharactersWithMotion(model); });
}

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence,
    bool embedResource,
    const FileSaveOptions& options) {
  GltfBuilder fileBuilder;
  fileBuilder.setAnimationInterpolation(options.gltfAnimationInterpolation);

  const auto kCharacterIsEmpty = character.skeleton.joints.empty() && character.mesh == nullptr;
  if (!kCharacterIsEmpty) {
    fileBuilder.addCharacter(character, Vector3f::Zero(), Quaternionf::Identity(), options);
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

} // namespace

namespace momentum {

fx::gltf::Document loadGltfDocument(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  fx::gltf::Document model;
  // Files up to 4 GB, the glb limit (its sizes are uint32).
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  constexpr fx::gltf::ReadQuotas kQuotas = {
      .MaxBufferCount = 8, .MaxFileSize = kMax, .MaxBufferByteLength = kMax};

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
        } else if constexpr (std::is_same_v<T, std::span<const std::byte>>) {
          ispanstream inputStream(arg);
          model = fx::gltf::LoadFromBinary(inputStream, "", kQuotas);
        }
      },
      input);

  return model;
}

std::string gltfLoadErrorMessage(std::string_view what) {
  constexpr std::string_view kContext = "Unable to load gltf : ";
  return what.starts_with(kContext) ? std::string(what) : std::string(kContext) + std::string(what);
}

namespace {

// The momentum "motion" object in `extensionsAndExtras`, or nullptr.
const nlohmann::json* findMotion(const nlohmann::json& extensionsAndExtras) {
  const auto& def = findMomentumExtension(extensionsAndExtras);
  const auto motion = def.find("motion");
  return motion == def.end() ? nullptr : &*motion;
}

} // namespace

const nlohmann::json&
getCharacterMotionJson(const fx::gltf::Document& model, uint32_t rootNode, bool isOnlyCharacter) {
  static const nlohmann::json kNoMotion = nlohmann::json::object();
  if (rootNode < model.nodes.size()) {
    if (const auto* motion = findMotion(model.nodes[rootNode].extensionsAndExtras)) {
      return *motion;
    }
  }
  if (!isOnlyCharacter || std::ranges::any_of(model.nodes, [](const fx::gltf::Node& node) {
        return findMotion(node.extensionsAndExtras) != nullptr;
      })) {
    return kNoMotion;
  }
  const auto* motion = findMotion(model.extensionsAndExtras);
  return motion != nullptr ? *motion : kNoMotion;
}

std::vector<CharacterWithNodeMap> loadGltfCharactersInternal(
    const fx::gltf::Document& model,
    GltfObjects objects) {
  try {
    checkRequiredExtensions(model);
    MT_LOGW_IF(
        model.scenes.size() > 1,
        "glTF file has {} scenes; momentum reads only scene {}.",
        model.scenes.size(),
        sceneToLoad(model));
    MT_LOGW_IF(
        !unreadAttributes(model).empty(),
        "glTF file has vertex attributes that momentum does not read: {}.",
        unreadAttributes(model));
    MT_LOGW_IF(
        !meshesWithUnreadableSkins(model).empty(),
        "glTF file has meshes whose skin momentum cannot read, which are not loaded: {}.",
        meshesWithUnreadableSkins(model));
    MT_LOGW_IF(
        !unreadRequiredExtensions(model).empty(),
        "glTF file requires extensions that momentum does not read: {}. Momentum reads the skeletons, meshes and animation without them.",
        unreadRequiredExtensions(model));
    const auto sceneParentMap = buildSceneParentMap(model);
    const auto skeletonRoots = gatherSkeletonRoots(model, sceneParentMap);
    // In a document from another tool, the meshes without a skin outside the skeletons are objects,
    // each a character of its own, unless the caller skips objects.
    const bool fromAnotherTool = !hasMarkedJoints(model) && !model.skins.empty();
    const bool loadObjects = fromAnotherTool && objects == GltfObjects::Load;
    const std::vector<uint32_t> objectRoots = loadObjects
        ? gatherObjectRoots(model, sceneParentMap, skeletonRoots)
        : std::vector<uint32_t>();
    std::vector<CharacterWithNodeMap> characters;
    characters.reserve(skeletonRoots.size() + objectRoots.size());
    for (const auto& roots : skeletonRoots) {
      auto& entry = characters.emplace_back();
      entry.rootNode =
          roots.empty() ? kInvalidNodeId : getTopLevelAncestor(roots.front(), sceneParentMap);
      entry.character = buildCharacter(
          model,
          roots,
          entry.rootNode,
          skeletonRoots.size() == 1 && !loadObjects,
          entry.nodeToObjectMap);
    }
    // A file whose object cannot be read still loads its other characters.
    for (const uint32_t objectRoot : objectRoots) {
      try {
        characters.push_back(buildObjectCharacter(model, objectRoot));
      } catch (const std::exception& e) {
        MT_LOGW("glTF object '{}' is not loaded: {}", model.nodes.at(objectRoot).name, e.what());
      }
    }
    if (fromAnotherTool) {
      nameCharactersThatShareATopLevelNode(model, sceneParentMap, skeletonRoots, characters);
    }
    if (loadObjects) {
      warnAboutUnloadedMeshes(model, characters);
    }
    return characters;
  } catch (std::runtime_error& err) {
    MT_THROW("{}", gltfLoadErrorMessage(err.what()));
  }
}

CharacterWithNodeMap& selectGltfCharacter(std::vector<CharacterWithNodeMap>& characters) {
  MT_THROW_IF(characters.empty(), "Unable to load gltf : no character found in the file.");
  MT_LOGW_IF(
      characters.size() > 1,
      "glTF file contains {} characters; loading the one with the most joints. Use "
      "loadGltfCharacters() to load all of them.",
      characters.size());
  return *std::ranges::max_element(characters, {}, [](const CharacterWithNodeMap& entry) {
    return entry.character.skeleton.joints.size();
  });
}

Character loadGltfCharacterInternal(
    const fx::gltf::Document& model,
    std::vector<size_t>& nodeToObjectMap) {
  auto characters = loadGltfCharactersInternal(model, GltfObjects::Skip);
  auto& loaded = selectGltfCharacter(characters);
  nodeToObjectMap = std::move(loaded.nodeToObjectMap);
  return std::move(loaded.character);
}

Character loadGltfCharacter(const fx::gltf::Document& model) {
  std::vector<size_t> nodeToObjectMap;
  return loadGltfCharacterInternal(model, nodeToObjectMap);
}

Character loadGltfCharacter(const filesystem::path& gltfFilename) {
  return loadModelAndCharacter(gltfFilename);
}

Character loadGltfCharacter(std::span<const std::byte> byteSpan) {
  return loadModelAndCharacter(byteSpan);
}

std::vector<Character> loadGltfCharacters(const fx::gltf::Document& model) {
  auto entries = loadGltfCharactersInternal(model, GltfObjects::Load);
  std::vector<Character> characters;
  characters.reserve(entries.size());
  std::ranges::transform(entries, std::back_inserter(characters), [](CharacterWithNodeMap& entry) {
    return std::move(entry.character);
  });
  return characters;
}

std::vector<Character> loadGltfCharacters(const filesystem::path& gltfFilename) {
  return loadModelAndCharacters(gltfFilename);
}

std::vector<Character> loadGltfCharacters(std::span<const std::byte> byteSpan) {
  return loadModelAndCharacters(byteSpan);
}

std::tuple<std::vector<CharacterMotion>, float> loadGltfCharactersWithMotion(
    const fx::gltf::Document& model) {
  auto characters = loadGltfCharactersInternal(model, GltfObjects::Load);
  // A motion stored only at the document level belongs to the only skinned character; objects have
  // no motion of their own.
  const auto numSkinned = std::ranges::count(characters, false, &CharacterWithNodeMap::isObject);
  std::vector<CharacterMotion> result;
  result.reserve(characters.size());
  for (auto& entry : characters) {
    const auto& motionJson =
        getCharacterMotionJson(model, entry.rootNode, !entry.isObject && numSkinned == 1);
    const auto [motion, identity] = getMotionFromJson(model, motionJson);
    CharacterMotion characterMotion;
    characterMotion.motion = mapMotionToCharacter(motion, entry.character);
    if (std::get<1>(identity).size() != 0) {
      characterMotion.identity = mapIdentityToCharacter(identity, entry.character);
    }
    characterMotion.timestamps = motionJson.value("timestamps", std::vector<int64_t>{});
    if (!characterMotion.timestamps.empty() &&
        characterMotion.timestamps.size() != static_cast<size_t>(characterMotion.motion.cols())) {
      MT_LOGW(
          "glTF character '{}' has {} timestamps for {} motion frames; ignoring the timestamps.",
          entry.character.name,
          characterMotion.timestamps.size(),
          characterMotion.motion.cols());
      characterMotion.timestamps.clear();
    }
    characterMotion.character = std::move(entry.character);
    result.push_back(std::move(characterMotion));
  }
  return {std::move(result), loadFpsFromModel(model)};
}

std::tuple<std::vector<CharacterMotion>, float> loadGltfCharactersWithMotion(
    const filesystem::path& gltfFilename) {
  return loadModelAndCharactersWithMotion(gltfFilename);
}

std::tuple<std::vector<CharacterMotion>, float> loadGltfCharactersWithMotion(
    std::span<const std::byte> byteSpan) {
  return loadModelAndCharactersWithMotion(byteSpan);
}

namespace {

// The skeleton states of a loaded character from the first animation that animates it, with their
// frame times; both are empty when no animation animates it. An object whose animation cannot be
// read loads without it, with a warning, so that it does not stop the load of the other characters.
std::tuple<std::vector<SkeletonState>, std::vector<float>> loadCharacterAnimation(
    const fx::gltf::Document& model,
    const CharacterWithNodeMap& entry) {
  const auto load = [&]() -> std::tuple<std::vector<SkeletonState>, std::vector<float>> {
    const auto* animation =
        findAnimationForCharacter(model, entry.character, entry.nodeToObjectMap);
    if (animation == nullptr) {
      return {};
    }
    return loadSkeletonStatesForCharacter(
        model, *animation, entry.character, entry.nodeToObjectMap);
  };
  if (!entry.isObject) {
    return load();
  }
  try {
    return load();
  } catch (const std::exception& e) {
    MT_LOGW("glTF object '{}' is loaded without its animation: {}", entry.character.name, e.what());
    return {};
  }
}

} // namespace

std::vector<CharacterSkeletonStates> loadGltfCharactersWithSkeletonStates(
    const fx::gltf::Document& model) {
  auto characters = loadGltfCharactersInternal(model, GltfObjects::Load);
  MT_LOGW_IF(
      !unreadChannels(model, characters).empty(),
      "glTF file has animation channels that momentum does not read: {}.",
      unreadChannels(model, characters));
  std::vector<CharacterSkeletonStates> result;
  result.reserve(characters.size());
  for (auto& entry : characters) {
    CharacterSkeletonStates loaded;
    std::tie(loaded.skeletonStates, loaded.timestamps) = loadCharacterAnimation(model, entry);
    loaded.character = std::move(entry.character);
    result.push_back(std::move(loaded));
  }
  return result;
}

std::vector<CharacterSkeletonStates> loadGltfCharactersWithSkeletonStates(
    const filesystem::path& gltfFilename) {
  return loadFromInput(gltfFilename, [](const fx::gltf::Document& model) {
    return loadGltfCharactersWithSkeletonStates(model);
  });
}

std::vector<CharacterSkeletonStates> loadGltfCharactersWithSkeletonStates(
    std::span<const std::byte> byteSpan) {
  return loadFromInput(byteSpan, [](const fx::gltf::Document& model) {
    return loadGltfCharactersWithSkeletonStates(model);
  });
}

std::vector<MatrixXf> loadGltfCharactersChannelMotion(const fx::gltf::Document& model) {
  std::vector<MatrixXf> motions;
  const auto characters = loadGltfCharactersInternal(model, GltfObjects::Load);
  MT_LOGW_IF(
      !unreadChannels(model, characters).empty(),
      "glTF file has animation channels that momentum does not read: {}.",
      unreadChannels(model, characters));
  for (const auto& entry : characters) {
    // One character's skeleton states at a time, so they are freed before the next character's.
    const auto [states, timestamps] = loadCharacterAnimation(model, entry);
    const Character& character = entry.character;
    MatrixXf& motion = motions.emplace_back(
        character.parameterTransform.numJointParameters(),
        static_cast<Eigen::Index>(states.size()));
    for (size_t frame = 0; frame < states.size(); ++frame) {
      // The loader sets each joint's local transform from the channels, so this reads them as they
      // are rather than recomputing them from the world transforms.
      motion.col(static_cast<Eigen::Index>(frame)) =
          skeletonStateToJointParameters(states[frame], character.skeleton).v;
    }
  }
  return motions;
}

std::vector<MatrixXf> loadGltfCharactersChannelMotion(const filesystem::path& gltfFilename) {
  return loadFromInput(gltfFilename, [](const fx::gltf::Document& model) {
    return loadGltfCharactersChannelMotion(model);
  });
}

std::vector<MatrixXf> loadGltfCharactersChannelMotion(std::span<const std::byte> byteSpan) {
  return loadFromInput(byteSpan, [](const fx::gltf::Document& model) {
    return loadGltfCharactersChannelMotion(model);
  });
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename) {
  // ---------------------------------------------
  // load model, parse motion data
  // ---------------------------------------------
  try {
    fx::gltf::Document model = loadGltfDocument(gltfFilename);
    return loadMotionFromModel(model);
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf from file '{}'. Error: {}", gltfFilename.string(), err.what());
  }
}

std::vector<int64_t> loadMotionTimestamps(const filesystem::path& gltfFilename) {
  try {
    fx::gltf::Document model = loadGltfDocument(gltfFilename);
    const auto* motion = findMotion(model.extensionsAndExtras);

    // Check if motion section exists and has timestamps
    if (motion == nullptr) {
      MT_LOGW("No motion data found in gltf file when loading timestamps");
      return {};
    }
    if (!motion->contains("timestamps")) {
      MT_LOGW("No timestamps found in gltf file");
      return {};
    }

    // Extract timestamps from the JSON array
    return motion->at("timestamps").get<std::vector<int64_t>>();
  } catch (const std::exception& err) {
    MT_THROW(
        "Unable to load timestamps from gltf file '{}'. Error: {}",
        gltfFilename.string(),
        err.what());
  }
}

std::tuple<Character, MatrixXf, JointParameters, float> loadCharacterWithMotion(
    const filesystem::path& gltfFilename) {
  return loadCharacterWithMotionCommon(gltfFilename);
}

std::tuple<Character, MatrixXf, JointParameters, float> loadCharacterWithMotion(
    std::span<const std::byte> byteSpan) {
  return loadCharacterWithMotionCommon(byteSpan);
}

namespace {

std::tuple<Character, MatrixXf, ModelParameters, float>
loadCharacterWithMotionModelParameterScalesCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  auto [character, motion, jointIdentity, fps] = loadCharacterWithMotionCommon(input, true);
  MT_THROW_IF(
      motion.rows() != character.parameterTransform.numAllModelParameters(),
      "Mismatch between character and motion; character has {} model parameters while motion has {}.",
      character.parameterTransform.numAllModelParameters(),
      motion.rows());

  auto modelIdentity =
      applyModelParameterScales(character.parameterTransform, motion, jointIdentity);
  return {character, motion, modelIdentity, fps};
}

} // namespace

std::tuple<Character, MatrixXf, ModelParameters, float> loadCharacterWithMotionModelParameterScales(
    const filesystem::path& gltfFilename) {
  return loadCharacterWithMotionModelParameterScalesCommon(gltfFilename);
}

std::tuple<Character, MatrixXf, ModelParameters, float> loadCharacterWithMotionModelParameterScales(
    std::span<const std::byte> byteSpan) {
  return loadCharacterWithMotionModelParameterScalesCommon(byteSpan);
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(std::span<const std::byte> byteSpan) {
  return loadCharacterWithSkeletonStatesCommon(byteSpan);
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(const filesystem::path& gltfFilename) {
  return loadCharacterWithSkeletonStatesCommon(gltfFilename);
}

std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacter(
    const filesystem::path& gltfFilename,
    const Character& character) {
  return loadMotionOnCharacterCommon(gltfFilename, character);
}

std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacter(
    const std::span<const std::byte> byteSpan,
    const Character& character) {
  return loadMotionOnCharacterCommon(byteSpan, character);
}

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    const float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    std::span<const std::vector<Marker>> markerSequence,
    bool embedResource,
    const FileSaveOptions& options,
    std::span<const int64_t> timestamps) {
  GltfBuilder fileBuilder;
  fileBuilder.setAnimationInterpolation(options.gltfAnimationInterpolation);
  const auto kCharacterIsEmpty = character.skeleton.joints.empty() && character.mesh == nullptr;
  if (!kCharacterIsEmpty) {
    fileBuilder.addCharacter(character, Vector3f::Zero(), Quaternionf::Identity(), options);
  }
  // Add potential motion or offsets, even if the character is empty
  // (it could be a motion database for example)
  if ((!std::get<0>(motion).empty()) || (!std::get<0>(offsets).empty())) {
    fileBuilder.addMotion(
        character, fps, motion, offsets, options.extensions, "default", timestamps);
  }
  if (!markerSequence.empty()) {
    fileBuilder.addMarkerSequence(fps, markerSequence);
  }
  if (embedResource) {
    fileBuilder.forceEmbedResources();
  }

  return fileBuilder.getDocument();
}

namespace {

MarkerSequence loadMarkerSequenceFromModel(fx::gltf::Document model) {
  MarkerSequence result;

  if (model.animations.empty()) {
    return result;
  }

  const auto& def = findMomentumExtension(model.extensionsAndExtras);
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
    if (channel.target.node < 0 || static_cast<size_t>(channel.target.node) >= model.nodes.size()) {
      continue;
    }
    const auto& node = model.nodes.at(channel.target.node);

    // ignore skins and cameras; don't ignore meshes because a marker node may have a mesh for viz
    if (node.skin >= 0 || node.camera >= 0) {
      continue;
    }

    const auto& extension = findMomentumExtension(node.extensionsAndExtras);
    const std::string type = extension.value("type", "");
    if (type == "marker") {
      // found a marker, get the sampler
      const auto& sampler = animation.samplers[channel.sampler];

      // get the data from the sampler
      const auto timestamps = copyAccessorBuffer<float>(model, sampler.input);
      auto positions = copyAccessorBuffer<Vector3f>(model, sampler.output);
      toMomentumVec3f(positions);
      if (positions.empty()) {
        continue;
      }
      MT_CHECK(
          timestamps.size() == positions.size(),
          "timestamps: {}, positions: {}",
          timestamps.size(),
          positions.size());

      // resize the output array if necessary
      const size_t length = static_cast<size_t>(std::lround(timestamps.back() * fps)) + 1;
      if (length > result.frames.size()) {
        result.frames.resize(length);
      }

      // go over all data and enter into the output array
      const auto& confArray =
          extension.count("confidence") != 0 ? extension["confidence"] : nlohmann::json();
      for (size_t i = 0; i < positions.size(); i++) {
        const auto index = static_cast<size_t>(std::lround(timestamps[i] * fps));
        result.frames[index].emplace_back();
        auto& marker = result.frames[index].back();

        marker.name = node.name;
        marker.occluded = false;
        marker.pos = positions[i].cast<double>();
        if (confArray.is_array() && i < confArray.size()) {
          marker.confidence = confArray[i].get<float>();
        }
      }
    }
  }

  return result;
}

} // namespace

MarkerSequence loadMarkerSequence(const filesystem::path& filename) {
  return loadMarkerSequenceFromModel(loadGltfDocument(filename));
}

MarkerSequence loadMarkerSequence(std::span<const std::byte> byteSpan) {
  return loadMarkerSequenceFromModel(loadGltfDocument(byteSpan));
}

void saveGltfCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options,
    std::span<const int64_t> timestamps) {
  constexpr auto kEmbedResources = false;
  fx::gltf::Document model = makeCharacterDocument(
      character, fps, motion, offsets, markerSequence, kEmbedResources, options, timestamps);

  GltfBuilder::save(model, filename, options.gltfFileFormat, kEmbedResources);
}

void saveGltfCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  constexpr auto kEmbedResources = false;
  fx::gltf::Document model = ::makeCharacterDocument(
      character, fps, skeletonStates, markerSequence, kEmbedResources, options);

  GltfBuilder::save(model, filename, options.gltfFileFormat, kEmbedResources);
}

std::vector<std::byte> saveCharacterToBytes(
    const Character& character,
    float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options,
    std::span<const int64_t> timestamps) {
  constexpr auto kEmbedResources = false;
  fx::gltf::Document model = makeCharacterDocument(
      character, fps, motion, offsets, markerSequence, kEmbedResources, options, timestamps);

  std::ostringstream output(std::ios::binary | std::ios::out);
  ensureDocumentHasBuffer(model);
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
