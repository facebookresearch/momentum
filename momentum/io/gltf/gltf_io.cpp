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

nlohmann::json getMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") != 0 &&
      extensionsAndExtras["extensions"].count("FB_momentum") != 0) {
    return extensionsAndExtras["extensions"]["FB_momentum"];
  } else {
    return nlohmann::json::object();
  }
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
    if (def.count("metadata") > 0) {
      character.metadata = def.value<std::string>("metadata", {});
      // ensure metadata is valid JSON
      try {
        auto json = nlohmann::json::parse(character.metadata);
      } catch (const nlohmann::json::parse_error& e) {
        MT_LOGW("Failed to parse metadata: {}", e.what());
      }
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

  BlendShape_u loadedBlendShape;
  std::tie(result.mesh, result.skinWeights, loadedBlendShape) =
      loadSkinnedMesh(model, remainingNodes, nodeToObjectMap);

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

  loadGlobalExtensions(model, result);

  // Finalize jointMap and inverseBind pose
  result.resetJointMap();
  result.initInverseBindPose();

  return result;
}

fx::gltf::Document loadModel(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
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
        } else if constexpr (std::is_same_v<T, std::span<const std::byte>>) {
          ispanstream inputStream(arg);
          model = fx::gltf::LoadFromBinary(inputStream, "", kQuotas);
        }
      },
      input);

  return model;
}

Character loadModelAndCharacter(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
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

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence,
    bool embedResource,
    const FileSaveOptions& options) {
  GltfBuilder fileBuilder;

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

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename) {
  // ---------------------------------------------
  // load model, parse motion data
  // ---------------------------------------------
  try {
    fx::gltf::Document model = loadModel(gltfFilename);
    return loadMotionFromModel(model);
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf from file '{}'. Error: {}", gltfFilename.string(), err.what());
  }
}

std::vector<int64_t> loadMotionTimestamps(const filesystem::path& gltfFilename) {
  try {
    fx::gltf::Document model = loadModel(gltfFilename);
    const auto& def = getMomentumExtension(model.extensionsAndExtras);

    // Check if motion section exists and has timestamps
    if (def.count("motion") == 0) {
      MT_LOGW("No motion data found in gltf file when loading timestamps");
      return {};
    }

    const auto& motion = def["motion"];
    if (!motion.contains("timestamps")) {
      MT_LOGW("No timestamps found in gltf file");
      return {};
    }

    // Extract timestamps from the JSON array
    return motion["timestamps"].get<std::vector<int64_t>>();
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

std::tuple<Character, MatrixXf, ModelParameters, float> loadCharacterWithMotionModelParameterScales(
    const filesystem::path& gltfFilename) {
  // Read the file into a buffer
  std::ifstream file(gltfFilename, std::ios::binary | std::ios::ate);
  MT_THROW_IF(!file.is_open(), "Unable to open file: {}", gltfFilename.string());

  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<std::byte> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    MT_THROW("Unable to read file: {}", gltfFilename.string());
  }

  // Delegate to the byteSpan overload to avoid code duplication
  return loadCharacterWithMotionModelParameterScales(std::span<const std::byte>(buffer));
}

std::tuple<Character, MatrixXf, ModelParameters, float> loadCharacterWithMotionModelParameterScales(
    std::span<const std::byte> byteSpan) {
  auto [character, motion, jointIdentity, fps] = loadCharacterWithMotionCommon(byteSpan, true);
  MT_THROW_IF(
      motion.rows() != character.parameterTransform.numAllModelParameters(),
      "Mismatch between character and motion; character has {} model parameters while motion has {}.",
      character.parameterTransform.numAllModelParameters(),
      motion.rows());

  auto modelIdentity =
      applyModelParameterScales(character.parameterTransform, motion, jointIdentity);
  return {character, motion, modelIdentity, fps};
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
      for (size_t i = 0; i < positions.size(); i++) {
        const auto index = static_cast<size_t>(std::lround(timestamps[i] * fps));
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
