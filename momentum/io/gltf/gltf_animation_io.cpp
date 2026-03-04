/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_animation_io.h"

#include "momentum/character/character_state.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/gltf/gltf_io_internal.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/math/constants.h"

#include <algorithm>
#include <fstream>
#include <limits>

namespace momentum {

namespace {

nlohmann::json getMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") != 0 &&
      extensionsAndExtras["extensions"].count("FB_momentum") != 0) {
    return extensionsAndExtras["extensions"]["FB_momentum"];
  } else {
    return nlohmann::json::object();
  }
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

} // namespace

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

float calculateFpsFromAnimationData(const fx::gltf::Document& model) {
  if (model.animations.empty()) {
    return 0.0f;
  }

  // Collect FPS values from each sampler
  std::vector<float> fpsValues;
  fpsValues.reserve(32);

  // Iterate through all animations and their samplers
  for (const auto& animation : model.animations) {
    for (const auto& sampler : animation.samplers) {
      const int32_t inputAccessorIndex = sampler.input;
      if (inputAccessorIndex < 0 ||
          inputAccessorIndex >= static_cast<int32_t>(model.accessors.size())) {
        continue;
      }

      const auto& accessor = model.accessors[inputAccessorIndex];
      if (accessor.count < 2) {
        continue;
      }

      // Get time range from accessor min/max
      if (accessor.min.empty() || accessor.max.empty()) {
        continue;
      }

      const float minTime = accessor.min[0];
      const float maxTime = accessor.max[0];
      const float duration = maxTime - minTime;

      if (duration <= 0.0f) {
        continue;
      }

      // Calculate FPS for this sampler: (keyframes - 1) / duration
      const float fps = static_cast<float>(accessor.count - 1) / duration;
      fpsValues.push_back(fps);
    }
  }

  if (fpsValues.empty()) {
    return 0.0f;
  }

  // Use median FPS to be robust against outliers
  std::sort(fpsValues.begin(), fpsValues.end());
  const float medianFps = fpsValues[fpsValues.size() / 2];
  return medianFps;
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotionFromModel(
    fx::gltf::Document& model) {
  // parse motion data
  try {
    // load motion
    const auto& def = getMomentumExtension(model.extensionsAndExtras);
    float fps = def.value("fps", 0.0f);

    // If FPS is not found in momentum metadata, try to calculate from animation data
    if (fps == 0.0f) {
      fps = calculateFpsFromAnimationData(model);
      if (fps > 0.0f) {
        MT_LOGD("FPS not found in momentum metadata, calculated from animation data: {}", fps);
      } else {
        // Fall back to default if calculation fails
        fps = 120.0f;
        MT_LOGW("FPS not found in momentum metadata or animation data, using default: {}", fps);
      }
    }

    const auto& [motion, identity] = getMotionFromModel(model);
    return {motion, identity, fps};
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to parse motion data  : {}", err.what());
  }

  return {};
}

std::tuple<Character, MatrixXf, JointParameters, float> loadCharacterWithMotionCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    bool mapParameters) {
  // ---------------------------------------------
  // load Skeleton and Mesh
  // ---------------------------------------------
  try {
    // Set maximum filesize to 4 gigabyte (glb hard limit due to uint32).
    // Need to fork gltf load to support dynamic loading at some point if files get too big
    fx::gltf::Document model = loadModel(input);
    Character character = loadGltfCharacter(model);

    // load motion
    auto [motion, identity, fps] = loadMotionFromModel(model);

    // check that there is actual motion data within the glb / gltf
    MT_LOGW_IF(std::get<1>(motion).cols() == 0, "No motion data found in gltf file");
    return {
        character,
        mapParameters ? mapMotionToCharacter(motion, character) : std::get<1>(motion),
        mapParameters ? mapIdentityToCharacter(identity, character) : std::get<1>(identity),
        fps};
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to load gltf : {}", err.what());
  }

  return {};
}

std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacterCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    const Character& character) {
  const auto [loadedChar, motion, identity, fps] = loadCharacterWithMotionCommon(input);
  return {
      mapMotionToCharacter({loadedChar.parameterTransform.name, motion}, character),
      mapIdentityToCharacter({loadedChar.skeleton.getJointNames(), identity}, character),
      fps};
}

namespace {

// Validates a channel's target node and resolves it to a bone index.
// Returns kInvalidIndex if the channel should be skipped.
// Throws on bone index out of range or name mismatch.
size_t validateAndResolveChannel(
    const fx::gltf::Animation::Channel& channel,
    const fx::gltf::Document& model,
    const std::vector<size_t>& nodeToObjectMap,
    const Skeleton& skeleton) {
  if (channel.target.node >= nodeToObjectMap.size() ||
      nodeToObjectMap.at(channel.target.node) == kInvalidIndex) {
    return kInvalidIndex;
  }

  const auto boneIdx = nodeToObjectMap[channel.target.node];
  MT_THROW_IF(
      boneIdx >= skeleton.joints.size(),
      "Bone index {} is larger than the number of joints in the skeleton ({})",
      boneIdx,
      skeleton.joints.size());

  const auto& node = model.nodes[channel.target.node];
  MT_THROW_IF(
      node.name != skeleton.joints[boneIdx].name,
      "Bone mismatch between gltf and character; GLTF: {}, Character: {}",
      node.name,
      skeleton.joints[boneIdx].name);

  return boneIdx;
}

// Reads translation/rotation/scale data from a channel sampler and applies it to skeleton states.
// Returns false if the channel's target path is unrecognized and should be skipped.
bool applyChannelDataToStates(
    const fx::gltf::Document& model,
    const fx::gltf::Animation::Sampler& sampler,
    const std::string& targetPath,
    size_t boneIdx,
    const Character& character,
    std::vector<SkeletonState>& states,
    std::vector<float>& timestamps) {
  // Read the appropriate transform component from the sampler output
  std::vector<Vector3f> translations_m;
  std::vector<Quaternionf> rotations;
  std::vector<Vector3f> scales;
  if (targetPath == "translation") {
    translations_m = copyAccessorBuffer<Vector3f>(model, sampler.output);
  } else if (targetPath == "rotation") {
    rotations = copyAccessorBuffer<Quaternionf>(model, sampler.output);
  } else if (targetPath == "scale") {
    scales = copyAccessorBuffer<Vector3f>(model, sampler.output);
  } else {
    return false;
  }

  // Validate and store timestamps
  auto timestampsCur = copyAccessorBuffer<float>(model, sampler.input);
  if (timestamps.empty()) {
    timestamps = timestampsCur;
  } else {
    MT_THROW_IF(
        timestamps.size() != timestampsCur.size(),
        "Timestamps do not match between channels, got {} and {}",
        timestamps.size(),
        timestampsCur.size());
  }

  // Lazily initialize states on first channel
  if (states.empty()) {
    states.resize(
        timestamps.size(),
        SkeletonState(
            JointParameters::Zero(character.parameterTransform.numJointParameters()),
            character.skeleton));
  }

  MT_THROW_IF(states.size() != timestamps.size(), "States and timestamps do not match in size");

  // Apply the transform component to each frame
  if (translations_m.size() == states.size()) {
    for (size_t i = 0; i < states.size(); ++i) {
      states[i].jointState[boneIdx].localTransform.translation = toCm() * translations_m[i];
    }
  }
  if (rotations.size() == states.size()) {
    for (size_t i = 0; i < states.size(); ++i) {
      states[i].jointState[boneIdx].localTransform.rotation = rotations[i];
    }
  }
  if (scales.size() == states.size()) {
    for (size_t i = 0; i < states.size(); ++i) {
      states[i].jointState[boneIdx].localTransform.scale = scales[i].x();
    }
  }

  return true;
}

// Computes global transforms from local transforms by walking the joint hierarchy.
void computeGlobalTransforms(const Skeleton& skeleton, std::vector<SkeletonState>& states) {
  for (auto& s : states) {
    for (size_t iJoint = 0; iJoint < skeleton.joints.size(); ++iJoint) {
      Transform parentTransform;
      const auto parentJoint = skeleton.joints[iJoint].parent;
      if (parentJoint != kInvalidIndex) {
        parentTransform = s.jointState[parentJoint].transform;
      }
      s.jointState[iJoint].transform = parentTransform * s.jointState[iJoint].localTransform;
    }
  }
}

} // namespace

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStatesCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  const fx::gltf::Document model = loadModel(input);
  std::vector<size_t> nodeToObjectMap;
  Character character = loadGltfCharacterInternal(model, nodeToObjectMap);

  if (model.animations.empty()) {
    MT_LOGW("No animation found in gltf file");
    return {character, {}, {}};
  }
  if (model.animations.size() > 1) {
    MT_LOGW("Multiple animations found in gltf file. Only the first one will be used.");
  }

  const auto& animation = model.animations.front();
  std::vector<float> timestamps;
  std::vector<SkeletonState> states;

  for (const auto& channel : animation.channels) {
    const auto boneIdx =
        validateAndResolveChannel(channel, model, nodeToObjectMap, character.skeleton);
    if (boneIdx == kInvalidIndex) {
      continue;
    }

    const auto& sampler = animation.samplers[channel.sampler];
    if (!applyChannelDataToStates(
            model, sampler, channel.target.path, boneIdx, character, states, timestamps)) {
      continue;
    }
  }

  computeGlobalTransforms(character.skeleton, states);

  return {character, states, timestamps};
}

} // namespace momentum
