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
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/gltf/gltf_io_internal.h"
#include "momentum/io/gltf/gltf_skeleton_io.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/io/gltf/utils/momentum_extension.h"
#include "momentum/math/constants.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace momentum {

std::tuple<MotionParameters, IdentityParameters> getMotionFromJson(
    const fx::gltf::Document& model,
    const nlohmann::json& motion) {
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

std::tuple<MotionParameters, IdentityParameters> getMotionFromModel(fx::gltf::Document& model) {
  const auto& def = findMomentumExtension(model.extensionsAndExtras);
  return getMotionFromJson(model, def.value("motion", nlohmann::json::object()));
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

float loadFpsFromModel(const fx::gltf::Document& model) {
  float fps = findMomentumExtension(model.extensionsAndExtras).value("fps", 0.0f);

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
  return fps;
}

std::tuple<MotionParameters, IdentityParameters, float> loadMotionFromModel(
    fx::gltf::Document& model) {
  // parse motion data
  try {
    const float fps = loadFpsFromModel(model);
    const auto& [motion, identity] = getMotionFromModel(model);
    return {motion, identity, fps};
  } catch (std::runtime_error& err) {
    MT_THROW("Unable to parse motion data  : {}", err.what());
  }
}

std::tuple<Character, MatrixXf, JointParameters, float> loadCharacterWithMotionCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    bool mapParameters) {
  // ---------------------------------------------
  // load Skeleton and Mesh
  // ---------------------------------------------
  try {
    fx::gltf::Document model = loadGltfDocument(input);
    auto characters = loadGltfCharactersInternal(model, GltfObjects::Skip);
    auto& loaded = selectGltfCharacter(characters);
    Character& character = loaded.character;

    // load the motion of that character
    const auto [motion, identity] = getMotionFromJson(
        model, getCharacterMotionJson(model, loaded.rootNode, characters.size() == 1));
    const float fps = loadFpsFromModel(model);

    // check that there is actual motion data within the glb / gltf
    MT_LOGW_IF(std::get<1>(motion).cols() == 0, "No motion data found in gltf file");
    return {
        character,
        mapParameters ? mapMotionToCharacter(motion, character) : std::get<1>(motion),
        mapParameters ? mapIdentityToCharacter(identity, character) : std::get<1>(identity),
        fps};
  } catch (std::runtime_error& err) {
    MT_THROW("{}", gltfLoadErrorMessage(err.what()));
  }
}

std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacterCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    const Character& character) {
  const auto [loadedChar, motion, identity, fps] = loadCharacterWithMotionCommon(input);
  // glTF files that store motion without per-joint identity offsets surface as an empty `identity`
  // vector. Skip mapIdentityToCharacter in that case to preserve the "no identity present" signal
  // and to avoid violating its size precondition (joint names paired with an empty values vector).
  const JointParameters mappedIdentity = identity.size() != 0
      ? mapIdentityToCharacter({loadedChar.skeleton.getJointNames(), identity}, character)
      : identity;
  return {
      mapMotionToCharacter({loadedChar.parameterTransform.name, motion}, character),
      mappedIdentity,
      fps};
}

namespace {

// Validates a channel's target node and resolves it to a bone index.
// Returns kInvalidIndex if the channel should be skipped.
// Throws on bone index out of range or name mismatch.
size_t validateAndResolveChannel(
    const fx::gltf::Animation::Channel& channel,
    const fx::gltf::Document& model,
    std::span<const size_t> nodeToObjectMap,
    const Skeleton& skeleton) {
  if (channel.target.node >= nodeToObjectMap.size() ||
      nodeToObjectMap[channel.target.node] == kInvalidIndex) {
    return kInvalidIndex;
  }

  // Bounds already validated above
  const auto boneIdx = nodeToObjectMap[channel.target.node];
  MT_THROW_IF(
      boneIdx >= skeleton.joints.size(),
      "Bone index {} is larger than the number of joints in the skeleton ({})",
      boneIdx,
      skeleton.joints.size());

  const auto& node = model.nodes.at(channel.target.node);
  MT_THROW_IF(
      node.name != skeleton.joints.at(boneIdx).name,
      "Bone mismatch between gltf and character; GLTF: {}, Character: {}",
      node.name,
      skeleton.joints[boneIdx].name);

  return boneIdx;
}

// The keys of a channel that animates a joint transform: the key times and, for each key, the
// value (x, y, z for a translation in meters or a scale, x, y, z, w for a rotation) and, for a
// cubic spline, the in- and out-tangents.
struct ChannelKeys {
  size_t joint = kInvalidIndex;
  std::string path;
  fx::gltf::Animation::Sampler::Type interpolation = fx::gltf::Animation::Sampler::Type::Linear;
  std::vector<float> times;
  std::vector<Vector4f> values;
  std::vector<Vector4f> inTangents;
  std::vector<Vector4f> outTangents;
};

// The keys of `sampler`, which animates the `path` of joint `joint`. A channel with an
// unrecognized target path, or whose values do not match its key times, has no keys.
ChannelKeys readChannelKeys(
    const fx::gltf::Document& model,
    const fx::gltf::Animation::Sampler& sampler,
    const std::string& path,
    size_t joint) {
  std::vector<Vector4f> outputs;
  if (path == "rotation") {
    std::ranges::transform(
        copyAccessorBuffer<Quaternionf>(model, sampler.output),
        std::back_inserter(outputs),
        [](const Quaternionf& rotation) { return rotation.coeffs(); });
  } else if (path == "translation" || path == "scale") {
    std::ranges::transform(
        copyAccessorBuffer<Vector3f>(model, sampler.output),
        std::back_inserter(outputs),
        [](const Vector3f& value) { return Vector4f(value.x(), value.y(), value.z(), 0.0f); });
  } else {
    return {};
  }

  ChannelKeys keys;
  keys.joint = joint;
  keys.path = path;
  keys.interpolation = sampler.interpolation;
  keys.times = copyAccessorBuffer<float>(model, sampler.input);
  if (!std::ranges::all_of(keys.times, [](float time) { return std::isfinite(time); })) {
    MT_LOGW(
        "glTF animation channel for the {} of joint {} has key times that are not finite numbers; it is not read.",
        path,
        joint);
    return {};
  }
  const size_t numKeys = keys.times.size();
  const bool cubic = keys.interpolation == fx::gltf::Animation::Sampler::Type::CubicSpline;
  // A cubic spline stores an in-tangent, a value and an out-tangent for each key.
  if (numKeys == 0 || outputs.size() != (cubic ? 3 * numKeys : numKeys)) {
    MT_LOGW(
        "glTF animation channel for the {} of joint {} has {} values for {} keys; it is not read.",
        path,
        joint,
        outputs.size(),
        numKeys);
    return {};
  }
  if (!cubic) {
    keys.values = std::move(outputs);
    return keys;
  }
  for (size_t key = 0; key < numKeys; ++key) {
    keys.inTangents.push_back(outputs.at(3 * key));
    keys.values.push_back(outputs.at(3 * key + 1));
    keys.outTangents.push_back(outputs.at(3 * key + 2));
  }
  return keys;
}

// Key times closer than this are the same frame time (see frameTimes()).
constexpr float kKeyTimeTolerance = 1e-5f;

// The value of `keys` at `time`, interpolated as the glTF format defines. Before the first key and
// after the last key, and within kKeyTimeTolerance of a key, the value is that of the key.
Vector4f sampleChannel(const ChannelKeys& keys, float time) {
  const auto& times = keys.times;
  const size_t lastKey = times.size() - 1;
  if (time <= times.at(0)) {
    return keys.values.at(0);
  }
  if (time >= times.at(lastKey)) {
    return keys.values.at(lastKey);
  }
  const auto next = static_cast<size_t>(std::ranges::upper_bound(times, time) - times.begin());
  const size_t key = next - 1;
  if (times.at(next) - time < kKeyTimeTolerance) {
    return keys.values.at(next);
  }
  if (time - times.at(key) < kKeyTimeTolerance ||
      keys.interpolation == fx::gltf::Animation::Sampler::Type::Step) {
    return keys.values.at(key);
  }
  const float duration = times.at(next) - times.at(key);
  const float s = (time - times.at(key)) / duration;
  const bool rotation = keys.path == "rotation";
  if (keys.interpolation == fx::gltf::Animation::Sampler::Type::Linear) {
    if (rotation) {
      return Quaternionf(keys.values.at(key)).slerp(s, Quaternionf(keys.values.at(next))).coeffs();
    }
    return (1.0f - s) * keys.values.at(key) + s * keys.values.at(next);
  }
  // A cubic Hermite spline, with tangents per unit of time.
  const float s2 = s * s;
  const float s3 = s2 * s;
  Vector4f value = (2.0f * s3 - 3.0f * s2 + 1.0f) * keys.values.at(key) +
      duration * (s3 - 2.0f * s2 + s) * keys.outTangents.at(key) +
      (-2.0f * s3 + 3.0f * s2) * keys.values.at(next) +
      duration * (s3 - s2) * keys.inTangents.at(next);
  if (rotation) {
    // A spline value of zero, such as halfway between the keys q and -q with zero tangents, has no
    // direction to normalize (Eigen's normalize() leaves a vector whose squared norm is not
    // positive unchanged), so it gets the spherical interpolation of its keys.
    if (!(value.squaredNorm() > 0.0f)) {
      return Quaternionf(keys.values.at(key)).slerp(s, Quaternionf(keys.values.at(next))).coeffs();
    }
    value.normalize();
  }
  return value;
}

// The frame times of `channels`: their key times when all channels have the same key times, and
// otherwise the key times of the channels whose key times increase, in order, with times closer
// than kKeyTimeTolerance merged. applyChannels() does not read the other channels.
std::vector<float> frameTimes(std::span<const ChannelKeys> channels) {
  if (channels.empty()) {
    return {};
  }
  if (std::ranges::all_of(channels, [&](const ChannelKeys& keys) {
        return keys.times == channels.front().times;
      })) {
    // Channels that share their key times give their keys in order, as before, also when the times
    // do not increase as the glTF format requires.
    const auto& times = channels.front().times;
    if (std::ranges::adjacent_find(times, std::greater_equal<>()) != times.end()) {
      MT_LOGW(
          "glTF animation has key times that do not increase, which the glTF format requires; momentum reads its keys in order.");
    }
    return times;
  }
  std::vector<float> result;
  for (const auto& keys : channels) {
    if (std::ranges::adjacent_find(keys.times, std::greater_equal<>()) != keys.times.end()) {
      continue;
    }
    result.insert(result.end(), keys.times.begin(), keys.times.end());
  }
  std::ranges::sort(result);
  const auto [first, last] = std::ranges::unique(
      result, [](float earlier, float later) { return later - earlier < kKeyTimeTolerance; });
  result.erase(first, last);
  return result;
}

// Sets the local transforms of the joints of `states`, one state per frame at `times`, from the
// channels' values at these times. A channel whose keys are at the frame times gives its key values
// in order; another channel is interpolated, and is not read if its key times do not increase, as
// the glTF format requires.
void applyChannels(
    std::span<const ChannelKeys> channels,
    std::span<const float> times,
    std::vector<SkeletonState>& states) {
  for (const auto& keys : channels) {
    const bool atFrames = std::ranges::equal(keys.times, times);
    if (!atFrames &&
        std::ranges::adjacent_find(keys.times, std::greater_equal<>()) != keys.times.end()) {
      MT_LOGW(
          "glTF animation channel for the {} of joint {} has key times that do not increase; it is not read.",
          keys.path,
          keys.joint);
      continue;
    }
    for (size_t frame = 0; frame < states.size(); ++frame) {
      MT_THROW_IF(
          keys.joint >= states[frame].jointState.size(),
          "Bone index {} exceeds jointState size {}",
          keys.joint,
          states[frame].jointState.size());
      auto& local = states[frame].jointState[keys.joint].localTransform;
      const Vector4f value = atFrames ? keys.values.at(frame) : sampleChannel(keys, times[frame]);
      if (keys.path == "translation") {
        local.translation = toCm() * value.head<3>();
      } else if (keys.path == "rotation") {
        local.rotation = Quaternionf(value);
      } else {
        local.scale = value.x();
      }
    }
  }
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

namespace {

// Whether `channel` animates a joint transform, the only channels readChannelKeys() reads.
bool isTransformChannel(const fx::gltf::Animation::Channel& channel) {
  const auto& path = channel.target.path;
  return path == "translation" || path == "rotation" || path == "scale";
}

// Whether `animation` has a transform channel that targets one of `character`'s joints.
bool animatesCharacter(
    const fx::gltf::Document& model,
    const fx::gltf::Animation& animation,
    const Character& character,
    std::span<const size_t> nodeToObjectMap) {
  return std::ranges::any_of(animation.channels, [&](const fx::gltf::Animation::Channel& channel) {
    return isTransformChannel(channel) &&
        validateAndResolveChannel(channel, model, nodeToObjectMap, character.skeleton) !=
        kInvalidIndex;
  });
}

} // namespace

const fx::gltf::Animation* findAnimationForCharacter(
    const fx::gltf::Document& model,
    const Character& character,
    std::span<const size_t> nodeToObjectMap) {
  const auto animates = [&](const fx::gltf::Animation& animation) {
    return animatesCharacter(model, animation, character, nodeToObjectMap);
  };
  const auto it = std::ranges::find_if(model.animations, animates);
  if (it == model.animations.end()) {
    return nullptr;
  }
  // The warning must not change what loads, so an animation that cannot be read does not count.
  [[maybe_unused]] const auto alsoAnimates = [&](const fx::gltf::Animation& animation) {
    try {
      return animates(animation);
    } catch (const std::exception&) {
      return false;
    }
  };
  MT_LOGW_IF(
      std::any_of(std::next(it), model.animations.end(), alsoAnimates),
      "glTF character '{}' has more than one animation; momentum reads only the first one, '{}'.",
      character.name,
      it->name);
  return &*it;
}

std::tuple<std::vector<SkeletonState>, std::vector<float>> loadSkeletonStatesForCharacter(
    const fx::gltf::Document& model,
    const fx::gltf::Animation& animation,
    const Character& character,
    std::span<const size_t> nodeToObjectMap) {
  std::vector<float> timestamps;
  std::vector<SkeletonState> states;

  // The channels give the local transforms of the joints' nodes. In a document from another tool,
  // these differ from the joints' local transforms (see JointFileTransform), so a joint without a
  // channel starts at its node's transform, and the transforms are mapped after the channels.
  const bool useFileTransforms = !hasMarkedJoints(model);
  std::vector<JointFileTransform> fileJoints;
  std::vector<Transform> restLocals;
  restLocals.reserve(character.skeleton.joints.size());
  if (useFileTransforms) {
    fileJoints = jointFileTransforms(model, character.skeleton, nodeToObjectMap);
    for (size_t jointId = 0; jointId < fileJoints.size(); ++jointId) {
      const uint32_t node = fileJoints[jointId].node;
      const Joint& joint = character.skeleton.joints.at(jointId);
      restLocals.push_back(
          node == kInvalidNodeId ? Transform(joint.translationOffset, joint.preRotation)
                                 : nodeLocalTransform(model.nodes.at(node)));
    }
  } else {
    const SkeletonState restState(
        JointParameters::Zero(character.parameterTransform.numJointParameters()),
        character.skeleton);
    for (const auto& jointState : restState.jointState) {
      restLocals.push_back(jointState.localTransform);
    }
  }

  std::vector<ChannelKeys> channels;
  for (const auto& channel : animation.channels) {
    const auto boneIdx =
        validateAndResolveChannel(channel, model, nodeToObjectMap, character.skeleton);
    if (boneIdx == kInvalidIndex) {
      continue;
    }
    ChannelKeys keys = readChannelKeys(
        model, animation.samplers.at(channel.sampler), channel.target.path, boneIdx);
    if (!keys.times.empty()) {
      channels.push_back(std::move(keys));
    }
  }

  // Channels may have their own key times; each is sampled at the frame times.
  timestamps = frameTimes(channels);
  states.assign(
      timestamps.size(),
      SkeletonState(
          JointParameters::Zero(character.parameterTransform.numJointParameters()),
          character.skeleton));
  for (auto& state : states) {
    for (size_t jointId = 0; jointId < restLocals.size() && jointId < state.jointState.size();
         ++jointId) {
      state.jointState.at(jointId).localTransform = restLocals.at(jointId);
    }
  }
  applyChannels(channels, timestamps, states);

  if (useFileTransforms) {
    for (auto& state : states) {
      for (size_t jointId = 0; jointId < fileJoints.size(); ++jointId) {
        if (fileJoints[jointId].node != kInvalidNodeId) {
          auto& local = state.jointState[jointId].localTransform;
          local = fileJoints[jointId].pre * local * fileJoints[jointId].post;
        }
      }
    }
  }
  computeGlobalTransforms(character.skeleton, states);
  return {std::move(states), std::move(timestamps)};
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStatesCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input) {
  return loadFromInput(
      input,
      [](const fx::gltf::Document& model)
          -> std::tuple<Character, std::vector<SkeletonState>, std::vector<float>> {
        std::vector<size_t> nodeToObjectMap;
        Character character = loadGltfCharacterInternal(model, nodeToObjectMap);

        const auto* animation = findAnimationForCharacter(model, character, nodeToObjectMap);
        if (animation == nullptr) {
          MT_LOGW("No animation of the character found in gltf file");
          return {std::move(character), std::vector<SkeletonState>(), std::vector<float>()};
        }
        // Only the animations after the selected one can also animate the character.
        const auto later =
            std::span(model.animations)
                .subspan(static_cast<size_t>(animation - model.animations.data()) + 1);
        if (std::ranges::any_of(later, [&](const fx::gltf::Animation& candidate) {
              return animatesCharacter(model, candidate, character, nodeToObjectMap);
            })) {
          MT_LOGW(
              "More than one animation in the gltf file animates the character. Only the first one will "
              "be used.");
        }

        auto [states, timestamps] =
            loadSkeletonStatesForCharacter(model, *animation, character, nodeToObjectMap);
        return {std::move(character), std::move(states), std::move(timestamps)};
      });
}

} // namespace momentum
