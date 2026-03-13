/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/bvh/bvh_io.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/exception.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <Eigen/Geometry>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace momentum {

namespace {

// Represents a BVH channel type
enum class BvhChannel { Xposition, Yposition, Zposition, Xrotation, Yrotation, Zrotation };

// Internal representation of a BVH joint during parsing
struct BvhJointData {
  std::string name;
  size_t parentIndex{0};
  Vector3f offset;
  std::vector<BvhChannel> channels;
  size_t channelOffset{0}; // Starting index in the motion data row
};

// Trim leading and trailing whitespace from a string
std::string trim(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// Parse the BVH channel type from a string
BvhChannel parseChannelType(const std::string& token) {
  if (token == "Xposition") {
    return BvhChannel::Xposition;
  } else if (token == "Yposition") {
    return BvhChannel::Yposition;
  } else if (token == "Zposition") {
    return BvhChannel::Zposition;
  } else if (token == "Xrotation") {
    return BvhChannel::Xrotation;
  } else if (token == "Yrotation") {
    return BvhChannel::Yrotation;
  } else if (token == "Zrotation") {
    return BvhChannel::Zrotation;
  } else {
    MT_THROW("Unknown BVH channel type: {}", token);
  }
}

// Returns the BVH channel name string
const char* channelTypeName(BvhChannel ch) {
  switch (ch) {
    case BvhChannel::Xposition:
      return "Xposition";
    case BvhChannel::Yposition:
      return "Yposition";
    case BvhChannel::Zposition:
      return "Zposition";
    case BvhChannel::Xrotation:
      return "Xrotation";
    case BvhChannel::Yrotation:
      return "Yrotation";
    case BvhChannel::Zrotation:
      return "Zrotation";
  }
}

// Returns the Eigen axis index for a rotation channel (0=X, 1=Y, 2=Z)
int rotationAxisIndex(BvhChannel ch) {
  switch (ch) {
    case BvhChannel::Xrotation:
      return 0;
    case BvhChannel::Yrotation:
      return 1;
    case BvhChannel::Zrotation:
      return 2;
    case BvhChannel::Xposition:
    case BvhChannel::Yposition:
    case BvhChannel::Zposition:
      MT_THROW("Not a rotation channel");
  }
}

// Returns the joint parameter index for a position channel (TX=0, TY=1, TZ=2)
int positionParameterIndex(BvhChannel ch) {
  switch (ch) {
    case BvhChannel::Xposition:
      return TX;
    case BvhChannel::Yposition:
      return TY;
    case BvhChannel::Zposition:
      return TZ;
    case BvhChannel::Xrotation:
    case BvhChannel::Yrotation:
    case BvhChannel::Zrotation:
      MT_THROW("Not a position channel");
  }
}

bool isRotationChannel(BvhChannel ch) {
  return ch == BvhChannel::Xrotation || ch == BvhChannel::Yrotation || ch == BvhChannel::Zrotation;
}

bool isPositionChannel(BvhChannel ch) {
  return ch == BvhChannel::Xposition || ch == BvhChannel::Yposition || ch == BvhChannel::Zposition;
}

struct BvhParseResult {
  std::vector<BvhJointData> joints;
  size_t totalChannels{0};
  std::vector<VectorXf> frames;
  float fps{30.0f};
};

// ---------------------------------------------------------------------------
// Hierarchy parsing helpers
// ---------------------------------------------------------------------------

// Process a joint definition line (ROOT or JOINT keyword)
void parseJointDefinition(
    std::istringstream& iss,
    std::stack<size_t>& jointStack,
    BvhParseResult& result) {
  std::string jointName;
  iss >> jointName;
  MT_THROW_IF(jointName.empty(), "BVH joint name is empty");

  BvhJointData joint;
  joint.name = jointName;
  joint.parentIndex = jointStack.empty() ? kInvalidIndex : jointStack.top();
  joint.offset = Vector3f::Zero();
  joint.channelOffset = result.totalChannels;

  const size_t jointIndex = result.joints.size();
  result.joints.push_back(joint);
  jointStack.push(jointIndex);
}

// Process an OFFSET line
void parseOffset(std::istringstream& iss, std::stack<size_t>& jointStack, BvhParseResult& result) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  iss >> x >> y >> z;
  // Only apply offset to real joints (not End Sites)
  if (jointStack.empty() || jointStack.top() == kInvalidIndex) {
    return;
  }
  const auto topIdx = jointStack.top();
  if (topIdx < result.joints.size()) {
    result.joints[topIdx].offset = Vector3f(x, y, z);
  }
}

// Process a CHANNELS line
void parseChannels(
    std::istringstream& iss,
    std::stack<size_t>& jointStack,
    BvhParseResult& result) {
  int numChannels = 0;
  iss >> numChannels;
  MT_THROW_IF(
      jointStack.empty() || jointStack.top() == kInvalidIndex ||
          jointStack.top() >= result.joints.size(),
      "CHANNELS found outside of a joint definition");
  const auto topIdx = jointStack.top();
  MT_THROW_IF(topIdx >= result.joints.size(), "Joint index out of range");
  auto& joint = result.joints[topIdx];
  for (int i = 0; i < numChannels; ++i) {
    std::string channelName;
    iss >> channelName;
    joint.channels.push_back(parseChannelType(channelName));
    result.totalChannels++;
  }
}

// Parse the HIERARCHY section of a BVH stream
void parseBvhHierarchy(std::istream& stream, BvhParseResult& result) {
  std::string line;
  std::stack<size_t> jointStack;

  while (stream) {
    GetLineCrossPlatform(stream, line);
    const std::string trimmedLine = trim(line);
    if (trimmedLine.empty()) {
      continue;
    }

    std::istringstream iss(trimmedLine);
    std::string keyword;
    iss >> keyword;

    if (keyword == "ROOT" || keyword == "JOINT") {
      parseJointDefinition(iss, jointStack, result);
    } else if (keyword == "End") {
      // End Site - push a sentinel value
      jointStack.push(kInvalidIndex);
    } else if (keyword == "OFFSET") {
      parseOffset(iss, jointStack, result);
    } else if (keyword == "CHANNELS") {
      parseChannels(iss, jointStack, result);
    } else if (keyword == "{") {
      continue;
    } else if (keyword == "}") {
      MT_THROW_IF(jointStack.empty(), "Unexpected closing brace in BVH hierarchy");
      jointStack.pop();
    } else if (keyword == "MOTION") {
      // Transition to motion section
      return;
    }
  }
}

// Parse the MOTION section of a BVH stream
void parseBvhMotion(std::istream& stream, BvhParseResult& result) {
  std::string line;

  while (stream) {
    GetLineCrossPlatform(stream, line);
    const std::string trimmedLine = trim(line);
    if (trimmedLine.empty()) {
      continue;
    }

    std::istringstream iss(trimmedLine);
    std::string keyword;
    iss >> keyword;

    if (keyword == "Frames:") {
      size_t numFrames = 0;
      iss >> numFrames;
      result.frames.reserve(numFrames);
    } else if (keyword == "Frame") {
      // "Frame Time: X.XXXX"
      std::string timeKeyword;
      float frameTime = 0.0f;
      iss >> timeKeyword >> frameTime;
      if (frameTime > 0.0f) {
        result.fps = 1.0f / frameTime;
      }
    } else {
      // Motion data line - parse all channel values
      std::istringstream dataIss(trimmedLine);
      VectorXf frameData(result.totalChannels);
      for (size_t i = 0; i < result.totalChannels; ++i) {
        MT_THROW_IF(!dataIss, "Unexpected end of motion data line");
        dataIss >> frameData[i];
      }
      result.frames.push_back(frameData);
    }
  }
}

// Parse the hierarchy and motion sections from a BVH stream
BvhParseResult parseBvhStream(std::istream& stream) {
  BvhParseResult result;

  // Find the HIERARCHY keyword
  std::string line;
  while (stream) {
    GetLineCrossPlatform(stream, line);
    const std::string trimmedLine = trim(line);
    if (trimmedLine == "HIERARCHY") {
      break;
    }
  }

  parseBvhHierarchy(stream, result);
  parseBvhMotion(stream, result);

  MT_THROW_IF(result.joints.empty(), "No joints found in BVH file");

  return result;
}

// ---------------------------------------------------------------------------
// Motion conversion helpers (load direction: BVH -> Momentum)
// ---------------------------------------------------------------------------

// Convert BVH channel values to a Momentum rotation quaternion for a single joint
Quaternionf convertBvhRotationToQuaternion(
    const BvhJointData& bvhJoint,
    const VectorXf& frameData) {
  Quaternionf q = Quaternionf::Identity();
  for (size_t chIdx = 0; chIdx < bvhJoint.channels.size(); ++chIdx) {
    const auto ch = bvhJoint.channels[chIdx];
    if (!isRotationChannel(ch)) {
      continue;
    }
    const auto dataIdx = static_cast<Eigen::Index>(bvhJoint.channelOffset + chIdx);
    const float angleRad = frameData[dataIdx] * pi<float>() / 180.0f;
    q = q * Quaternionf(Eigen::AngleAxisf(angleRad, Vector3f::Unit(rotationAxisIndex(ch))));
  }
  return q;
}

// Map a single BVH joint's channel values to Momentum model parameters for one frame
void mapBvhJointToModelParams(
    const BvhJointData& bvhJoint,
    const VectorXf& frameData,
    Eigen::Index jointParamBase,
    Eigen::Index frameIdx,
    MatrixXf& motion) {
  // Map position channels directly
  for (size_t chIdx = 0; chIdx < bvhJoint.channels.size(); ++chIdx) {
    const auto ch = bvhJoint.channels[chIdx];
    if (!isPositionChannel(ch)) {
      continue;
    }
    const auto dataIdx = static_cast<Eigen::Index>(bvhJoint.channelOffset + chIdx);
    motion(jointParamBase + positionParameterIndex(ch), frameIdx) = frameData[dataIdx];
  }

  // Convert BVH rotation to Momentum's ZYX convention
  const Quaternionf q = convertBvhRotationToQuaternion(bvhJoint, frameData);
  if (!q.isApprox(Quaternionf::Identity())) {
    // Decompose to Momentum's ZYX convention: [rx, ry, rz]
    // Momentum applies rotations as R = Rz(rz) * Ry(ry) * Rx(rx)
    // so we decompose as ZYX and reverse to get [rx, ry, rz]
    const Vector3f eulerAngles = rotationMatrixToEulerZYX(q.toRotationMatrix()).reverse();
    motion(jointParamBase + RX, frameIdx) = eulerAngles[0];
    motion(jointParamBase + RY, frameIdx) = eulerAngles[1];
    motion(jointParamBase + RZ, frameIdx) = eulerAngles[2];
  }
}

// Build a Character from BVH joint data using an identity ParameterTransform
Character buildCharacterFromBvh(const std::vector<BvhJointData>& bvhJoints) {
  Skeleton skeleton;
  skeleton.joints.reserve(bvhJoints.size());

  std::vector<std::string> jointNames;
  jointNames.reserve(bvhJoints.size());

  for (const auto& bvhJoint : bvhJoints) {
    Joint joint;
    joint.name = bvhJoint.name;
    joint.parent = bvhJoint.parentIndex;
    joint.preRotation = Quaternionf::Identity();
    joint.translationOffset = bvhJoint.offset;
    skeleton.joints.push_back(joint);
    jointNames.push_back(bvhJoint.name);
  }

  auto parameterTransform = ParameterTransform::identity(jointNames);

  return {skeleton, parameterTransform};
}

// Map BVH motion frames to model parameters
// Each frame is converted to a vector of 7*numJoints parameters
MatrixXf mapBvhMotion(
    const std::vector<BvhJointData>& bvhJoints,
    const std::vector<VectorXf>& frames,
    size_t numJoints) {
  const auto numModelParams = static_cast<Eigen::Index>(numJoints * kParametersPerJoint);
  const auto numFrames = static_cast<Eigen::Index>(frames.size());

  MatrixXf motion = MatrixXf::Zero(numModelParams, numFrames);

  for (Eigen::Index frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
    for (size_t jointIdx = 0; jointIdx < bvhJoints.size(); ++jointIdx) {
      const auto jointParamBase = static_cast<Eigen::Index>(jointIdx * kParametersPerJoint);
      mapBvhJointToModelParams(
          bvhJoints[jointIdx], frames[frameIdx], jointParamBase, frameIdx, motion);
    }
  }

  return motion;
}

std::tuple<Character, MotionParameters, float> loadBvhFromStream(std::istream& stream) {
  auto parseResult = parseBvhStream(stream);

  Character character = buildCharacterFromBvh(parseResult.joints);

  MatrixXf motion = mapBvhMotion(parseResult.joints, parseResult.frames, parseResult.joints.size());

  MotionParameters motionParams = std::make_tuple(character.parameterTransform.name, motion);

  return std::make_tuple(std::move(character), std::move(motionParams), parseResult.fps);
}

// ---------------------------------------------------------------------------
// Save helpers (Momentum -> BVH)
// ---------------------------------------------------------------------------

// Build BvhJointData from a Character's skeleton, using the standard BVH channel layout:
// Root joint gets 6 channels (Xposition, Yposition, Zposition, Zrotation, Xrotation, Yrotation),
// non-root joints get 3 rotation channels (Zrotation, Xrotation, Yrotation).
std::vector<BvhJointData> buildBvhJointsFromCharacter(const Character& character) {
  const auto& skeleton = character.skeleton;
  std::vector<BvhJointData> bvhJoints;
  bvhJoints.reserve(skeleton.joints.size());

  size_t channelOffset = 0;
  for (const auto& joint : skeleton.joints) {
    BvhJointData bvhJoint;
    bvhJoint.name = joint.name;
    bvhJoint.parentIndex = joint.parent;
    bvhJoint.offset = joint.translationOffset;
    bvhJoint.channelOffset = channelOffset;

    if (joint.parent == kInvalidIndex) {
      // Root joint: position + rotation channels
      bvhJoint.channels = {
          BvhChannel::Xposition,
          BvhChannel::Yposition,
          BvhChannel::Zposition,
          BvhChannel::Zrotation,
          BvhChannel::Xrotation,
          BvhChannel::Yrotation};
      channelOffset += 6;
    } else {
      // Non-root joint: rotation channels only
      bvhJoint.channels = {BvhChannel::Zrotation, BvhChannel::Xrotation, BvhChannel::Yrotation};
      channelOffset += 3;
    }

    bvhJoints.push_back(std::move(bvhJoint));
  }

  return bvhJoints;
}

// Collect children for each joint in the skeleton
std::vector<std::vector<size_t>> buildChildMap(const Skeleton& skeleton) {
  const auto numJoints = skeleton.joints.size();
  std::vector<std::vector<size_t>> children(numJoints);
  for (size_t i = 0; i < numJoints; ++i) {
    const auto parent = skeleton.joints[i].parent;
    if (parent != kInvalidIndex && parent < numJoints) {
      children[parent].push_back(i);
    }
  }
  return children;
}

// Write a single joint and its subtree recursively
void writeBvhJoint(
    std::ostream& stream,
    const Skeleton& skeleton,
    const std::vector<BvhJointData>& bvhJoints,
    const std::vector<std::vector<size_t>>& children,
    size_t jointIdx,
    int depth) {
  MT_THROW_IF(jointIdx >= skeleton.joints.size(), "Joint index out of range");
  const auto depthSz = static_cast<std::string::size_type>(depth);
  const auto indent = std::string(depthSz * 2, ' ');
  const auto innerIndent = std::string((depthSz + 1) * 2, ' ');

  // Write joint header
  if (skeleton.joints[jointIdx].parent == kInvalidIndex) {
    stream << "ROOT " << skeleton.joints[jointIdx].name << "\n";
  } else {
    stream << indent << "JOINT " << skeleton.joints[jointIdx].name << "\n";
  }
  stream << indent << "{\n";

  // Write offset
  const auto& offset = bvhJoints[jointIdx].offset;
  stream << innerIndent << "OFFSET " << std::fixed << std::setprecision(4) << offset[0] << " "
         << offset[1] << " " << offset[2] << "\n";

  // Write channels
  const auto& channels = bvhJoints[jointIdx].channels;
  stream << innerIndent << "CHANNELS " << channels.size();
  for (const auto& ch : channels) {
    stream << " " << channelTypeName(ch);
  }
  stream << "\n";

  if (children[jointIdx].empty()) {
    // Leaf joint: write an End Site with zero offset
    stream << innerIndent << "End Site\n";
    stream << innerIndent << "{\n";
    stream << std::string((depthSz + 2) * 2, ' ') << "OFFSET 0.0000 0.0000 0.0000\n";
    stream << innerIndent << "}\n";
  } else {
    // Recurse into children
    for (const auto childIdx : children[jointIdx]) {
      writeBvhJoint(stream, skeleton, bvhJoints, children, childIdx, depth + 1);
    }
  }

  stream << indent << "}\n";
}

// Write the HIERARCHY section of a BVH file to a stream
void writeBvhHierarchy(
    std::ostream& stream,
    const Character& character,
    const std::vector<BvhJointData>& bvhJoints) {
  const auto& skeleton = character.skeleton;
  stream << "HIERARCHY\n";

  const auto children = buildChildMap(skeleton);

  // Find and write root joints
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    if (skeleton.joints[i].parent == kInvalidIndex) {
      writeBvhJoint(stream, skeleton, bvhJoints, children, i, 0);
    }
  }
}

// Convert Momentum model parameters to BVH channel values for one frame
VectorXf convertModelParamsToBvhFrame(
    const std::vector<BvhJointData>& bvhJoints,
    const MatrixXf& motion,
    Eigen::Index frameIdx) {
  // Count total channels
  size_t totalChannels = 0;
  for (const auto& joint : bvhJoints) {
    totalChannels += joint.channels.size();
  }

  VectorXf frameData = VectorXf::Zero(totalChannels);

  for (size_t jointIdx = 0; jointIdx < bvhJoints.size(); ++jointIdx) {
    const auto& bvhJoint = bvhJoints[jointIdx];
    const auto jointParamBase = static_cast<Eigen::Index>(jointIdx * kParametersPerJoint);

    // Get Momentum rotation values (rx, ry, rz in radians)
    const float rx = motion(jointParamBase + RX, frameIdx);
    const float ry = motion(jointParamBase + RY, frameIdx);
    const float rz = motion(jointParamBase + RZ, frameIdx);

    // Convert Momentum ZYX rotation to quaternion: R = Rz(rz) * Ry(ry) * Rx(rx)
    const Quaternionf q = Quaternionf(Eigen::AngleAxisf(rz, Vector3f::UnitZ())) *
        Quaternionf(Eigen::AngleAxisf(ry, Vector3f::UnitY())) *
        Quaternionf(Eigen::AngleAxisf(rx, Vector3f::UnitX()));

    // Collect rotation and position channel indices
    std::vector<int> rotAxes;
    std::vector<size_t> rotChannelIndices;
    for (size_t chIdx = 0; chIdx < bvhJoint.channels.size(); ++chIdx) {
      const auto ch = bvhJoint.channels[chIdx];
      if (isRotationChannel(ch)) {
        rotAxes.push_back(rotationAxisIndex(ch));
        rotChannelIndices.push_back(bvhJoint.channelOffset + chIdx);
      } else if (isPositionChannel(ch)) {
        const auto dataIdx = static_cast<Eigen::Index>(bvhJoint.channelOffset + chIdx);
        frameData[dataIdx] = motion(jointParamBase + positionParameterIndex(ch), frameIdx);
      }
    }

    if (rotAxes.size() == 3 && rotChannelIndices.size() == 3) {
      // Decompose the quaternion into Euler angles matching the BVH channel order
      const Vector3f angles = rotationMatrixToEuler(
          q.toRotationMatrix(), rotAxes[0], rotAxes[1], rotAxes[2], EulerConvention::Intrinsic);

      const float rad2deg = 180.0f / pi<float>();
      for (size_t k = 0; k < 3; ++k) {
        frameData[rotChannelIndices[k]] = angles[k] * rad2deg;
      }
    }
  }

  return frameData;
}

// Write the MOTION section of a BVH file to a stream
void writeBvhMotion(
    std::ostream& stream,
    const std::vector<BvhJointData>& bvhJoints,
    const MatrixXf& motion,
    float fps) {
  MT_THROW_IF(fps <= 0.0f, "fps must be positive for BVH export");

  const auto numFrames = motion.cols();

  stream << "MOTION\n";
  stream << "Frames: " << numFrames << "\n";
  stream << "Frame Time: " << std::fixed << std::setprecision(6) << (1.0f / fps) << "\n";

  for (Eigen::Index frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
    const VectorXf frameData = convertModelParamsToBvhFrame(bvhJoints, motion, frameIdx);

    for (Eigen::Index i = 0; i < frameData.size(); ++i) {
      if (i > 0) {
        stream << " ";
      }
      stream << std::fixed << std::setprecision(4) << frameData[i];
    }
    stream << "\n";
  }
}

// Write a Character with motion to a BVH stream
void saveBvhToStream(
    std::ostream& stream,
    const Character& character,
    const MatrixXf& motion,
    float fps) {
  const auto bvhJoints = buildBvhJointsFromCharacter(character);

  writeBvhHierarchy(stream, character, bvhJoints);
  writeBvhMotion(stream, bvhJoints, motion, fps);
}

} // namespace

Character loadBvhCharacter(const filesystem::path& filepath) {
  std::ifstream stream(filepath);
  MT_THROW_IF(!stream.is_open(), "Failed to open BVH file: {}", filepath.string());
  auto [character, motionParams, fps] = loadBvhFromStream(stream);
  return character;
}

Character loadBvhCharacter(std::span<const std::byte> bytes) {
  ispanstream stream(bytes);
  auto [character, motionParams, fps] = loadBvhFromStream(stream);
  return character;
}

std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    const filesystem::path& filepath) {
  std::ifstream stream(filepath);
  MT_THROW_IF(!stream.is_open(), "Failed to open BVH file: {}", filepath.string());
  return loadBvhFromStream(stream);
}

std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    std::span<const std::byte> bytes) {
  ispanstream stream(bytes);
  return loadBvhFromStream(stream);
}

void saveBvhCharacter(const filesystem::path& filepath, const Character& character) {
  std::ofstream stream(filepath);
  MT_THROW_IF(!stream.is_open(), "Failed to open BVH file for writing: {}", filepath.string());

  // Write with zero motion (single frame, all zeros)
  const auto numParams =
      static_cast<Eigen::Index>(character.skeleton.joints.size() * kParametersPerJoint);
  MatrixXf motion = MatrixXf::Zero(numParams, 1);
  saveBvhToStream(stream, character, motion, 30.0f);
}

void saveBvhCharacterWithMotion(
    const filesystem::path& filepath,
    const Character& character,
    const MotionParameters& motionParams,
    float fps) {
  std::ofstream stream(filepath);
  MT_THROW_IF(!stream.is_open(), "Failed to open BVH file for writing: {}", filepath.string());

  const auto& [paramNames, motion] = motionParams;
  saveBvhToStream(stream, character, motion, fps);
}

std::vector<std::byte> saveBvhCharacterWithMotion(
    const Character& character,
    const MotionParameters& motionParams,
    float fps) {
  const auto& [paramNames, motion] = motionParams;

  std::ostringstream oss;
  saveBvhToStream(oss, character, motion, fps);

  const std::string str = oss.str();
  std::vector<std::byte> bytes(str.size());
  std::memcpy(bytes.data(), str.data(), str.size());
  return bytes;
}

} // namespace momentum
