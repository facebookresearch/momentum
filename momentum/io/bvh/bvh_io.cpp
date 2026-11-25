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
#include "momentum/common/log.h"
#include "momentum/io/common/stream_utils.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <Eigen/Geometry>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
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
  MT_THROW("Invalid BVH channel: {}", static_cast<int>(ch));
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
  MT_THROW("Invalid BVH channel: {}", static_cast<int>(ch));
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
  MT_THROW("Invalid BVH channel: {}", static_cast<int>(ch));
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
    // Decompose into Momentum's Extrinsic XYZ convention: [rx, ry, rz]
    const Vector3f eulerAngles =
        rotationMatrixToEulerXYZ(q.toRotationMatrix(), EulerConvention::Extrinsic);
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

// Channel values are written with 4 decimals, so a smaller joint value is written as zero.
constexpr float kWrittenAsZero = 5e-5f;

constexpr std::array<BvhChannel, 3> kPositionChannels = {
    BvhChannel::Xposition,
    BvhChannel::Yposition,
    BvhChannel::Zposition};
constexpr std::array<BvhChannel, 3> kRotationChannels = {
    BvhChannel::Zrotation,
    BvhChannel::Xrotation,
    BvhChannel::Yrotation};

// Whether each joint gets position channels: root joints, and joints that `jointMotion` (joint
// parameters x frames) translates in some frame.
std::vector<bool> jointsWithPositionChannels(
    const Skeleton& skeleton,
    const MatrixXf& jointMotion) {
  std::vector<bool> result(skeleton.joints.size());
  for (size_t joint = 0; joint < skeleton.joints.size(); ++joint) {
    const auto base = static_cast<Eigen::Index>(joint * kParametersPerJoint);
    result.at(joint) = skeleton.joints.at(joint).parent == kInvalidIndex ||
        !jointMotion.middleRows(base + TX, 3).isZero(kWrittenAsZero);
  }
  return result;
}

// Appends `name` to the comma-separated `list`.
void appendName(std::string& list, std::string_view name) {
  if (!list.empty()) {
    list += ", ";
  }
  list += name;
}

// Warns about the data of `character` and the joint scales of `jointMotion` (joint parameters x
// frames), which a BVH file cannot store.
void warnAboutDataThatIsNotSaved(const Character& character, const MatrixXf& jointMotion) {
  std::string notSaved;
  if (character.mesh != nullptr) {
    appendName(notSaved, "mesh");
  }
  if (!character.locators.empty()) {
    appendName(notSaved, "locators");
  }
  if (character.collision != nullptr && !character.collision->empty()) {
    appendName(notSaved, "collision geometry");
  }
  if (!character.physicalProperties.empty()) {
    appendName(notSaved, "physical properties");
  }
  // "{}" is empty metadata, as isEmptyMetadata() in io_json_utils says, which io_bvh does not link.
  if (!character.metadata.empty() && character.metadata != "{}") {
    appendName(notSaved, "metadata");
  }
  if (!notSaved.empty()) {
    MT_LOGW(
        "BVH export: a BVH file stores only a skeleton and its motion; not saved: {}", notSaved);
  }

  std::string scaledJoints;
  for (size_t joint = 0; joint < character.skeleton.joints.size(); ++joint) {
    const auto row = static_cast<Eigen::Index>(joint * kParametersPerJoint + SC);
    if (!jointMotion.row(row).isZero(kWrittenAsZero)) {
      appendName(scaledJoints, character.skeleton.joints[joint].name);
    }
  }
  if (!scaledJoints.empty()) {
    MT_LOGW(
        "BVH export: BVH cannot store joint scales; the scales of these joints are not saved: {}",
        scaledJoints);
  }
}

// Writes `joint` and the joints below it to the HIERARCHY section, and appends them to `order` in
// the order in which the section lists them, which is the order of their channels in a frame.
void writeBvhJoint(
    std::ostream& stream,
    const Skeleton& skeleton,
    const std::vector<std::vector<size_t>>& children,
    const std::vector<bool>& hasPositionChannels,
    size_t joint,
    int depth,
    std::vector<size_t>& order) {
  order.push_back(joint);
  const auto depthSz = static_cast<std::string::size_type>(depth);
  const auto indent = std::string(depthSz * 2, ' ');
  const auto innerIndent = std::string((depthSz + 1) * 2, ' ');

  // Write joint header
  const auto& skeletonJoint = skeleton.joints.at(joint);
  if (skeletonJoint.parent == kInvalidIndex) {
    stream << "ROOT " << skeletonJoint.name << "\n";
  } else {
    stream << indent << "JOINT " << skeletonJoint.name << "\n";
  }
  stream << indent << "{\n";

  // Write offset
  const auto& offset = skeletonJoint.translationOffset;
  stream << innerIndent << "OFFSET " << std::fixed << std::setprecision(4) << offset[0] << " "
         << offset[1] << " " << offset[2] << "\n";

  // Write channels
  const bool positions = hasPositionChannels.at(joint);
  stream << innerIndent << "CHANNELS " << (positions ? 6 : 3);
  if (positions) {
    for (const auto channel : kPositionChannels) {
      stream << " " << channelTypeName(channel);
    }
  }
  for (const auto channel : kRotationChannels) {
    stream << " " << channelTypeName(channel);
  }
  stream << "\n";

  if (children.at(joint).empty()) {
    // Leaf joint: write an End Site with zero offset
    stream << innerIndent << "End Site\n";
    stream << innerIndent << "{\n";
    stream << std::string((depthSz + 2) * 2, ' ') << "OFFSET 0.0000 0.0000 0.0000\n";
    stream << innerIndent << "}\n";
  } else {
    // Recurse into children
    for (const auto child : children.at(joint)) {
      writeBvhJoint(stream, skeleton, children, hasPositionChannels, child, depth + 1, order);
    }
  }

  stream << indent << "}\n";
}

// Writes the HIERARCHY section, one ROOT per root joint, and returns the joints in the order in
// which it lists them.
std::vector<size_t> writeBvhHierarchy(
    std::ostream& stream,
    const Skeleton& skeleton,
    const std::vector<bool>& hasPositionChannels) {
  stream << "HIERARCHY\n";
  const auto children = buildChildMap(skeleton);
  std::vector<size_t> order;
  order.reserve(skeleton.joints.size());
  for (size_t joint = 0; joint < skeleton.joints.size(); ++joint) {
    if (skeleton.joints[joint].parent == kInvalidIndex) {
      writeBvhJoint(stream, skeleton, children, hasPositionChannels, joint, 0, order);
    }
  }
  return order;
}

// Writes the channel values of one frame of `jointMotion` (joint parameters x frames): for each
// joint in `order`, its translation joint parameters when it has position channels, then its local
// rotation as Euler angles in degrees in the order of the rotation channels. BVH joints have no
// pre-rotation, so each joint's pre-rotation is folded into its rotation channels.
void writeBvhFrame(
    std::ostream& stream,
    const Skeleton& skeleton,
    const MatrixXf& jointMotion,
    Eigen::Index frame,
    const std::vector<size_t>& order,
    const std::vector<bool>& hasPositionChannels) {
  constexpr float kRadiansToDegrees = 180.0f / pi<float>();
  bool first = true;
  const auto write = [&](float value) {
    if (!first) {
      stream << " ";
    }
    first = false;
    stream << std::fixed << std::setprecision(4) << value;
  };
  for (const size_t joint : order) {
    const auto base = static_cast<Eigen::Index>(joint * kParametersPerJoint);
    if (hasPositionChannels.at(joint)) {
      for (const auto channel : kPositionChannels) {
        write(jointMotion(base + positionParameterIndex(channel), frame));
      }
    }

    // The local rotation of a Momentum joint (see JointState): R = Rpre * Rz(rz) * Ry(ry) * Rx(rx)
    const Quaternionf q = skeleton.joints.at(joint).preRotation *
        Quaternionf(Eigen::AngleAxisf(jointMotion(base + RZ, frame), Vector3f::UnitZ())) *
        Quaternionf(Eigen::AngleAxisf(jointMotion(base + RY, frame), Vector3f::UnitY())) *
        Quaternionf(Eigen::AngleAxisf(jointMotion(base + RX, frame), Vector3f::UnitX()));
    const Vector3f angles = rotationMatrixToEuler(
        q.toRotationMatrix(),
        rotationAxisIndex(kRotationChannels[0]),
        rotationAxisIndex(kRotationChannels[1]),
        rotationAxisIndex(kRotationChannels[2]),
        EulerConvention::Intrinsic);
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      write(angles[axis] * kRadiansToDegrees);
    }
  }
  stream << "\n";
}

// The joint parameters (joint parameters x frames) of `motion`, the model parameters of
// `character` (model parameters x frames), with the identity `offsets` added as
// ParameterTransform::apply(CharacterParameters) adds them.
MatrixXf
toJointMotion(const Character& character, const MatrixXf& motion, const VectorXf& offsets) {
  const auto& parameterTransform = character.parameterTransform;
  MT_THROW_IF(
      motion.rows() != parameterTransform.numAllModelParameters(),
      "BVH export: the motion has {} rows, but the character has {} model parameters",
      motion.rows(),
      parameterTransform.numAllModelParameters());
  MT_THROW_IF(
      offsets.size() != 0 && offsets.size() != parameterTransform.numJointParameters(),
      "BVH export: the offsets have {} values, but the character has {} joint parameters",
      offsets.size(),
      parameterTransform.numJointParameters());
  MatrixXf jointMotion = parameterTransform.transform * motion;
  if (offsets.size() != 0) {
    jointMotion.colwise() += offsets;
  }
  return jointMotion;
}

// The text of a BVH file with `character` and its `jointMotion` (joint parameters x frames).
std::string bvhText(const Character& character, const MatrixXf& jointMotion, float fps) {
  const auto& skeleton = character.skeleton;
  MT_THROW_IF(skeleton.joints.empty(), "BVH export: the character has no joints");
  MT_THROW_IF(
      jointMotion.rows() != static_cast<Eigen::Index>(skeleton.joints.size() * kParametersPerJoint),
      "BVH export: the motion has {} joint parameters, but the skeleton has {}",
      jointMotion.rows(),
      skeleton.joints.size() * kParametersPerJoint);
  MT_THROW_IF(fps <= 0.0f, "BVH export: fps must be positive, not {}", fps);
  warnAboutDataThatIsNotSaved(character, jointMotion);

  std::ostringstream stream;
  const auto hasPositionChannels = jointsWithPositionChannels(skeleton, jointMotion);
  const auto order = writeBvhHierarchy(stream, skeleton, hasPositionChannels);
  stream << "MOTION\n";
  stream << "Frames: " << jointMotion.cols() << "\n";
  stream << "Frame Time: " << std::fixed << std::setprecision(6) << (1.0f / fps) << "\n";
  for (Eigen::Index frame = 0; frame < jointMotion.cols(); ++frame) {
    writeBvhFrame(stream, skeleton, jointMotion, frame, order, hasPositionChannels);
  }
  return stream.str();
}

// Writes the BVH `text` to `filepath`. Writing the text only when it is complete leaves the file
// unchanged when the export throws.
void writeBvhFile(const filesystem::path& filepath, const std::string& text) {
  std::ofstream stream(filepath, std::ios::binary);
  MT_THROW_IF(!stream.is_open(), "Failed to open BVH file for writing: {}", filepath.string());
  stream << text;
  stream.close();
  MT_THROW_IF(stream.fail(), "Failed to write BVH file: {}", filepath.string());
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
  // The rest pose: a single frame of zero model parameters
  const MatrixXf motion = MatrixXf::Zero(character.parameterTransform.numAllModelParameters(), 1);
  writeBvhFile(filepath, bvhText(character, toJointMotion(character, motion, VectorXf()), 30.0f));
}

void saveBvhCharacterWithMotion(
    const filesystem::path& filepath,
    const Character& character,
    const MotionParameters& motionParams,
    float fps,
    const VectorXf& offsets) {
  const auto& motion = std::get<1>(motionParams);
  writeBvhFile(filepath, bvhText(character, toJointMotion(character, motion, offsets), fps));
}

void saveBvhCharacterWithJointParams(
    const filesystem::path& filepath,
    const Character& character,
    const MatrixXf& jointParams,
    float fps) {
  writeBvhFile(filepath, bvhText(character, jointParams, fps));
}

std::vector<std::byte> saveBvhCharacterWithMotion(
    const Character& character,
    const MotionParameters& motionParams,
    float fps,
    const VectorXf& offsets) {
  const auto& motion = std::get<1>(motionParams);
  const std::string text = bvhText(character, toJointMotion(character, motion, offsets), fps);
  std::vector<std::byte> bytes(text.size());
  std::memcpy(bytes.data(), text.data(), text.size());
  return bytes;
}

} // namespace momentum
