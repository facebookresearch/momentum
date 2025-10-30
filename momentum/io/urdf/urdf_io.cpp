/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_io.h"
#include "momentum/math/constants.h"

#include <urdf_model/link.h>
#include <urdf_model/pose.h>
#include <urdf_parser/urdf_parser.h>

namespace momentum {

namespace {

template <typename T>
struct ParsingData {
  Skeleton skeleton;
  ParameterTransform parameterTransform;
  std::vector<Eigen::Triplet<float>> triplets;
  ParameterLimits limits;
  size_t totalDoFs = 0;
};

template <typename S>
[[nodiscard]] Vector3<S> toMomentumVector3(const urdf::Vector3& urdfVector3) {
  return Vector3<S>(urdfVector3.x, urdfVector3.y, urdfVector3.z);
}

template <typename S>
[[nodiscard]] Quaternion<S> toMomentumQuaternion(const urdf::Rotation& urdfRotation) {
  return Quaternion<S>(urdfRotation.w, urdfRotation.x, urdfRotation.y, urdfRotation.z);
}

template <typename S>
[[nodiscard]] TransformT<S> toMomentumTransform(const urdf::Pose& urdfPose) {
  return TransformT<S>(
      toMomentumVector3<S>(urdfPose.position), toMomentumQuaternion<S>(urdfPose.rotation));
}

[[nodiscard]] std::string_view toString(int jointType) {
  switch (jointType) {
    case urdf::Joint::REVOLUTE:
      return "REVOLUTE";
    case urdf::Joint::CONTINUOUS:
      return "CONTINUOUS";
    case urdf::Joint::PRISMATIC:
      return "PRISMATIC";
    case urdf::Joint::FLOATING:
      return "FLOATING";
    case urdf::Joint::PLANAR:
      return "PLANAR";
    default:
      return "UNKNOWN";
  }
}

template <typename T>
bool loadUrdfSkeletonRecursive(
    ParsingData<T>& data,
    size_t parentJointId,
    const Quaternionf& parentJointAxis,
    const urdf::ModelInterface* urdfModel,
    const urdf::Link* urdfLink) {
  MT_THROW_IF(urdfLink == nullptr, "URDF link is null.");

  //-------------
  // Parse joint
  //-------------

  auto* urdfJoint = urdfLink->parent_joint.get();
  MT_THROW_IF(
      parentJointId != kInvalidIndex && urdfJoint == nullptr,
      "URDF parent joint is null for a non root link ({}).",
      urdfLink->name);

  Joint joint;
  joint.name = urdfLink->name; // Use link name or joint name?
  joint.parent = parentJointId;

  const Eigen::Index jointId = data.skeleton.joints.size();
  const Eigen::Index jointParamsBaseIndex = jointId * kParametersPerJoint;
  const Eigen::Index modelParamsBaseIndex = data.totalDoFs;

  //---------------------------
  // Parse Parameter transform
  //---------------------------

  Quaternionf jointAxis = Quaternionf::Identity();

  if (urdfJoint != nullptr) {
    // Set parameter transform based on joint type
    switch (urdfJoint->type) {
      case urdf::Joint::PRISMATIC: {
        data.parameterTransform.name.push_back(fmt::format("joint{}_tx", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 0, modelParamsBaseIndex, 1.0f);
        data.totalDoFs += 1;
        break;
      }
      case urdf::Joint::REVOLUTE:
      case urdf::Joint::CONTINUOUS: {
        data.parameterTransform.name.push_back(fmt::format("joint{}_rx", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 3, modelParamsBaseIndex, 1.0f);
        data.totalDoFs += 1;
        break;
      }
      case urdf::Joint::FLOATING: {
        data.parameterTransform.name.push_back(fmt::format("joint{}_tx", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 0, modelParamsBaseIndex + 0, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_ty", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 1, modelParamsBaseIndex + 1, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_tz", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 2, modelParamsBaseIndex + 2, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_rx", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 3, modelParamsBaseIndex + 3, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_ry", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 4, modelParamsBaseIndex + 4, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_rz", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 5, modelParamsBaseIndex + 5, 1.0f);
        data.totalDoFs += 6;
        break;
      }
      case urdf::Joint::PLANAR: {
        data.parameterTransform.name.push_back(fmt::format("joint{}_tx", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 1, modelParamsBaseIndex + 0, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_ty", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 2, modelParamsBaseIndex + 1, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("joint{}_tz", jointId));
        data.triplets.emplace_back(jointParamsBaseIndex + 3, modelParamsBaseIndex + 2, 1.0f);
        data.totalDoFs += 3;
        break;
      }
      case urdf::Joint::FIXED: {
        // Do nothing
        break;
      }
      default: {
        MT_THROW("Unsupported joint type: {}", toString(urdfJoint->type));
      }
    }

    // Update pre-rotation based on joint axis
    switch (urdfJoint->type) {
      case urdf::Joint::REVOLUTE:
      case urdf::Joint::CONTINUOUS:
      case urdf::Joint::PRISMATIC:
      case urdf::Joint::PLANAR: {
        jointAxis = Quaternionf::FromTwoVectors(
            Vector3f::UnitX(), toMomentumVector3<float>(urdfJoint->axis));
        break;
      }
      default: {
        // Do nothing
        break;
      }
    }

    // Parse joint limits (required only for revolute and prismatic joint)
    if (auto urdfJointLimits = urdfJoint->limits) {
      switch (urdfJoint->type) {
        case urdf::Joint::REVOLUTE: {
          ParameterLimit jointLimits;
          jointLimits.type = MinMax;
          jointLimits.data.minMax.parameterIndex = modelParamsBaseIndex;
          jointLimits.data.minMax.limits[0] = urdfJointLimits->lower; // rad
          jointLimits.data.minMax.limits[1] = urdfJointLimits->upper;
          jointLimits.weight = 1.0f;
          data.limits.push_back(jointLimits);
          break;
        }
        case urdf::Joint::PRISMATIC: {
          ParameterLimit jointLimits;
          jointLimits.type = MinMax;
          jointLimits.data.minMax.parameterIndex = modelParamsBaseIndex;
          // meters in URDF, centimeters in Momentum
          jointLimits.data.minMax.limits[0] = toCm<float>(urdfJointLimits->lower);
          jointLimits.data.minMax.limits[1] = toCm<float>(urdfJointLimits->upper);
          jointLimits.weight = 1.0f;
          data.limits.push_back(jointLimits);
          break;
        }
        default: {
          // Do nothing
          break;
        }
      }
    }
  }

  // Set Joint Offset
  //
  // In Momentum, the joint transformation from parent to child is defined as:
  //   T(x; p) = T_offset(p) * T(x)
  // where:
  //   x represents the model parameters, corresponding to the URDF's joint angles.
  //   p represents the joint properties, parsed from the URDF joint properties.
  //
  // The URDF defines joint transformations differently, involving both link frames and joint
  // frames. The link frame transformation is defined as:
  //   T_child(x; p) = T_parent(x; p) * T_offset(p)
  // when x is assumed to be zero.
  //
  // The joint frame transformation is defined as:
  //   T_parent(x; p) * T_offset(p) * T_joint_axis(p)
  //
  // The final joint transformation from parent to child is:
  //   T_child = T_parent * T_offset(p) * T(x)
  //
  // However, T(x) is applied in the "joint frame," defined by:
  //   T_parent * T_joint_origin * T_joint_axis
  // rather than just:
  //   T_parent * T_joint_origin
  //
  // Therefore, the final joint transformation from parent to child is:
  //   T = (T_parent_axis(p))^(-1) * T_joint_origin(p) * T_axis(p) * T(x)
  //
  // To align with Momentum's convention, T_offset should be calculated as:
  //   T_offset = (T_parent_axis(p))^(-1) * T_joint_origin(p) * T_axis(p)
  //
  // For 1-DOF joints, such as revolute and prismatic joints, the joint frame is aligned with the
  // axis along its x-axis, according to the URDF convention. Similarly, for planar joints, the
  // plane normal is aligned with the x-axis.
  //
  // Reference: https://wiki.ros.org/urdf/XML/joint
  if (urdfJoint != nullptr) {
    const urdf::Pose& urdfPose = urdfJoint->parent_to_joint_origin_transform;
    joint.preRotation =
        parentJointAxis.inverse() * toMomentumQuaternion<float>(urdfPose.rotation) * jointAxis;
    joint.translationOffset =
        parentJointAxis.inverse() * toMomentumVector3<float>(urdfPose.position) * toCm<float>();
  } else {
    joint.preRotation = parentJointAxis.inverse() * jointAxis;
    joint.translationOffset.setZero();
  }

  data.skeleton.joints.push_back(joint);

  // Continue parsing child links and joints
  for (const auto& childLink : urdfLink->child_links) {
    if (!loadUrdfSkeletonRecursive(data, jointId, jointAxis, urdfModel, childLink.get())) {
      return false;
    }
  }

  return true;
}

template <typename T>
CharacterT<T> loadUrdfCharacterFromUrdfModel(urdf::ModelInterfaceSharedPtr urdfModel) {
  const urdf::Link* root = urdfModel->getRoot().get();
  MT_THROW_IF(!root, "Failed to parse URDF file from. No root link found.");

  ParsingData<T> data;

  // Special Case: If the root link is named "world", it is treated as a world link. In this case,
  // the actual root link is the first child link. Otherwise, the root link itself is considered the
  // actual root link, and it is assumed to have a fixed joint (hence, no degrees of freedom).
  if (root->name == "world") {
    if (root->child_links.empty()) {
      MT_THROW(
          "Failed to parse URDF. The world link must have at least one child link, but it has none.");
    } else if (root->child_links.size() > 1) {
      MT_THROW(
          "Failed to parse URDF. The world link must have only one child link, but it has {}.",

          root->child_links.size());
    }
    root = root->child_links[0].get();
  }

  if (!loadUrdfSkeletonRecursive(
          data, kInvalidIndex, Quaternionf::Identity(), urdfModel.get(), root)) {
    MT_THROW("Failed to parse URDF.");
  }

  Skeleton& skeleton = data.skeleton;
  ParameterTransform& parameterTransform = data.parameterTransform;
  auto& triplets = data.triplets;

  const size_t numJoints = skeleton.joints.size();
  const size_t numJointParameters = numJoints * kParametersPerJoint;
  const size_t numModelParameters = data.totalDoFs;
  parameterTransform.offsets.setZero(numJointParameters);
  parameterTransform.transform.resize(numJointParameters, numModelParameters);
  parameterTransform.transform.setFromTriplets(triplets.begin(), triplets.end());
  parameterTransform.activeJointParams = parameterTransform.computeActiveJointParams();

  // TODO: Parse collision geometries

  return CharacterT<T>(data.skeleton, data.parameterTransform, data.limits);
}

} // namespace

template <typename T>
CharacterT<T> loadUrdfCharacter(const filesystem::path& filepath) {
  urdf::ModelInterfaceSharedPtr urdfModel;

  try {
    urdfModel = urdf::parseURDFFile(filepath.string());
  } catch (const std::exception& e) {
    MT_THROW("Failed to parse URDF file from: {}. Error: {}", filepath.string(), e.what());
  } catch (...) {
    MT_THROW("Failed to parse URDF file from: {}", filepath.string());
  }

  try {
    return loadUrdfCharacterFromUrdfModel<T>(urdfModel);
  } catch (const std::exception& e) {
    MT_THROW(
        "Failed to create Character from URDF file: {}. Error: {}", filepath.string(), e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF file: {}", filepath.string());
  }
}

template CharacterT<float> loadUrdfCharacter(const filesystem::path& filepath);
template CharacterT<double> loadUrdfCharacter(const filesystem::path& filepath);

template <typename T>
CharacterT<T> loadUrdfCharacter(std::span<const std::byte> bytes) {
  urdf::ModelInterfaceSharedPtr urdfModel;

  try {
    // Convert the byte span to a string
    std::string urdfString(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    // Parse the URDF from the string
    urdfModel = urdf::parseURDF(urdfString);
  } catch (const std::exception& e) {
    MT_THROW("Failed to read URDF file from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to read URDF file from URDF bytes");
  }

  try {
    return loadUrdfCharacterFromUrdfModel<T>(urdfModel);
  } catch (const std::exception& e) {
    MT_THROW("Failed to create Character from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF bytes");
  }
}

template CharacterT<float> loadUrdfCharacter(std::span<const std::byte> bytes);
template CharacterT<double> loadUrdfCharacter(std::span<const std::byte> bytes);

} // namespace momentum
