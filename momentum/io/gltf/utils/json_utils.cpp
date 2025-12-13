/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/utils/json_utils.h"

#include "momentum/character/character.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <nlohmann/json.hpp>

namespace momentum {

void parameterSetsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::object();
  for (const auto& set : character.parameterTransform.parameterSets) {
    j[set.first] = nlohmann::json::array();
    for (size_t i = 0; i < character.parameterTransform.name.size(); i++) {
      if (set.second.test(i)) {
        j[set.first].push_back(character.parameterTransform.name[i]);
      }
    }
  }
}

ParameterSets parameterSetsFromJson(const Character& character, const nlohmann::json& j) {
  ParameterSets result;

  for (const auto& parameterSet : j.items()) {
    auto& set = result[parameterSet.key()];
    for (const auto& parameter : parameterSet.value().items()) {
      const auto parameterIndex =
          character.parameterTransform.getParameterIdByName(parameter.value());

      if (parameterIndex != kInvalidIndex) {
        set.set(parameterIndex);
      }
    }
  }

  return result;
}

void poseConstraintsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::object();
  for (const auto& csts : character.parameterTransform.poseConstraints) {
    for (const auto& pc : csts.second.parameterIdValue) {
      if (!character.parameterTransform.name.empty() &&
          pc.first < character.parameterTransform.name.size()) {
        j[csts.first][character.parameterTransform.name[pc.first]] = pc.second;
      }
    }
  }
}

PoseConstraints poseConstraintsFromJson(const Character& character, const nlohmann::json& j) {
  PoseConstraints result;

  for (const auto& constraint : j.items()) {
    auto& pose = result[constraint.key()];
    for (const auto& constraintPair : constraint.value().items()) {
      const auto parameterIndex =
          character.parameterTransform.getParameterIdByName(constraintPair.key());

      const float value = constraintPair.value();
      if (parameterIndex != kInvalidIndex) {
        pose.parameterIdValue.emplace_back(parameterIndex, value);
      }
    }
  }

  return result;
}

void parameterTransformToJson(const Character& character, nlohmann::json& j) {
  j["parameters"] = character.parameterTransform.name;

  for (size_t jt = 0; jt < character.skeleton.joints.size(); jt++) {
    const auto& jointName = character.skeleton.joints[jt].name;
    const size_t paramIndex = jt * kParametersPerJoint;
    for (size_t d = 0; d < kParametersPerJoint; d++) {
      for (auto index = character.parameterTransform.transform.outerIndexPtr()[paramIndex + d];
           index < character.parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        const auto& parameterIndex = character.parameterTransform.transform.innerIndexPtr()[index];
        const auto& parameterValue = character.parameterTransform.transform.valuePtr()[index];

        if (parameterIndex < character.parameterTransform.name.size()) {
          j["joints"][jointName][kJointParameterNames[d]]
           [character.parameterTransform.name[parameterIndex]] = parameterValue;
        }
      }
    }
  }
  // write blendshape parameters
  const auto blendShapeParameters = character.parameterTransform.blendShapeParameters;
  const auto faceExpressionParameters = character.parameterTransform.faceExpressionParameters;
  auto bs = nlohmann::json::array();
  for (int i = 0; i < blendShapeParameters.size(); i++) {
    bs.push_back(blendShapeParameters[i]);
  }
  auto fe = nlohmann::json::array();
  for (int i = 0; i < faceExpressionParameters.size(); i++) {
    fe.push_back(faceExpressionParameters[i]);
  }
  j["blendShapeParameters"] = bs;
  j["faceExpressionParameters"] = fe;
}

ParameterTransform parameterTransformFromJson(const Character& character, const nlohmann::json& j) {
  ParameterTransform pt;
  try {
    pt.activeJointParams =
        VectorX<bool>::Constant(character.skeleton.joints.size() * kParametersPerJoint, false);
    pt.offsets = VectorXf::Zero(character.skeleton.joints.size() * kParametersPerJoint);

    // load all the parameter names
    auto parameterItr = j.find("parameters");
    MT_THROW_IF(parameterItr == j.end(), "No 'parameters' found in parameter transform.");

    for (const auto& p : *parameterItr) {
      pt.name.push_back(p.get<std::string>());
    }

    // triplet list
    std::vector<Eigen::Triplet<float>> triplets;

    auto jointItr = j.find("joints");
    MT_THROW_IF(jointItr == j.end(), "No 'joints' found in parameter transform.");

    for (const auto& joint : jointItr->items()) {
      const std::string& jointName = joint.key();

      // get the right joint to modify
      const size_t jointIndex = character.skeleton.getJointIdByName(jointName);
      MT_THROW_IF(jointIndex == kInvalidIndex, "Unknown joint name in expression : {}", jointName);

      for (const auto& jointParameter : joint.value().items()) {
        const std::string& jointParameterName = jointParameter.key();

        // the first pToken is the name of the joint and it's attribute type
        size_t attributeIndex = kInvalidIndex;
        for (size_t jt = 0; jt < kParametersPerJoint; jt++) {
          if (jointParameterName == kJointParameterNames[jt]) {
            attributeIndex = jt;
            break;
          }
        }

        // if we didn't find a right name exit with an error
        MT_THROW_IF(
            attributeIndex == kInvalidIndex,
            "Unknown channel name in expression : {}",
            jointParameterName);

        // enable the attribute in the skeleton if we have a parameter controlling it
        pt.activeJointParams[jointIndex * kParametersPerJoint + attributeIndex] = 1;

        // split the parameter names
        for (const auto& parameter : jointParameter.value().items()) {
          // get weight and name
          const float weight = parameter.value();
          const std::string& parameterName = parameter.key();

          auto parameterIndex = pt.getParameterIdByName(parameterName);

          MT_THROW_IF(
              parameterIndex == kInvalidIndex,
              "Unknown parameter name in expression : {}",
              parameterName);

          // add triplet
          triplets.emplace_back(
              static_cast<int>(jointIndex) * kParametersPerJoint + static_cast<int>(attributeIndex),
              static_cast<int>(parameterIndex),
              weight);
        }
      }
    }

    // resize the Transform matrix to the correct size
    pt.transform.resize(
        static_cast<int>(character.skeleton.joints.size()) * kParametersPerJoint,
        static_cast<int>(pt.name.size()));

    // finish parameter setup by creating sparse matrix
    pt.transform.setFromTriplets(triplets.begin(), triplets.end());
  } catch (...) {
    // create an identity parameter transform
    pt = ParameterTransform::identity(character.skeleton.getJointNames());
  }

  // read blendshape parameters
  auto blendShapeParameters = j.find("blendShapeParameters");
  if (blendShapeParameters != j.end()) {
    pt.blendShapeParameters.resize(blendShapeParameters->size());
    for (int i = 0; i < blendShapeParameters->size(); i++) {
      pt.blendShapeParameters[i] = (*blendShapeParameters)[i].get<int>();
    }
  } else {
    pt.blendShapeParameters.resize(0);
  }
  auto faceExpressionParameters = j.find("faceExpressionParameters");
  if (faceExpressionParameters != j.end()) {
    pt.faceExpressionParameters.resize(faceExpressionParameters->size());
    for (int i = 0; i < faceExpressionParameters->size(); i++) {
      pt.faceExpressionParameters[i] = (*faceExpressionParameters)[i].get<int>();
    }
  } else {
    pt.faceExpressionParameters.resize(0);
  }

  return pt;
}

void parameterLimitsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::array();
  for (const auto& lim : character.parameterLimits) {
    j.push_back(nlohmann::json::object());
    auto& li = j.back();
    li["weight"] = lim.weight;

    switch (lim.type) {
      case MinMax:
        li["type"] = "minmax";
        MT_THROW_IF(
            character.parameterTransform.name.empty() ||
                lim.data.minMax.parameterIndex >= character.parameterTransform.name.size(),
            "MinMax parameter index {} is out of bounds (name size: {})",
            lim.data.minMax.parameterIndex,
            character.parameterTransform.name.size());
        li["parameter"] = character.parameterTransform.name[lim.data.minMax.parameterIndex];
        li["limits"] = lim.data.minMax.limits;
        break;
      case MinMaxJoint:
        li["type"] = "minmax_joint";
        if (!character.skeleton.joints.empty() &&
            lim.data.minMaxJoint.jointIndex < character.skeleton.joints.size()) {
          li["jointIndex"] = character.skeleton.joints[lim.data.minMaxJoint.jointIndex].name;
        }
        li["jointParameter"] = kJointParameterNames[lim.data.minMaxJoint.jointParameter];
        li["limits"] = lim.data.minMaxJoint.limits;
        break;
      case MinMaxJointPassive:
        li["type"] = "minmax_joint_passive";
        if (!character.skeleton.joints.empty() &&
            lim.data.minMaxJoint.jointIndex < character.skeleton.joints.size()) {
          li["jointIndex"] = character.skeleton.joints[lim.data.minMaxJoint.jointIndex].name;
        }
        li["jointParameter"] = kJointParameterNames[lim.data.minMaxJoint.jointParameter];
        li["limits"] = lim.data.minMaxJoint.limits;
        break;
      case Linear:
        li["type"] = "linear";
        MT_THROW_IF(
            character.parameterTransform.name.empty() ||
                lim.data.linear.referenceIndex >= character.parameterTransform.name.size(),
            "Linear referenceIndex {} is out of bounds (name size: {})",
            lim.data.linear.referenceIndex,
            character.parameterTransform.name.size());
        li["referenceParameter"] =
            character.parameterTransform.name[lim.data.linear.referenceIndex];
        MT_THROW_IF(
            character.parameterTransform.name.empty() ||
                lim.data.linear.targetIndex >= character.parameterTransform.name.size(),
            "Linear targetIndex {} is out of bounds (name size: {})",
            lim.data.linear.targetIndex,
            character.parameterTransform.name.size());
        li["targetParameter"] = character.parameterTransform.name[lim.data.linear.targetIndex];
        li["scale"] = lim.data.linear.scale;
        li["offset"] = lim.data.linear.offset;
        if (lim.data.linear.rangeMin != -std::numeric_limits<float>::max()) {
          li["rangeMin"] = lim.data.linear.rangeMin;
        }
        if (lim.data.linear.rangeMax != std::numeric_limits<float>::max()) {
          li["rangeMax"] = lim.data.linear.rangeMax;
        }
        break;
      case LinearJoint:
        li["type"] = "linear_joint";
        if (!character.skeleton.joints.empty() &&
            lim.data.linearJoint.referenceJointIndex < character.skeleton.joints.size()) {
          li["referenceJoint"] =
              character.skeleton.joints[lim.data.linearJoint.referenceJointIndex].name;
        }
        li["referenceJointParameter"] = lim.data.linearJoint.referenceJointParameter;
        if (!character.skeleton.joints.empty() &&
            lim.data.linearJoint.targetJointIndex < character.skeleton.joints.size()) {
          li["targetJoint"] = character.skeleton.joints[lim.data.linearJoint.targetJointIndex].name;
        }
        li["targetJointParameter"] = lim.data.linearJoint.targetJointParameter;
        li["scale"] = lim.data.linearJoint.scale;
        li["offset"] = lim.data.linearJoint.offset;
        if (lim.data.linearJoint.rangeMin != -std::numeric_limits<float>::max()) {
          li["rangeMin"] = lim.data.linearJoint.rangeMin;
        }
        if (lim.data.linearJoint.rangeMax != std::numeric_limits<float>::max()) {
          li["rangeMax"] = lim.data.linearJoint.rangeMax;
        }
        break;
      case HalfPlane:
        li["type"] = "half_plane";
        MT_THROW_IF(
            character.parameterTransform.name.empty() ||
                lim.data.halfPlane.param1 >= character.parameterTransform.name.size(),
            "HalfPlane param1 index {} is out of bounds (name size: {})",
            lim.data.halfPlane.param1,
            character.parameterTransform.name.size());
        li["param1"] = character.parameterTransform.name[lim.data.halfPlane.param1];
        MT_THROW_IF(
            character.parameterTransform.name.empty() ||
                lim.data.halfPlane.param2 >= character.parameterTransform.name.size(),
            "HalfPlane param2 index {} is out of bounds (name size: {})",
            lim.data.halfPlane.param2,
            character.parameterTransform.name.size());
        li["param2"] = character.parameterTransform.name[lim.data.halfPlane.param2];
        li["normal"] = lim.data.halfPlane.normal;
        li["offset"] = lim.data.halfPlane.offset;
        break;
      case Ellipsoid: {
        li["type"] = "ellipsoid";
        if (!character.skeleton.joints.empty() &&
            lim.data.ellipsoid.parent < character.skeleton.joints.size()) {
          li["parent"] = character.skeleton.joints[lim.data.ellipsoid.parent].name;
        }
        if (!character.skeleton.joints.empty() &&
            lim.data.ellipsoid.ellipsoidParent < character.skeleton.joints.size()) {
          li["ellipsoidParent"] =
              character.skeleton.joints[lim.data.ellipsoid.ellipsoidParent].name;
        }
        li["offset"] = lim.data.ellipsoid.offset * toM();
        auto eli = lim.data.ellipsoid.ellipsoid;
        eli.translation() *= toM();
        toJson(eli.matrix(), li["ellipsoid"]);
        break;
      }
      default: {
        MT_LOGE(
            "Unknown parameter limit type '{}' from character name '{}'",
            toString(lim.type),
            character.name);
        break;
      }
    }
  }
}

ParameterLimits parameterLimitsFromJson(const Character& character, const nlohmann::json& j) {
  ParameterLimits result;

  for (const auto& element : j) {
    const std::string type = element.value("type", "");
    ParameterLimit l;
    l.weight = element.value("weight", 0.0f);
    if (type == "minmax") {
      l.type = MinMax;
      l.data.minMax.parameterIndex =
          character.parameterTransform.getParameterIdByName(element.value("parameter", ""));
      l.data.minMax.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "minmax_joint") {
      l.type = MinMaxJoint;
      l.data.minMaxJoint.jointIndex =
          character.skeleton.getJointIdByName(element.value("jointIndex", ""));
      const std::string attribute = element.value("jointParameter", "");
      size_t attributeIndex = kInvalidIndex;
      for (size_t t = 0; t < kParametersPerJoint; t++) {
        if (attribute == kJointParameterNames[t]) {
          attributeIndex = t;
          break;
        }
      }
      l.data.minMaxJoint.jointParameter = attributeIndex;
      l.data.minMaxJoint.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "minmax_joint_passive") {
      l.type = MinMaxJointPassive;
      l.data.minMaxJoint.jointIndex =
          character.skeleton.getJointIdByName(element.value("jointIndex", ""));
      const std::string attribute = element.value("jointParameter", "");
      size_t attributeIndex = kInvalidIndex;
      for (size_t t = 0; t < kParametersPerJoint; t++) {
        if (attribute == kJointParameterNames[t]) {
          attributeIndex = t;
          break;
        }
      }
      l.data.minMaxJoint.jointParameter = attributeIndex;
      l.data.minMaxJoint.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "linear") {
      l.type = Linear;
      l.data.linear.referenceIndex = character.parameterTransform.getParameterIdByName(
          element.value("referenceParameter", ""));
      l.data.linear.targetIndex =
          character.parameterTransform.getParameterIdByName(element.value("targetParameter", ""));
      l.data.linear.scale = element["scale"];
      l.data.linear.offset = element["offset"];
      l.data.linear.rangeMin = element.value("rangeMin", -std::numeric_limits<float>::max());
      l.data.linear.rangeMax = element.value("rangeMax", std::numeric_limits<float>::max());
    } else if (type == "linear_joint") {
      l.type = LinearJoint;
      l.data.linearJoint.referenceJointIndex =
          character.skeleton.getJointIdByName(element.value("referenceJoint", ""));
      l.data.linearJoint.targetJointIndex =
          character.skeleton.getJointIdByName(element.value("targetJoint", ""));
      l.data.linearJoint.referenceJointParameter = element["referenceJointParameter"].get<size_t>();
      l.data.linearJoint.targetJointParameter = element["targetJointParameter"].get<size_t>();
      l.data.linearJoint.scale = element["scale"];
      l.data.linearJoint.offset = element["offset"];
      l.data.linearJoint.rangeMin = element.value("rangeMin", -std::numeric_limits<float>::max());
      l.data.linearJoint.rangeMax = element.value("rangeMax", std::numeric_limits<float>::max());
    } else if (type == "half_plane") {
      l.type = HalfPlane;
      l.data.halfPlane.param1 =
          character.parameterTransform.getParameterIdByName(element.value("param1", ""));
      l.data.halfPlane.param2 =
          character.parameterTransform.getParameterIdByName(element.value("param2", ""));
      l.data.halfPlane.normal = fromJson<Vector2f>(element["normal"]);
      l.data.halfPlane.offset = element["offset"];
    } else if (type == "ellipsoid") {
      l.type = Ellipsoid;
      l.data.ellipsoid.parent = character.skeleton.getJointIdByName(element.value("parent", ""));
      l.data.ellipsoid.ellipsoidParent =
          character.skeleton.getJointIdByName(element.value("ellipsoidParent", ""));
      l.data.ellipsoid.offset = fromJson<Vector3f>(element["offset"]);
      l.data.ellipsoid.offset /= toM();
      l.data.ellipsoid.ellipsoid.matrix() = fromJson<Matrix4f>(element["ellipsoid"]);
      l.data.ellipsoid.ellipsoid.translation() /= toM();
      l.data.ellipsoid.ellipsoidInv = l.data.ellipsoid.ellipsoid.inverse();
    } else if (type == "elipsoid") {
      // TODO: Remove once all the model files are migrated to ellipsoid
      MT_LOGW_ONCE(
          "Deprecated parameter limit type: {} (typo). Please use 'ellipsoid' instead.", type);
      l.type = Ellipsoid;
      l.data.ellipsoid.parent = character.skeleton.getJointIdByName(element.value("parent", ""));
      l.data.ellipsoid.ellipsoidParent =
          character.skeleton.getJointIdByName(element.value("elipsoidParent", ""));
      l.data.ellipsoid.offset = fromJson<Vector3f>(element["offset"]);
      l.data.ellipsoid.offset /= toM();
      l.data.ellipsoid.ellipsoid.matrix() = fromJson<Matrix4f>(element["elipsoid"]);
      l.data.ellipsoid.ellipsoid.translation() /= toM();
      l.data.ellipsoid.ellipsoidInv = l.data.ellipsoid.ellipsoid.inverse();
    } else {
      MT_THROW("Unknown parameter limit type '{}' from character name '{}'.", type, character.name);
    }
    result.push_back(l);
  }

  return result;
}

} // namespace momentum
