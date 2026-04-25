/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_io.h"
#include "momentum/io/urdf/urdf_mesh_io.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"

#include <urdf_model/link.h>
#include <urdf_model/model.h>
#include <urdf_model/pose.h>
#include <urdf_parser/urdf_parser.h>

#include <unordered_map>

namespace momentum {

namespace {

/// Per-link visual data collected during skeleton parsing.
struct LinkVisualData {
  size_t jointIndex{};
  std::vector<urdf::VisualSharedPtr> visuals;
};

/// Deferred mimic joint data, resolved after the full skeleton is built.
struct MimicData {
  std::string mimickedJointName; // URDF joint name to mimic
  float multiplier{};
  float offset{}; // radians for revolute, meters for prismatic
  Eigen::Index jointParamsBaseIndex{}; // this joint's parameter base index
  int axisIdx{}; // principal axis index (0=X, 1=Y, 2=Z)
  float sign{}; // axis sign (+1/-1)
  bool isRotation{}; // true for revolute/continuous, false for prismatic
};

template <typename T>
struct ParsingData {
  Skeleton skeleton;
  ParameterTransform parameterTransform;
  std::vector<Eigen::Triplet<float>> triplets;
  ParameterLimits limits;
  size_t totalDoFs = 0;

  std::vector<LinkVisualData> linkVisuals;

  /// Maps URDF joint name → model parameter index (for mimic joint resolution).
  std::unordered_map<std::string, Eigen::Index> urdfJointNameToModelParamIdx;

  /// Mimic joints deferred until all joints are processed.
  std::vector<MimicData> mimicJoints;
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

/// Returns the principal axis index (0=X, 1=Y, 2=Z) and sign (+1/-1) for a URDF axis vector.
///
/// The imported Momentum node represents the child link carrying its incoming URDF joint. Since
/// the local frame is kept equal to the URDF child link frame, we do not spend preRotation on
/// aligning arbitrary URDF axes to a canonical axis. Instead, we only accept URDF axes already
/// aligned with a principal axis and map the DOF onto the matching Momentum parameter.
///
/// For negative axes (e.g., (0,0,-1)), the sign tracks how the URDF scalar joint coordinate maps
/// into the preserved child-link frame's local principal-axis parameter. The imported model
/// parameter itself stays equal to the URDF joint scalar, so limits remain in the same coordinate.
/// This also means the exposed model parameter is not always numerically equal to the local
/// positive-axis rx/ry/rz slot used by Momentum FK; for a negative axis, the local slot receives
/// the negated value through the parameter transform.
///
/// @return A pair of (axis index, sign): axis index 0=X, 1=Y, 2=Z; sign is +1 or -1.
/// @throws If the axis is not aligned with a principal axis (e.g., (0.707, 0, 0.707)).
[[nodiscard]] std::pair<int, float> getPrincipalAxis(const urdf::Vector3& axis) {
  const auto ax = static_cast<float>(axis.x);
  const auto ay = static_cast<float>(axis.y);
  const auto az = static_cast<float>(axis.z);
  constexpr float kTol = 1e-4f;

  if (std::abs(std::abs(ax) - 1.0f) < kTol && std::abs(ay) < kTol && std::abs(az) < kTol) {
    return {0, ax > 0 ? 1.0f : -1.0f};
  }
  if (std::abs(ax) < kTol && std::abs(std::abs(ay) - 1.0f) < kTol && std::abs(az) < kTol) {
    return {1, ay > 0 ? 1.0f : -1.0f};
  }
  if (std::abs(ax) < kTol && std::abs(ay) < kTol && std::abs(std::abs(az) - 1.0f) < kTol) {
    return {2, az > 0 ? 1.0f : -1.0f};
  }

  MT_THROW(
      "Non-principal joint axis ({}, {}, {}) is not supported. "
      "Only axes aligned with X, Y, or Z are supported.",
      ax,
      ay,
      az);
}

/// Loads a single mesh from a URDF visual geometry element.
///
/// For mesh file references, resolves the path and loads STL or OBJ files.
/// For primitive geometries (box, cylinder, sphere), generates the mesh procedurally.
/// Returns std::nullopt if the geometry type is unsupported or the mesh file cannot be loaded.
std::optional<Mesh> loadVisualMesh(
    const urdf::Geometry& geometry,
    const filesystem::path& urdfDir) {
  switch (geometry.type) {
    case urdf::Geometry::MESH: {
      const auto& urdfMesh = dynamic_cast<const urdf::Mesh&>(geometry);
      if (urdfDir.empty()) {
        MT_LOGW("Cannot resolve mesh path without URDF directory: {}", urdfMesh.filename);
        return std::nullopt;
      }
      const auto resolvedPath = resolveMeshPath(urdfMesh.filename, urdfDir);
      if (!resolvedPath) {
        return std::nullopt;
      }

      const auto ext = resolvedPath->extension().string();
      std::optional<Mesh> mesh;
      if (ext == ".stl" || ext == ".STL") {
        mesh = loadStlMesh(*resolvedPath);
      } else if (ext == ".obj" || ext == ".OBJ") {
        mesh = loadObjMesh(*resolvedPath);
      } else {
        MT_LOGW("Unsupported mesh format '{}': {}", ext, resolvedPath->string());
        return std::nullopt;
      }

      if (!mesh) {
        return std::nullopt;
      }

      // Apply scale from URDF mesh element
      const Eigen::Vector3f scale(
          static_cast<float>(urdfMesh.scale.x),
          static_cast<float>(urdfMesh.scale.y),
          static_cast<float>(urdfMesh.scale.z));
      if (!scale.isApprox(Eigen::Vector3f::Ones())) {
        for (auto& v : mesh->vertices) {
          v = v.cwiseProduct(scale);
        }
      }
      return mesh;
    }
    case urdf::Geometry::BOX: {
      const auto& box = dynamic_cast<const urdf::Box&>(geometry);
      return generateBoxMesh(
          static_cast<float>(box.dim.x),
          static_cast<float>(box.dim.y),
          static_cast<float>(box.dim.z));
    }
    case urdf::Geometry::CYLINDER: {
      const auto& cyl = dynamic_cast<const urdf::Cylinder&>(geometry);
      return generateCylinderMesh(static_cast<float>(cyl.radius), static_cast<float>(cyl.length));
    }
    case urdf::Geometry::SPHERE: {
      const auto& sphere = dynamic_cast<const urdf::Sphere&>(geometry);
      return generateSphereMesh(static_cast<float>(sphere.radius));
    }
    default:
      MT_LOGW("Unknown URDF geometry type: {}", static_cast<int>(geometry.type));
      return std::nullopt;
  }
}

constexpr std::array<const char*, 3> kTranslationNames = {"tx", "ty", "tz"};
constexpr std::array<const char*, 3> kRotationNames = {"rx", "ry", "rz"};

template <typename T>
bool loadUrdfSkeletonRecursive(
    ParsingData<T>& data,
    size_t parentJointId,
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
  // The imported Momentum node is identified by the child link. It stores the incoming URDF
  // joint that moves that link, rather than creating a separate "joint frame" node.
  joint.name = urdfLink->name;
  joint.parent = parentJointId;

  const Eigen::Index jointId = data.skeleton.joints.size();
  const Eigen::Index jointParamsBaseIndex = jointId * kParametersPerJoint;
  const Eigen::Index modelParamsBaseIndex = data.totalDoFs;

  //---------------------------
  // Parse Parameter transform
  //---------------------------
  //
  // URDF expresses motion on a joint between links, but attached data is authored in the child
  // link frame. Our imported Momentum node therefore keeps the child link frame as its local
  // frame and stores the incoming URDF joint's DOF on that node. We do not introduce a separate
  // internal axis-alignment frame.
  //
  // As a result, we map the URDF DOF directly to the corresponding principal-axis parameter
  // (rx/ry/rz for revolute, tx/ty/tz for prismatic) inside the preserved link frame. For
  // negative URDF axes, the parameter transform carries a negative coefficient into the local
  // principal axis so the imported model parameter remains the URDF joint scalar coordinate
  // rather than the local positive-axis rotation used internally by Momentum FK.

  if (urdfJoint != nullptr) {
    // Check if this joint mimics another joint. Mimic joints don't create their own model
    // parameter — they reuse the mimicked joint's parameter with a multiplier and offset.
    const bool isMimic = urdfJoint->mimic != nullptr;

    switch (urdfJoint->type) {
      case urdf::Joint::PRISMATIC: {
        const auto [axisIdx, sign] = getPrincipalAxis(urdfJoint->axis);
        if (isMimic) {
          MimicData mimic;
          mimic.mimickedJointName = urdfJoint->mimic->joint_name;
          mimic.multiplier = static_cast<float>(urdfJoint->mimic->multiplier);
          mimic.offset = static_cast<float>(urdfJoint->mimic->offset);
          mimic.jointParamsBaseIndex = jointParamsBaseIndex;
          mimic.axisIdx = axisIdx;
          mimic.sign = sign;
          mimic.isRotation = false;
          data.mimicJoints.push_back(mimic);
        } else {
          data.parameterTransform.name.push_back(urdfJoint->name);
          data.triplets.emplace_back(jointParamsBaseIndex + axisIdx, modelParamsBaseIndex, sign);
          data.urdfJointNameToModelParamIdx[urdfJoint->name] = modelParamsBaseIndex;
          data.totalDoFs += 1;
        }
        break;
      }
      case urdf::Joint::REVOLUTE:
      case urdf::Joint::CONTINUOUS: {
        const auto [axisIdx, sign] = getPrincipalAxis(urdfJoint->axis);
        if (isMimic) {
          MimicData mimic;
          mimic.mimickedJointName = urdfJoint->mimic->joint_name;
          mimic.multiplier = static_cast<float>(urdfJoint->mimic->multiplier);
          mimic.offset = static_cast<float>(urdfJoint->mimic->offset);
          mimic.jointParamsBaseIndex = jointParamsBaseIndex;
          mimic.axisIdx = axisIdx;
          mimic.sign = sign;
          mimic.isRotation = true;
          data.mimicJoints.push_back(mimic);
        } else {
          data.parameterTransform.name.push_back(urdfJoint->name);
          data.triplets.emplace_back(
              jointParamsBaseIndex + 3 + axisIdx, modelParamsBaseIndex, sign);
          data.urdfJointNameToModelParamIdx[urdfJoint->name] = modelParamsBaseIndex;
          data.totalDoFs += 1;
        }
        break;
      }
      case urdf::Joint::FLOATING: {
        // Floating joints have 6 DOFs; use the URDF joint name with axis suffixes.
        // Translation parameters use a 10x scaling factor so that a unit change in
        // model parameter space produces 10 cm of translation. This makes translation
        // easier to change relative to rotation in optimization.
        constexpr float kFloatingTranslationScale = 10.0f;
        const auto& jn = urdfJoint->name;
        data.parameterTransform.name.push_back(fmt::format("{}_tx", jn));
        data.triplets.emplace_back(
            jointParamsBaseIndex + 0, modelParamsBaseIndex + 0, kFloatingTranslationScale);
        data.parameterTransform.name.push_back(fmt::format("{}_ty", jn));
        data.triplets.emplace_back(
            jointParamsBaseIndex + 1, modelParamsBaseIndex + 1, kFloatingTranslationScale);
        data.parameterTransform.name.push_back(fmt::format("{}_tz", jn));
        data.triplets.emplace_back(
            jointParamsBaseIndex + 2, modelParamsBaseIndex + 2, kFloatingTranslationScale);
        data.parameterTransform.name.push_back(fmt::format("{}_rx", jn));
        data.triplets.emplace_back(jointParamsBaseIndex + 3, modelParamsBaseIndex + 3, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("{}_ry", jn));
        data.triplets.emplace_back(jointParamsBaseIndex + 4, modelParamsBaseIndex + 4, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("{}_rz", jn));
        data.triplets.emplace_back(jointParamsBaseIndex + 5, modelParamsBaseIndex + 5, 1.0f);
        data.totalDoFs += 6;
        break;
      }
      case urdf::Joint::PLANAR: {
        // The axis defines the plane normal. DOFs are translation along the two in-plane
        // axes and rotation around the normal.
        const int normalIdx = getPrincipalAxis(urdfJoint->axis).first;
        const int planeAxis0 = (normalIdx + 1) % 3;
        const int planeAxis1 = (normalIdx + 2) % 3;
        const auto& jn = urdfJoint->name;
        data.parameterTransform.name.push_back(
            fmt::format("{}_{}", jn, kTranslationNames[planeAxis0]));
        data.triplets.emplace_back(
            jointParamsBaseIndex + planeAxis0, modelParamsBaseIndex + 0, 1.0f);
        data.parameterTransform.name.push_back(
            fmt::format("{}_{}", jn, kTranslationNames[planeAxis1]));
        data.triplets.emplace_back(
            jointParamsBaseIndex + planeAxis1, modelParamsBaseIndex + 1, 1.0f);
        data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kRotationNames[normalIdx]));
        data.triplets.emplace_back(
            jointParamsBaseIndex + 3 + normalIdx, modelParamsBaseIndex + 2, 1.0f);
        data.totalDoFs += 3;
        break;
      }
      case urdf::Joint::FIXED: {
        // No degrees of freedom
        break;
      }
      default: {
        MT_THROW("Unsupported joint type: {}", toString(urdfJoint->type));
      }
    }

    // Parse joint limits (required only for non-mimic revolute and prismatic joints).
    // Mimic joints are driven by the mimicked joint's parameter, so their limits don't
    // correspond to an independent model parameter.
    if (!isMimic && urdfJoint->limits) {
      auto urdfJointLimits = urdfJoint->limits;
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
  // The preRotation is simply the URDF joint origin rotation — the transform from the parent
  // link frame to the child link frame at bind pose. No extra axis-alignment rotation is baked
  // in, because the imported Momentum node is meant to coincide with the URDF child link frame.
  //
  // If we axis-aligned the local frame here, every quantity authored in the child link frame
  // (visuals, collisions, inertials, descendant joint origins, external animation data) would
  // need an additional frame conversion.
  if (urdfJoint != nullptr) {
    const urdf::Pose& urdfPose = urdfJoint->parent_to_joint_origin_transform;
    joint.preRotation = toMomentumQuaternion<float>(urdfPose.rotation);
    joint.translationOffset = toMomentumVector3<float>(urdfPose.position) * toCm<float>();
  } else {
    joint.preRotation = Quaternionf::Identity();
    joint.translationOffset.setZero();
  }

  data.skeleton.joints.push_back(joint);

  // Collect visual elements for mesh loading
  if (!urdfLink->visual_array.empty()) {
    LinkVisualData visualData;
    visualData.jointIndex = static_cast<size_t>(jointId);
    visualData.visuals = urdfLink->visual_array;
    data.linkVisuals.push_back(std::move(visualData));
  }

  // Continue parsing child links and joints
  for (const auto& childLink : urdfLink->child_links) {
    if (!loadUrdfSkeletonRecursive(data, jointId, urdfModel, childLink.get())) {
      return false;
    }
  }

  return true;
}

/// Resolves mimic joints by mapping each mimic joint's DOF to the mimicked joint's model
/// parameter via the parameter transform.
template <typename T>
void resolveMimicJoints(ParsingData<T>& data) {
  for (const auto& mimic : data.mimicJoints) {
    auto it = data.urdfJointNameToModelParamIdx.find(mimic.mimickedJointName);
    if (it == data.urdfJointNameToModelParamIdx.end()) {
      MT_LOGW("Mimic joint references unknown joint '{}', skipping.", mimic.mimickedJointName);
      continue;
    }
    const Eigen::Index mimickedModelParamIdx = it->second;
    const float coefficient = mimic.sign * mimic.multiplier;

    if (mimic.isRotation) {
      data.triplets.emplace_back(
          mimic.jointParamsBaseIndex + 3 + mimic.axisIdx, mimickedModelParamIdx, coefficient);
      data.parameterTransform.offsets[mimic.jointParamsBaseIndex + 3 + mimic.axisIdx] =
          mimic.sign * mimic.offset;
    } else {
      data.triplets.emplace_back(
          mimic.jointParamsBaseIndex + mimic.axisIdx, mimickedModelParamIdx, coefficient);
      data.parameterTransform.offsets[mimic.jointParamsBaseIndex + mimic.axisIdx] =
          mimic.sign * toCm<float>(mimic.offset);
    }
  }
}

/// Extracts per-vertex color from a URDF visual's material, returning white if no color is set.
/// Sets hasColor to true if the material has a non-black color.
Eigen::Vector3b extractVertexColor(const urdf::Visual& visual, bool& hasColor) {
  if (visual.material &&
      (visual.material->color.r > 0 || visual.material->color.g > 0 ||
       visual.material->color.b > 0)) {
    const auto& c = visual.material->color;
    hasColor = true;
    return {
        static_cast<uint8_t>(std::clamp(c.r, 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(std::clamp(c.g, 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(std::clamp(c.b, 0.0f, 1.0f) * 255.0f)};
  }
  return {255, 255, 255};
}

/// Builds a combined mesh and skin weights from per-link visual data.
/// Returns nullopt if no visual meshes were loaded.
template <typename T>
std::optional<std::pair<Mesh, SkinWeights>> buildCombinedMesh(
    const ParsingData<T>& data,
    const filesystem::path& urdfDir) {
  if (data.linkVisuals.empty()) {
    return std::nullopt;
  }

  const SkeletonStateT<float> bindState(data.parameterTransform.bindPose(), data.skeleton, false);

  Mesh combinedMesh;
  SkinWeights skinWeights;
  bool hasAnyColor = false;

  for (const auto& linkVisual : data.linkVisuals) {
    const size_t jointIdx = linkVisual.jointIndex;
    const auto& jointWorldTransform = bindState.jointState[jointIdx].transform;

    for (const auto& visual : linkVisual.visuals) {
      if (!visual || !visual->geometry) {
        continue;
      }

      auto meshOpt = loadVisualMesh(*visual->geometry, urdfDir);
      if (!meshOpt) {
        continue;
      }

      // Convert from meters (URDF) to centimeters (Momentum) and apply transforms
      for (auto& v : meshOpt->vertices) {
        v *= toCm<float>();
      }
      const auto visualOrigin = toMomentumTransform<float>(visual->origin);
      const Eigen::Vector3f visualTranslation = visualOrigin.translation * toCm<float>();
      for (auto& v : meshOpt->vertices) {
        v = visualOrigin.rotation * v + visualTranslation;
      }
      for (auto& v : meshOpt->vertices) {
        v = jointWorldTransform.rotation * v + jointWorldTransform.translation;
      }

      const Eigen::Vector3b vertexColor = extractVertexColor(*visual, hasAnyColor);

      // Merge into the combined mesh
      const auto vertexOffset = static_cast<int32_t>(combinedMesh.vertices.size());
      combinedMesh.vertices.insert(
          combinedMesh.vertices.end(), meshOpt->vertices.begin(), meshOpt->vertices.end());
      for (const auto& face : meshOpt->faces) {
        combinedMesh.faces.push_back(
            {face[0] + vertexOffset, face[1] + vertexOffset, face[2] + vertexOffset});
      }
      combinedMesh.colors.resize(combinedMesh.vertices.size(), vertexColor);

      // Extend skin weights: all new vertices are rigidly bound to this joint
      const auto prevVertexCount = skinWeights.index.rows();
      const auto newVertexCount =
          prevVertexCount + static_cast<Eigen::Index>(meshOpt->vertices.size());
      skinWeights.index.conservativeResize(newVertexCount, Eigen::NoChange);
      skinWeights.weight.conservativeResize(newVertexCount, Eigen::NoChange);
      skinWeights.index.bottomRows(newVertexCount - prevVertexCount).setZero();
      skinWeights.weight.bottomRows(newVertexCount - prevVertexCount).setZero();
      for (auto vi = prevVertexCount; vi < newVertexCount; ++vi) {
        skinWeights.index(vi, 0) = static_cast<uint32_t>(jointIdx);
        skinWeights.weight(vi, 0) = 1.0f;
      }
    }
  }

  if (!hasAnyColor) {
    combinedMesh.colors.clear();
  }

  if (combinedMesh.vertices.empty()) {
    return std::nullopt;
  }

  combinedMesh.updateNormals();
  return std::make_pair(std::move(combinedMesh), std::move(skinWeights));
}

template <typename T>
CharacterT<T> loadUrdfCharacterFromUrdfModel(
    urdf::ModelInterfaceSharedPtr urdfModel,
    const filesystem::path& urdfDir) {
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

  if (!loadUrdfSkeletonRecursive(data, kInvalidIndex, urdfModel.get(), root)) {
    MT_THROW("Failed to parse URDF.");
  }

  const size_t numJoints = data.skeleton.joints.size();
  const size_t numJointParameters = numJoints * kParametersPerJoint;
  const size_t numModelParameters = data.totalDoFs;
  data.parameterTransform.offsets.setZero(numJointParameters);

  resolveMimicJoints(data);

  data.parameterTransform.transform.resize(numJointParameters, numModelParameters);
  data.parameterTransform.transform.setFromTriplets(data.triplets.begin(), data.triplets.end());
  data.parameterTransform.activeJointParams = data.parameterTransform.computeActiveJointParams();

  const std::string robotName = urdfModel->getName();

  auto meshResult = buildCombinedMesh(data, urdfDir);
  if (meshResult) {
    auto& [mesh, skinWeights] = *meshResult;
    return CharacterT<T>(
        data.skeleton,
        data.parameterTransform,
        data.limits,
        LocatorList{},
        &mesh,
        &skinWeights,
        nullptr, // collision
        nullptr, // poseShapes
        {}, // blendShapes
        {}, // faceExpressionBlendShapes
        robotName);
  }

  return CharacterT<T>(
      data.skeleton,
      data.parameterTransform,
      data.limits,
      LocatorList{},
      nullptr, // mesh
      nullptr, // skinWeights
      nullptr, // collision
      nullptr, // poseShapes
      {}, // blendShapes
      {}, // faceExpressionBlendShapes
      robotName);
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
    return loadUrdfCharacterFromUrdfModel<T>(urdfModel, filepath.parent_path());
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
    return loadUrdfCharacterFromUrdfModel<T>(urdfModel, {});
  } catch (const std::exception& e) {
    MT_THROW("Failed to create Character from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF bytes");
  }
}

template CharacterT<float> loadUrdfCharacter(std::span<const std::byte> bytes);
template CharacterT<double> loadUrdfCharacter(std::span<const std::byte> bytes);

template <typename T>
CharacterT<T> loadUrdfCharacter(
    std::span<const std::byte> bytes,
    const filesystem::path& meshBasePath) {
  urdf::ModelInterfaceSharedPtr urdfModel;

  try {
    std::string urdfString(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    urdfModel = urdf::parseURDF(urdfString);
  } catch (const std::exception& e) {
    MT_THROW("Failed to read URDF file from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to read URDF file from URDF bytes");
  }

  try {
    return loadUrdfCharacterFromUrdfModel<T>(urdfModel, meshBasePath);
  } catch (const std::exception& e) {
    MT_THROW("Failed to create Character from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF bytes");
  }
}

template CharacterT<float> loadUrdfCharacter(
    std::span<const std::byte> bytes,
    const filesystem::path& meshBasePath);
template CharacterT<double> loadUrdfCharacter(
    std::span<const std::byte> bytes,
    const filesystem::path& meshBasePath);

} // namespace momentum
