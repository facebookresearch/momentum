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

#include <fmt/format.h>
#include <urdf_model/link.h>
#include <urdf_model/model.h>
#include <urdf_model/pose.h>
#include <urdf_parser/urdf_parser.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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
  std::vector<std::pair<Eigen::Index, float>> jointParamCoefficients;
  bool isRotation{}; // true for revolute/continuous, false for prismatic
};

struct ParsingData {
  Skeleton skeleton;
  ParameterTransform parameterTransform;
  std::vector<Eigen::Triplet<float>> triplets;
  ParameterLimits limits;
  PhysicalProperties physicalProperties;
  CollisionGeometry collision;
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

[[nodiscard]] Matrix3f toMomentumInertia(const urdf::Inertial& inertial) {
  constexpr float kInertiaScale = toCm<float>() * toCm<float>();
  Matrix3f result;
  result << static_cast<float>(inertial.ixx), static_cast<float>(inertial.ixy),
      static_cast<float>(inertial.ixz), static_cast<float>(inertial.ixy),
      static_cast<float>(inertial.iyy), static_cast<float>(inertial.iyz),
      static_cast<float>(inertial.ixz), static_cast<float>(inertial.iyz),
      static_cast<float>(inertial.izz);
  return (result * kInertiaScale).eval();
}

void addPhysicalPropertiesForUrdfLink(
    PhysicalProperties& physicalProperties,
    const urdf::Link& urdfLink,
    const size_t jointIndex) {
  if (!urdfLink.inertial) {
    return;
  }

  JointPhysicalProperties jointProperties;
  jointProperties.jointName = urdfLink.name;
  jointProperties.jointIndex = jointIndex;
  jointProperties.mass = static_cast<float>(urdfLink.inertial->mass);
  jointProperties.centerOfMassOffset =
      toMomentumVector3<float>(urdfLink.inertial->origin.position) * toCm<float>();
  jointProperties.inertia = toMomentumInertia(*urdfLink.inertial);
  jointProperties.inertiaRotation =
      toMomentumQuaternion<float>(urdfLink.inertial->origin.rotation).normalized();
  physicalProperties.push_back(std::move(jointProperties));
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

[[nodiscard]] std::string_view toString(const urdf::Geometry& geometry) {
  switch (geometry.type) {
    case urdf::Geometry::SPHERE:
      return "SPHERE";
    case urdf::Geometry::BOX:
      return "BOX";
    case urdf::Geometry::CYLINDER:
      return "CYLINDER";
    case urdf::Geometry::MESH:
      return "MESH";
  }

  return "UNKNOWN";
}

struct PrincipalAxis {
  int index{}; // principal axis index (0=X, 1=Y, 2=Z)
  float sign{}; // axis sign (+1/-1)
};

struct JointDofMapping {
  std::vector<std::pair<Eigen::Index, float>> jointParamCoefficients;
  bool isRotation{};
};

struct JointFrameData {
  Eigen::Index modelParamsBaseIndex{};
  Eigen::Index axisJointId{};
  Eigen::Index linkJointId{};
  Eigen::Index motionJointParamsBaseIndex{};
  Quaternionf originRotation = Quaternionf::Identity();
  Vector3f originTranslation = Vector3f::Zero();
  // For a revolute/continuous joint whose axis is a principal axis, the resolved axis is cached
  // here so the DOF mapping does not recompute it. Null for arbitrary axes (which use
  // arbitraryAxisFrameRotation instead) and for non-rotational joints.
  std::optional<PrincipalAxis> principalRotationAxis;
  std::optional<Quaternionf> arbitraryAxisFrameRotation;

  [[nodiscard]] bool hasArbitraryRotationAxis() const {
    return arbitraryAxisFrameRotation.has_value();
  }
};

[[nodiscard]] Vector3f normalizedAxis(const urdf::Vector3& axis, std::string_view jointName) {
  const Vector3f result{
      static_cast<float>(axis.x), static_cast<float>(axis.y), static_cast<float>(axis.z)};
  MT_THROW_IF(
      !result.allFinite(),
      "Joint '{}' has invalid non-finite axis ({}, {}, {}).",
      jointName,
      result.x(),
      result.y(),
      result.z());
  const float norm = result.norm();
  MT_THROW_IF(
      norm < 1e-8f,
      "Joint '{}' has invalid zero-length axis ({}, {}, {}).",
      jointName,
      result.x(),
      result.y(),
      result.z());
  return result / norm;
}

[[nodiscard]] std::optional<PrincipalAxis> getPrincipalAxis(const Vector3f& axis) {
  constexpr float kTol = 1e-4f;

  const float ax = axis.x();
  const float ay = axis.y();
  const float az = axis.z();
  if (std::abs(std::abs(ax) - 1.0f) < kTol && std::abs(ay) < kTol && std::abs(az) < kTol) {
    return PrincipalAxis{0, ax > 0 ? 1.0f : -1.0f};
  }
  if (std::abs(ax) < kTol && std::abs(std::abs(ay) - 1.0f) < kTol && std::abs(az) < kTol) {
    return PrincipalAxis{1, ay > 0 ? 1.0f : -1.0f};
  }
  if (std::abs(ax) < kTol && std::abs(ay) < kTol && std::abs(std::abs(az) - 1.0f) < kTol) {
    return PrincipalAxis{2, az > 0 ? 1.0f : -1.0f};
  }

  return std::nullopt;
}

[[nodiscard]] PrincipalAxis requirePrincipalAxis(
    const urdf::Vector3& axis,
    std::string_view jointName) {
  const Vector3f normalized = normalizedAxis(axis, jointName);
  const auto principalAxis = getPrincipalAxis(normalized);
  if (principalAxis) {
    return *principalAxis;
  }

  MT_THROW(
      "Joint '{}' has non-principal planar axis ({}, {}, {}). "
      "Only axes aligned with X, Y, or Z are supported for planar joints.",
      jointName,
      normalized.x(),
      normalized.y(),
      normalized.z());
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

[[nodiscard]] TaperedCapsule loadCollisionCapsule(
    const urdf::Cylinder& cylinder,
    const TransformT<float>& collisionOrigin,
    size_t jointIndex) {
  // Convention conversion: URDF cylinders are centered at the origin with their axis along
  // +Z, while Momentum TaperedCapsules start at the origin with their axis along +X.
  //   - Rotation: compose the URDF collision origin rotation with a UnitX→UnitZ rotation so the
  //     local +X axis of the Momentum capsule maps to the local +Z axis of the URDF cylinder.
  //   - Translation: convert the URDF center position to centimeters, then shift along the
  //     URDF cylinder's local +Z by -length/2 so the capsule starts where the cylinder ends.
  TaperedCapsule capsule;
  capsule.parent = jointIndex;
  capsule.length = static_cast<float>(cylinder.length) * toCm<float>();
  capsule.radius = Eigen::Vector2f::Constant(static_cast<float>(cylinder.radius) * toCm<float>());
  capsule.transformation.rotation = collisionOrigin.rotation *
      Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitZ());
  capsule.transformation.translation = collisionOrigin.translation * toCm<float>() +
      collisionOrigin.rotation * Eigen::Vector3f(0.0f, 0.0f, -0.5f * capsule.length);
  return capsule;
}

[[nodiscard]] CollisionEllipsoid loadCollisionSphere(
    const urdf::Sphere& sphere,
    const TransformT<float>& collisionOrigin,
    size_t jointIndex) {
  // A URDF sphere has a single radius and is centered at the collision origin. It maps directly to
  // a Momentum ellipsoid with equal radii along all three local axes (a sphere is the special case
  // of an ellipsoid). The collision origin becomes the ellipsoid center and orientation.
  CollisionEllipsoid ellipsoid;
  ellipsoid.parent = jointIndex;
  ellipsoid.radii = Eigen::Vector3f::Constant(static_cast<float>(sphere.radius) * toCm<float>());
  ellipsoid.transformation.rotation = collisionOrigin.rotation;
  ellipsoid.transformation.translation = collisionOrigin.translation * toCm<float>();
  return ellipsoid;
}

[[nodiscard]] CollisionBox loadCollisionBox(
    const urdf::Box& box,
    const TransformT<float>& collisionOrigin,
    size_t jointIndex) {
  // A URDF box is centered at the collision origin with full extents box.dim along its local axes,
  // whereas a Momentum box stores half extents. The collision origin becomes the box center and
  // orientation.
  CollisionBox collisionBox;
  collisionBox.parent = jointIndex;
  collisionBox.halfExtents = Eigen::Vector3f(
                                 static_cast<float>(box.dim.x),
                                 static_cast<float>(box.dim.y),
                                 static_cast<float>(box.dim.z)) *
      (0.5f * toCm<float>());
  collisionBox.transformation.rotation = collisionOrigin.rotation;
  collisionBox.transformation.translation = collisionOrigin.translation * toCm<float>();
  return collisionBox;
}

/// Imports a single URDF collision geometry as a Momentum collision primitive.
///
/// URDF cylinders become tapered capsules, spheres become ellipsoids with equal radii, and boxes
/// become boxes. Mesh collision geometry is not supported and is skipped with a warning.
[[nodiscard]] std::optional<CollisionPrimitive> loadCollisionPrimitive(
    const urdf::Collision& collision,
    std::string_view linkName,
    size_t jointIndex) {
  if (!collision.geometry) {
    MT_LOGW("Ignoring URDF collision on link '{}' because it has no geometry.", linkName);
    return std::nullopt;
  }

  const auto& geometry = *collision.geometry;
  const auto collisionOrigin = toMomentumTransform<float>(collision.origin);

  switch (geometry.type) {
    case urdf::Geometry::CYLINDER:
      return CollisionPrimitive(loadCollisionCapsule(
          dynamic_cast<const urdf::Cylinder&>(geometry), collisionOrigin, jointIndex));
    case urdf::Geometry::SPHERE:
      return CollisionPrimitive(loadCollisionSphere(
          dynamic_cast<const urdf::Sphere&>(geometry), collisionOrigin, jointIndex));
    case urdf::Geometry::BOX:
      return CollisionPrimitive(
          loadCollisionBox(dynamic_cast<const urdf::Box&>(geometry), collisionOrigin, jointIndex));
    case urdf::Geometry::MESH:
      MT_LOGW(
          "Ignoring unsupported URDF mesh collision geometry on link '{}'. Only cylinder, sphere, "
          "and box collision geometry can be imported.",
          linkName);
      return std::nullopt;
  }

  MT_LOGW(
      "Ignoring unknown URDF collision geometry '{}' on link '{}'.", toString(geometry), linkName);
  return std::nullopt;
}

constexpr std::array<const char*, 3> kTranslationNames = {"tx", "ty", "tz"};
constexpr std::array<const char*, 3> kRotationNames = {"rx", "ry", "rz"};

void addModelDof(
    ParsingData& data,
    const std::string& urdfJointName,
    const JointDofMapping& mapping,
    Eigen::Index modelParamsBaseIndex) {
  data.parameterTransform.name.push_back(urdfJointName);
  for (const auto& [jointParamIndex, coefficient] : mapping.jointParamCoefficients) {
    data.triplets.emplace_back(jointParamIndex, modelParamsBaseIndex, coefficient);
  }
  data.urdfJointNameToModelParamIdx[urdfJointName] = modelParamsBaseIndex;
  data.totalDoFs += 1;
}

void addMimicDof(ParsingData& data, const urdf::Joint& urdfJoint, const JointDofMapping& mapping) {
  MimicData mimic;
  mimic.mimickedJointName = urdfJoint.mimic->joint_name;
  mimic.multiplier = static_cast<float>(urdfJoint.mimic->multiplier);
  mimic.offset = static_cast<float>(urdfJoint.mimic->offset);
  mimic.jointParamCoefficients = mapping.jointParamCoefficients;
  mimic.isRotation = mapping.isRotation;
  data.mimicJoints.push_back(std::move(mimic));
}

[[nodiscard]] JointDofMapping rotationDofMapping(
    Eigen::Index jointParamsBaseIndex,
    const PrincipalAxis& axis) {
  JointDofMapping result;
  result.jointParamCoefficients.emplace_back(jointParamsBaseIndex + RX + axis.index, axis.sign);
  result.isRotation = true;
  return result;
}

[[nodiscard]] JointDofMapping prismaticDofMapping(
    Eigen::Index jointParamsBaseIndex,
    const Vector3f& axisInParentFrame) {
  JointDofMapping result;
  result.isRotation = false;
  constexpr float kTol = 1e-6f;
  for (int index = 0; index < 3; ++index) {
    if (std::abs(axisInParentFrame[index]) > kTol) {
      result.jointParamCoefficients.emplace_back(
          jointParamsBaseIndex + index, axisInParentFrame[index]);
    }
  }
  return result;
}

[[nodiscard]] bool isRevoluteLike(const urdf::Joint& urdfJoint) {
  return urdfJoint.type == urdf::Joint::REVOLUTE || urdfJoint.type == urdf::Joint::CONTINUOUS;
}

[[nodiscard]] JointFrameData makeJointFrameData(
    const ParsingData& data,
    const urdf::Joint* urdfJoint) {
  JointFrameData result;
  result.modelParamsBaseIndex = data.totalDoFs;
  result.axisJointId = data.skeleton.joints.size();

  const urdf::Pose* urdfPose =
      urdfJoint != nullptr ? &urdfJoint->parent_to_joint_origin_transform : nullptr;
  result.originRotation = urdfPose != nullptr ? toMomentumQuaternion<float>(urdfPose->rotation)
                                              : Quaternionf::Identity();
  result.originTranslation = urdfPose != nullptr
      ? (toMomentumVector3<float>(urdfPose->position) * toCm<float>()).eval()
      : Vector3f::Zero();

  if (urdfJoint != nullptr && isRevoluteLike(*urdfJoint)) {
    const Vector3f axis = normalizedAxis(urdfJoint->axis, urdfJoint->name);
    result.principalRotationAxis = getPrincipalAxis(axis);
    if (!result.principalRotationAxis) {
      result.arbitraryAxisFrameRotation =
          Quaternionf::FromTwoVectors(Vector3f::UnitX(), axis).normalized();
    }
  }

  result.linkJointId = result.axisJointId + (result.hasArbitraryRotationAxis() ? 1 : 0);
  // The motion DOF always lands on the first node this joint pushes: the axis-aligned helper
  // when present, otherwise the link node — both share the same index (axisJointId).
  result.motionJointParamsBaseIndex = result.axisJointId * kParametersPerJoint;
  return result;
}

void addModelOrMimicDof(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    const JointDofMapping& mapping,
    Eigen::Index modelParamsBaseIndex) {
  if (urdfJoint.mimic != nullptr) {
    addMimicDof(data, urdfJoint, mapping);
  } else {
    addModelDof(data, urdfJoint.name, mapping, modelParamsBaseIndex);
  }
}

void addPrismaticJointDof(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    const JointFrameData& frame) {
  const Vector3f axis = normalizedAxis(urdfJoint.axis, urdfJoint.name);
  const JointDofMapping mapping =
      prismaticDofMapping(frame.motionJointParamsBaseIndex, frame.originRotation * axis);
  addModelOrMimicDof(data, urdfJoint, mapping, frame.modelParamsBaseIndex);
}

void addRevoluteJointDof(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    const JointFrameData& frame) {
  JointDofMapping mapping;
  if (frame.principalRotationAxis) {
    mapping = rotationDofMapping(frame.motionJointParamsBaseIndex, *frame.principalRotationAxis);
  } else {
    // Arbitrary axis: the inserted axis-aligned helper node carries a single rotation about its
    // local X axis.
    mapping.jointParamCoefficients.emplace_back(frame.motionJointParamsBaseIndex + RX, 1.0f);
    mapping.isRotation = true;
  }

  addModelOrMimicDof(data, urdfJoint, mapping, frame.modelParamsBaseIndex);
}

void addFloatingJointDofs(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    const JointFrameData& frame) {
  // Floating joints have 6 DOFs (3 translation, then 3 rotation); use the URDF joint name with
  // axis suffixes. Translation parameters use a 10x scaling factor so that a unit change in
  // model parameter space produces 10 cm of translation. This makes translation easier to
  // change relative to rotation in optimization.
  constexpr float kFloatingTranslationScale = 10.0f;
  const auto& jn = urdfJoint.name;
  for (int index = 0; index < 3; ++index) {
    data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kTranslationNames[index]));
    data.triplets.emplace_back(
        frame.motionJointParamsBaseIndex + TX + index,
        frame.modelParamsBaseIndex + index,
        kFloatingTranslationScale);
  }
  for (int index = 0; index < 3; ++index) {
    data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kRotationNames[index]));
    data.triplets.emplace_back(
        frame.motionJointParamsBaseIndex + RX + index,
        frame.modelParamsBaseIndex + 3 + index,
        1.0f);
  }
  data.totalDoFs += 6;
}

void addPlanarJointDofs(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    const JointFrameData& frame) {
  // The axis defines the plane normal. DOFs are translation along the two in-plane
  // axes and rotation around the normal.
  const int normalIdx = requirePrincipalAxis(urdfJoint.axis, urdfJoint.name).index;
  const int planeAxis0 = (normalIdx + 1) % 3;
  const int planeAxis1 = (normalIdx + 2) % 3;
  const auto& jn = urdfJoint.name;
  data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kTranslationNames[planeAxis0]));
  data.triplets.emplace_back(
      frame.motionJointParamsBaseIndex + planeAxis0, frame.modelParamsBaseIndex + 0, 1.0f);
  data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kTranslationNames[planeAxis1]));
  data.triplets.emplace_back(
      frame.motionJointParamsBaseIndex + planeAxis1, frame.modelParamsBaseIndex + 1, 1.0f);
  data.parameterTransform.name.push_back(fmt::format("{}_{}", jn, kRotationNames[normalIdx]));
  data.triplets.emplace_back(
      frame.motionJointParamsBaseIndex + RX + normalIdx, frame.modelParamsBaseIndex + 2, 1.0f);
  data.totalDoFs += 3;
}

void addJointDofs(ParsingData& data, const urdf::Joint& urdfJoint, const JointFrameData& frame) {
  //---------------------------
  // Parse Parameter transform
  //---------------------------
  //
  // URDF expresses motion on a joint between links, but attached data is authored in the child
  // link frame. Our imported Momentum node therefore keeps the child link frame as its local
  // frame whenever the URDF axis already matches a Momentum principal axis. For arbitrary
  // revolute axes, we insert an internal axis-aligned helper followed by a fixed child-link node:
  //   parent -> origin * axisFrame -> revolute X(q) -> axisFrame^-1 -> child link
  // so the composite transform is exactly origin * Rot(axis, q) while the visible child-link
  // node remains in the URDF child-link frame.
  //
  // Revolute model parameters stay in the URDF scalar coordinate. Prismatic parameters keep the
  // existing Momentum convention of centimeters. Negative principal axes, arbitrary prismatic
  // direction vectors, and mimic offsets are represented by coefficients in the parameter
  // transform rather than by changing the underlying joint frame.

  switch (urdfJoint.type) {
    case urdf::Joint::PRISMATIC: {
      addPrismaticJointDof(data, urdfJoint, frame);
      break;
    }
    case urdf::Joint::REVOLUTE:
    case urdf::Joint::CONTINUOUS: {
      addRevoluteJointDof(data, urdfJoint, frame);
      break;
    }
    case urdf::Joint::FLOATING: {
      addFloatingJointDofs(data, urdfJoint, frame);
      break;
    }
    case urdf::Joint::PLANAR: {
      addPlanarJointDofs(data, urdfJoint, frame);
      break;
    }
    case urdf::Joint::FIXED: {
      // No degrees of freedom
      break;
    }
    case urdf::Joint::UNKNOWN: {
      MT_THROW("Unsupported joint type: {}", toString(urdfJoint.type));
    }
  }
}

void addJointLimits(
    ParsingData& data,
    const urdf::Joint& urdfJoint,
    Eigen::Index modelParamsBaseIndex) {
  // Mimic joints are driven by the mimicked joint's parameter, so their limits don't
  // correspond to an independent model parameter.
  if (urdfJoint.mimic != nullptr || !urdfJoint.limits) {
    return;
  }

  switch (urdfJoint.type) {
    case urdf::Joint::REVOLUTE: {
      ParameterLimit jointLimits;
      jointLimits.type = MinMax;
      jointLimits.data.minMax.parameterIndex = modelParamsBaseIndex;
      jointLimits.data.minMax.limits[0] = urdfJoint.limits->lower; // rad
      jointLimits.data.minMax.limits[1] = urdfJoint.limits->upper;
      jointLimits.weight = 1.0f;
      data.limits.push_back(jointLimits);
      break;
    }
    case urdf::Joint::PRISMATIC: {
      ParameterLimit jointLimits;
      jointLimits.type = MinMax;
      jointLimits.data.minMax.parameterIndex = modelParamsBaseIndex;
      jointLimits.data.minMax.limits[0] = toCm<float>(urdfJoint.limits->lower);
      jointLimits.data.minMax.limits[1] = toCm<float>(urdfJoint.limits->upper);
      jointLimits.weight = 1.0f;
      data.limits.push_back(jointLimits);
      break;
    }
    case urdf::Joint::UNKNOWN:
    case urdf::Joint::CONTINUOUS:
    case urdf::Joint::FLOATING:
    case urdf::Joint::PLANAR:
    case urdf::Joint::FIXED: {
      // No independent min/max parameter limit to add.
      break;
    }
  }
}

void appendSkeletonJointsForUrdfLink(
    ParsingData& data,
    size_t parentJointId,
    const urdf::Link& urdfLink,
    const urdf::Joint* urdfJoint,
    const JointFrameData& frame) {
  if (frame.hasArbitraryRotationAxis()) {
    MT_THROW_IF(urdfJoint == nullptr, "Internal URDF axis node requires a URDF joint.");
    Joint axisJoint;
    axisJoint.name = fmt::format("{}_urdf_axis", urdfJoint->name);
    axisJoint.parent = parentJointId;
    axisJoint.preRotation = frame.originRotation * *frame.arbitraryAxisFrameRotation;
    axisJoint.translationOffset = frame.originTranslation;
    data.skeleton.joints.push_back(axisJoint);
  }

  Joint linkJoint;
  linkJoint.name = urdfLink.name;
  linkJoint.parent =
      frame.hasArbitraryRotationAxis() ? static_cast<size_t>(frame.axisJointId) : parentJointId;
  linkJoint.preRotation = frame.hasArbitraryRotationAxis()
      ? frame.arbitraryAxisFrameRotation->inverse()
      : frame.originRotation;
  linkJoint.translationOffset =
      frame.hasArbitraryRotationAxis() ? Vector3f::Zero() : frame.originTranslation;
  data.skeleton.joints.push_back(linkJoint);
}

void collectLinkVisuals(ParsingData& data, const urdf::Link& urdfLink, size_t linkJointId) {
  if (urdfLink.visual_array.empty()) {
    return;
  }

  LinkVisualData visualData;
  visualData.jointIndex = linkJointId;
  visualData.visuals = urdfLink.visual_array;
  data.linkVisuals.push_back(std::move(visualData));
}

void collectCollisionGeometry(ParsingData& data, const urdf::Link& urdfLink, size_t linkJointId) {
  for (const auto& collision : urdfLink.collision_array) {
    if (!collision) {
      MT_LOGW("Ignoring null URDF collision on link '{}'.", urdfLink.name);
      continue;
    }
    auto primitive = loadCollisionPrimitive(*collision, urdfLink.name, linkJointId);
    if (primitive) {
      data.collision.push_back(*primitive);
    }
  }
}

bool loadUrdfSkeletonRecursive(
    ParsingData& data,
    size_t parentJointId,
    const urdf::Link* urdfLink) {
  MT_THROW_IF(urdfLink == nullptr, "URDF link is null.");

  auto* urdfJoint = urdfLink->parent_joint.get();
  MT_THROW_IF(
      parentJointId != kInvalidIndex && urdfJoint == nullptr,
      "URDF parent joint is null for a non root link ({}).",
      urdfLink->name);

  const JointFrameData frame = makeJointFrameData(data, urdfJoint);
  if (urdfJoint != nullptr) {
    addJointDofs(data, *urdfJoint, frame);
    addJointLimits(data, *urdfJoint, frame.modelParamsBaseIndex);
  }

  appendSkeletonJointsForUrdfLink(data, parentJointId, *urdfLink, urdfJoint, frame);
  addPhysicalPropertiesForUrdfLink(
      data.physicalProperties, *urdfLink, static_cast<size_t>(frame.linkJointId));

  collectLinkVisuals(data, *urdfLink, static_cast<size_t>(frame.linkJointId));
  collectCollisionGeometry(data, *urdfLink, static_cast<size_t>(frame.linkJointId));

  for (const auto& childLink : urdfLink->child_links) {
    if (!loadUrdfSkeletonRecursive(data, static_cast<size_t>(frame.linkJointId), childLink.get())) {
      return false;
    }
  }

  return true;
}

/// Resolves mimic joints by mapping each mimic joint's DOF to the mimicked joint's model
/// parameter via the parameter transform.
void resolveMimicJoints(ParsingData& data) {
  for (const auto& mimic : data.mimicJoints) {
    auto it = data.urdfJointNameToModelParamIdx.find(mimic.mimickedJointName);
    if (it == data.urdfJointNameToModelParamIdx.end()) {
      MT_LOGW("Mimic joint references unknown joint '{}', skipping.", mimic.mimickedJointName);
      continue;
    }
    const Eigen::Index mimickedModelParamIdx = it->second;
    const float offset = mimic.isRotation ? mimic.offset : toCm<float>(mimic.offset);

    for (const auto& [jointParamIndex, coefficient] : mimic.jointParamCoefficients) {
      data.triplets.emplace_back(
          jointParamIndex, mimickedModelParamIdx, coefficient * mimic.multiplier);
      data.parameterTransform.offsets[jointParamIndex] = coefficient * offset;
    }
  }
}

[[nodiscard]] Eigen::Vector3b toVertexColor(const urdf::Color& color) {
  return {
      static_cast<uint8_t>(std::clamp(color.r, 0.0f, 1.0f) * 255.0f),
      static_cast<uint8_t>(std::clamp(color.g, 0.0f, 1.0f) * 255.0f),
      static_cast<uint8_t>(std::clamp(color.b, 0.0f, 1.0f) * 255.0f)};
}

/// Extracts per-vertex color from a URDF visual's material.
std::optional<Eigen::Vector3b> extractVertexColor(const urdf::Visual& visual) {
  if (!visual.material) {
    return std::nullopt;
  }

  const auto& material = *visual.material;
  const bool looksLikeTextureOnlyMaterial = !material.texture_filename.empty() &&
      material.color.r == 0.0f && material.color.g == 0.0f && material.color.b == 0.0f &&
      material.color.a == 1.0f;
  if (looksLikeTextureOnlyMaterial) {
    return std::nullopt;
  }

  return toVertexColor(material.color);
}

/// Builds a combined mesh and skin weights from per-link visual data.
/// Returns nullopt if no visual meshes were loaded.
std::optional<std::pair<Mesh, SkinWeights>> buildCombinedMesh(
    const ParsingData& data,
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

      const auto vertexColor = extractVertexColor(*visual);
      hasAnyColor = hasAnyColor || vertexColor.has_value();

      // Merge into the combined mesh
      const auto vertexOffset = static_cast<int32_t>(combinedMesh.vertices.size());
      combinedMesh.vertices.insert(
          combinedMesh.vertices.end(), meshOpt->vertices.begin(), meshOpt->vertices.end());
      for (const auto& face : meshOpt->faces) {
        combinedMesh.faces.emplace_back(
            face[0] + vertexOffset, face[1] + vertexOffset, face[2] + vertexOffset);
      }
      combinedMesh.colors.resize(
          combinedMesh.vertices.size(), vertexColor.value_or(Eigen::Vector3b{255, 255, 255}));

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

Character loadUrdfCharacterFromUrdfModel(
    urdf::ModelInterfaceSharedPtr urdfModel,
    const filesystem::path& urdfDir) {
  const urdf::Link* root = urdfModel->getRoot().get();
  MT_THROW_IF(!root, "Failed to parse URDF file from. No root link found.");

  ParsingData data;

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

  if (!loadUrdfSkeletonRecursive(data, kInvalidIndex, root)) {
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
  const CollisionGeometry* collision = data.collision.empty() ? nullptr : &data.collision;

  auto meshResult = buildCombinedMesh(data, urdfDir);
  if (meshResult) {
    auto& [mesh, skinWeights] = *meshResult;
    return {
        data.skeleton,
        data.parameterTransform,
        data.limits,
        LocatorList{},
        &mesh,
        &skinWeights,
        collision,
        nullptr, // poseShapes
        {}, // blendShapes
        {}, // faceExpressionBlendShapes
        robotName,
        {}, // inverseBindPose
        {}, // skinnedLocators
        {}, // metadata
        data.physicalProperties};
  }

  return {
      data.skeleton,
      data.parameterTransform,
      data.limits,
      LocatorList{},
      nullptr, // mesh
      nullptr, // skinWeights
      collision,
      nullptr, // poseShapes
      {}, // blendShapes
      {}, // faceExpressionBlendShapes
      robotName,
      {}, // inverseBindPose
      {}, // skinnedLocators
      {}, // metadata
      data.physicalProperties};
}

} // namespace

Character loadUrdfCharacter(const filesystem::path& filepath) {
  urdf::ModelInterfaceSharedPtr urdfModel;

  try {
    urdfModel = urdf::parseURDFFile(filepath.string());
  } catch (const std::exception& e) {
    MT_THROW("Failed to parse URDF file from: {}. Error: {}", filepath.string(), e.what());
  } catch (...) {
    MT_THROW("Failed to parse URDF file from: {}", filepath.string());
  }

  try {
    return loadUrdfCharacterFromUrdfModel(urdfModel, filepath.parent_path());
  } catch (const std::exception& e) {
    MT_THROW(
        "Failed to create Character from URDF file: {}. Error: {}", filepath.string(), e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF file: {}", filepath.string());
  }
}

Character loadUrdfCharacter(std::span<const std::byte> bytes) {
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
    return loadUrdfCharacterFromUrdfModel(urdfModel, {});
  } catch (const std::exception& e) {
    MT_THROW("Failed to create Character from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF bytes");
  }
}

Character loadUrdfCharacter(
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
    return loadUrdfCharacterFromUrdfModel(urdfModel, meshBasePath);
  } catch (const std::exception& e) {
    MT_THROW("Failed to create Character from URDF bytes. Error: {}", e.what());
  } catch (...) {
    MT_THROW("Failed to create Character from URDF bytes");
  }
}

} // namespace momentum
