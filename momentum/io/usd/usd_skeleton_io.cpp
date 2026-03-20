/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_skeleton_io.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

namespace {

Eigen::Matrix4f toEigenMatrix4f(const GfMatrix4d& gfMatrix) {
  return Eigen::Map<const Eigen::Matrix4d>(gfMatrix.GetArray()).cast<float>();
}

GfMatrix4d toGfMatrix4d(const Eigen::Matrix4f& eigenMatrix) {
  GfMatrix4d gfMatrix{};
  Eigen::Map<Eigen::Matrix4d>(gfMatrix.GetArray()) = eigenMatrix.cast<double>();
  return gfMatrix;
}

// Build a local-space transform matrix from a joint's preRotation and translationOffset.
Eigen::Matrix4f jointToLocalTransform(const Joint& joint) {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = joint.preRotation.toRotationMatrix();
  transform.block<3, 1>(0, 3) = joint.translationOffset;
  return transform;
}

// Compute world-space bind transform for a joint by walking up the parent chain.
Eigen::Matrix4f computeWorldTransform(const Skeleton& skeleton, size_t jointIndex) {
  Eigen::Matrix4f world = jointToLocalTransform(skeleton.joints[jointIndex]);
  size_t parent = skeleton.joints[jointIndex].parent;
  while (parent != kInvalidIndex) {
    world = jointToLocalTransform(skeleton.joints[parent]) * world;
    parent = skeleton.joints[parent].parent;
  }
  return world;
}

// Extract the leaf name from a hierarchical joint path (e.g., "Root/Spine/Chest" -> "Chest").
std::string leafName(const std::string& path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

// Build hierarchical joint paths from flat names and parent indices.
// E.g., joint 0 "root" (no parent) -> "root"
//       joint 1 "spine" (parent 0) -> "root/spine"
//       joint 2 "chest" (parent 1) -> "root/spine/chest"
std::vector<std::string> buildHierarchicalPaths(const Skeleton& skeleton) {
  std::vector<std::string> paths(skeleton.joints.size());
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    if (skeleton.joints[i].parent == kInvalidIndex) {
      paths[i] = skeleton.joints[i].name;
    } else {
      paths[i] = paths[skeleton.joints[i].parent] + "/" + skeleton.joints[i].name;
    }
  }
  return paths;
}

} // namespace

Skeleton loadSkeletonFromUsd(const UsdStageRefPtr& stage) {
  Skeleton skeleton;

  UsdSkelCache skelCache;

  for (const auto& prim : stage->Traverse()) {
    if (!prim.IsA<UsdSkelSkeleton>()) {
      continue;
    }

    UsdSkelSkeleton skelPrim(prim);
    UsdSkelSkeletonQuery skelQuery = skelCache.GetSkelQuery(skelPrim);
    if (!skelQuery) {
      continue;
    }

    VtArray<TfToken> jointNames = skelQuery.GetJointOrder();
    if (jointNames.empty()) {
      continue;
    }

    skeleton.joints.reserve(jointNames.size());

    // Get parent topology (derived from hierarchical joint paths)
    VtArray<int> topology = skelQuery.GetTopology().GetParentIndices();

    for (size_t i = 0; i < jointNames.size(); ++i) {
      Joint joint;
      // Extract the leaf name from the hierarchical path
      joint.name = leafName(jointNames[i].GetString());

      if (i < topology.size() && topology[i] >= 0) {
        joint.parent = topology[i];
      } else {
        joint.parent = kInvalidIndex;
      }

      joint.preRotation = Eigen::Quaternionf::Identity();
      joint.translationOffset = Eigen::Vector3f::Zero();

      skeleton.joints.push_back(joint);
    }

    // Prefer local rest transforms when available (preserves exact joint data)
    VtArray<GfMatrix4d> restTransforms;
    if (skelPrim.GetRestTransformsAttr().Get(&restTransforms) &&
        restTransforms.size() == skeleton.joints.size()) {
      for (size_t i = 0; i < restTransforms.size(); ++i) {
        auto matrix = toEigenMatrix4f(restTransforms[i]);
        skeleton.joints[i].translationOffset = matrix.block<3, 1>(0, 3);
        Eigen::Matrix3f rotMatrix = matrix.block<3, 3>(0, 0);
        skeleton.joints[i].preRotation = Eigen::Quaternionf(rotMatrix).normalized();
      }
    } else {
      // Fall back to world-space bind transforms
      VtArray<GfMatrix4d> bindTransforms;
      if (skelQuery.GetJointWorldBindTransforms(&bindTransforms)) {
        MT_CHECK(bindTransforms.size() == skeleton.joints.size());

        for (size_t i = 0; i < bindTransforms.size(); ++i) {
          Eigen::Matrix4f worldMatrix = toEigenMatrix4f(bindTransforms[i]);

          // Convert world-space to local-space by removing parent's world transform
          Eigen::Matrix4f localMatrix;
          if (skeleton.joints[i].parent != kInvalidIndex) {
            Eigen::Matrix4f parentWorld =
                toEigenMatrix4f(bindTransforms[skeleton.joints[i].parent]);
            localMatrix = parentWorld.inverse() * worldMatrix;
          } else {
            localMatrix = worldMatrix;
          }

          skeleton.joints[i].translationOffset = localMatrix.block<3, 1>(0, 3);
          Eigen::Matrix3f rotMatrix = localMatrix.block<3, 3>(0, 0);
          skeleton.joints[i].preRotation = Eigen::Quaternionf(rotMatrix).normalized();
        }
      }
    }

    break; // Use first skeleton found
  }

  return skeleton;
}

void saveSkeletonToUsd(const Skeleton& skeleton, UsdSkelSkeleton& skelPrim) {
  // Write joint names as hierarchical paths (encodes parent topology)
  const auto paths = buildHierarchicalPaths(skeleton);
  VtArray<TfToken> jointTokens;
  jointTokens.reserve(skeleton.joints.size());
  for (const auto& path : paths) {
    jointTokens.push_back(TfToken(path));
  }
  skelPrim.GetJointsAttr().Set(jointTokens);

  // Write local rest transforms (preserves exact joint data)
  VtArray<GfMatrix4d> restTransforms;
  restTransforms.reserve(skeleton.joints.size());
  for (const auto& joint : skeleton.joints) {
    restTransforms.push_back(toGfMatrix4d(jointToLocalTransform(joint)));
  }
  skelPrim.GetRestTransformsAttr().Set(restTransforms);

  // Write world-space bind transforms (for interop with standard USD viewers)
  VtArray<GfMatrix4d> bindTransforms;
  bindTransforms.reserve(skeleton.joints.size());
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    bindTransforms.push_back(toGfMatrix4d(computeWorldTransform(skeleton, i)));
  }
  skelPrim.GetBindTransformsAttr().Set(bindTransforms);
}

void saveCollisionGeometryToUsd(
    const CollisionGeometry& collision,
    const Skeleton& skeleton,
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath) {
  if (collision.empty() || skeleton.joints.empty()) {
    return;
  }

  auto collisionScope = UsdGeomScope::Define(stage, skelRootPath.AppendChild(TfToken("Collision")));

  for (size_t i = 0; i < collision.size(); ++i) {
    const auto& capsule = collision[i];
    const std::string jointName = (capsule.parent < skeleton.joints.size())
        ? skeleton.joints[capsule.parent].name
        : "unknown";
    const std::string primName = jointName + "_col_" + std::to_string(i);

    auto primPath = collisionScope.GetPath().AppendChild(TfToken(primName));
    auto prim = stage->DefinePrim(primPath);

    prim.CreateAttribute(TfToken("momentum:type"), SdfValueTypeNames->String)
        .Set(std::string("collision_capsule"));
    prim.CreateAttribute(TfToken("momentum:parent"), SdfValueTypeNames->String).Set(jointName);
    prim.CreateAttribute(TfToken("momentum:length"), SdfValueTypeNames->Float).Set(capsule.length);
    prim.CreateAttribute(TfToken("momentum:radius"), SdfValueTypeNames->Float2)
        .Set(GfVec2f(capsule.radius.x(), capsule.radius.y()));

    const auto& t = capsule.transformation.translation;
    prim.CreateAttribute(TfToken("momentum:translation"), SdfValueTypeNames->Float3)
        .Set(GfVec3f(t.x(), t.y(), t.z()));

    const auto& q = capsule.transformation.rotation;
    prim.CreateAttribute(TfToken("momentum:rotation"), SdfValueTypeNames->Quatf)
        .Set(GfQuatf(q.w(), q.x(), q.y(), q.z()));
  }
}

CollisionGeometry loadCollisionGeometryFromUsd(
    const UsdStageRefPtr& stage,
    const Skeleton& skeleton) {
  CollisionGeometry result;

  for (const auto& prim : stage->Traverse()) {
    auto typeAttr = prim.GetAttribute(TfToken("momentum:type"));
    if (!typeAttr) {
      continue;
    }

    std::string type;
    if (!typeAttr.Get(&type) || type != "collision_capsule") {
      continue;
    }

    TaperedCapsule capsule;

    std::string parentName;
    if (auto attr = prim.GetAttribute(TfToken("momentum:parent")); attr) {
      attr.Get(&parentName);
      capsule.parent = skeleton.getJointIdByName(parentName);
    }

    if (auto attr = prim.GetAttribute(TfToken("momentum:length")); attr) {
      attr.Get(&capsule.length);
    }

    GfVec2f radius(0.0f, 0.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:radius")); attr) {
      attr.Get(&radius);
      capsule.radius = Eigen::Vector2f(radius[0], radius[1]);
    }

    GfVec3f translation(0.0f, 0.0f, 0.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:translation")); attr) {
      attr.Get(&translation);
      capsule.transformation.translation =
          Eigen::Vector3f(translation[0], translation[1], translation[2]);
    }

    GfQuatf rotation(1.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:rotation")); attr) {
      attr.Get(&rotation);
      capsule.transformation.rotation = Eigen::Quaternionf(
          rotation.GetReal(),
          rotation.GetImaginary()[0],
          rotation.GetImaginary()[1],
          rotation.GetImaginary()[2]);
    }

    result.push_back(capsule);
  }

  return result;
}

void saveLocatorsToUsd(
    const LocatorList& locators,
    const Skeleton& skeleton,
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath) {
  if (locators.empty() || skeleton.joints.empty()) {
    return;
  }

  auto locatorScope = UsdGeomScope::Define(stage, skelRootPath.AppendChild(TfToken("Locators")));

  for (size_t i = 0; i < locators.size(); ++i) {
    const auto& loc = locators[i];
    const std::string baseName = loc.name.empty() ? "locator" : loc.name;
    const std::string primName = baseName + "_" + std::to_string(i);

    auto primPath = locatorScope.GetPath().AppendChild(TfToken(primName));
    auto prim = stage->DefinePrim(primPath);

    prim.CreateAttribute(TfToken("momentum:type"), SdfValueTypeNames->String)
        .Set(std::string("locator"));
    prim.CreateAttribute(TfToken("momentum:name"), SdfValueTypeNames->String).Set(loc.name);

    const std::string parentName =
        (loc.parent < skeleton.joints.size()) ? skeleton.joints[loc.parent].name : "";
    prim.CreateAttribute(TfToken("momentum:parent"), SdfValueTypeNames->String).Set(parentName);

    prim.CreateAttribute(TfToken("momentum:offset"), SdfValueTypeNames->Float3)
        .Set(GfVec3f(loc.offset.x(), loc.offset.y(), loc.offset.z()));

    prim.CreateAttribute(TfToken("momentum:weight"), SdfValueTypeNames->Float).Set(loc.weight);

    prim.CreateAttribute(TfToken("momentum:locked"), SdfValueTypeNames->Int3)
        .Set(GfVec3i(loc.locked.x(), loc.locked.y(), loc.locked.z()));

    prim.CreateAttribute(TfToken("momentum:limitOrigin"), SdfValueTypeNames->Float3)
        .Set(GfVec3f(loc.limitOrigin.x(), loc.limitOrigin.y(), loc.limitOrigin.z()));

    prim.CreateAttribute(TfToken("momentum:limitWeight"), SdfValueTypeNames->Float3)
        .Set(GfVec3f(loc.limitWeight.x(), loc.limitWeight.y(), loc.limitWeight.z()));

    if (loc.attachedToSkin) {
      prim.CreateAttribute(TfToken("momentum:attachedToSkin"), SdfValueTypeNames->Bool).Set(true);
      prim.CreateAttribute(TfToken("momentum:skinOffset"), SdfValueTypeNames->Float)
          .Set(loc.skinOffset);
    }
  }
}

LocatorList loadLocatorsFromUsd(const UsdStageRefPtr& stage, const Skeleton& skeleton) {
  LocatorList result;

  for (const auto& prim : stage->Traverse()) {
    auto typeAttr = prim.GetAttribute(TfToken("momentum:type"));
    if (!typeAttr) {
      continue;
    }

    std::string type;
    if (!typeAttr.Get(&type) || type != "locator") {
      continue;
    }

    Locator loc;
    if (auto attr = prim.GetAttribute(TfToken("momentum:name")); attr) {
      attr.Get(&loc.name);
    } else {
      loc.name = prim.GetName().GetString();
    }

    std::string parentName;
    if (auto attr = prim.GetAttribute(TfToken("momentum:parent")); attr) {
      attr.Get(&parentName);
      if (!parentName.empty()) {
        loc.parent = skeleton.getJointIdByName(parentName);
      }
    }

    GfVec3f offset(0.0f, 0.0f, 0.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:offset")); attr) {
      attr.Get(&offset);
      loc.offset = Eigen::Vector3f(offset[0], offset[1], offset[2]);
    }

    if (auto attr = prim.GetAttribute(TfToken("momentum:weight")); attr) {
      attr.Get(&loc.weight);
    }

    GfVec3i locked(0, 0, 0);
    if (auto attr = prim.GetAttribute(TfToken("momentum:locked")); attr) {
      attr.Get(&locked);
      loc.locked = Eigen::Vector3i(locked[0], locked[1], locked[2]);
    }

    GfVec3f limitOrigin(0.0f, 0.0f, 0.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:limitOrigin")); attr) {
      attr.Get(&limitOrigin);
      loc.limitOrigin = Eigen::Vector3f(limitOrigin[0], limitOrigin[1], limitOrigin[2]);
    }

    GfVec3f limitWeight(0.0f, 0.0f, 0.0f);
    if (auto attr = prim.GetAttribute(TfToken("momentum:limitWeight")); attr) {
      attr.Get(&limitWeight);
      loc.limitWeight = Eigen::Vector3f(limitWeight[0], limitWeight[1], limitWeight[2]);
    }

    if (auto attr = prim.GetAttribute(TfToken("momentum:attachedToSkin")); attr) {
      attr.Get(&loc.attachedToSkin);
    }

    if (auto attr = prim.GetAttribute(TfToken("momentum:skinOffset")); attr) {
      attr.Get(&loc.skinOffset);
    }

    result.push_back(std::move(loc));
  }

  return result;
}

} // namespace momentum
