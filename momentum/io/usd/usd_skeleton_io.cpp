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
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
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

} // namespace momentum
