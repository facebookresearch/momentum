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

    VtArray<int> topology = skelQuery.GetTopology().GetParentIndices();

    for (size_t i = 0; i < jointNames.size(); ++i) {
      Joint joint;
      joint.name = jointNames[i].GetString();

      if (i < topology.size() && topology[i] >= 0) {
        joint.parent = topology[i];
      } else {
        joint.parent = kInvalidIndex;
      }

      joint.preRotation = Eigen::Quaternionf::Identity();
      joint.translationOffset = Eigen::Vector3f::Zero();

      skeleton.joints.push_back(joint);
    }

    // Get bind transforms (rest pose)
    VtArray<GfMatrix4d> bindTransforms;
    if (skelQuery.GetJointWorldBindTransforms(&bindTransforms)) {
      MT_CHECK(bindTransforms.size() == skeleton.joints.size());

      for (size_t i = 0; i < bindTransforms.size(); ++i) {
        auto matrix = toEigenMatrix4f(bindTransforms[i]);

        // Extract translation
        skeleton.joints[i].translationOffset = matrix.block<3, 1>(0, 3);

        // Extract rotation (simplified - could be improved)
        Eigen::Matrix3f rotMatrix = matrix.block<3, 3>(0, 0);
        Eigen::Quaternionf quat(rotMatrix);
        skeleton.joints[i].preRotation = quat.normalized();
      }
    }

    break; // Use first skeleton found
  }

  return skeleton;
}

void saveSkeletonToUsd(const Skeleton& skeleton, UsdSkelSkeleton& skelPrim) {
  VtArray<TfToken> jointNames;
  jointNames.reserve(skeleton.joints.size());
  for (const auto& joint : skeleton.joints) {
    jointNames.push_back(TfToken(joint.name));
  }
  skelPrim.GetJointsAttr().Set(jointNames);

  VtArray<GfMatrix4d> bindTransforms;
  bindTransforms.reserve(skeleton.joints.size());
  for (const auto& joint : skeleton.joints) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 1>(0, 3) = joint.translationOffset;
    bindTransforms.push_back(toGfMatrix4d(transform));
  }
  skelPrim.GetBindTransformsAttr().Set(bindTransforms);
}

} // namespace momentum
