/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Internal helpers shared between fbx_io.cpp and fbx_builder.cpp.
// This header is NOT part of the public API.

#pragma once

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/marker.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/io/file_save_options.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <fbxsdk/scene/geometry/fbxcluster.h>

// **FBX SDK**
// They do the most awful things to isnan in here
#include <fbxsdk.h>
#include <fbxsdk/fileio/fbxiosettings.h>

#ifdef isnan
#undef isnan
#endif

#include <span>
#include <unordered_map>

namespace momentum::fbx_internal {

// ============================================================================
// Coordinate system helpers
// ============================================================================

[[nodiscard]] inline ::fbxsdk::FbxAxisSystem::EUpVector toFbx(const FbxUpVector upVector) {
  switch (upVector) {
    case FbxUpVector::XAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eXAxis;
    case FbxUpVector::YAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eYAxis;
    case FbxUpVector::ZAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eZAxis;
    default:
      MT_THROW("Unsupported up vector");
  }
}

[[nodiscard]] inline ::fbxsdk::FbxAxisSystem::EFrontVector toFbx(const FbxFrontVector frontVector) {
  switch (frontVector) {
    case FbxFrontVector::ParityEven:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityEven;
    case FbxFrontVector::ParityOdd:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityOdd;
    default:
      MT_THROW("Unsupported front vector");
  }
}

[[nodiscard]] inline ::fbxsdk::FbxAxisSystem::ECoordSystem toFbx(const FbxCoordSystem coordSystem) {
  switch (coordSystem) {
    case FbxCoordSystem::RightHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eRightHanded;
    case FbxCoordSystem::LeftHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eLeftHanded;
    default:
      MT_THROW("Unsupported coordinate system");
  }
}

[[nodiscard]] inline ::fbxsdk::FbxAxisSystem toFbx(const FbxCoordSystemInfo& coordSystemInfo) {
  return {
      toFbx(coordSystemInfo.upVector),
      toFbx(coordSystemInfo.frontVector),
      toFbx(coordSystemInfo.coordSystem)};
}

// ============================================================================
// Frame rate
// ============================================================================

inline void setFrameRate(::fbxsdk::FbxScene* scene, const double framerate) {
  if (std::abs(framerate - 30.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames30);
  } else if (std::abs(framerate - 24.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames24);
  } else if (std::abs(framerate - 48.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames48);
  } else if (std::abs(framerate - 50.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames50);
  } else if (std::abs(framerate - 60.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames60);
  } else if (std::abs(framerate - 72.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames72);
  } else if (std::abs(framerate - 96.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames96);
  } else if (std::abs(framerate - 100.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames100);
  } else if (std::abs(framerate - 120.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames120);
  } else if (std::abs(framerate - 1000.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames1000);
  } else {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eCustom);
    scene->GetGlobalSettings().SetCustomFrameRate(framerate);
  }
}

// ============================================================================
// Skeleton
// ============================================================================

struct SkeletonNodeResult {
  std::vector<::fbxsdk::FbxNode*> nodes;
  std::unordered_map<size_t, fbxsdk::FbxNode*> jointToNodeMap;
  ::fbxsdk::FbxNode* rootNode = nullptr;
};

inline SkeletonNodeResult createSkeletonNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene) {
  SkeletonNodeResult result;

  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    ::fbxsdk::FbxNode* skeletonNode = ::fbxsdk::FbxNode::Create(scene, joint.name.c_str());
    ::fbxsdk::FbxSkeleton* skeletonAttribute =
        ::fbxsdk::FbxSkeleton::Create(scene, joint.name.c_str());

    if (joint.parent == kInvalidIndex) {
      result.rootNode = skeletonNode;
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eRoot);
    } else {
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eLimbNode);
    }
    skeletonNode->SetNodeAttribute(skeletonAttribute);
    result.jointToNodeMap[i] = skeletonNode;

    skeletonNode->LclTranslation.Set(FbxDouble3(
        joint.translationOffset[0], joint.translationOffset[1], joint.translationOffset[2]));

    const auto angles = rotationMatrixToEulerZYX(joint.preRotation.toRotationMatrix());
    skeletonNode->SetPivotState(FbxNode::eSourcePivot, FbxNode::ePivotActive);
    skeletonNode->SetRotationActive(true);
    skeletonNode->SetPreRotation(
        ::fbxsdk::FbxNode::eSourcePivot,
        FbxDouble3(toDeg(angles[2]), toDeg(angles[1]), toDeg(angles[0])));

    result.nodes.emplace_back(skeletonNode);
  }

  // Second pass: handle the parenting, in case the parents are not in order
  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];
    auto* skeletonNode = result.jointToNodeMap[i];
    if (joint.parent != kInvalidIndex) {
      MT_THROW_IF(
          joint.parent >= result.nodes.size(),
          "Joint parent index {} exceeds nodes size {}",
          joint.parent,
          result.nodes.size());
      result.nodes[joint.parent]->AddChild(skeletonNode);
    }
  }

  return result;
}

// ============================================================================
// Metadata, locators, collision geometry
// ============================================================================

inline void addMetaData(::fbxsdk::FbxNode* skeletonRootNode, const Character& character) {
  if (skeletonRootNode != nullptr) {
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "metadata")
        .Set(FbxString(character.metadata.c_str()));
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "name")
        .Set(FbxString(character.name.c_str()));
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "RigName")
        .Set(FbxString(character.name.c_str()));
  }
}

inline void createLocatorNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes) {
  for (const auto& loc : character.locators) {
    ::fbxsdk::FbxMarker* markerAttribute = ::fbxsdk::FbxMarker::Create(scene, loc.name.c_str());
    markerAttribute->Look.Set(::fbxsdk::FbxMarker::ELook::eHardCross);

    ::fbxsdk::FbxNode* locatorNode = ::fbxsdk::FbxNode::Create(scene, loc.name.c_str());
    locatorNode->SetNodeAttribute(markerAttribute);
    locatorNode->LclTranslation.Set(FbxVector4(loc.offset[0], loc.offset[1], loc.offset[2]));

    if (loc.parent != kInvalidIndex) {
      MT_THROW_IF(
          loc.parent >= skeletonNodes.size(),
          "Locator parent index {} exceeds skeletonNodes size {}",
          loc.parent,
          skeletonNodes.size());
      skeletonNodes[loc.parent]->AddChild(locatorNode);
    }
  }
}

inline void createCollisionGeometryNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes) {
  if (!character.collision) {
    MT_LOGD(
        "No collision geometry found in character, skipping creation of collision geometry nodes");
    return;
  }

  const auto& collisions = *character.collision;
  for (auto i = 0u; i < collisions.size(); ++i) {
    const TaperedCapsule& collision = collisions[i];

    ::fbxsdk::FbxNode* collisionNode =
        ::fbxsdk::FbxNode::Create(scene, ("Collision " + std::to_string(i)).c_str());
    auto* nullNodeAttr = ::fbxsdk::FbxNull::Create(scene, "Null");
    collisionNode->SetNodeAttribute(nullNodeAttr);

    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxBoolDT, "Col_Type").Set(true);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Length")
        .Set(collision.length);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Rad_A")
        .Set(collision.radius[0]);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Rad_B")
        .Set(collision.radius[1]);

    collisionNode->LclTranslation.Set(FbxVector4(
        collision.transformation.translation.x(),
        collision.transformation.translation.y(),
        collision.transformation.translation.z()));
    const Vector3f rot = rotationMatrixToEulerXYZ<float>(
        collision.transformation.rotation.toRotationMatrix(), EulerConvention::Extrinsic);
    collisionNode->LclRotation.Set(FbxDouble3(toDeg(rot.x()), toDeg(rot.y()), toDeg(rot.z())));
    collisionNode->LclScaling.Set(FbxDouble3(1));

    if (collision.parent != kInvalidIndex) {
      MT_THROW_IF(
          collision.parent >= skeletonNodes.size(),
          "Collision parent index {} exceeds skeletonNodes size {}",
          collision.parent,
          skeletonNodes.size());
      skeletonNodes[collision.parent]->AddChild(collisionNode);
    } else {
      MT_LOGE("Found a collision node with no parent");
    }
  }
}

// ============================================================================
// Animation curves
// ============================================================================

inline std::pair<::fbxsdk::FbxAnimStack*, ::fbxsdk::FbxAnimLayer*> getOrCreateAnimStackAndLayer(
    ::fbxsdk::FbxScene* scene,
    const char* stackName) {
  ::fbxsdk::FbxAnimStack* animStack = nullptr;
  if (scene->GetSrcObjectCount<::fbxsdk::FbxAnimStack>() > 0) {
    animStack = scene->GetSrcObject<::fbxsdk::FbxAnimStack>(0);
  } else {
    animStack = ::fbxsdk::FbxAnimStack::Create(scene, stackName);
  }

  ::fbxsdk::FbxAnimLayer* animBaseLayer = nullptr;
  if (animStack->GetMemberCount<::fbxsdk::FbxAnimLayer>() > 0) {
    animBaseLayer = animStack->GetMember<::fbxsdk::FbxAnimLayer>(0);
  } else {
    animBaseLayer = ::fbxsdk::FbxAnimLayer::Create(scene, "Layer0");
    animStack->AddMember(animBaseLayer);
  }

  return {animStack, animBaseLayer};
}

struct JointCurveSetup {
  std::vector<::fbxsdk::FbxAnimCurve*> animCurves;
  std::vector<size_t> animCurvesIndex;
};

// Set up animation curves for each active joint parameter.
inline JointCurveSetup setupJointAnimCurves(
    const Character& character,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes,
    ::fbxsdk::FbxAnimLayer* animBaseLayer,
    const bool skipActiveJointParamCheck) {
  const auto& aj = character.parameterTransform.activeJointParams;

  JointCurveSetup setup;
  setup.animCurves.resize(character.skeleton.joints.size() * 9, nullptr);

  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const size_t jointIndex = i * kParametersPerJoint;
    const size_t index = i * 9;
    skeletonNodes[i]->LclTranslation.GetCurveNode(true);
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    if (skipActiveJointParamCheck || aj[jointIndex + 0]) {
      setup.animCurves[index + 0] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      setup.animCurvesIndex.push_back(index + 0);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 1]) {
      setup.animCurves[index + 1] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      setup.animCurvesIndex.push_back(index + 1);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 2]) {
      setup.animCurves[index + 2] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      setup.animCurvesIndex.push_back(index + 2);
    }
    skeletonNodes[i]->LclRotation.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 3]) {
      setup.animCurves[index + 3] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      setup.animCurvesIndex.push_back(index + 3);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 4]) {
      setup.animCurves[index + 4] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      setup.animCurvesIndex.push_back(index + 4);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 5]) {
      setup.animCurves[index + 5] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      setup.animCurvesIndex.push_back(index + 5);
    }
    skeletonNodes[i]->LclScaling.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 6]) {
      setup.animCurves[index + 6] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      setup.animCurvesIndex.push_back(index + 6);
      setup.animCurves[index + 7] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      setup.animCurvesIndex.push_back(index + 7);
      setup.animCurves[index + 8] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      setup.animCurvesIndex.push_back(index + 8);
    }
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
  }

  return setup;
}

// jointValues: (numJointParameters x numFrames) matrix of joint values
inline void createAnimationCurves(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes,
    const MatrixXf& jointValues,
    const double framerate,
    const bool skipActiveJointParamCheck) {
  setFrameRate(scene, framerate);

  const auto& aj = character.parameterTransform.activeJointParams;

  auto [animStack, animBaseLayer] = getOrCreateAnimStackAndLayer(scene, "Skeleton Animation Stack");

  auto [animCurves, animCurvesIndex] =
      setupJointAnimCurves(character, skeletonNodes, animBaseLayer, skipActiveJointParamCheck);

  ::fbxsdk::FbxTime time;
  for (const auto ai : animCurvesIndex) {
    const size_t jointIndex = ai / 9;
    const size_t jointOffset = ai % 9;
    const size_t parameterIndex =
        jointIndex * kParametersPerJoint + std::min(jointOffset, size_t(6));
    if (!skipActiveJointParamCheck && aj[parameterIndex] == 0) {
      continue;
    }

    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    animCurves[ai]->KeyModifyBegin();
    for (size_t f = 0; f < jointValues.cols(); f++) {
      time.SetSecondDouble(static_cast<double>(f) / framerate);

      float jointVal = jointValues(parameterIndex, f);

      if (jointOffset < 3 && jointIndex < character.skeleton.joints.size()) {
        jointVal += character.skeleton.joints[jointIndex].translationOffset[jointOffset];
      } else if (jointOffset >= 3 && jointOffset <= 5) {
        jointVal = toDeg(jointVal);
      } else {
        jointVal = std::pow(2.0f, jointVal);
      }

      const auto keyIndex = animCurves[ai]->KeyAdd(time);
      animCurves[ai]->KeySet(keyIndex, time, jointVal);
    }
    animCurves[ai]->KeyModifyEnd();
  }
  // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
}

// ============================================================================
// Skin weights
// ============================================================================

inline void saveSkinWeightsToFbx(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    ::fbxsdk::FbxMesh* mesh,
    const std::unordered_map<size_t, fbxsdk::FbxNode*>& jointToNodeMap) {
  ::fbxsdk::FbxSkin* fbxskin = ::fbxsdk::FbxSkin::Create(scene, "meshskinning");
  fbxskin->SetSkinningType(::fbxsdk::FbxSkin::eLinear);
  fbxskin->SetGeometry(mesh);
  FbxAMatrix meshTransform;
  meshTransform.SetIdentity();
  for (const auto& jointNode : jointToNodeMap) {
    size_t jointIdx = jointNode.first;
    auto* fbxJointNode = jointNode.second;

    std::ostringstream s;
    s << "skinningcluster_" << jointIdx;
    FbxCluster* pCluster = ::fbxsdk::FbxCluster::Create(scene, s.str().c_str());
    pCluster->SetLinkMode(::fbxsdk::FbxCluster::ELinkMode::eNormalize);
    pCluster->SetLink(fbxJointNode);

    ::fbxsdk::FbxAMatrix globalMatrix = fbxJointNode->EvaluateLocalTransform();
    ::fbxsdk::FbxNode* pParent = fbxJointNode->GetParent();
    while (pParent != nullptr) {
      globalMatrix = pParent->EvaluateLocalTransform() * globalMatrix;
      pParent = pParent->GetParent();
    }
    pCluster->SetTransformLinkMatrix(globalMatrix);
    pCluster->SetTransformMatrix(meshTransform);

    for (int i = 0; i < character.skinWeights->index.rows(); i++) {
      for (int j = 0; j < character.skinWeights->index.cols(); j++) {
        auto boneIndex = character.skinWeights->index(i, j);
        if (boneIndex == jointNode.first && character.skinWeights->weight(i, j) > 0) {
          pCluster->AddControlPointIndex(i, character.skinWeights->weight(i, j));
        }
      }
    }
    fbxskin->AddCluster(pCluster);
  }
  mesh->AddDeformer(fbxskin);
}

// ============================================================================
// Polygon / UV helpers
// ============================================================================

inline void writePolygonsToFbxMesh(const Mesh& mesh, ::fbxsdk::FbxMesh* lMesh) {
  if (!mesh.polyFaces.empty() && !mesh.polyFaceSizes.empty()) {
    uint32_t offset = 0;
    for (const auto polySize : mesh.polyFaceSizes) {
      lMesh->BeginPolygon();
      for (uint32_t i = 0; i < polySize; ++i) {
        lMesh->AddPolygon(mesh.polyFaces[offset + i]);
      }
      lMesh->EndPolygon();
      offset += polySize;
    }
  } else {
    for (const auto& face : mesh.faces) {
      lMesh->BeginPolygon();
      for (int i = 0; i < 3; i++) {
        lMesh->AddPolygon(face[i]);
      }
      lMesh->EndPolygon();
    }
  }
}

inline void writeTextureUVIndicesToFbxMesh(
    const Mesh& mesh,
    ::fbxsdk::FbxMesh* lMesh,
    fbxsdk::FbxLayerElement::EType uvType) {
  if (!mesh.polyFaces.empty() && !mesh.polyFaceSizes.empty()) {
    uint32_t offset = 0;
    int polyIdx = 0;
    for (const auto polySize : mesh.polyFaceSizes) {
      for (uint32_t i = 0; i < polySize; ++i) {
        const uint32_t texIdx = mesh.polyTexcoordFaces.empty() ? mesh.polyFaces[offset + i]
                                                               : mesh.polyTexcoordFaces[offset + i];
        lMesh->SetTextureUVIndex(polyIdx, i, texIdx, uvType);
      }
      offset += polySize;
      polyIdx++;
    }
  } else {
    for (int faceIdx = 0; faceIdx < static_cast<int>(mesh.texcoord_faces.size()); ++faceIdx) {
      const auto& texcoords = mesh.texcoord_faces[faceIdx];
      lMesh->SetTextureUVIndex(faceIdx, 0, texcoords[0], uvType);
      lMesh->SetTextureUVIndex(faceIdx, 1, texcoords[1], uvType);
      lMesh->SetTextureUVIndex(faceIdx, 2, texcoords[2], uvType);
    }
  }
}

// ============================================================================
// Mesh with blend shapes
// ============================================================================

struct MeshBlendShapeResult {
  std::vector<::fbxsdk::FbxBlendShapeChannel*> blendShapeChannels;
  std::vector<::fbxsdk::FbxBlendShapeChannel*> faceExprChannels;
};

inline std::vector<::fbxsdk::FbxBlendShapeChannel*> saveBlendShapeGeometryToFbx(
    const BlendShapeBase& blendShape,
    ::fbxsdk::FbxScene* scene,
    ::fbxsdk::FbxMesh* mesh,
    const std::string& deformerName,
    const std::string& channelPrefix) {
  std::vector<::fbxsdk::FbxBlendShapeChannel*> channels;

  ::fbxsdk::FbxBlendShape* fbxBlendShape =
      ::fbxsdk::FbxBlendShape::Create(scene, deformerName.c_str());
  fbxBlendShape->SetGeometry(mesh);

  const auto& shapes = blendShape.getShapeVectors();
  const auto& names = blendShape.getShapeNames();
  const int numVertices = mesh->GetControlPointsCount();

  for (Eigen::Index i = 0; i < blendShape.shapeSize(); i++) {
    ::fbxsdk::FbxBlendShapeChannel* channelPtr =
        ::fbxsdk::FbxBlendShapeChannel::Create(scene, (channelPrefix + std::to_string(i)).c_str());
    fbxBlendShape->AddBlendShapeChannel(channelPtr);
    channels.push_back(channelPtr);

    ::fbxsdk::FbxShape* shape = ::fbxsdk::FbxShape::Create(scene, names[i].c_str());
    shape->SetControlPointCount(numVertices);
    for (int j = 0; j < numVertices; j++) {
      const auto basePoint = mesh->GetControlPointAt(j);
      if (j * 3 + 2 < shapes.rows()) {
        const Eigen::Vector3f delta = shapes.block<3, 1>(j * 3, i);
        shape->SetControlPointAt(
            FbxVector4(
                basePoint[0] + delta.x(), basePoint[1] + delta.y(), basePoint[2] + delta.z()),
            j);
      } else {
        shape->SetControlPointAt(basePoint, j);
      }
      shape->AddControlPointIndex(j);
    }
    channelPtr->AddTargetShape(shape, 100.0);
  }

  mesh->AddDeformer(fbxBlendShape);
  return channels;
}

inline void createBlendShapeAnimationCurves(
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxBlendShapeChannel*>& channels,
    const MatrixXf& weights,
    const double framerate) {
  if (channels.empty() || weights.cols() == 0) {
    return;
  }

  setFrameRate(scene, framerate);

  auto [animStack, animBaseLayer] =
      getOrCreateAnimStackAndLayer(scene, "BlendShape Animation Stack");

  const Eigen::Index numChannels =
      std::min(static_cast<Eigen::Index>(channels.size()), weights.rows());

  for (Eigen::Index i = 0; i < numChannels; i++) {
    ::fbxsdk::FbxAnimCurve* curve = channels[i]->DeformPercent.GetCurve(animBaseLayer, true);
    if (curve == nullptr) {
      continue;
    }

    curve->KeyModifyBegin();
    for (Eigen::Index f = 0; f < weights.cols(); f++) {
      ::fbxsdk::FbxTime time;
      time.SetSecondDouble(static_cast<double>(f) / framerate);
      const float value = weights(i, f) * 100.0f;
      const auto keyIndex = curve->KeyAdd(time);
      curve->KeySet(keyIndex, time, value);
    }
    curve->KeyModifyEnd();
  }
}

// Create the mesh node with vertices, normals, UVs, skinning, and blend shapes.
// parentNode is the node to parent the mesh under.
inline MeshBlendShapeResult createMeshNode(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    ::fbxsdk::FbxNode* parentNode,
    const std::unordered_map<size_t, fbxsdk::FbxNode*>& jointToNodeMap,
    Permissive permissive,
    const std::string& meshName = "body_mesh") {
  MeshBlendShapeResult result;
  const auto numVertices = static_cast<int>(character.mesh.get()->vertices.size());
  ::fbxsdk::FbxNode* meshNode = ::fbxsdk::FbxNode::Create(scene, meshName.c_str());
  ::fbxsdk::FbxMesh* lMesh = ::fbxsdk::FbxMesh::Create(scene, "mesh");
  lMesh->SetControlPointCount(numVertices);
  lMesh->InitNormals(numVertices);
  for (int i = 0; i < numVertices; i++) {
    FbxVector4 point(
        character.mesh.get()->vertices[i].x(),
        character.mesh.get()->vertices[i].y(),
        character.mesh.get()->vertices[i].z());
    FbxVector4 normal(
        character.mesh.get()->normals[i].x(),
        character.mesh.get()->normals[i].y(),
        character.mesh.get()->normals[i].z());
    lMesh->SetControlPointAt(point, normal, i);
  }
  writePolygonsToFbxMesh(*character.mesh, lMesh);
  lMesh->BuildMeshEdgeArray();
  meshNode->SetNodeAttribute(lMesh);

  // Add texture coordinates
  if (!character.mesh->texcoords.empty()) {
    const fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;
    lMesh->InitTextureUV(0, uvType);
    lMesh->InitTextureUVIndices(::fbxsdk::FbxLayerElement::EMappingMode::eByPolygonVertex, uvType);
    for (const auto& texcoords : character.mesh->texcoords) {
      lMesh->AddTextureUV(::fbxsdk::FbxVector2(texcoords[0], 1.0f - texcoords[1]), uvType);
    }
    writeTextureUVIndicesToFbxMesh(*character.mesh, lMesh, uvType);
  }

  // Add skinning
  MT_THROW_IF(
      permissive == Permissive::No && !character.skinWeights,
      "The character '{}' has no skinning weights and permissive mode is not enabled.",
      character.name);

  if (character.skinWeights != nullptr) {
    saveSkinWeightsToFbx(character, scene, lMesh, jointToNodeMap);
  }

  // Add blend shapes
  if (character.blendShape && character.blendShape->shapeSize() > 0) {
    const auto& base = character.blendShape->getBaseShape();
    for (int j = 0; j < numVertices; j++) {
      FbxVector4 point(base[j].x(), base[j].y(), base[j].z());
      lMesh->SetControlPointAt(point, j);
    }
    result.blendShapeChannels =
        saveBlendShapeGeometryToFbx(*character.blendShape, scene, lMesh, "blendshape", "channel_");
  }

  // Add face expression blend shapes
  if (character.faceExpressionBlendShape && character.faceExpressionBlendShape->shapeSize() > 0) {
    result.faceExprChannels = saveBlendShapeGeometryToFbx(
        *character.faceExpressionBlendShape,
        scene,
        lMesh,
        "face_expression_blendshape",
        "face_expr_channel_");
  }

  parentNode->AddChild(meshNode);
  return result;
}

// ============================================================================
// Blend shape weight extraction
// ============================================================================

inline MatrixXf extractBlendShapeWeights(const Character& character, const MatrixXf& poses) {
  const auto& pt = character.parameterTransform;
  if (poses.cols() == 0 || pt.blendShapeParameters.size() == 0 || !character.blendShape ||
      character.blendShape->shapeSize() == 0) {
    return {};
  }

  const Eigen::Index numShapes = pt.blendShapeParameters.size();
  MatrixXf weights(numShapes, poses.cols());
  for (Eigen::Index i = 0; i < numShapes; i++) {
    const auto paramIdx = pt.blendShapeParameters(i);
    if (paramIdx >= 0 && paramIdx < poses.rows()) {
      weights.row(i) = poses.row(paramIdx);
    } else {
      weights.row(i).setZero();
    }
  }
  return weights;
}

inline MatrixXf extractFaceExpressionWeights(const Character& character, const MatrixXf& poses) {
  const auto& pt = character.parameterTransform;
  if (poses.cols() == 0 || pt.faceExpressionParameters.size() == 0 ||
      !character.faceExpressionBlendShape || character.faceExpressionBlendShape->shapeSize() == 0) {
    return {};
  }

  const Eigen::Index numExprs = pt.faceExpressionParameters.size();
  MatrixXf weights(numExprs, poses.cols());
  for (Eigen::Index i = 0; i < numExprs; i++) {
    const auto paramIdx = pt.faceExpressionParameters(i);
    if (paramIdx >= 0 && paramIdx < poses.rows()) {
      weights.row(i) = poses.row(paramIdx);
    } else {
      weights.row(i).setZero();
    }
  }
  return weights;
}

// ============================================================================
// Marker nodes
// ============================================================================

// Custom property names for marker identification
constexpr const char* kMomentumMarkersRootProperty = "Momentum_Markers_Root";
constexpr const char* kMomentumMarkerProperty = "Momentum_Marker";

inline std::vector<::fbxsdk::FbxNode*> createMarkerNodes(
    ::fbxsdk::FbxScene* scene,
    std::span<const std::vector<Marker>> markerSequence,
    const double framerate) {
  std::vector<::fbxsdk::FbxNode*> markerNodes;
  if (markerSequence.empty()) {
    return markerNodes;
  }

  setFrameRate(scene, framerate);

  ::fbxsdk::FbxNode* markersRootNode = ::fbxsdk::FbxNode::Create(scene, "Markers");
  ::fbxsdk::FbxNull* markersRootAttr = ::fbxsdk::FbxNull::Create(scene, "MarkersRootNull");
  markersRootNode->SetNodeAttribute(markersRootAttr);

  ::fbxsdk::FbxProperty::Create(markersRootNode, ::fbxsdk::FbxBoolDT, kMomentumMarkersRootProperty)
      .Set(true);

  std::map<std::string, size_t> markerNameToIndex;
  std::vector<std::string> markerNames;
  std::vector<std::vector<float>> timestamps;
  std::vector<std::vector<Vector3d>> markerPositions;

  for (size_t frameIndex = 0; frameIndex < markerSequence.size(); ++frameIndex) {
    const float timestamp = static_cast<float>(frameIndex) / static_cast<float>(framerate);
    for (const auto& marker : markerSequence[frameIndex]) {
      if (marker.occluded) {
        continue;
      }

      if (markerNameToIndex.count(marker.name) == 0) {
        const auto index = timestamps.size();
        timestamps.emplace_back();
        markerPositions.emplace_back();
        markerNameToIndex[marker.name] = index;
        markerNames.emplace_back(marker.name);
      }

      const auto& index = markerNameToIndex.at(marker.name);
      MT_THROW_IF(
          index >= timestamps.size() || index >= markerPositions.size(),
          "Marker index {} exceeds container size",
          index);
      timestamps[index].push_back(timestamp);
      markerPositions[index].push_back(marker.pos);
    }
  }

  for (const auto& markerName : markerNames) {
    ::fbxsdk::FbxMarker* markerAttribute = ::fbxsdk::FbxMarker::Create(scene, markerName.c_str());
    markerAttribute->Look.Set(::fbxsdk::FbxMarker::ELook::eHardCross);

    ::fbxsdk::FbxNode* markerNode = ::fbxsdk::FbxNode::Create(scene, markerName.c_str());
    markerNode->SetNodeAttribute(markerAttribute);

    ::fbxsdk::FbxProperty::Create(markerNode, ::fbxsdk::FbxBoolDT, kMomentumMarkerProperty)
        .Set(true);

    markerNode->LclTranslation.Set(FbxVector4(0.0, 0.0, 0.0));
    markersRootNode->AddChild(markerNode);
    markerNodes.push_back(markerNode);
  }

  scene->GetRootNode()->AddChild(markersRootNode);

  if (!timestamps.empty() && !timestamps[0].empty()) {
    auto [animStack, animBaseLayer] = getOrCreateAnimStackAndLayer(scene, "Marker Animation Stack");

    for (size_t j = 0; j < markerNames.size(); ++j) {
      if (timestamps[j].empty()) {
        continue;
      }

      auto* markerNode = markerNodes.at(j);

      markerNode->LclTranslation.GetCurveNode(animBaseLayer, true);
      ::fbxsdk::FbxAnimCurve* curveX =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      ::fbxsdk::FbxAnimCurve* curveY =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      ::fbxsdk::FbxAnimCurve* curveZ =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

      curveX->KeyModifyBegin();
      curveY->KeyModifyBegin();
      curveZ->KeyModifyBegin();

      for (size_t k = 0; k < timestamps[j].size(); ++k) {
        MT_THROW_IF(
            j >= markerPositions.size() || k >= markerPositions[j].size(),
            "Marker position index out of bounds: j={}, k={}",
            j,
            k);
        ::fbxsdk::FbxTime time;
        time.SetSecondDouble(timestamps[j][k]);

        const auto& pos = markerPositions[j][k];

        const auto keyIndexX = curveX->KeyAdd(time);
        curveX->KeySet(keyIndexX, time, static_cast<float>(pos.x()));

        const auto keyIndexY = curveY->KeyAdd(time);
        curveY->KeySet(keyIndexY, time, static_cast<float>(pos.y()));

        const auto keyIndexZ = curveZ->KeyAdd(time);
        curveZ->KeySet(keyIndexZ, time, static_cast<float>(pos.z()));
      }

      curveX->KeyModifyEnd();
      curveY->KeyModifyEnd();
      curveZ->KeyModifyEnd();
    }
  }

  return markerNodes;
}

// ============================================================================
// Namespace prefix
// ============================================================================

inline void prependNamespaceToAllNodes(
    ::fbxsdk::FbxNode* node,
    const std::string& namespacePrefix) {
  if (namespacePrefix.empty() || node == nullptr) {
    return;
  }

  const std::string currentName = node->GetName();
  const std::string newName = namespacePrefix + currentName;
  node->SetName(newName.c_str());

  for (int i = 0; i < node->GetChildCount(); ++i) {
    prependNamespaceToAllNodes(node->GetChild(i), namespacePrefix);
  }
}

} // namespace momentum::fbx_internal
