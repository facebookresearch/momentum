/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"
#include "momentum/io/fbx/openfbx_loader.h"

#ifdef MOMENTUM_WITH_FBX_SDK
#include "momentum/character/blend_shape.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/marker.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/fbx/fbx_memory_stream.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
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

#include <variant>
#endif // MOMENTUM_WITH_FBX_SDK

namespace momentum {

#ifdef MOMENTUM_WITH_FBX_SDK

namespace {

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EUpVector toFbx(const FbxUpVector upVector) {
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

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EFrontVector toFbx(const FbxFrontVector frontVector) {
  switch (frontVector) {
    case FbxFrontVector::ParityEven:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityEven;
    case FbxFrontVector::ParityOdd:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityOdd;
    default:
      MT_THROW("Unsupported front vector");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem::ECoordSystem toFbx(const FbxCoordSystem coordSystem) {
  switch (coordSystem) {
    case FbxCoordSystem::RightHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eRightHanded;
    case FbxCoordSystem::LeftHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eLeftHanded;
    default:
      MT_THROW("Unsupported coordinate system");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem toFbx(const FbxCoordSystemInfo& coordSystemInfo) {
  return {
      toFbx(coordSystemInfo.upVector),
      toFbx(coordSystemInfo.frontVector),
      toFbx(coordSystemInfo.coordSystem)};
}

void createLocatorNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes) {
  for (const auto& loc : character.locators) {
    ::fbxsdk::FbxMarker* markerAttribute = ::fbxsdk::FbxMarker::Create(scene, loc.name.c_str());
    markerAttribute->Look.Set(::fbxsdk::FbxMarker::ELook::eHardCross);

    // create the node
    ::fbxsdk::FbxNode* locatorNode = ::fbxsdk::FbxNode::Create(scene, loc.name.c_str());
    locatorNode->SetNodeAttribute(markerAttribute);

    // set translation offset
    locatorNode->LclTranslation.Set(FbxVector4(loc.offset[0], loc.offset[1], loc.offset[2]));

    // set parent if it has one
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

void createCollisionGeometryNodes(
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
    auto* nullNodeAttr =
        ::fbxsdk::FbxNull::Create(scene, "Null"); // TODO: Find a good node attribute
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

void setFrameRate(::fbxsdk::FbxScene* scene, const double framerate) {
  // enumerate common frame rates first, then resort to custom framerate
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

// jointValues: (numJointParameters x numFrames) matrix of joint values
void createAnimationCurves(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes,
    const MatrixXf& jointValues,
    const double framerate,
    const bool skipActiveJointParamCheck) {
  // set the framerate
  setFrameRate(scene, framerate);

  const auto& aj = character.parameterTransform.activeJointParams;

  // create animation stack
  ::fbxsdk::FbxAnimStack* animStack =
      ::fbxsdk::FbxAnimStack::Create(scene, "Skeleton Animation Stack");
  ::fbxsdk::FbxAnimLayer* animBaseLayer = ::fbxsdk::FbxAnimLayer::Create(scene, "Layer0");
  animStack->AddMember(animBaseLayer);

  // create anim curves for each joint and store them in an array
  std::vector<::fbxsdk::FbxAnimCurve*> animCurves(character.skeleton.joints.size() * 9, nullptr);
  std::vector<size_t> animCurvesIndex;
  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const size_t jointIndex = i * kParametersPerJoint;
    const size_t index = i * 9;
    skeletonNodes[i]->LclTranslation.GetCurveNode(true);
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    if (skipActiveJointParamCheck || aj[jointIndex + 0]) {
      animCurves[index + 0] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 0);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 1]) {
      animCurves[index + 1] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 1);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 2]) {
      animCurves[index + 2] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 2);
    }
    skeletonNodes[i]->LclRotation.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 3]) {
      animCurves[index + 3] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 3);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 4]) {
      animCurves[index + 4] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 4);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 5]) {
      animCurves[index + 5] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 5);
    }
    skeletonNodes[i]->LclScaling.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 6]) {
      animCurves[index + 6] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 6);
      animCurves[index + 7] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 7);
      animCurves[index + 8] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 8);
    }
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
  }

  // calculate the actual motion and set the keyframes
  ::fbxsdk::FbxTime time;
  // now go over each animCurveIndex and generate the curve
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
      // set keyframe time
      time.SetSecondDouble(static_cast<double>(f) / framerate);

      // get joint value
      float jointVal = jointValues(parameterIndex, f);

      // add translation offset for tx values
      if (jointOffset < 3 && jointIndex < character.skeleton.joints.size()) {
        jointVal += character.skeleton.joints[jointIndex].translationOffset[jointOffset];
      }
      // convert to degrees
      else if (jointOffset >= 3 && jointOffset <= 5) {
        jointVal = toDeg(jointVal);
      }
      // convert to non-exponential scaling
      else {
        jointVal = std::pow(2.0f, jointVal);
      }

      const auto keyIndex = animCurves[ai]->KeyAdd(time);
      animCurves[ai]->KeySet(keyIndex, time, jointVal);
    }
    animCurves[ai]->KeyModifyEnd();
  }
  // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
}

// Get or create the animation stack and base layer for the scene.
// Returns a pair of (animStack, animBaseLayer).
std::pair<::fbxsdk::FbxAnimStack*, ::fbxsdk::FbxAnimLayer*> getOrCreateAnimStackAndLayer(
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

// Post-process function to prepend namespace to all nodes in the scene
void prependNamespaceToAllNodes(::fbxsdk::FbxNode* node, const std::string& namespacePrefix) {
  if (namespacePrefix.empty() || node == nullptr) {
    return;
  }

  // Prepend namespace to the node name
  const std::string currentName = node->GetName();
  const std::string newName = namespacePrefix + currentName;
  node->SetName(newName.c_str());

  // Recursively process all children
  for (int i = 0; i < node->GetChildCount(); ++i) {
    prependNamespaceToAllNodes(node->GetChild(i), namespacePrefix);
  }
}

// Create marker nodes under a "Markers" hierarchy and add animation for their translations
std::vector<::fbxsdk::FbxNode*> createMarkerNodes(
    ::fbxsdk::FbxScene* scene,
    std::span<const std::vector<Marker>> markerSequence,
    const double framerate) {
  std::vector<::fbxsdk::FbxNode*> markerNodes;

  if (markerSequence.empty()) {
    return markerNodes;
  }

  // Set the framerate for the scene
  setFrameRate(scene, framerate);

  // Create a root node for all markers
  ::fbxsdk::FbxNode* markersRootNode = ::fbxsdk::FbxNode::Create(scene, "Markers");
  ::fbxsdk::FbxNull* markersRootAttr = ::fbxsdk::FbxNull::Create(scene, "MarkersRootNull");
  markersRootNode->SetNodeAttribute(markersRootAttr);

  // Add custom property to identify this as a Momentum markers root
  ::fbxsdk::FbxProperty::Create(markersRootNode, ::fbxsdk::FbxBoolDT, kMomentumMarkersRootProperty)
      .Set(true);

  // Collect unique marker names and organize data per marker
  std::map<std::string, size_t> markerNameToIndex;
  std::vector<std::string> markerNames;
  std::vector<std::vector<float>> timestamps;
  std::vector<std::vector<Vector3d>> markerPositions;

  for (size_t frameIndex = 0; frameIndex < markerSequence.size(); ++frameIndex) {
    const float timestamp = static_cast<float>(frameIndex) / static_cast<float>(framerate);
    for (const auto& marker : markerSequence[frameIndex]) {
      // Skip occluded markers
      if (marker.occluded) {
        continue;
      }

      // Create new arrays if marker is unknown
      if (markerNameToIndex.count(marker.name) == 0) {
        const auto index = timestamps.size();
        timestamps.emplace_back();
        markerPositions.emplace_back();
        markerNameToIndex[marker.name] = index;
        markerNames.emplace_back(marker.name);
      }

      // Add timestamp and position for this marker
      const auto& index = markerNameToIndex.at(marker.name);
      MT_THROW_IF(
          index >= timestamps.size() || index >= markerPositions.size(),
          "Marker index {} exceeds container size",
          index);
      timestamps[index].push_back(timestamp);
      markerPositions[index].push_back(marker.pos);
    }
  }

  // Create a marker node for each unique marker name
  for (const auto& markerName : markerNames) {
    // Create FbxMarker attribute for visualization
    ::fbxsdk::FbxMarker* markerAttribute = ::fbxsdk::FbxMarker::Create(scene, markerName.c_str());
    markerAttribute->Look.Set(::fbxsdk::FbxMarker::ELook::eHardCross);

    // Create the node
    ::fbxsdk::FbxNode* markerNode = ::fbxsdk::FbxNode::Create(scene, markerName.c_str());
    markerNode->SetNodeAttribute(markerAttribute);

    // Add custom property to identify this as a Momentum marker
    ::fbxsdk::FbxProperty::Create(markerNode, ::fbxsdk::FbxBoolDT, kMomentumMarkerProperty)
        .Set(true);

    // Initialize at origin
    markerNode->LclTranslation.Set(FbxVector4(0.0, 0.0, 0.0));

    // Add to markers root
    markersRootNode->AddChild(markerNode);
    markerNodes.push_back(markerNode);
  }

  // Add markers root to scene root
  scene->GetRootNode()->AddChild(markersRootNode);

  // Create animation stack if we have motion data
  if (!timestamps.empty() && !timestamps[0].empty()) {
    // Get or create animation stack and layer
    auto [animStack, animBaseLayer] = getOrCreateAnimStackAndLayer(scene, "Marker Animation Stack");

    // Create animation curves for each marker
    for (size_t j = 0; j < markerNames.size(); ++j) {
      if (timestamps[j].empty()) {
        continue;
      }

      auto* markerNode = markerNodes.at(j);

      // Create curves for X, Y, Z translation
      markerNode->LclTranslation.GetCurveNode(animBaseLayer, true);
      ::fbxsdk::FbxAnimCurve* curveX =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      ::fbxsdk::FbxAnimCurve* curveY =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      ::fbxsdk::FbxAnimCurve* curveZ =
          markerNode->LclTranslation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

      // Add keyframes for each timestamp
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

void saveSkinWeightsToFbx(
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
    // TODO: should use inverse bind transform from character instead.
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

// Extract blend shape weights from model parameters.
// Returns a (numBlendShapes x numFrames) matrix with values in [0, 1].
MatrixXf extractBlendShapeWeights(const Character& character, const MatrixXf& poses) {
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

// Extract face expression weights from model parameters.
// Returns a (numFaceExpressions x numFrames) matrix with values in [0, 1].
MatrixXf extractFaceExpressionWeights(const Character& character, const MatrixXf& poses) {
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

// Save blend shape geometry to FBX and return channel pointers for animation.
std::vector<::fbxsdk::FbxBlendShapeChannel*> saveBlendShapeGeometryToFbx(
    const BlendShapeBase& blendShape,
    ::fbxsdk::FbxScene* scene,
    ::fbxsdk::FbxMesh* mesh,
    const std::string& deformerName,
    const std::string& channelPrefix) {
  std::vector<::fbxsdk::FbxBlendShapeChannel*> channels;

  // create blendshape deformer
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

    // add blendshape target
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

// Create animation curves for blend shape channel weights.
// weights: (numChannels x numFrames) matrix where values are in [0, 1] range
void createBlendShapeAnimationCurves(
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxBlendShapeChannel*>& channels,
    const MatrixXf& weights,
    const double framerate) {
  if (channels.empty() || weights.cols() == 0) {
    return;
  }

  setFrameRate(scene, framerate);

  // Get or create animation stack and layer
  auto [animStack, animBaseLayer] =
      getOrCreateAnimStackAndLayer(scene, "BlendShape Animation Stack");

  const Eigen::Index numChannels =
      std::min(static_cast<Eigen::Index>(channels.size()), weights.rows());

  for (Eigen::Index i = 0; i < numChannels; i++) {
    // Animate the DeformPercent property on each channel
    ::fbxsdk::FbxAnimCurve* curve = channels[i]->DeformPercent.GetCurve(animBaseLayer, true);
    if (curve == nullptr) {
      continue;
    }

    curve->KeyModifyBegin();
    for (Eigen::Index f = 0; f < weights.cols(); f++) {
      ::fbxsdk::FbxTime time;
      time.SetSecondDouble(static_cast<double>(f) / framerate);

      // FBX DeformPercent is in [0, 100] range, momentum weights are in [0, 1]
      const float value = weights(i, f) * 100.0f;

      const auto keyIndex = curve->KeyAdd(time);
      curve->KeySet(keyIndex, time, value);
    }
    curve->KeyModifyEnd();
  }
}

void addMetaData(::fbxsdk::FbxNode* skeletonRootNode, const Character& character) {
  // add metadata
  if (skeletonRootNode != nullptr) {
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "metadata")
        .Set(FbxString(character.metadata.c_str()));
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "name")
        .Set(FbxString(character.name.c_str()));
    ::fbxsdk::FbxProperty::Create(skeletonRootNode, ::fbxsdk::FbxStringDT, "RigName")
        .Set(FbxString(character.name.c_str()));
  }
}

void writePolygonsToFbxMesh(const Mesh& mesh, ::fbxsdk::FbxMesh* lMesh) {
  if (!mesh.polyFaces.empty() && !mesh.polyFaceSizes.empty()) {
    // Write original polygon topology
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
    // Fall back to triangulated faces
    for (const auto& face : mesh.faces) {
      lMesh->BeginPolygon();
      for (int i = 0; i < 3; i++) {
        lMesh->AddPolygon(face[i]);
      }
      lMesh->EndPolygon();
    }
  }
}

void writeTextureUVIndicesToFbxMesh(
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

// Result of creating skeleton nodes in the FBX scene.
struct SkeletonNodeResult {
  std::vector<::fbxsdk::FbxNode*> nodes;
  std::unordered_map<size_t, fbxsdk::FbxNode*> jointToNodeMap;
  ::fbxsdk::FbxNode* rootNode = nullptr;
};

// Create skeleton nodes for all joints in the character and set up parenting.
SkeletonNodeResult createSkeletonNodes(const Character& character, ::fbxsdk::FbxScene* scene) {
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

// Result of creating a mesh node with blend shapes in the FBX scene.
struct MeshBlendShapeResult {
  std::vector<::fbxsdk::FbxBlendShapeChannel*> blendShapeChannels;
  std::vector<::fbxsdk::FbxBlendShapeChannel*> faceExprChannels;
};

// Create the mesh node with vertices, normals, UVs, skinning, and blend shapes.
MeshBlendShapeResult createMeshNode(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    ::fbxsdk::FbxNode* root,
    const std::unordered_map<size_t, fbxsdk::FbxNode*>& jointToNodeMap,
    Permissive permissive,
    const filesystem::path& filename) {
  MeshBlendShapeResult result;
  const auto numVertices = static_cast<int>(character.mesh.get()->vertices.size());
  ::fbxsdk::FbxNode* meshNode = ::fbxsdk::FbxNode::Create(scene, "body_mesh");
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
      " Failed to save the character '{}' to {}. The character has no skinning weights and permissive mode is not enabled. Only mesh-only characters are allowed in permissive mode.",
      character.name,
      filename.string());

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

  root->AddChild(meshNode);
  return result;
}

void saveFbxCommon(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointValues,
    const double framerate,
    const bool saveMesh,
    const bool skipActiveJointParamCheck,
    const FbxCoordSystemInfo& coordSystemInfo,
    Permissive permissive,
    std::span<const std::vector<Marker>> markerSequence,
    std::string_view fbxNamespace,
    const MatrixXf& poses) {
  // ---------------------------------------------
  // initialize FBX SDK and prepare for export
  // ---------------------------------------------
  auto* manager = ::fbxsdk::FbxManager::Create();
  auto* ios = ::fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
  manager->SetIOSettings(ios);

  // Create an exporter.
  ::fbxsdk::FbxExporter* lExporter = ::fbxsdk::FbxExporter::Create(manager, "");

  // Declare the path and filename of the file containing the scene.
  // In this case, we are assuming the file is in the same directory as the executable.
  // Going through string() because on windows, wchar_t (native filesystem path) are different from
  // char https://en.cppreference.com/w/cpp/language/types This avoids a build error on windows
  // only.
  std::string sFilename = filename.string();
  const char* lFilename = sFilename.c_str();

  // Initialize the exporter.
  bool lExportStatus = lExporter->Initialize(lFilename, -1, manager->GetIOSettings());

  MT_THROW_IF(
      !lExportStatus,
      "Unable to initialize fbx exporter {}",
      lExporter->GetStatus().GetErrorString());

  // Normalize namespace: ensure it ends with ':' if not empty
  std::string namespacePrefix(fbxNamespace);
  if (!namespacePrefix.empty() && namespacePrefix.back() != ':') {
    namespacePrefix += ":";
  }

  // ---------------------------------------------
  // create the scene
  // ---------------------------------------------
  ::fbxsdk::FbxScene* scene = ::fbxsdk::FbxScene::Create(manager, "momentum_scene");
  ::fbxsdk::FbxNode* root = scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  // set the coordinate system
  ::fbxsdk::FbxAxisSystem axis = toFbx(coordSystemInfo);
  axis.ConvertScene(scene);

  // Create skeleton hierarchy
  auto skeletonResult = createSkeletonNodes(character, scene);
  addMetaData(skeletonResult.rootNode, character);
  createLocatorNodes(character, scene, skeletonResult.nodes);
  createCollisionGeometryNodes(character, scene, skeletonResult.nodes);

  // Create mesh with blend shapes
  MeshBlendShapeResult meshResult;
  if (saveMesh && character.mesh != nullptr) {
    meshResult =
        createMeshNode(character, scene, root, skeletonResult.jointToNodeMap, permissive, filename);
  }

  // Add skeleton to scene root
  if (!skeletonResult.nodes.empty()) {
    root->AddChild(skeletonResult.rootNode);
  }

  // ---------------------------------------------
  // create animation curves if we have motion
  // ---------------------------------------------
  if (jointValues.cols() != 0) {
    if (jointValues.rows() == character.parameterTransform.numJointParameters()) {
      createAnimationCurves(
          character,
          scene,
          skeletonResult.nodes,
          jointValues,
          framerate,
          skipActiveJointParamCheck);
    } else {
      MT_LOGE(
          "Rows of joint values {} do not match joint parameter dimension {} so not saving any motion.",
          jointValues.rows(),
          character.parameterTransform.numJointParameters());
    }
  }

  // ---------------------------------------------
  // create marker nodes and animation
  // ---------------------------------------------
  if (!markerSequence.empty()) {
    createMarkerNodes(scene, markerSequence, framerate);
  }

  // ---------------------------------------------
  // create blend shape animation curves
  // Auto-detect blend shapes from model parameters
  // ---------------------------------------------
  if (saveMesh) {
    const MatrixXf blendShapeWeights = extractBlendShapeWeights(character, poses);
    if (blendShapeWeights.cols() > 0 && !meshResult.blendShapeChannels.empty()) {
      createBlendShapeAnimationCurves(
          scene, meshResult.blendShapeChannels, blendShapeWeights, framerate);
    }
    const MatrixXf faceExpressionWeights = extractFaceExpressionWeights(character, poses);
    if (faceExpressionWeights.cols() > 0 && !meshResult.faceExprChannels.empty()) {
      createBlendShapeAnimationCurves(
          scene, meshResult.faceExprChannels, faceExpressionWeights, framerate);
    }
  }

  // ---------------------------------------------
  // apply namespace prefix to all nodes
  // ---------------------------------------------
  if (!namespacePrefix.empty()) {
    prependNamespaceToAllNodes(scene->GetRootNode(), namespacePrefix);
  }

  // ---------------------------------------------
  // close the fbx exporter
  // ---------------------------------------------

  // finally export the scene
  lExporter->Export(scene);
  lExporter->Destroy();

  // destroy the scene and the manager
  if (scene != nullptr) {
    scene->Destroy();
  }
  manager->Destroy();
}

} // namespace

#endif // MOMENTUM_WITH_FBX_SDK

Character loadFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

MarkerSequence loadFbxMarkerSequence(const filesystem::path& filename, bool stripNamespaces) {
  return loadOpenFbxMarkerSequence(filename, stripNamespaces);
}

MarkerSequence loadFbxMarkerSequence(std::span<const std::byte> inputSpan, bool stripNamespaces) {
  return loadOpenFbxMarkerSequence(inputSpan, stripNamespaces);
}

#ifdef MOMENTUM_WITH_FBX_SDK

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses,
    const VectorXf& identity,
    const double framerate,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  CharacterParameters params;
  if (identity.size() == character.parameterTransform.numJointParameters()) {
    params.offsets = identity;
  } else {
    params.offsets = character.parameterTransform.bindPose();
  }

  CharacterState state;
  MatrixXf jointValues;
  if (poses.cols() > 0) {
    params.pose = poses.col(0);
    state.set(params, character, false, false, false);

    jointValues.resize(state.skeletonState.jointParameters.v.size(), poses.cols());

    jointValues.col(0) = state.skeletonState.jointParameters.v;

    for (Eigen::Index f = 1; f < poses.cols(); f++) {
      params.pose = poses.col(f);
      state.set(params, character, false, false, false);
      jointValues.col(f) = state.skeletonState.jointParameters.v;
    }
  }

  saveFbxCommon(
      filename,
      character,
      jointValues,
      framerate,
      options.mesh,
      false,
      options.coordSystemInfo,
      options.permissive,
      markerSequence,
      options.fbxNamespace,
      poses);
}

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams,
    const double framerate,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  saveFbxCommon(
      filename,
      character,
      jointParams,
      framerate,
      options.mesh,
      true,
      options.coordSystemInfo,
      options.permissive,
      markerSequence,
      options.fbxNamespace,
      MatrixXf());
}

void saveFbxWithSkeletonStates(
    const filesystem::path& filename,
    const Character& character,
    std::span<const SkeletonState> skeletonStates,
    const double framerate,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  const size_t nFrames = skeletonStates.size();
  MatrixXf jointParams(character.parameterTransform.zero().v.size(), nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    jointParams.col(iFrame) =
        skeletonStateToJointParameters(skeletonStates[iFrame], character.skeleton).v;
  }

  saveFbxCommon(
      filename,
      character,
      jointParams,
      framerate,
      options.mesh,
      true,
      options.coordSystemInfo,
      options.permissive,
      markerSequence,
      options.fbxNamespace,
      MatrixXf());
}

void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FileSaveOptions& options) {
  saveFbx(filename, character, MatrixXf(), VectorXf(), 120.0, {}, options);
}

MatrixXf loadFbxBlendShapeWeights(const filesystem::path& filename) {
  // Initialize FBX SDK
  auto* manager = ::fbxsdk::FbxManager::Create();
  auto* ios = ::fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
  manager->SetIOSettings(ios);

  // RAII-like cleanup for FBX SDK objects
  ::fbxsdk::FbxScene* scene = nullptr;
  const auto cleanup = [&]() {
    if (scene != nullptr) {
      scene->Destroy();
    }
    manager->Destroy();
  };

  auto* importer = ::fbxsdk::FbxImporter::Create(manager, "");
  const std::string sFilename = filename.string();
  if (!importer->Initialize(sFilename.c_str(), -1, manager->GetIOSettings())) {
    cleanup();
    return {};
  }

  scene = ::fbxsdk::FbxScene::Create(manager, "scene");
  importer->Import(scene);
  importer->Destroy();

  // Find the animation stack
  if (scene->GetSrcObjectCount<::fbxsdk::FbxAnimStack>() == 0) {
    cleanup();
    return {};
  }
  auto* animStack = scene->GetSrcObject<::fbxsdk::FbxAnimStack>(0);
  if (animStack->GetMemberCount<::fbxsdk::FbxAnimLayer>() == 0) {
    cleanup();
    return {};
  }
  auto* animLayer = animStack->GetMember<::fbxsdk::FbxAnimLayer>(0);

  // Collect all blend shape channels from all meshes
  std::vector<::fbxsdk::FbxBlendShapeChannel*> allChannels;
  for (int iMesh = 0; iMesh < scene->GetSrcObjectCount<::fbxsdk::FbxMesh>(); ++iMesh) {
    auto* mesh = scene->GetSrcObject<::fbxsdk::FbxMesh>(iMesh);
    for (int iDeformer = 0; iDeformer < mesh->GetDeformerCount(::fbxsdk::FbxDeformer::eBlendShape);
         ++iDeformer) {
      auto* blendShape = static_cast<::fbxsdk::FbxBlendShape*>(
          mesh->GetDeformer(iDeformer, ::fbxsdk::FbxDeformer::eBlendShape));
      for (int iChannel = 0; iChannel < blendShape->GetBlendShapeChannelCount(); ++iChannel) {
        allChannels.push_back(blendShape->GetBlendShapeChannel(iChannel));
      }
    }
  }

  if (allChannels.empty()) {
    cleanup();
    return {};
  }

  // Determine the number of frames from the first animated channel
  Eigen::Index numFrames = 0;
  for (auto* channel : allChannels) {
    auto* curve = channel->DeformPercent.GetCurve(animLayer);
    if (curve != nullptr) {
      numFrames = std::max(numFrames, static_cast<Eigen::Index>(curve->KeyGetCount()));
    }
  }

  if (numFrames == 0) {
    cleanup();
    return {};
  }

  // Read weights from animation curves
  const auto numChannels = static_cast<Eigen::Index>(allChannels.size());
  MatrixXf weights = MatrixXf::Zero(numChannels, numFrames);

  for (Eigen::Index i = 0; i < numChannels; i++) {
    auto* curve = allChannels[i]->DeformPercent.GetCurve(animLayer);
    if (curve == nullptr) {
      continue;
    }

    const int keyCount = curve->KeyGetCount();
    for (int k = 0; k < keyCount && k < numFrames; k++) {
      // FBX DeformPercent is in [0, 100], convert to [0, 1]
      weights(i, k) = curve->KeyGetValue(k) / 100.0f;
    }
  }

  cleanup();
  return weights;
}

#else // !MOMENTUM_WITH_FBX_SDK

void saveFbx(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const MatrixXf& /* poses */,
    const VectorXf& /* identity */,
    const double /* framerate */,
    std::span<const std::vector<Marker>> /* markerSequence */,
    const FileSaveOptions& /* options */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

void saveFbxWithJointParams(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const MatrixXf& /* jointParams */,
    const double /* framerate */,
    std::span<const std::vector<Marker>> /* markerSequence */,
    const FileSaveOptions& /* options */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

void saveFbxWithSkeletonStates(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    std::span<const SkeletonState> /* skeletonStates */,
    const double /* framerate */,
    std::span<const std::vector<Marker>> /* markerSequence */,
    const FileSaveOptions& /* options */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

void saveFbxModel(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const FileSaveOptions& /* options */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

MatrixXf loadFbxBlendShapeWeights(const filesystem::path& /* filename */) {
  MT_THROW("Loading blend shape weights requires the full Autodesk FBX SDK.");
}

#endif // MOMENTUM_WITH_FBX_SDK

} // namespace momentum
