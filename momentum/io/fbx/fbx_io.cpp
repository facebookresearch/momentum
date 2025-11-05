/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/marker.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/exception.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/fbx/fbx_memory_stream.h"
#include "momentum/io/fbx/openfbx_loader.h"
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

namespace momentum {

namespace {

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EUpVector toFbx(const FBXUpVector upVector) {
  switch (upVector) {
    case FBXUpVector::XAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eXAxis;
    case FBXUpVector::YAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eYAxis;
    case FBXUpVector::ZAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eZAxis;
    default:
      MT_THROW("Unsupported up vector");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EFrontVector toFbx(const FBXFrontVector frontVector) {
  switch (frontVector) {
    case FBXFrontVector::ParityEven:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityEven;
    case FBXFrontVector::ParityOdd:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityOdd;
    default:
      MT_THROW("Unsupported front vector");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem::ECoordSystem toFbx(const FBXCoordSystem coordSystem) {
  switch (coordSystem) {
    case FBXCoordSystem::RightHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eRightHanded;
    case FBXCoordSystem::LeftHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eLeftHanded;
    default:
      MT_THROW("Unsupported coordinate system");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem toFbx(const FBXCoordSystemInfo& coordSystemInfo) {
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
    const std::vector<std::vector<Marker>>& markerSequence,
    const double framerate) {
  std::vector<::fbxsdk::FbxNode*> markerNodes;

  if (markerSequence.empty()) {
    return markerNodes;
  }

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
    // Get or create animation stack
    ::fbxsdk::FbxAnimStack* animStack = nullptr;
    if (scene->GetSrcObjectCount<::fbxsdk::FbxAnimStack>() > 0) {
      // Use existing animation stack (created for skeleton animation)
      animStack = scene->GetSrcObject<::fbxsdk::FbxAnimStack>(0);
    } else {
      // Create new animation stack
      animStack = ::fbxsdk::FbxAnimStack::Create(scene, "Marker Animation Stack");
    }

    // Get or create base layer
    ::fbxsdk::FbxAnimLayer* animBaseLayer = nullptr;
    if (animStack->GetMemberCount<::fbxsdk::FbxAnimLayer>() > 0) {
      animBaseLayer = animStack->GetMember<::fbxsdk::FbxAnimLayer>(0);
    } else {
      animBaseLayer = ::fbxsdk::FbxAnimLayer::Create(scene, "Layer0");
      animStack->AddMember(animBaseLayer);
    }

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

void saveFbxCommon(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointValues,
    const double framerate,
    const bool saveMesh,
    const bool skipActiveJointParamCheck,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive,
    const std::vector<std::vector<Marker>>& markerSequence,
    std::string_view fbxNamespace) {
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

  // set the coordinate system
  ::fbxsdk::FbxAxisSystem axis = toFbx(coordSystemInfo);
  axis.ConvertScene(scene);

  // ---------------------------------------------
  // create the skeleton nodes
  // ---------------------------------------------
  std::vector<::fbxsdk::FbxNode*> skeletonNodes;
  std::unordered_map<size_t, fbxsdk::FbxNode*> jointToNodeMap;

  ::fbxsdk::FbxNode* skeletonRootNode = nullptr;

  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    // create the node
    ::fbxsdk::FbxNode* skeletonNode = ::fbxsdk::FbxNode::Create(scene, joint.name.c_str());
    // create node attribute
    ::fbxsdk::FbxSkeleton* skeletonAttribute =
        ::fbxsdk::FbxSkeleton::Create(scene, joint.name.c_str());

    if (joint.parent == kInvalidIndex) {
      skeletonRootNode = skeletonNode;
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eRoot);
    } else {
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eLimbNode);
    }
    skeletonNode->SetNodeAttribute(skeletonAttribute);
    jointToNodeMap[i] = skeletonNode;

    // set translation offset
    skeletonNode->LclTranslation.Set(FbxDouble3(
        joint.translationOffset[0], joint.translationOffset[1], joint.translationOffset[2]));

    // set pre-rotation
    const auto angles = rotationMatrixToEulerZYX(joint.preRotation.toRotationMatrix());
    skeletonNode->SetPivotState(FbxNode::eSourcePivot, FbxNode::ePivotActive);
    skeletonNode->SetRotationActive(true);
    skeletonNode->SetPreRotation(
        ::fbxsdk::FbxNode::eSourcePivot,
        FbxDouble3(toDeg(angles[2]), toDeg(angles[1]), toDeg(angles[0])));

    // add to list
    skeletonNodes.emplace_back(skeletonNode);
  }

  // Second pass: handle the parenting, in case the parents are not in order
  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    // set parent if it has one
    auto* skeletonNode = jointToNodeMap[i];
    if (joint.parent != kInvalidIndex) {
      skeletonNodes[joint.parent]->AddChild(skeletonNode);
    }
  }

  // ---------------------------------------------
  // create the locator nodes
  // ---------------------------------------------
  createLocatorNodes(character, scene, skeletonNodes);

  // ---------------------------------------------
  // create the collision geometry nodes
  // ---------------------------------------------
  createCollisionGeometryNodes(character, scene, skeletonNodes);

  // ---------------------------------------------
  // create the mesh nodes
  // ---------------------------------------------
  if (saveMesh && character.mesh != nullptr) {
    // Add the mesh
    const int numVertices = character.mesh.get()->vertices.size();
    const int numFaces = character.mesh.get()->faces.size();
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
    // Add polygons to lMesh
    for (int iFace = 0; iFace < numFaces; iFace++) {
      lMesh->BeginPolygon();
      for (int i = 0; i < 3; i++) { // We have tris for models. This could be extended for
                                    // supporting Quads or npoly if needed.
        lMesh->AddPolygon(character.mesh.get()->faces[iFace][i]);
      }
      lMesh->EndPolygon();
    }
    lMesh->BuildMeshEdgeArray();
    meshNode->SetNodeAttribute(lMesh);

    // ---------------------------------------------
    // add texture coordinates
    // ---------------------------------------------
    if (!character.mesh->texcoords.empty()) {
      const fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;

      // Initialize UV set and indices first. Both functions must be called before adding UVs
      // and UV indices.
      lMesh->InitTextureUV(0, uvType);
      lMesh->InitTextureUVIndices(
          ::fbxsdk::FbxLayerElement::EMappingMode::eByPolygonVertex, uvType);

      // Add UVs
      for (const auto& texcoords : character.mesh->texcoords) {
        // flip y back to fbx convention - refer to reading code
        lMesh->AddTextureUV(::fbxsdk::FbxVector2(texcoords[0], 1.0f - texcoords[1]), uvType);
      }

      // Set UV indices for each face. We only have triangles.
      int faceCount = 0;
      for (const auto& texcoords : character.mesh->texcoord_faces) {
        lMesh->SetTextureUVIndex(faceCount, 0, texcoords[0], uvType);
        lMesh->SetTextureUVIndex(faceCount, 1, texcoords[1], uvType);
        lMesh->SetTextureUVIndex(faceCount++, 2, texcoords[2], uvType);
      }
    }

    // ---------------------------------------------
    // create the skinning
    // ---------------------------------------------
    // Add the mesh skinning
    // Momentum skinning is saved in two matrices: index and weight (size numvertices x
    // not-ordered-joints). The index contains the joint index and the weight is the normalized
    // weight the vertex for that joint.

    MT_THROW_IF(
        !permissive && !character.skinWeights,
        " Failed to save the character '{}' to {}. The character has no skinning weights and permissive mode is not enabled. Only mesh-only characters are allowed in permissive mode.",
        character.name,
        filename.string());

    if (character.skinWeights != nullptr) {
      ::fbxsdk::FbxSkin* fbxskin = ::fbxsdk::FbxSkin::Create(scene, "meshskinning");
      fbxskin->SetSkinningType(::fbxsdk::FbxSkin::eLinear);
      fbxskin->SetGeometry(lMesh);
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
      lMesh->AddDeformer(fbxskin);
    }
    // Add the mesh under the root
    root->AddChild(meshNode);
  }

  // ---------------------------------------------
  // add the skeleton to the root
  // ---------------------------------------------
  if (!skeletonNodes.empty()) {
    root->AddChild(skeletonRootNode);
  }

  // ---------------------------------------------
  // create animation curves if we have motion
  // ---------------------------------------------
  if (jointValues.cols() != 0) {
    if (jointValues.rows() == character.parameterTransform.numJointParameters()) {
      createAnimationCurves(
          character, scene, skeletonNodes, jointValues, framerate, skipActiveJointParamCheck);
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

Character loadFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses, // model parameters
    const VectorXf& identity,
    const double framerate,
    const bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive,
    const std::vector<std::vector<Marker>>& markerSequence,
    std::string_view fbxNamespace) {
  CharacterParameters params;
  if (identity.size() == character.parameterTransform.numJointParameters()) {
    params.offsets = identity;
  } else {
    params.offsets = character.parameterTransform.bindPose();
  }

  // first convert model parameters to joint values
  CharacterState state;
  MatrixXf jointValues;
  if (poses.cols() > 0) {
    // Set the initial pose to initialize the state
    params.pose = poses.col(0);
    state.set(params, character, false, false, false);

    // Resize the jointValues matrix based on the size of joint parameters and number of poses
    jointValues.resize(state.skeletonState.jointParameters.v.size(), poses.cols());

    // Store the joint parameters for the initial pose
    jointValues.col(0) = state.skeletonState.jointParameters.v;

    // Iterate through each subsequent pose
    for (Eigen::Index f = 1; f < poses.cols(); f++) {
      // set the current pose
      params.pose = poses.col(f);
      state.set(params, character, false, false, false);
      jointValues.col(f) = state.skeletonState.jointParameters.v;
    }
  }

  // Call the helper function to save FBX file with joint values
  saveFbxCommon(
      filename,
      character,
      jointValues,
      framerate,
      saveMesh,
      false,
      coordSystemInfo,
      permissive,
      markerSequence,
      fbxNamespace);
}

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams,
    const double framerate,
    const bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive,
    const std::vector<std::vector<Marker>>& markerSequence,
    std::string_view fbxNamespace) {
  // Call the helper function to save FBX file with joint values.
  // Set skipActiveJointParamCheck=true to skip the active joint param check as the joint params are
  // passed in directly from user.
  saveFbxCommon(
      filename,
      character,
      jointParams,
      framerate,
      saveMesh,
      true,
      coordSystemInfo,
      permissive,
      markerSequence,
      fbxNamespace);
}

void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive,
    std::string_view fbxNamespace) {
  saveFbx(
      filename,
      character,
      MatrixXf(),
      VectorXf(),
      120.0,
      true,
      coordSystemInfo,
      permissive,
      {},
      fbxNamespace);
}

MarkerSequence loadFbxMarkerSequence(const filesystem::path& filename, bool stripNamespaces) {
  return loadOpenFbxMarkerSequence(filename, stripNamespaces);
}

} // namespace momentum
