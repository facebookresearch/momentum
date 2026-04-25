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
#include "momentum/character/character_state.h"
#include "momentum/io/fbx/fbx_io_internal.h"
#include "momentum/io/fbx/fbx_memory_stream.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"

#include <variant>
#endif // MOMENTUM_WITH_FBX_SDK

namespace momentum {

#ifdef MOMENTUM_WITH_FBX_SDK

namespace {

using namespace fbx_internal; // NOLINT(google-build-using-namespace)

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
    meshResult = createMeshNode(character, scene, root, skeletonResult.jointToNodeMap, permissive);
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
