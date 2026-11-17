/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/io/fbx/openfbx_loader.h"

#include <algorithm>
#include <numeric>

#ifdef MOMENTUM_WITH_FBX_SDK
#include "momentum/character/skeleton_state.h"
#include "momentum/io/fbx/fbx_builder.h"
#include "momentum/io/fbx/fbx_io_internal.h"
#include "momentum/io/fbx/fbx_memory_stream.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"

#include <variant>
#endif // MOMENTUM_WITH_FBX_SDK

namespace momentum {

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

std::vector<Character> loadFbxCharacters(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacters(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::vector<Character> loadFbxCharacters(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacters(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<std::vector<std::pair<Character, std::vector<MatrixXf>>>, float>
loadFbxCharactersWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharactersWithMotion(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<std::vector<std::pair<Character, std::vector<MatrixXf>>>, float>
loadFbxCharactersWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    Permissive permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharactersWithMotion(
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

std::vector<MatrixXf> joinTakesOnSharedTimeline(
    std::span<const std::vector<MatrixXf>> takesPerCharacter) {
  std::vector<Eigen::Index> takeLengths;
  for (const auto& takes : takesPerCharacter) {
    takeLengths.resize(std::max(takeLengths.size(), takes.size()), 0);
    auto length = takeLengths.begin();
    for (const MatrixXf& take : takes) {
      *length = std::max(*length, take.cols());
      ++length;
    }
  }
  const Eigen::Index numFrames =
      std::accumulate(takeLengths.begin(), takeLengths.end(), Eigen::Index{0});

  std::vector<MatrixXf> joined;
  joined.reserve(takesPerCharacter.size());
  for (const auto& takes : takesPerCharacter) {
    const auto animated =
        std::ranges::find_if(takes, [](const MatrixXf& take) { return take.cols() > 0; });
    if (animated == takes.end()) {
      joined.emplace_back();
      continue;
    }
    VectorXf pose = VectorXf::Zero(animated->rows());
    MatrixXf motion(pose.size(), numFrames);
    Eigen::Index frame = 0;
    auto take = takes.begin();
    for (const Eigen::Index length : takeLengths) {
      const Eigen::Index takeFrames = take != takes.end() ? take->cols() : 0;
      if (takeFrames > 0) {
        MT_THROW_IF(
            take->rows() != pose.size(),
            "The takes of one character have {} and {} joint parameters.",
            take->rows(),
            pose.size());
        motion.middleCols(frame, takeFrames) = *take;
        pose = take->col(takeFrames - 1);
      }
      motion.middleCols(frame + takeFrames, length - takeFrames).colwise() = pose;
      frame += length;
      if (take != takes.end()) {
        ++take;
      }
    }
    joined.push_back(std::move(motion));
  }
  return joined;
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
  FbxBuilder builder;
  builder.addCharacter(character, options);
  builder.addMotion(character, framerate, poses, identity);
  builder.addMarkerSequence(framerate, markerSequence);
  builder.save(filename);
}

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams,
    const double framerate,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  FbxBuilder builder;
  builder.addCharacter(character, options);
  if (jointParams.cols() != 0 &&
      jointParams.rows() != character.parameterTransform.numJointParameters()) {
    MT_LOGE(
        "Rows of joint values {} do not match joint parameter dimension {} so not saving any motion.",
        jointParams.rows(),
        character.parameterTransform.numJointParameters());
  } else {
    builder.addMotionWithJointParams(character, framerate, jointParams);
  }
  builder.addMarkerSequence(framerate, markerSequence);
  builder.save(filename);
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

  FbxBuilder builder;
  builder.addCharacter(character, options);
  builder.addMotionWithJointParams(character, framerate, jointParams);
  builder.addMarkerSequence(framerate, markerSequence);
  builder.save(filename);
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
